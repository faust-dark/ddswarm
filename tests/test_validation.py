"""Tests for validation_minion: reproducibility + consistency checks over a run."""

from ddswarm.agents import (
    DesignerMinion,
    ReportMinion,
    ResearcherMinion,
    ScreenerMinion,
    ValidationMinion,
)
from ddswarm.agents.base import AgentContext
from ddswarm.contracts import (
    AdmetPrediction,
    AdmetSummary,
    ClinicalSimulation,
    ConcentrationPoint,
    EndpointPrediction,
    MoleculeAdmetResult,
    PKParameters,
    PKSimulation,
    ProjectSpec,
    ValidationReport,
)
from ddswarm.orchestrator import Runner, Store, TaskDefinition


def _tasks():
    return [
        TaskDefinition(task_id="research", agent="researcher_minion", params={"seed_entries": []}),
        TaskDefinition(task_id="design", agent="designer_minion", depends_on=["research"]),
        TaskDefinition(task_id="screen", agent="screener_minion", depends_on=["design"]),
        TaskDefinition(task_id="report", agent="report_minion", depends_on=["screen"]),
        TaskDefinition(task_id="validation", agent="validation_minion", depends_on=["report"]),
    ]


def _agents(store):
    return {
        "researcher_minion": ResearcherMinion(store),
        "designer_minion": DesignerMinion(store),
        "screener_minion": ScreenerMinion(store),
        "report_minion": ReportMinion(store),
        "validation_minion": ValidationMinion(store),
    }


def _spec():
    return ProjectSpec(project_id="p", input_type="mixed", objective="inhibit EGFR",
                       targets=["EGFR"], starting_materials=["c1ccccc1"], status="approved")


def _run_pipeline(tmp_path):
    store = Store(tmp_path)
    runner = Runner(store, _agents(store))
    manifest = runner.create_run(_spec(), _tasks())
    final = runner.run(manifest.run_id, project_id="p")
    return store, final


def _load_report(store, result):
    assert result.artifacts
    return store.load_artifact(result.artifacts[0].uri, ValidationReport)


def test_validation_all_checks_pass(tmp_path):
    store, final = _run_pipeline(tmp_path)
    assert final.status.value == "completed"
    res = store.load_latest_agent_result("p", final.run_id, "validation")
    vr = _load_report(store, res)
    assert vr.summary.total_checks == 8
    assert vr.summary.failed == 0
    assert vr.status.value == "success"
    assert vr.reproducibility["n_artifacts_checked"] >= 4


def test_validation_detects_tamper(tmp_path):
    store, final = _run_pipeline(tmp_path)
    # tamper one artifact file on disk
    files = list(store.artifacts_dir("p", final.run_id).glob("*.json"))
    assert files
    files[0].write_text(files[0].read_text() + "\n// tampered")

    ctx = AgentContext(project_id="p", run_id=final.run_id, task_id="validation",
                       spec=store.load_project_spec("p"), params={}, inputs={})
    result = ValidationMinion(store).run(ctx)
    vr = _load_report(store, result)
    hashes = [c for c in vr.checks if c.check_id == "hash_integrity"]
    assert hashes and hashes[0].status == "fail"
    assert result.status.value == "partial"


def test_validation_empty_store_warns(tmp_path):
    store = Store(tmp_path)
    runner = Runner(store, _agents(store))
    manifest = runner.create_run(_spec(), _tasks())
    # do not run the pipeline; validation sees no artifacts
    ctx = AgentContext(project_id="p", run_id=manifest.run_id, task_id="validation",
                       spec=store.load_project_spec("p"), params={}, inputs={})
    result = ValidationMinion(store).run(ctx)
    vr = _load_report(store, result)
    assert vr.summary.total_checks == 1
    assert vr.checks[0].check_id == "no_artifacts"
    assert vr.checks[0].status == "warn"


# ---- mechanistic-constraint sanity (mass balance / monotonicity / justification) ----
def _pk_sim(auc: float, *, cmax: float = 0.028571, curve=None):
    import math as _m
    if curve is None:
        curve = [ConcentrationPoint(time_hours=t, concentration_mg_l=cmax * _m.exp(-0.006 * t))
                 for t in (0.0, 1.0, 2.0, 4.0, 8.0)]
    return PKSimulation(
        molecule_id="m1", canonical_smiles="CCO", dose_mg=100.0,
        model="one_compartment",
        parameters=PKParameters(clearance_ml_min_kg=5.0,
                                volume_of_distribution_l_kg=50.0,
                                bioavailability=1.0),
        half_life_hours=115.5, cmax_mg_l=cmax, auc_mg_h_l=auc,
        concentration_curve=curve,
    )


def _cs(sims):
    return ClinicalSimulation(project_id="p", status="success", simulations=sims)


def test_mechanistic_constraints_accepts_mass_balanced_pk():
    # Cmax*V/(CL*0.06) = 0.028571*50/0.3 = 4.7618
    issues = ValidationMinion._mechanistic_constraints(
        {"clinical_simulation": [_cs([_pk_sim(4.7619)])]})
    assert issues == []


def test_mechanistic_constraints_flags_mass_balance_violation():
    issues = ValidationMinion._mechanistic_constraints(
        {"clinical_simulation": [_cs([_pk_sim(9.99)])]})
    assert any("violates mass balance" in i for i in issues)


def test_mechanistic_constraints_flags_non_monotonic_curve():
    curve = [ConcentrationPoint(time_hours=t, concentration_mg_l=c)
             for t, c in ((0.0, 0.02), (1.0, 0.03), (2.0, 0.01), (4.0, 0.005))]
    issues = ValidationMinion._mechanistic_constraints(
        {"clinical_simulation": [_cs([_pk_sim(4.7619, curve=curve)])]})
    assert any("non-monotonic" in i for i in issues)


def test_mechanistic_constraints_flags_uncited_prediction():
    uncited = MoleculeAdmetResult(
        id="m1", canonical_smiles="CCO",
        endpoints=[EndpointPrediction(endpoint="Clearance_ml_min_kg", model="fake_ml",
                                      model_version="1", prediction_status="predicted",
                                      predicted_value=5.0, citation=None)],
    )
    cited = MoleculeAdmetResult(
        id="m2", canonical_smiles="CC",
        endpoints=[EndpointPrediction(endpoint="LogP", model="rule_admet",
                                      model_version="1", prediction_status="predicted",
                                      predicted_value=1.2, citation="Wildman-Crippen 1999")],
    )
    ap = AdmetPrediction(project_id="p", status="success",
                         summary=AdmetSummary(), molecules=[uncited, cited])
    issues = ValidationMinion._mechanistic_constraints({"admet_prediction": [ap]})
    assert any("without a citable rule" in i and "Clearance" in i for i in issues)
    assert not any("LogP" in i for i in issues)
