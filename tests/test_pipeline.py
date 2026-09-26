"""Integration test: research -> design -> screen."""

from rdkit import Chem

from ddswarm.agents import DesignerMinion, PredictorMinion, ResearcherMinion, ScreenerMinion
from ddswarm.contracts import AdmetPrediction, ProjectSpec, RunStatus, ScreenedBatch
from ddswarm.orchestrator import Runner, Store, TaskDefinition

GEFITINIB = "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4"


def _agents(store):
    return {
        "researcher_minion": ResearcherMinion(store),
        "designer_minion": DesignerMinion(store),
        "screener_minion": ScreenerMinion(store),
        "predictor_minion": PredictorMinion(store),
    }


def test_research_design_screen(tmp_path):
    store = Store(tmp_path)
    runner = Runner(store, _agents(store))
    spec = ProjectSpec(
        project_id="p1", input_type="mixed", objective="inhibit EGFR",
        targets=["EGFR"], starting_materials=[GEFITINIB], status="approved",
    )
    tasks = [
        TaskDefinition(task_id="research", agent="researcher_minion", params={"seed_entries": [{
            "id": "ref", "type": "reference", "source": "PubChem",
            "url": "https://pubchem.ncbi.nlm.nih.gov/compound/123631",
            "identifier": "CID 123631", "observed_value": "gefitinib: approved reversible EGFR TKI",
            "evidence_level": "observed",
        }]}),
        TaskDefinition(task_id="design", agent="designer_minion", depends_on=["research"]),
        TaskDefinition(task_id="screen", agent="screener_minion", depends_on=["design"]),
    ]
    manifest = runner.create_run(spec, tasks)
    final = runner.run(manifest.run_id, project_id="p1")
    assert final.status == RunStatus.completed

    res = store.load_latest_agent_result("p1", manifest.run_id, "screen")
    assert res.status.value == "success"
    batch = store.load_artifact(res.artifacts[0].uri, ScreenedBatch)
    # designer now generates real analogs of gefitinib (Cl->F/Br/I, methyl->ethyl):
    # each is a genuinely NEW structure, all valid. Brenk is advisory by default:
    # all pass, carrying Aliphatic_long_chain advisories.
    parent = Chem.MolToSmiles(Chem.MolFromSmiles(GEFITINIB))
    assert batch.summary.total >= 1
    assert all(m.valid for m in batch.molecules)
    assert batch.summary.invalid == 0
    assert batch.summary.passed >= 1  # most analogs pass; the iodine analog (MW>500) is rejected
    assert batch.summary.passed + batch.summary.rejected == batch.summary.total
    assert all(m.canonical_smiles != parent for m in batch.molecules)


def test_research_design_screen_predict(tmp_path):
    store = Store(tmp_path)
    runner = Runner(store, _agents(store))
    spec = ProjectSpec(
        project_id="p1", input_type="mixed", objective="inhibit EGFR",
        targets=["EGFR"], starting_materials=[GEFITINIB], status="approved",
    )
    tasks = [
        TaskDefinition(task_id="research", agent="researcher_minion", params={"seed_entries": [{
            "id": "ref", "type": "reference", "source": "PubChem",
            "url": "https://pubchem.ncbi.nlm.nih.gov/compound/123631",
            "identifier": "CID 123631", "observed_value": "gefitinib: approved reversible EGFR TKI",
            "evidence_level": "observed",
        }]}),
        TaskDefinition(task_id="design", agent="designer_minion", depends_on=["research"]),
        # Brenk is advisory by default, so gefitinib passes screening and flows to prediction
        TaskDefinition(task_id="screen", agent="screener_minion", depends_on=["design"]),
        TaskDefinition(task_id="predict", agent="predictor_minion", depends_on=["screen"],
                       params={"providers": ["null_admet"]}),
    ]
    manifest = runner.create_run(spec, tasks)
    final = runner.run(manifest.run_id, project_id="p1")
    assert final.status == RunStatus.completed

    res = store.load_latest_agent_result("p1", manifest.run_id, "predict")
    assert res.status.value == "success"
    batch = store.load_artifact(res.artifacts[0].uri, AdmetPrediction)
    # every gefitinib analog passes screening (Brenk advisory) and flows to the
    # null provider: 10 not_available endpoints per molecule.
    assert batch.summary.total_molecules >= 1
    assert batch.summary.total_endpoints == 10 * batch.summary.total_molecules
    assert batch.summary.not_available == batch.summary.total_endpoints
    assert batch.summary.predicted == 0


def test_screen_soft_gate_falls_back_to_starting_material(tmp_path):
    store = Store(tmp_path)
    runner = Runner(store, _agents(store))
    spec = ProjectSpec(
        project_id="p1", input_type="mixed", objective="inhibit EGFR",
        targets=["EGFR"], starting_materials=["c1ccccc1"], status="approved",
    )
    tasks = [
        TaskDefinition(task_id="design", agent="designer_minion"),
        # max_mw=1 rejects every molecule, so 0 pass screening (soft gate)
        TaskDefinition(task_id="screen", agent="screener_minion", depends_on=["design"],
                       params={"filters": {"max_mw": 1.0}}),
        TaskDefinition(task_id="predict", agent="predictor_minion", depends_on=["screen"],
                       params={"providers": ["null_admet"]}),
    ]
    manifest = runner.create_run(spec, tasks)
    final = runner.run(manifest.run_id, project_id="p1")
    assert final.status == RunStatus.completed

    sres = store.load_latest_agent_result("p1", manifest.run_id, "screen")
    sbatch = store.load_artifact(sres.artifacts[0].uri, ScreenedBatch)
    assert sbatch.summary.passed == 0

    # downstream is not blocked: it falls back to the starting material
    pres = store.load_latest_agent_result("p1", manifest.run_id, "predict")
    pbatch = store.load_artifact(pres.artifacts[0].uri, AdmetPrediction)
    assert pbatch.summary.total_molecules == 1
