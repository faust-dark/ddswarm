"""Tests for formulation_minion + deterministic developability rules (offline)."""

from rdkit import Chem

from ddswarm.agents import FormulationMinion
from ddswarm.agents.base import AgentContext
from ddswarm.chemistry.formulation import assess_formulation
from ddswarm.contracts import (
    FormulationAssessment,
    MoleculeScreenResult,
    ProjectSpec,
    ScreenedBatch,
    ScreenSummary,
)
from ddswarm.orchestrator import Store

GEFITINIB = "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4"
BENZOIC_ACID = "O=C(O)c1ccccc1"
MORPHOLINE = "C1COCCN1"


def _spec():
    return ProjectSpec(project_id="p", input_type="mixed", objective="inhibit EGFR",
                       targets=["EGFR"], starting_materials=[GEFITINIB], status="approved")


def _run(store, molecules=None, inputs=None):
    ctx = AgentContext(project_id="p", run_id="r", task_id="formulation", spec=_spec(),
                       params={"molecules": molecules or []}, inputs=inputs or {})
    return FormulationMinion(store).run(ctx)


def _loaded(store, result):
    assert result.artifacts
    return store.load_artifact(result.artifacts[0].uri, FormulationAssessment)


# ---- chemistry rules ----
def test_gefitinib_rules():
    a = assess_formulation(Chem.MolFromSmiles(GEFITINIB))
    assert a["solubility_class"] == "moderate"
    assert a["permeability_class"] == "high"
    assert a["bcs_class"] == "I"
    assert any("basic" in g for g in a["ionizable_groups"])
    assert "basic center" in a["salt_suggestion"]


def test_benzoic_acid_acidic_salt():
    a = assess_formulation(Chem.MolFromSmiles(BENZOIC_ACID))
    assert any("carboxylic acid" in g for g in a["ionizable_groups"])
    assert "acidic center" in a["salt_suggestion"]


def test_morpholine_high_solubility():
    a = assess_formulation(Chem.MolFromSmiles(MORPHOLINE))
    assert a["solubility_class"] == "high"
    assert a["bcs_class"] == "III"


def test_benzene_no_ionizable():
    a = assess_formulation(Chem.MolFromSmiles("c1ccccc1"))
    assert a["ionizable_groups"] == []
    assert a["salt_suggestion"] is None


# ---- agent ----
def test_agent_deterministic(tmp_path):
    store = Store(tmp_path)
    result = _run(store, [{"id": "m0", "smiles": GEFITINIB}])
    assert result.status.value == "success"
    fa = _loaded(store, result)
    assert fa.summary.total_molecules == 1
    assert fa.summary.bcs_counts == {"I": 1}
    m = fa.molecules[0]
    assert m.bcs_class == "I"
    assert m.evidence_level.value == "predicted"
    assert m.note == "rule-based estimate (logP/TPSA heuristics); not measured properties"


def test_agent_consumes_screened_survivors(tmp_path):
    store = Store(tmp_path)
    sb = ScreenedBatch(
        project_id="p", status="success",
        molecules=[MoleculeScreenResult(id="passed", input_smiles="c1ccccc1",
                                        canonical_smiles="c1ccccc1", valid=True, passed=True)],
        summary=ScreenSummary(total=1, passed=1, rejected=0, invalid=0),
    )
    result = _run(store, inputs={"screened_batch": [sb]})
    fa = _loaded(store, result)
    assert fa.summary.total_molecules == 1
    assert fa.molecules[0].id == "passed"


def test_agent_no_molecules_insufficient(tmp_path):
    store = Store(tmp_path)
    spec = ProjectSpec(project_id="p", input_type="mixed", objective="x", status="approved")
    ctx = AgentContext(project_id="p", run_id="r", task_id="formulation", spec=spec,
                       params={"molecules": []}, inputs={})
    result = FormulationMinion(store).run(ctx)
    assert result.status.value == "insufficient_evidence"
