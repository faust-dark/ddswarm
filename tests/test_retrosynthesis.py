"""Tests for retrosynthesis_minion + SA-score chemistry (offline)."""

import json

from rdkit import Chem

from ddswarm.agents import RetrosynthesisMinion
from ddswarm.agents.base import AgentContext
from ddswarm.chemistry.sa_score import classify_accessibility, compute_sa_score
from ddswarm.contracts import ProjectSpec, RetrosynthesisPlan, ScreenedBatch, ScreenSummary, MoleculeScreenResult
from ddswarm.orchestrator import Store

GEFITINIB = "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4"


def _spec():
    return ProjectSpec(project_id="p", input_type="mixed", objective="inhibit EGFR",
                       targets=["EGFR"], starting_materials=["c1ccccc1"], status="approved")


def _run(store, molecules=None, inputs=None, params=None):
    ctx = AgentContext(project_id="p", run_id="r", task_id="retrosynth", spec=_spec(),
                       params=params or {"molecules": molecules or []}, inputs=inputs or {})
    return RetrosynthesisMinion(store).run(ctx)


def _plan(store, result):
    assert result.artifacts
    return store.load_artifact(result.artifacts[0].uri, RetrosynthesisPlan)


# ---- chemistry ---- 
def test_sa_score_and_classification():
    assert compute_sa_score(Chem.MolFromSmiles("c1ccccc1")) == 1.0
    assert classify_accessibility(1.0) == "easy"
    assert classify_accessibility(3.0) == "easy"
    assert classify_accessibility(3.01) == "moderate"
    assert classify_accessibility(6.0) == "moderate"
    assert classify_accessibility(6.01) == "hard"


def test_gefitinib_easy():
    sa = compute_sa_score(Chem.MolFromSmiles(GEFITINIB))
    assert sa < 3.0
    assert classify_accessibility(sa) == "easy"


# ---- agent ----
def test_retrosynth_deterministic(tmp_path):
    store = Store(tmp_path)
    result = _run(store, molecules=[{"id": "m0", "smiles": GEFITINIB}])
    assert result.status.value == "success"
    plan = _plan(store, result)
    assert plan.summary.total_molecules == 1
    assert plan.summary.easy == 1
    e = plan.molecules[0]
    assert e.sa_score > 0
    assert e.evidence_level.value == "calculated"
    assert e.routes == []  # no backend configured


def test_retrosynth_consumes_screened_survivors(tmp_path):
    store = Store(tmp_path)
    sb = ScreenedBatch(
        project_id="p", status="success",
        molecules=[
            MoleculeScreenResult(id="passed", input_smiles="c1ccccc1",
                                 canonical_smiles="c1ccccc1", valid=True, passed=True),
            MoleculeScreenResult(id="rejected", input_smiles="CCO",
                                 canonical_smiles="CCO", valid=True, passed=False),
        ],
        summary=ScreenSummary(total=2, passed=1, rejected=1, invalid=0),
    )
    result = _run(store, inputs={"screened_batch": [sb]})
    plan = _plan(store, result)
    assert plan.summary.total_molecules == 1
    assert plan.molecules[0].id == "passed"


def test_retrosynth_no_molecules_insufficient(tmp_path):
    store = Store(tmp_path)
    spec = ProjectSpec(project_id="p", input_type="mixed", objective="x", status="approved")
    ctx = AgentContext(project_id="p", run_id="r", task_id="retrosynth", spec=spec,
                       params={"molecules": []}, inputs={})
    result = RetrosynthesisMinion(store).run(ctx)
    assert result.status.value == "insufficient_evidence"


def test_retrosynth_with_reasoning_routes(tmp_path):
    payload = [{"molecule_id": "m0", "steps": ["alkylate", "cyclize"],
                "key_reagents": ["K2CO3"], "feasibility_note": "feasible"}]
    script = tmp_path / "fake_hermes.sh"
    script.write_text(f"#!/bin/sh\necho '{json.dumps(payload)}'\n")
    script.chmod(0o755)
    store = Store(tmp_path)
    result = _run(store, molecules=[{"id": "m0", "smiles": GEFITINIB}],
                  params={"molecules": [{"id": "m0", "smiles": GEFITINIB}],
                          "reasoning_backend": "hermes", "reasoning_command": str(script)})
    assert result.status.value == "success"
    plan = _plan(store, result)
    e = plan.molecules[0]
    assert len(e.routes) == 1
    assert e.routes[0].steps == ["alkylate", "cyclize"]
    assert e.routes[0].evidence_level.value == "hypothetical"
    assert e.evidence_level.value == "calculated"  # SA score stays calculated
    assert plan.summary.molecules_with_routes == 1
    assert any("hypothetical" in w for w in result.warnings)
