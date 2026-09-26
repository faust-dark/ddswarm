"""Tests for screener_minion: deterministic filters and honest rejection."""

from ddswarm.agents import ScreenerMinion
from ddswarm.agents.base import AgentContext
from ddswarm.contracts import ProjectSpec, ScreenedBatch
from ddswarm.orchestrator import Store

GEFITINIB = "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4"
BENZENE = "c1ccccc1"
RHODANINE = "O=C1CSC(=S)N1"


def _run(store, molecules, filters=None):
    agent = ScreenerMinion(store)
    spec = ProjectSpec(project_id="p1", input_type="mixed", objective="screen", status="approved")
    ctx = AgentContext(
        project_id="p1", run_id="r1", task_id="screen", spec=spec,
        params={"molecules": molecules, "filters": filters or {}},
    )
    return agent.run(ctx)


def _batch(store, result):
    assert result.artifacts, "expected a screened_batch artifact"
    return store.load_artifact(result.artifacts[0].uri, ScreenedBatch)


def test_invalid_smiles(tmp_path):
    store = Store(tmp_path)
    batch = _batch(store, _run(store, [{"id": "bad", "smiles": "CC(C"}]))
    assert batch.summary.invalid == 1
    m = batch.molecules[0]
    assert not m.valid and m.sanitization_error and not m.passed


def test_ro5_rejection(tmp_path):
    store = Store(tmp_path)
    batch = _batch(store, _run(store, [{"id": "big", "smiles": "C" * 45}]))
    m = batch.molecules[0]
    assert m.valid and not m.passed
    assert "MW" in m.rule_of_five_violations
    assert "logP" in m.rule_of_five_violations


def test_pains_rejection(tmp_path):
    store = Store(tmp_path)
    batch = _batch(store, _run(store, [{"id": "pains", "smiles": RHODANINE}]))
    m = batch.molecules[0]
    assert m.valid and not m.passed
    assert any("PAINS:" in a for a in m.structural_alerts)


def test_brenk_advisory_by_default(tmp_path):
    store = Store(tmp_path)
    batch = _batch(store, _run(store, [{"id": "g", "smiles": GEFITINIB}]))
    m = batch.molecules[0]
    # Brenk is advisory: reported, but does not eliminate by default
    assert m.valid and m.passed
    assert any("BRENK:Aliphatic_long_chain" in a for a in m.brenk_alerts)
    assert m.structural_alerts == []  # no hard PAINS alert


def test_brenk_hard_rejection(tmp_path):
    store = Store(tmp_path)
    batch = _batch(store, _run(store, [{"id": "g", "smiles": GEFITINIB}], filters={"brenk_hard": True}))
    m = batch.molecules[0]
    assert m.valid and not m.passed
    assert any("BRENK:Aliphatic_long_chain" in a for a in m.brenk_alerts)
    assert any("Brenk alert(s)" in r for r in m.elimination_reasons)


def test_clean_passes(tmp_path):
    store = Store(tmp_path)
    batch = _batch(store, _run(store, [{"id": "ok", "smiles": BENZENE}]))
    m = batch.molecules[0]
    assert m.valid and m.passed and m.elimination_reasons == []


def test_no_molecules_insufficient(tmp_path):
    store = Store(tmp_path)
    result = _run(store, [])
    assert result.status.value == "insufficient_evidence"


def test_deterministic(tmp_path):
    store = Store(tmp_path)
    b1 = _batch(store, _run(store, [{"id": "x", "smiles": GEFITINIB}]))
    b2 = _batch(store, _run(store, [{"id": "x", "smiles": GEFITINIB}]))
    assert b1.molecules[0].model_dump() == b2.molecules[0].model_dump()
