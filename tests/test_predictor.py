"""Tests for predictor_minion and the ADMET provider abstraction."""

from ddswarm.agents import PredictorMinion
from ddswarm.agents.base import AgentContext
from ddswarm.contracts import AdmetPrediction, ProjectSpec
from ddswarm.orchestrator import Store
from ddswarm.predictors import NullAdmetProvider

GEFITINIB = "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4"


def _run(store, molecules=None, providers=None):
    agent = PredictorMinion(store)
    spec = ProjectSpec(project_id="p1", input_type="mixed", objective="predict", status="approved")
    params = {"molecules": molecules or []}
    if providers is not None:
        params["providers"] = providers
    ctx = AgentContext(project_id="p1", run_id="r1", task_id="predict", spec=spec, params=params)
    return agent.run(ctx)


def _batch(store, result):
    assert result.artifacts
    return store.load_artifact(result.artifacts[0].uri, AdmetPrediction)


def test_null_provider_returns_not_available():
    p = NullAdmetProvider()
    preds = p.predict("c1ccccc1")
    assert len(preds) == len(NullAdmetProvider.supported_endpoints)
    assert all(e.prediction_status == "not_available" for e in preds)
    assert all(e.predicted_value is None for e in preds)
    assert all(e.evidence_level.value == "predicted" for e in preds)


def test_predict_success_all_not_available(tmp_path):
    store = Store(tmp_path)
    result = _run(store, molecules=[{"id": "x", "smiles": "c1ccccc1"}], providers=["null_admet"])
    assert result.status.value == "success"
    batch = _batch(store, result)
    assert batch.summary.total_molecules == 1
    assert batch.summary.total_endpoints == len(NullAdmetProvider.supported_endpoints)
    assert batch.summary.not_available == batch.summary.total_endpoints
    assert batch.summary.predicted == 0


def test_no_provider_insufficient(tmp_path):
    store = Store(tmp_path)
    result = _run(store, molecules=[{"id": "x", "smiles": "c1ccccc1"}], providers=[])
    assert result.status.value == "insufficient_evidence"


def test_no_molecules_insufficient(tmp_path):
    store = Store(tmp_path)
    result = _run(store, providers=["null_admet"])
    assert result.status.value == "insufficient_evidence"


def test_unknown_provider_failed(tmp_path):
    store = Store(tmp_path)
    result = _run(store, molecules=[{"id": "x", "smiles": "c1ccccc1"}], providers=["does_not_exist"])
    assert result.status.value == "failed"


def test_deterministic(tmp_path):
    store = Store(tmp_path)
    r1 = _run(store, molecules=[{"id": "x", "smiles": GEFITINIB}], providers=["null_admet"])
    r2 = _run(store, molecules=[{"id": "x", "smiles": GEFITINIB}], providers=["null_admet"])
    b1 = _batch(store, r1)
    b2 = _batch(store, r2)
    assert b1.molecules[0].model_dump() == b2.molecules[0].model_dump()
