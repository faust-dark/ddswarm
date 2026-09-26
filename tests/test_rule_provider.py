"""Tests for RuleBasedAdmetProvider (deterministic RDKit descriptor rules)."""

import pytest

from ddswarm.agents import PredictorMinion
from ddswarm.agents.base import AgentContext
from ddswarm.contracts import AdmetPrediction, ProjectSpec
from ddswarm.orchestrator import Store
from ddswarm.predictors import RuleBasedAdmetProvider

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
HEXADECANE = "CCCCCCCCCCCCCCCC"
SUCROSE = "C(C1C(C(C(C(O1)OC2(C(C(C(O2)CO)O)O)CO)O)O)O)O"

PROVIDER = RuleBasedAdmetProvider()


def _by_endpoint(smiles):
    preds = PROVIDER.predict(smiles)
    return {p.endpoint: p for p in preds}


def test_all_endpoints_present_and_cited():
    preds = PROVIDER.predict(ASPIRIN)
    assert {p.endpoint for p in preds} == set(PROVIDER.supported_endpoints)
    assert len(preds) == 5
    for p in preds:
        assert p.model == "rule_admet"
        assert p.prediction_status == "predicted"
        assert p.evidence_level.value == "predicted"
        assert p.citation  # every rule carries a literature source


def test_esol_aspirin():
    p = _by_endpoint(ASPIRIN)["LogS_ESOL"]
    assert p.predicted_value == pytest.approx(-1.992, abs=1e-2)
    assert p.units == "log10(mol/L)"
    assert p.uncertainty.type == "std"
    assert p.uncertainty.value == pytest.approx(0.75)


def test_qed_aspirin():
    p = _by_endpoint(ASPIRIN)["QED"]
    assert p.predicted_value == pytest.approx(0.550, abs=1e-2)
    assert 0.0 <= p.predicted_value <= 1.0


def test_lipinski_aspirin_zero():
    p = _by_endpoint(ASPIRIN)["Lipinski_Violations"]
    assert p.predicted_value == 0.0
    assert p.predicted_label == "0 violations"


def test_good_permeability_rules_aspirin():
    assert _by_endpoint(ASPIRIN)["Veber_Bioavailability"].predicted_label == "good"
    assert _by_endpoint(ASPIRIN)["Egan_Absorption"].predicted_label == "good"


def test_poor_rules_hexadecane():
    p = _by_endpoint(HEXADECANE)
    assert p["Veber_Bioavailability"].predicted_label == "poor"  # 13 rotatable bonds
    assert p["Egan_Absorption"].predicted_label == "poor"        # clogP ~6.5
    assert p["Lipinski_Violations"].predicted_value >= 1.0        # clogP > 5


def test_polar_molecule_sucrose():
    p = _by_endpoint(SUCROSE)
    assert p["Lipinski_Violations"].predicted_value == 2.0  # HBD>5 and HBA>10
    assert p["Veber_Bioavailability"].predicted_label == "poor"   # TPSA ~189
    assert p["Egan_Absorption"].predicted_label == "poor"         # TPSA > 131.6


def test_out_of_domain_invalid_smiles():
    preds = PROVIDER.predict("not_a_smiles")
    assert len(preds) == 5
    assert all(p.prediction_status == "out_of_domain" for p in preds)
    assert all(p.applicability_domain.status == "out_of_domain" for p in preds)


def test_deterministic():
    a = [p.model_dump() for p in PROVIDER.predict(ASPIRIN)]
    b = [p.model_dump() for p in PROVIDER.predict(ASPIRIN)]
    assert a == b


def test_predictor_minion_integration(tmp_path):
    store = Store(tmp_path)
    spec = ProjectSpec(project_id="p", input_type="mixed", objective="inhibit X",
                       targets=["X"], starting_materials=[ASPIRIN], status="approved")
    ctx = AgentContext(project_id="p", run_id="r", task_id="predict", spec=spec,
                       params={"providers": ["rule_admet"]}, inputs={})
    result = PredictorMinion(store).run(ctx)
    assert result.status.value == "success"
    batch = store.load_artifact(result.artifacts[0].uri, AdmetPrediction)
    assert batch.summary.total_molecules == 1
    assert batch.summary.total_endpoints == 5
    assert batch.summary.predicted == 5
    assert batch.summary.not_available == 0
