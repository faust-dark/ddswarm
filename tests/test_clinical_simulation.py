"""Tests for clinical_minion's PK/PD simulation (ClinicalSimulation artifact)."""

import json
from pathlib import Path

import pytest

from ddswarm.agents import ClinicalMinion
from ddswarm.agents.base import AgentContext
from ddswarm.contracts import ClinicalSimulation, ProjectSpec
from ddswarm.orchestrator import Store


def _spec(**kw):
    kw.setdefault("dose_mg", 10.0)
    return ProjectSpec(project_id="p", input_type="mixed", objective="inhibit X",
                       targets=["X"], starting_materials=["c1ccccc1"], status="approved", **kw)


def _run(store, spec, params):
    ctx = AgentContext(project_id="p", run_id="r", task_id="clinical", spec=spec,
                       params=params, inputs={})
    return ClinicalMinion(store).run(ctx)


def _load(store, result, model, kind):
    arts = [a for a in result.artifacts if a.kind == kind]
    assert arts, f"no artifact of kind {kind!r}"
    return store.load_artifact(arts[0].uri, model)


def _fake_hermes(tmp_path: Path, payload) -> str:
    script = tmp_path / "fake_hermes.sh"
    script.write_text(f"#!/bin/sh\necho '{json.dumps(payload)}'\n")
    script.chmod(0o755)
    return str(script)


def test_pk_simulation_explicit_params(tmp_path):
    store = Store(tmp_path)
    params = {"pk_parameters": [
        {"molecule_id": "starting_material_0", "clearance_ml_min_kg": 12.0,
         "volume_of_distribution_l_kg": 4.5, "bioavailability": 0.9, "source": "DrugBank"},
    ]}
    result = _run(store, _spec(), params)
    assert result.status.value == "success"
    cs = _load(store, result, ClinicalSimulation, "clinical_simulation")
    assert cs.summary.total_molecules == 1
    assert cs.summary.simulated == 1
    assert cs.summary.insufficient_inputs == 0
    s = cs.simulations[0]
    assert s.molecule_id == "starting_material_0"
    assert s.half_life_hours == pytest.approx(4.331, abs=1e-2)
    assert s.cmax_mg_l == pytest.approx(0.0286, abs=1e-3)
    assert s.parameters.source == "user"
    assert s.parameters.source_ref == "DrugBank"
    assert len(s.concentration_curve) == 9  # default sampling times
    assert s.concentration_curve[0].concentration_mg_l == pytest.approx(s.cmax_mg_l, abs=1e-4)


def test_pk_simulation_from_half_life_and_vd(tmp_path):
    store = Store(tmp_path)
    params = {"pk_parameters": [
        {"molecule_id": "starting_material_0", "half_life_hours": 4.0,
         "volume_of_distribution_l_kg": 4.5, "bioavailability": 0.9},
    ]}
    result = _run(store, _spec(), params)
    assert result.status.value == "success"
    cs = _load(store, result, ClinicalSimulation, "clinical_simulation")
    s = cs.simulations[0]
    # CL recovered as ln2/t1/2 * Vd / 0.06 ~= 12.997 mL/min/kg
    assert s.parameters.clearance_ml_min_kg == pytest.approx(12.997, abs=1e-2)
    assert s.half_life_hours == pytest.approx(4.0, abs=1e-2)


def test_pk_simulation_reasoning_recall(tmp_path):
    store = Store(tmp_path)
    payload = [
        {"molecule_id": "starting_material_0", "clearance_ml_min_kg": 12.0,
         "volume_of_distribution_l_kg": 4.5, "bioavailability": 0.9,
         "half_life_hours": None, "source": "DrugBank", "notes": "label value"},
    ]
    params = {"reasoning_backend": "hermes",
              "reasoning_command": _fake_hermes(tmp_path, payload)}
    result = _run(store, _spec(), params)
    assert result.status.value == "success"
    cs = _load(store, result, ClinicalSimulation, "clinical_simulation")
    assert cs.summary.simulated == 1
    assert cs.simulations[0].parameters.source == "inferred"


def test_pk_parameters_backend_method(tmp_path):
    from ddswarm.reasoning import make_reasoning_backend
    payload = [
        {"molecule_id": "m1", "clearance_ml_min_kg": 12.0, "volume_of_distribution_l_kg": 4.5,
         "bioavailability": 0.9, "half_life_hours": None, "source": "DrugBank", "notes": ""},
    ]
    backend = make_reasoning_backend("hermes", command=_fake_hermes(tmp_path, payload))
    res = backend.pk_parameters(_spec(), [("m1", "c1ccccc1")], {})
    assert res.ok
    assert len(res.entries) == 1
    assert res.entries[0]["clearance_ml_min_kg"] == 12.0
    assert res.entries[0]["volume_of_distribution_l_kg"] == 4.5


# ---- model complexity selection (one- vs two-compartment) ----

def _two_comp_params():
    return [{"molecule_id": "m1", "clearance_ml_min_kg": 12.0,
             "central_volume_l_kg": 3.0, "intercompartmental_clearance_ml_min_kg": 10.0,
             "peripheral_volume_l_kg": 7.0, "bioavailability": 0.9, "source": "DrugBank"}]


def test_pk_two_compartment_mechanism_hint(tmp_path):
    store = Store(tmp_path)
    spec = _spec(adme_issue="extensive tissue distribution, biphasic kinetics")
    params = {"molecules": [{"id": "m1", "smiles": "c1ccccc1"}],
              "pk_parameters": _two_comp_params()}
    result = _run(store, spec, params)
    assert result.status.value == "success"
    cs = _load(store, result, ClinicalSimulation, "clinical_simulation")
    s = cs.simulations[0]
    assert s.model == "two_compartment"
    assert s.selection_reason == "mechanism_hint"
    assert s.distribution_half_life_hours == pytest.approx(1.435, abs=0.01)
    assert s.half_life_hours == pytest.approx(16.28, abs=0.1)


def test_pk_default_selection_is_one_compartment(tmp_path):
    store = Store(tmp_path)
    spec = _spec()  # no mechanism/adme hint
    params = {"molecules": [{"id": "m1", "smiles": "c1ccccc1"}],
              "pk_parameters": [{"molecule_id": "m1", "clearance_ml_min_kg": 12.0,
                                 "volume_of_distribution_l_kg": 4.5, "bioavailability": 0.9}]}
    result = _run(store, spec, params)
    cs = _load(store, result, ClinicalSimulation, "clinical_simulation")
    s = cs.simulations[0]
    assert s.model == "one_compartment"
    assert s.selection_reason == "default"
    assert s.selection_metrics is None


def test_pk_data_driven_selects_two_compartment(tmp_path):
    import math

    from ddswarm.chemistry.pkpd import simulate_two_compartment

    # observed data generated from a genuinely biphasic (two-compartment) curve
    r = simulate_two_compartment(dose_mg=10.0, v1_l_kg=3.0, clearance_ml_min_kg=12.0,
                                 intercompartmental_clearance_ml_min_kg=10.0, v2_l_kg=7.0)
    obs_times = [0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0]
    observed = [
        {"molecule_id": "m1", "time_hours": t,
         "concentration_mg_l": r.a_coefficient * math.exp(-r.alpha_per_hour * t)
         + r.b_coefficient * math.exp(-r.beta_per_hour * t)}
        for t in obs_times
    ]
    # both model families are feasible (1-comp via Vd = V1+V2, 2-comp via V1/Q/V2)
    params = {
        "molecules": [{"id": "m1", "smiles": "c1ccccc1"}],
        "pk_parameters": [{"molecule_id": "m1", "clearance_ml_min_kg": 12.0,
                           "volume_of_distribution_l_kg": 10.0,
                           "central_volume_l_kg": 3.0,
                           "intercompartmental_clearance_ml_min_kg": 10.0,
                           "peripheral_volume_l_kg": 7.0, "bioavailability": 1.0}],
        "observed_concentrations": observed,
    }
    result = _run(Store(tmp_path), _spec(), params)
    cs = _load(Store(tmp_path), result, ClinicalSimulation, "clinical_simulation")
    s = cs.simulations[0]
    assert s.model == "two_compartment"
    assert s.selection_reason == "data_driven"
    assert s.selection_metrics is not None
    assert s.selection_metrics.residual_sum_of_squares == 0.0  # observed == 2-comp curve
    assert "perfect fit" in s.selection_metrics.identifiability_note


# ---- PD Emax/Hill ----

def _pd_params(hill=None):
    p = {"molecule_id": "m1", "baseline_effect": 0.0, "max_effect": 100.0,
         "ec50_mg_l": 0.02, "source": "literature"}
    if hill is not None:
        p["hill_coefficient"] = hill
    return [p]


def test_pd_emax(tmp_path):
    store = Store(tmp_path)
    spec = _spec(mechanism_of_action="direct receptor agonist")
    params = {"molecules": [{"id": "m1", "smiles": "c1ccccc1"}],
              "pk_parameters": [{"molecule_id": "m1", "clearance_ml_min_kg": 12.0,
                                 "volume_of_distribution_l_kg": 4.5, "bioavailability": 0.9}],
              "pd_parameters": _pd_params(), "pd_model": "emax"}
    result = _run(store, spec, params)
    cs = _load(store, result, ClinicalSimulation, "clinical_simulation")
    assert cs.summary.pd_simulated == 1
    pd = cs.pd_simulations[0]
    assert pd.model == "emax"
    assert pd.selection_reason == "mechanism_hint"
    assert len(pd.effect_curve) == 9
    # effect saturates toward Emax at high concentration (Cmax ~0.0286 >> EC50 0.02)
    assert pd.effect_curve[0].effect > 50.0


def test_pd_hill(tmp_path):
    store = Store(tmp_path)
    params = {"molecules": [{"id": "m1", "smiles": "c1ccccc1"}],
              "pk_parameters": [{"molecule_id": "m1", "clearance_ml_min_kg": 12.0,
                                 "volume_of_distribution_l_kg": 4.5, "bioavailability": 0.9}],
              "pd_parameters": _pd_params(hill=1.2), "pd_model": "hill"}
    result = _run(store, _spec(), params)
    cs = _load(store, result, ClinicalSimulation, "clinical_simulation")
    pd = cs.pd_simulations[0]
    assert pd.model == "hill"
    assert pd.parameters.hill_coefficient == 1.2


def _pk_params():
    return [{"molecule_id": "m1", "clearance_ml_min_kg": 12.0,
             "volume_of_distribution_l_kg": 4.5, "bioavailability": 0.9}]


def test_pd_indirect_response(tmp_path):
    store = Store(tmp_path)
    spec = _spec(mechanism_of_action="turnover-mediated biomarker inhibition")
    params = {"molecules": [{"id": "m1", "smiles": "c1ccccc1"}],
              "pk_parameters": _pk_params(),
              "pd_parameters": [{"molecule_id": "m1", "baseline_effect": 100.0,
                                 "kout_per_hour": 0.5, "imax": 0.8, "ic50_mg_l": 0.02}]}
    result = _run(store, spec, params)
    cs = _load(store, result, ClinicalSimulation, "clinical_simulation")
    assert cs.summary.pd_simulated == 1
    pd = cs.pd_simulations[0]
    assert pd.model == "indirect_response"
    assert pd.selection_reason == "mechanism_hint"
    assert pd.parameters.kout_per_hour == 0.5
    assert pd.parameters.imax == 0.8
    # response starts at baseline and is inhibited below it
    assert pd.effect_curve[0].effect == pytest.approx(100.0, abs=0.01)
    assert pd.effect_curve[-1].effect < 100.0


def test_pd_effect_compartment(tmp_path):
    store = Store(tmp_path)
    spec = _spec(mechanism_of_action="delayed onset of action")
    params = {"molecules": [{"id": "m1", "smiles": "c1ccccc1"}],
              "pk_parameters": _pk_params(),
              "pd_parameters": [{"molecule_id": "m1", "baseline_effect": 0.0,
                                 "max_effect": 100.0, "ec50_mg_l": 0.02,
                                 "ke0_per_hour": 0.8}]}
    result = _run(store, spec, params)
    cs = _load(store, result, ClinicalSimulation, "clinical_simulation")
    pd = cs.pd_simulations[0]
    assert pd.model == "effect_compartment"
    assert pd.parameters.ke0_per_hour == 0.8
    assert pd.effect_curve[0].effect == pytest.approx(0.0, abs=0.01)  # delayed


def test_pd_receptor_occupancy(tmp_path):
    store = Store(tmp_path)
    spec = _spec(mechanism_of_action="target engagement at the receptor")
    params = {"molecules": [{"id": "m1", "smiles": "c1ccccc1"}],
              "pk_parameters": _pk_params(),
              "pd_parameters": [{"molecule_id": "m1", "kd_mg_l": 0.01}]}
    result = _run(store, spec, params)
    cs = _load(store, result, ClinicalSimulation, "clinical_simulation")
    pd = cs.pd_simulations[0]
    assert pd.model == "receptor_occupancy"
    assert pd.parameters.kd_mg_l == 0.01
    # fractional occupancy in [0, 1)
    assert all(0.0 <= pt.effect < 1.0 for pt in pd.effect_curve)


def test_pd_per_molecule_override(tmp_path):
    store = Store(tmp_path)
    params = {"molecules": [{"id": "a", "smiles": "c1ccccc1"},
                            {"id": "b", "smiles": "c1ccccc1"}],
              "pk_parameters": [{"molecule_id": "a", "clearance_ml_min_kg": 12.0,
                                 "volume_of_distribution_l_kg": 4.5, "bioavailability": 0.9},
                                {"molecule_id": "b", "clearance_ml_min_kg": 12.0,
                                 "volume_of_distribution_l_kg": 4.5, "bioavailability": 0.9}],
              "pd_parameters": [{"molecule_id": "a", "max_effect": 100.0, "ec50_mg_l": 0.02},
                                {"molecule_id": "b", "kd_mg_l": 0.01}],
              "pd_model": {"a": "emax", "b": "receptor_occupancy"}}
    result = _run(store, _spec(), params)
    cs = _load(store, result, ClinicalSimulation, "clinical_simulation")
    assert cs.summary.pd_simulated == 2
    models = {pd.molecule_id: pd.model for pd in cs.pd_simulations}
    assert models == {"a": "emax", "b": "receptor_occupancy"}


# ---- multi-compartment + tissue (organ) models ----

def test_pk_multi_compartment_explicit(tmp_path):
    store = Store(tmp_path)
    params = {"molecules": [{"id": "m1", "smiles": "c1ccccc1"}],
              "pk_parameters": [{"molecule_id": "m1", "clearance_ml_min_kg": 12.0,
                                 "central_volume_l_kg": 3.0,
                                 "peripheral_compartments": [
                                     {"volume_l_kg": 7.0, "intercompartmental_clearance_ml_min_kg": 10.0},
                                     {"volume_l_kg": 5.0, "intercompartmental_clearance_ml_min_kg": 4.0}],
                                 "bioavailability": 1.0}],
              "pk_model": "multi_compartment"}
    result = _run(store, _spec(), params)
    cs = _load(store, result, ClinicalSimulation, "clinical_simulation")
    s = cs.simulations[0]
    assert s.model == "multi_compartment"
    assert s.selection_reason == "mechanism_hint"
    assert len(s.parameters.peripheral_compartments) == 2
    assert s.half_life_hours == pytest.approx(21.4, abs=0.5)
    assert s.cmax_mg_l == pytest.approx(10.0 / (3.0 * 70.0), abs=1e-4)
    assert s.tissue is None  # no tissue hint


def test_tissue_well_stirred_hint(tmp_path):
    store = Store(tmp_path)
    spec = _spec(adme_issue="rapid well-stirred tissue equilibrium")
    params = {"molecules": [{"id": "m1", "smiles": "c1ccccc1"}],
              "pk_parameters": [{"molecule_id": "m1", "clearance_ml_min_kg": 12.0,
                                 "volume_of_distribution_l_kg": 4.5, "bioavailability": 0.9,
                                 "tissue_partition_coefficient": 3.0}]}
    result = _run(store, spec, params)
    cs = _load(store, result, ClinicalSimulation, "clinical_simulation")
    s = cs.simulations[0]
    assert s.model == "one_compartment"  # systemic model unchanged
    assert s.tissue is not None
    assert s.tissue.model == "well_stirred"
    assert s.tissue.parameters.tissue_partition_coefficient == 3.0
    # Ct(t) = Kp * C_plasma(t) (both rounded independently -> tolerance for rounding)
    assert s.tissue.tissue_curve[0].concentration_mg_l == pytest.approx(
        3.0 * s.concentration_curve[0].concentration_mg_l, abs=2e-4)


def test_tissue_blood_flow_limited_hint(tmp_path):
    store = Store(tmp_path)
    spec = _spec(adme_issue="perfusion-limited tissue uptake")
    params = {"molecules": [{"id": "m1", "smiles": "c1ccccc1"}],
              "pk_parameters": [{"molecule_id": "m1", "clearance_ml_min_kg": 12.0,
                                 "volume_of_distribution_l_kg": 4.5, "bioavailability": 0.9,
                                 "tissue_partition_coefficient": 3.0,
                                 "tissue_blood_flow_ml_min_kg": 20.0,
                                 "tissue_volume_l_kg": 1.0}]}
    result = _run(store, spec, params)
    cs = _load(store, result, ClinicalSimulation, "clinical_simulation")
    s = cs.simulations[0]
    assert s.tissue is not None
    assert s.tissue.model == "blood_flow_limited"
    assert s.tissue.parameters.tissue_blood_flow_ml_min_kg == 20.0


def test_tissue_permeability_limited_hint(tmp_path):
    store = Store(tmp_path)
    spec = _spec(adme_issue="membrane permeability barrier")
    params = {"molecules": [{"id": "m1", "smiles": "c1ccccc1"}],
              "pk_parameters": [{"molecule_id": "m1", "clearance_ml_min_kg": 12.0,
                                 "volume_of_distribution_l_kg": 4.5, "bioavailability": 0.9,
                                 "tissue_partition_coefficient": 3.0,
                                 "tissue_volume_l_kg": 1.0,
                                 "permeability_surface_product_ml_min_kg": 5.0}]}
    result = _run(store, spec, params)
    cs = _load(store, result, ClinicalSimulation, "clinical_simulation")
    s = cs.simulations[0]
    assert s.tissue is not None
    assert s.tissue.model == "permeability_limited"
    assert s.tissue.parameters.permeability_surface_product_ml_min_kg == 5.0
