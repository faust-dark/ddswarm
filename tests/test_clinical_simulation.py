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
