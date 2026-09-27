"""Tests for selectivity_minion + the thermodynamic selectivity math."""

import math

from ddswarm.agents import SelectivityMinion
from ddswarm.agents.base import AgentContext
from ddswarm.chemistry.receptors import expand_receptor
from ddswarm.chemistry.selectivity import (
    classify_sexual_adr,
    docking_ki,
    occupancy_at,
    select_adr_profile,
    selectivity_index,
)
from ddswarm.contracts import ProjectSpec
from ddswarm.orchestrator import Store


def test_docking_ki_monotonic_and_identity():
    ki_strong = docking_ki(-10.0)
    ki_weak = docking_ki(-7.0)
    assert ki_strong < ki_weak           # more negative ΔG -> tighter (smaller) Ki
    assert docking_ki(None) is None
    assert math.isclose(docking_ki(0.0), 1.0, rel_tol=1e-9)  # ΔG=0 -> Ki = 1 M


def test_occupancy_saturation_and_monotonicity():
    k = docking_ki(-8.0)
    o = occupancy_at(1e-8, k)
    assert 0.0 <= o <= 1.0
    o_lo = occupancy_at(1e-9, k)
    o_hi = occupancy_at(1e-6, k)
    assert o_lo <= o_hi < 1.0            # monotonic in C, saturates below 1
    assert occupancy_at(1e-8, None) is None
    assert occupancy_at(None, k) is None


def test_selectivity_index():
    assert math.isclose(selectivity_index(1e-6, 1e-9), 1000.0, rel_tol=1e-9)
    assert selectivity_index(None, 1e-9) is None
    assert selectivity_index(1e-6, None) is None


def test_classify_sexual_adr_labels():
    label, _ = classify_sexual_adr(sert_occupancy=0.9, htr2c_occupancy=0.0,
                                   reference_sert_occupancy=0.9)
    assert label == "class_effect"
    label, _ = classify_sexual_adr(sert_occupancy=0.4, htr2c_occupancy=0.0,
                                   reference_sert_occupancy=0.9)
    assert label == "efficacy_tradeoff"
    label, _ = classify_sexual_adr(sert_occupancy=0.9, htr2c_occupancy=0.5,
                                   reference_sert_occupancy=0.9)
    assert label == "direct_5ht2c_engagement"
    label, _ = classify_sexual_adr(sert_occupancy=None, htr2c_occupancy=None,
                                   reference_sert_occupancy=0.9)
    assert label == "insufficient_data"


def test_agent_no_molecules_insufficient(tmp_path):
    store = Store(tmp_path)
    spec = ProjectSpec(project_id="p", input_type="mixed", objective="o", status="approved")
    ctx = AgentContext(project_id="p", run_id="r1", task_id="sel", spec=spec, params={}, inputs={})
    res = SelectivityMinion(store).run(ctx)
    assert res.status.value == "insufficient_evidence"


def test_select_adr_profile_matches_only_ssri_panel():
    assert select_adr_profile("SERT", ["HTR2A", "HTR2C"]) == "sert_5ht2c_sexual_adr"
    assert select_adr_profile("SERT", ["HTR2A"]) is None
    assert select_adr_profile("DRD2", ["HTR2A", "HTR2C"]) is None
    assert select_adr_profile(None, []) is None


def test_expand_receptor_fills_pdb_from_registry():
    assert expand_receptor({"name": "DRD2"})["pdb_id"] == "6CM4"
    assert expand_receptor({"name": "SERT"})["pdb_id"] == "5I73"
    # explicit pdb_id always wins
    assert expand_receptor({"name": "DRD2", "pdb_id": "1ABC"})["pdb_id"] == "1ABC"
    # unregistered name stays untouched (no pdb_id invented)
    assert "pdb_id" not in expand_receptor({"name": "MADE_UP"})
    # registry entry with no curated pdb_id does not fabricate one
    assert "pdb_id" not in expand_receptor({"name": "COX1"})


def test_interpret_adr_only_for_ssri_profile():
    # a non-SSRI panel never gets an ADR label
    assert SelectivityMinion._interpret_adr(None, {"DRD2": 0.1}, 0.9, "CC") is None
    # SSRI profile with negligible 5-HT2C and retained SERT -> class effect
    r = SelectivityMinion._interpret_adr(
        "sert_5ht2c_sexual_adr", {"SERT": 0.9, "HTR2C": 0.0}, 0.9, "CC")
    assert r is not None and r.label == "class_effect"
