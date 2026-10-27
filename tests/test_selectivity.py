"""Tests for selectivity_minion + the thermodynamic selectivity math + ADR library."""

import math
from pathlib import Path

from ddswarm.agents import DockingMinion, SelectivityMinion
from ddswarm.agents.base import AgentContext
from ddswarm.chemistry.receptors import canonical_receptor_name, expand_receptor, registry_entry
from ddswarm.chemistry.selectivity import (
    classify_cox_gi,
    classify_gabaa_adr,
    classify_occupancy_adr,
    classify_sexual_adr,
    docking_ki,
    occupancy_at,
    select_adr_profile,
    select_adr_profiles,
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


def test_classify_gabaa_adr_labels():
    # alpha2 binds 100x tighter than alpha1 -> alpha2-selective (reduced sedation)
    label, rationale = classify_gabaa_adr(alpha1_ki=1e-6, alpha2_ki=1e-8)
    assert label == "alpha2_selective"
    assert "reduced sedation" in rationale
    # comparable binding -> non-selective
    label, _ = classify_gabaa_adr(alpha1_ki=1e-7, alpha2_ki=1e-7)
    assert label == "non_selective"
    # alpha1 preferred -> sedation-dominant
    label, _ = classify_gabaa_adr(alpha1_ki=1e-9, alpha2_ki=1e-7)
    assert label == "alpha1_preferring"
    # missing affinity -> insufficient_data
    label, _ = classify_gabaa_adr(alpha1_ki=None, alpha2_ki=1e-7)
    assert label == "insufficient_data"
    # alpha5 amnesia note is appended when occupancy crosses the threshold
    _, rationale = classify_gabaa_adr(alpha1_ki=1e-6, alpha2_ki=1e-8, alpha5_occupancy=0.8)
    assert "amnesia" in rationale


def test_gabaa_registry_entries_and_aliases():
    assert registry_entry("GABRA1")["pdb_id"] == "6X3X"
    assert registry_entry("GABRA2")["pdb_id"] == "9QK0"
    assert registry_entry("GABRA3")["pdb_id"] == "9TNB"
    assert registry_entry("GABRA5")["pdb_id"] == "8BHK"
    assert canonical_receptor_name("alpha2") == "GABRA2"
    assert canonical_receptor_name("GABRA5") == "GABRA5"


def test_selectivity_default_panel_gabaa():
    spec = ProjectSpec(project_id="p", input_type="mixed", objective="o",
                       status="approved", targets=["GABRA2"])
    ctx = AgentContext(project_id="p", run_id="r1", task_id="sel", spec=spec, params={}, inputs={})
    panel = SelectivityMinion._default_panel(ctx)
    assert [p["name"] for p in panel] == ["GABRA2", "GABRA1", "GABRA5"]


def test_selectivity_default_panel_primary_only():
    spec = ProjectSpec(project_id="p", input_type="mixed", objective="o",
                       status="approved", targets=["DRD2"])
    ctx = AgentContext(project_id="p", run_id="r1", task_id="sel", spec=spec, params={}, inputs={})
    panel = SelectivityMinion._default_panel(ctx)
    assert [p["name"] for p in panel] == ["DRD2"]


def test_docking_default_receptor_resolution():
    spec = ProjectSpec(project_id="p", input_type="mixed", objective="o",
                       status="approved", targets=["GABRA2"])
    ctx = AgentContext(project_id="p", run_id="r1", task_id="dock", spec=spec, params={}, inputs={})
    assert DockingMinion._default_receptor(ctx) == {"name": "GABRA2", "pdb_id": "9QK0"}
    # an unregistered target resolves to None (non-blocking gap in the agent)
    spec2 = ProjectSpec(project_id="p", input_type="mixed", objective="o",
                        status="approved", targets=["UNREGISTERED_X"])
    ctx2 = AgentContext(project_id="p", run_id="r1", task_id="dock", spec=spec2, params={}, inputs={})
    assert DockingMinion._default_receptor(ctx2) is None


def test_docking_receptor_source_resolves_pdb_id(tmp_path):
    # _default_receptor returns the registry shape {name, pdb_id}; the run()
    # path must read "pdb_id" (not only "identifier") or the RCSB URL is built
    # from an empty id -> ".pdb" -> PDBFixer "list index out of range".
    store = Store(tmp_path)
    agent = DockingMinion(store)
    spec = ProjectSpec(project_id="p", input_type="mixed", objective="o",
                       status="approved", targets=["GABRA1"])
    ctx = AgentContext(project_id="p", run_id="r1", task_id="dock", spec=spec, params={}, inputs={})
    src = agent._receptor_source(ctx)
    assert src is not None
    assert src.identifier == "6X3X"
    assert src.kind == "pdb_id"


def test_docking_receptor_source_explicit_pdb_id_shape(tmp_path):
    # explicit params['receptor'] using the registry's "pdb_id" key must resolve
    store = Store(tmp_path)
    agent = DockingMinion(store)
    spec = ProjectSpec(project_id="p", input_type="mixed", objective="o",
                       status="approved", targets=[])
    ctx = AgentContext(project_id="p", run_id="r1", task_id="dock", spec=spec,
                       params={"receptor": {"pdb_id": "6X3X"}}, inputs={})
    src = agent._receptor_source(ctx)
    assert src is not None and src.identifier == "6X3X"


def test_docking_receptor_source_unresolved(tmp_path):
    store = Store(tmp_path)
    agent = DockingMinion(store)
    spec = ProjectSpec(project_id="p", input_type="mixed", objective="o",
                       status="approved", targets=["UNREGISTERED_X"])
    ctx = AgentContext(project_id="p", run_id="r1", task_id="dock", spec=spec, params={}, inputs={})
    assert agent._receptor_source(ctx) is None


def test_select_adr_profiles_matches_all_profiles():
    # SSRI sexual-ADR profile
    assert select_adr_profiles("SERT", ["HTR2A", "HTR2C"]) == ["sert_5ht2c_sexual_adr"]
    assert select_adr_profiles("SERT", ["HTR2A"]) == []
    # NSAID COX GI profile (with alias resolution)
    assert select_adr_profiles("COX2", ["COX1"]) == ["cox_gi_risk"]
    assert select_adr_profiles("PTGS2", ["PTGS1"]) == ["cox_gi_risk"]
    # single antitarget profile
    assert select_adr_profiles("SERT", ["CHRM1"]) == ["chrm1_anticholinergic"]
    # multiple antitarget profiles (ordered: registry insertion order)
    assert select_adr_profiles("SERT", ["HRH1", "CHRM1"]) == ["chrm1_anticholinergic", "hrh1_sedation"]
    # a panel can match special + antitarget profiles at once
    assert select_adr_profiles("SERT", ["HTR2C", "CHRM1"]) == [
        "sert_5ht2c_sexual_adr", "chrm1_anticholinergic"]
    # no match
    assert select_adr_profiles(None, []) == []
    assert select_adr_profiles("DRD2", ["HTR2A"]) == []


def test_select_adr_profile_backward_compat():
    assert select_adr_profile("SERT", ["HTR2A", "HTR2C"]) == "sert_5ht2c_sexual_adr"
    assert select_adr_profile("SERT", ["HTR2A"]) is None
    assert select_adr_profile("DRD2", ["HTR2A", "HTR2C"]) is None
    assert select_adr_profile(None, []) is None


def test_classify_cox_gi_labels():
    # SI = Ki_COX1/Ki_COX2 = 1000 -> COX-2 preferred
    assert classify_cox_gi(1e-6, 1e-9)[0] == "cox2_preferential"
    # SI = 1 -> non-selective
    assert classify_cox_gi(1e-7, 1e-7)[0] == "non_selective_nsaid"
    # SI = 0.01 -> COX-1 preferred
    assert classify_cox_gi(1e-9, 1e-7)[0] == "cox1_preferring"
    # missing affinity
    assert classify_cox_gi(None, 1e-7)[0] == "insufficient_data"
    assert classify_cox_gi(1e-7, None)[0] == "insufficient_data"


def test_classify_occupancy_adr_labels():
    # blockade profile: high occupancy -> liability
    assert classify_occupancy_adr("chrm1_anticholinergic", 0.8)[0] == "liability"
    assert classify_occupancy_adr("chrm1_anticholinergic", 0.1)[0] == "low_risk"
    assert classify_occupancy_adr("chrm1_anticholinergic", None)[0] == "insufficient_data"
    # agonism profile: high occupancy -> sign-dependent engagement (not liability)
    assert classify_occupancy_adr("htr2b_valvulopathy", 0.8)[0] == "sign_dependent_engagement"


def test_canonical_receptor_name_aliases():
    assert canonical_receptor_name("PTGS2") == "COX2"
    assert canonical_receptor_name("COX-2") == "COX2"
    assert canonical_receptor_name("M1") == "CHRM1"
    assert canonical_receptor_name("5-HT2C") == "HTR2C"
    assert canonical_receptor_name("5HT2C") == "HTR2C"
    assert canonical_receptor_name("mu") == "OPRM1"
    assert canonical_receptor_name("MOR") == "OPRM1"
    assert canonical_receptor_name("alpha1A") == "ADRA1A"
    assert canonical_receptor_name("SLC6A4") == "SERT"
    assert canonical_receptor_name("hERG") == "KCNH2"
    # unknown names are normalized, never mapped to a fabricated entry
    assert canonical_receptor_name("UNKNOWN_THING") == "UNKNOWNTHING"


def test_expand_receptor_fills_pdb_from_registry():
    assert expand_receptor({"name": "DRD2"})["pdb_id"] == "6CM4"
    assert expand_receptor({"name": "SERT"})["pdb_id"] == "5I73"
    # explicit pdb_id always wins
    assert expand_receptor({"name": "DRD2", "pdb_id": "1ABC"})["pdb_id"] == "1ABC"
    # alias resolution fills a curated pdb_id (COX2 via PTGS2)
    assert expand_receptor({"name": "COX2"})["pdb_id"] == "3LN1"
    assert expand_receptor({"name": "PTGS2"})["pdb_id"] == "3LN1"
    # unregistered name stays untouched (no pdb_id invented)
    assert "pdb_id" not in expand_receptor({"name": "MADE_UP"})
    # registry entry with no curated pdb_id does not fabricate one
    assert "pdb_id" not in expand_receptor({"name": "ADRA2C"})


def test_interpret_adr_ssri_cox_and_antitarget():
    # no profiles -> empty list
    assert SelectivityMinion._interpret_adr([], {"DRD2": 0.1}, {}, 0.9, "CC") == []
    # SSRI profile -> class effect
    r = SelectivityMinion._interpret_adr(
        ["sert_5ht2c_sexual_adr"], {"SERT": 0.9, "HTR2C": 0.0}, {}, 0.9, "CC")
    assert len(r) == 1 and r[0].profile == "sert_5ht2c_sexual_adr" and r[0].label == "class_effect"
    # antitarget -> liability
    r = SelectivityMinion._interpret_adr(
        ["chrm1_anticholinergic"], {"CHRM1": 0.8}, {}, 0.9, "CC")
    assert len(r) == 1 and r[0].profile == "chrm1_anticholinergic" and r[0].label == "liability"
    # cox -> cox2_preferential (needs the ki dict, not occupancy)
    r = SelectivityMinion._interpret_adr(
        ["cox_gi_risk"], {}, {"COX1": 1e-6, "COX2": 1e-9}, 0.9, "CC")
    assert len(r) == 1 and r[0].profile == "cox_gi_risk" and r[0].label == "cox2_preferential"


def _run_selectivity(store, receptors, primary, affinities, monkeypatch):
    """Run the full selectivity agent against a fake deterministic docking engine."""
    from ddswarm import agents
    from ddswarm.docking.base import DockingEngine, DockPose, PreparedLigand, PreparedReceptor

    class Fake(DockingEngine):
        name = "fake"
        version = "1.0"

        def prepare_receptor(self, source, work_dir):
            # docking runs after ALL receptors are prepared (and concurrently),
            # so the fake keys on the receptor object, not on call order
            return PreparedReceptor(identifier=source.identifier)

        def prepare_ligand(self, smiles, work_dir):
            return PreparedLigand(canonical_smiles=smiles, pdbqt_path=Path("l.pdbqt"))

        def available(self):
            return True, "fake"

        def dock(self, receptor, ligand, protocol, work_dir):
            a = affinities.get((receptor.identifier, ligand.canonical_smiles))
            return [DockPose(rank=1, affinity_kcal_per_mol=a)] if a is not None else []

    monkeypatch.setattr(agents.selectivity, "make_engine", lambda name, vina_binary=None: Fake())
    spec = ProjectSpec(project_id="p", input_type="mixed", objective="o", status="approved",
                       starting_materials=["c1ccccc1"])
    params = {"receptors": receptors, "primary": primary, "calibrate": False,
              "box": {"center": [0, 0, 0], "size": [20, 20, 20]}}
    ctx = AgentContext(project_id="p", run_id="r1", task_id="sel", spec=spec,
                       params=params, inputs={})
    return SelectivityMinion(store).run(ctx)


def test_agent_e2e_cox_panel(tmp_path, monkeypatch):
    store = Store(tmp_path)
    res = _run_selectivity(
        store,
        receptors=[{"name": "COX2", "pdb_id": "3LN1"}, {"name": "COX1", "pdb_id": "6Y3C"}],
        primary="COX2",
        affinities={("3LN1", "c1ccccc1"): -9.5, ("6Y3C", "c1ccccc1"): -6.0},
        monkeypatch=monkeypatch)
    assert res.status.value == "success"
    assert any("cox_gi_risk:cox2_preferential" in f for f in res.findings)


def test_agent_e2e_tca_panel(tmp_path, monkeypatch):
    store = Store(tmp_path)
    res = _run_selectivity(
        store,
        receptors=[{"name": "SERT", "pdb_id": "5I73"},
                   {"name": "M1", "pdb_id": "5CXV"},
                   {"name": "H1", "pdb_id": "3RZE"}],
        primary="SERT",
        affinities={("5I73", "c1ccccc1"): -9.0,
                    ("5CXV", "c1ccccc1"): -11.5,
                    ("3RZE", "c1ccccc1"): -11.5},
        monkeypatch=monkeypatch)
    assert res.status.value == "success"
    # alias names M1/H1 canonicalised to CHRM1/HRH1 and interpreted as liabilities
    assert any("chrm1_anticholinergic:liability" in f for f in res.findings)
    assert any("hrh1_sedation:liability" in f for f in res.findings)


def _ctx(params):
    spec = ProjectSpec(project_id="p", input_type="mixed", objective="o", status="approved",
                       starting_materials=["c1ccccc1"])
    return AgentContext(project_id="p", run_id="r1", task_id="sel", spec=spec,
                        params=params, inputs={})


def test_calibration_ligands_static_wins():
    ligs, source = SelectivityMinion._calibration_ligands("SERT", _ctx({}))
    assert source == "static"
    assert ligs  # SERT is in the fixed CALIBRATION_SET


def test_calibration_ligands_on_demand_chembl(monkeypatch):
    from ddswarm import agents
    monkeypatch.setattr(agents.selectivity, "search_target", lambda gene: "CHEMBL217")
    monkeypatch.setattr(agents.selectivity, "receptor_reference_ligands",
                        lambda tid: [{"smiles": "CCN", "ki_molar": 1e-9, "citation": "ref Ki=1 nM"}])
    ligs, source = SelectivityMinion._calibration_ligands("DRD2", _ctx({"chembl_calibration": True}))
    assert source == "chembl"
    assert ligs == [("CCN", 1e-9, "ref Ki=1 nM")]


def test_calibration_ligands_none_when_off():
    # receptor not in the static set + chembl_calibration off -> no calibration
    ligs, source = SelectivityMinion._calibration_ligands("DRD2", _ctx({}))
    assert source == "none" and ligs == []


def test_calibration_ligands_none_on_chembl_failure(monkeypatch):
    from ddswarm import agents
    monkeypatch.setattr(agents.selectivity, "search_target", lambda gene: None)  # no target
    ligs, source = SelectivityMinion._calibration_ligands("DRD2", _ctx({"chembl_calibration": True}))
    assert source == "none" and ligs == []


def test_agent_e2e_chembl_calibration_on_demand(tmp_path, monkeypatch):
    """A receptor absent from the static set is calibrated from ChEMBL on-demand."""
    from ddswarm import agents
    from ddswarm.contracts.selectivity_assessment import SelectivityAssessment
    from ddswarm.docking.base import DockingEngine, DockPose, PreparedLigand, PreparedReceptor

    monkeypatch.setattr(agents.selectivity, "search_target", lambda gene: "CHEMBL217")
    monkeypatch.setattr(agents.selectivity, "receptor_reference_ligands",
                        lambda tid: [{"smiles": "CCN", "ki_molar": 1e-9, "citation": "ref Ki=1 nM"}])

    class Fake(DockingEngine):
        name = "fake"
        version = "1.0"

        def prepare_receptor(self, source, work_dir):
            # docking runs after ALL receptors are prepared (and concurrently),
            # so the fake keys on the receptor object, not on call order
            return PreparedReceptor(identifier=source.identifier)

        def prepare_ligand(self, smiles, work_dir):
            return PreparedLigand(canonical_smiles=smiles, pdbqt_path=Path("l.pdbqt"))

        def available(self):
            return True, "fake"

        def dock(self, receptor, ligand, protocol, work_dir):
            if receptor.identifier == "6CM4":
                if ligand.canonical_smiles == "CCN":  # the on-demand calibration ligand
                    return [DockPose(rank=1, affinity_kcal_per_mol=-9.0)]
                return [DockPose(rank=1, affinity_kcal_per_mol=-8.0)]
            return []

    monkeypatch.setattr(agents.selectivity, "make_engine", lambda name, vina_binary=None: Fake())
    store = Store(tmp_path)
    params = {"receptors": [{"name": "DRD2", "pdb_id": "6CM4"}], "primary": "DRD2",
              "calibrate": True, "chembl_calibration": True,
              "box": {"center": [0, 0, 0], "size": [20, 20, 20]}}
    res = SelectivityMinion(store).run(_ctx(params))
    assert res.status.value == "success"
    art = store.load_artifact(res.artifacts[0].uri, SelectivityAssessment)
    assert art.calibration.n_points["DRD2"] == 1
    assert "ON-DEMAND ChEMBL" in art.calibration.note
