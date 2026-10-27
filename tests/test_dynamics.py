"""Tests for dynamics_minion + MD engines (offline; OpenMM tests skip if absent)."""

from pathlib import Path

import pytest

from ddswarm.agents import DynamicsMinion
from ddswarm.agents.base import AgentContext
from ddswarm.contracts import DynamicsResult, MdMetrics, MdProtocol, MdRun, ProjectSpec
from ddswarm.dynamics import NullMdEngine, make_engine
from ddswarm.orchestrator import Store

try:
    from ddswarm.dynamics.openmm_engine import _OPENMM_AVAILABLE as _HAVE_OPENMM  # noqa: PLC0412
except Exception:  # noqa: BLE001
    _HAVE_OPENMM = False

FIXTURE = Path(__file__).parent / "fixtures" / "trp_cage.pdb"
requires_openmm = pytest.mark.skipif(not _HAVE_OPENMM, reason="OpenMM/PDBFixer not installed")


def _spec():
    return ProjectSpec(project_id="p", input_type="mixed", objective="inhibit EGFR",
                       targets=["EGFR"], starting_materials=["c1ccccc1"], status="approved")


def _run(store, params, inputs=None):
    ctx = AgentContext(project_id="p", run_id="r", task_id="dynamics", spec=_spec(),
                       params=params, inputs=inputs or {})
    return DynamicsMinion(store).run(ctx)


def _loaded(store, result):
    assert result.artifacts
    return store.load_artifact(result.artifacts[0].uri, DynamicsResult)


# ---- contract ----
def test_dynamics_result_roundtrip():
    r = DynamicsResult(
        project_id="p", status="success",
        runs=[MdRun(receptor_id="2ITY", status="success",
                    metrics=MdMetrics(backbone_rmsd_mean_angstrom=0.5,
                                      potential_energy_mean_kj_per_mol=-2000.0))],
    )
    r2 = DynamicsResult.model_validate_json(r.model_dump_json())
    assert r2.runs[0].metrics.backbone_rmsd_mean_angstrom == 0.5


# ---- engines ----
def test_null_engine_not_run():
    run = NullMdEngine().run_md(Path("x.pdb"), "2ITY", MdProtocol())
    assert run.status == "not_run"
    assert run.metrics.backbone_rmsd_mean_angstrom is None
    assert "no MD engine configured" in run.stability_note


def test_make_engine():
    assert make_engine("null").name == "null"
    assert make_engine("openmm").name == "openmm"
    assert make_engine(None).name == "null"


@requires_openmm
def test_openmm_engine_runs():
    from ddswarm.dynamics.openmm_engine import OpenMMMdEngine

    run = OpenMMMdEngine().run_md(FIXTURE, "trp_cage", MdProtocol(steps=200, seed=1))
    assert run.status == "success"
    assert run.metrics.backbone_rmsd_mean_angstrom is not None
    assert run.metrics.backbone_rmsd_mean_angstrom >= 0.0
    assert run.metrics.potential_energy_std_kj_per_mol is not None
    assert run.evidence_level.value == "calculated"


# ---- agent ----
def test_agent_no_receptor_insufficient(tmp_path):
    store = Store(tmp_path)
    res = _run(store, {"engine": "null"})
    assert res.status.value == "insufficient_evidence"


def test_agent_null_engine_not_run(tmp_path):
    store = Store(tmp_path)
    res = _run(store, {"engine": "null",
                       "receptor": {"kind": "file", "identifier": "trp_cage",
                                    "path": str(FIXTURE)}})
    assert res.status.value == "success"
    r = _loaded(store, res)
    assert r.summary.total_runs == 1
    assert r.runs[0].status == "not_run"
    assert any("no MD engine configured" in w for w in res.warnings)


@requires_openmm
def test_agent_openmm_real(tmp_path):
    store = Store(tmp_path)
    res = _run(store, {"engine": "openmm", "steps": 200, "seed": 1,
                       "receptor": {"kind": "file", "identifier": "trp_cage",
                                    "path": str(FIXTURE)}})
    assert res.status.value == "success"
    r = _loaded(store, res)
    assert r.runs[0].status == "success"
    assert r.runs[0].metrics.backbone_rmsd_mean_angstrom is not None
    assert r.protocol.engine == "openmm"
    assert r.protocol.note == "short relaxation trajectory; NOT production MD"


@requires_openmm
def test_agent_openmm_resume_extends(tmp_path):
    store = Store(tmp_path)
    params = {"engine": "openmm", "steps": 100, "seed": 1,
              "receptor": {"kind": "file", "identifier": "trp_cage",
                           "path": str(FIXTURE)}}
    # fresh run: 100 steps, checkpoint persisted
    res1 = _run(store, dict(params))
    assert res1.status.value == "success"
    r1 = _loaded(store, res1)
    assert r1.runs[0].total_steps == 100 and r1.runs[0].prior_steps == 0
    assert r1.runs[0].checkpoint_uri is not None
    assert r1.runs[0].checkpoint_hash is not None
    # resume: same store + key -> extends by another 100 steps
    res2 = _run(store, dict(params, resume=True))
    assert res2.status.value == "success"
    r2 = _loaded(store, res2)
    assert r2.runs[0].prior_steps == 100
    assert r2.runs[0].total_steps == 200
    assert any("resumed from" in f for f in res2.findings)


# ---- four-branch gate (predictor + docking + tox + formulation) ----
def _branch_inputs():
    from ddswarm.contracts import (
        AdmetPrediction,
        AdmetSummary,
        DockedPose,
        DockingResult,
        FormulationAssessment,
        FormulationSummary,
        MoleculeAdmetResult,
        MoleculeFormulation,
        MoleculeToxAssessment,
        ToxProfile,
        ToxSummary,
    )
    admet = AdmetPrediction(project_id="p", status="success",
                            summary=AdmetSummary(total_molecules=1, predicted=0, not_available=10),
                            molecules=[MoleculeAdmetResult(id="m1", canonical_smiles="c1ccccc1", endpoints=[])])
    docking = DockingResult(project_id="p", status="success",
                            poses=[DockedPose(molecule_id="m1", rank=1)])
    tox = ToxProfile(project_id="p", status="success",
                     summary=ToxSummary(total_alerts=0),
                     molecules=[MoleculeToxAssessment(id="m1", canonical_smiles="c1ccccc1", alerts=[])])
    formulation = FormulationAssessment(
        project_id="p", status="success",
        summary=FormulationSummary(total_molecules=1),
        molecules=[MoleculeFormulation(id="m1", canonical_smiles="c1ccccc1", bcs_class="I")])
    return {
        "admet_prediction": [admet],
        "docking_result": [docking],
        "tox_profile": [tox],
        "formulation_assessment": [formulation],
    }


def test_agent_gate_missing_branches(tmp_path):
    store = Store(tmp_path)
    res = _run(store, {"engine": "null", "gate_branches": True,
                       "receptor": {"kind": "file", "identifier": "trp_cage",
                                    "path": str(FIXTURE)}})
    assert res.status.value == "insufficient_evidence"
    assert any("gated" in w for w in res.warnings)


def test_agent_gate_all_branches_present(tmp_path):
    store = Store(tmp_path)
    res = _run(store, {"engine": "null", "gate_branches": True,
                       "receptor": {"kind": "file", "identifier": "trp_cage",
                                    "path": str(FIXTURE)}},
               inputs=_branch_inputs())
    assert res.status.value == "success"
    r = _loaded(store, res)
    assert r.runs[0].status == "not_run"  # null engine, but gate passed
