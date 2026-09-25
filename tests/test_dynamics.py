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
