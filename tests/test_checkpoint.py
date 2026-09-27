"""Tests for the content-addressed compute checkpoint / resume (Increment 27)."""

from pathlib import Path

from ddswarm.contracts.dynamics_result import MdProtocol, MdRun
from ddswarm.docking.base import (
    DockPose,
    DockProtocol,
    DockingEngine,
    PreparedLigand,
    PreparedReceptor,
)
from ddswarm.orchestrator import Store
from ddswarm.orchestrator.checkpoint import (
    ComputeCache,
    cached_dock,
    cached_run_md,
    poses_from_cache,
    poses_to_cache,
)


class FakeDockEngine(DockingEngine):
    name = "fake"
    version = "1.0"

    def __init__(self):
        self.calls = 0

    def prepare_receptor(self, source, work_dir):
        return PreparedReceptor()

    def prepare_ligand(self, smiles, work_dir):
        return PreparedLigand(canonical_smiles=smiles)

    def available(self):
        return True, "fake"

    def dock(self, receptor, ligand, protocol, work_dir):
        self.calls += 1
        return [DockPose(rank=1, affinity_kcal_per_mol=-8.0,
                         rmsd_lower_bound=0.0, rmsd_upper_bound=0.0,
                         pose_atoms={1: (1.0, 2.0, 3.0)})]


class FakeMdEngine:
    name = "fake"
    version = "1.0"

    def __init__(self):
        self.calls = 0

    def run_md(self, pdb_path, receptor_id, protocol):
        self.calls += 1
        return MdRun(receptor_id=receptor_id, status="success")


def _protocol():
    return DockProtocol(box_center=(0.0, 0.0, 0.0), box_size=(20.0, 20.0, 20.0),
                        exhaustiveness=8, num_modes=9, seed=42)


def test_cache_roundtrip_and_disable(tmp_path):
    cache = ComputeCache(Store(tmp_path), enabled=True)
    assert cache.get("dock", "k1") is None
    cache.put("dock", "k1", [{"a": 1}])
    assert cache.get("dock", "k1") == [{"a": 1}]
    # disabled cache never hits and never writes
    off = ComputeCache(Store(tmp_path), enabled=False)
    assert off.get("dock", "k1") is None
    off.put("dock", "k2", [{"b": 2}])
    assert off.get("dock", "k2") is None


def test_poses_roundtrip():
    poses = [DockPose(rank=1, affinity_kcal_per_mol=-8.0,
                      rmsd_lower_bound=0.5, rmsd_upper_bound=1.5,
                      pose_atoms={7: (1.0, 2.0, 3.0)})]
    restored = poses_from_cache(poses_to_cache(poses))
    assert restored[0].affinity_kcal_per_mol == -8.0
    assert restored[0].rmsd_lower_bound == 0.5
    assert restored[0].pose_atoms == {7: (1.0, 2.0, 3.0)}
    assert restored[0].pose_file is None  # work_dir path is not portable


def test_cached_dock_reuses_on_second_call(tmp_path):
    store = Store(tmp_path)
    cache = ComputeCache(store, enabled=True)
    engine = FakeDockEngine()
    rec = PreparedReceptor()
    lig = PreparedLigand(canonical_smiles="CC")
    proto = _protocol()
    out1 = cached_dock(engine, cache, "5I73", rec, lig, proto, Path(tmp_path) / "w")
    out2 = cached_dock(engine, cache, "5I73", rec, lig, proto, Path(tmp_path) / "w")
    assert engine.calls == 1
    assert out1[0].affinity_kcal_per_mol == out2[0].affinity_kcal_per_mol == -8.0


def test_cached_dock_recomputes_when_disabled(tmp_path):
    cache = ComputeCache(Store(tmp_path), enabled=False)
    engine = FakeDockEngine()
    rec, lig, proto = PreparedReceptor(), PreparedLigand(canonical_smiles="CC"), _protocol()
    cached_dock(engine, cache, "5I73", rec, lig, proto, Path(tmp_path) / "w")
    cached_dock(engine, cache, "5I73", rec, lig, proto, Path(tmp_path) / "w")
    assert engine.calls == 2


def test_cached_dock_keyed_on_inputs(tmp_path):
    cache = ComputeCache(Store(tmp_path), enabled=True)
    engine = FakeDockEngine()
    rec = PreparedReceptor()
    proto = _protocol()
    cached_dock(engine, cache, "5I73", rec, PreparedLigand(canonical_smiles="CC"), proto, Path(tmp_path) / "w")
    cached_dock(engine, cache, "5I73", rec, PreparedLigand(canonical_smiles="CCC"), proto, Path(tmp_path) / "w")
    cached_dock(engine, cache, "6A93", rec, PreparedLigand(canonical_smiles="CC"), proto, Path(tmp_path) / "w")
    assert engine.calls == 3  # three distinct (receptor, ligand) pairs


def test_cached_run_md_reuses(tmp_path):
    cache = ComputeCache(Store(tmp_path), enabled=True)
    engine = FakeMdEngine()
    proto = MdProtocol(engine="fake", engine_version="1.0", temperature_kelvin=300.0,
                       timestep_femtoseconds=2.0, steps=100, seed=42,
                       minimize_max_iterations=200)
    pdb = Path(tmp_path) / "r.pdb"
    pdb.write_text("END")
    r1 = cached_run_md(engine, cache, pdb, "5I73", proto)
    r2 = cached_run_md(engine, cache, pdb, "5I73", proto)
    assert engine.calls == 1
    assert r1.receptor_id == r2.receptor_id == "5I73"


def test_inject_resume_marks_all_tasks():
    from ddswarm.cli import _inject_resume
    from ddswarm.contracts import RunTask

    tasks = [RunTask(task_id="dock", agent="docking_minion"),
             RunTask(task_id="report", agent="report_minion")]
    out = _inject_resume(tasks)
    assert all(t.params.get("resume") is True for t in out)
    # the input list is not mutated
    assert all(t.params == {} for t in tasks)
