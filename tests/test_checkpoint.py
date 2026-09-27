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
    load_md_state,
    md_state_key,
    md_state_path,
    poses_from_cache,
    poses_to_cache,
    write_md_meta,
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
        self.last_resume_state = None
        self.last_prior_steps = 0

    def run_md(self, pdb_path, receptor_id, protocol, resume_state=None,
               prior_steps=0, checkpoint_path=None):
        self.calls += 1
        self.last_resume_state = resume_state
        self.last_prior_steps = prior_steps
        if checkpoint_path is not None:
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            Path(checkpoint_path).write_text("<State/>")
        return MdRun(receptor_id=receptor_id, status="success",
                     prior_steps=prior_steps, total_steps=prior_steps + protocol.steps)


def _protocol():
    return DockProtocol(box_center=(0.0, 0.0, 0.0), box_size=(20.0, 20.0, 20.0),
                        exhaustiveness=8, num_modes=9, seed=42)


def _md_protocol(steps=100):
    return MdProtocol(engine="openmm", engine_version="8.6.1", temperature_kelvin=300.0,
                      timestep_femtoseconds=2.0, steps=steps, seed=42,
                      minimize_max_iterations=200)


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


def test_md_state_roundtrip(tmp_path):
    store = Store(tmp_path)
    key = md_state_key("5I73", _md_protocol(), "openmm", "8.6.1")
    state_path = md_state_path(store, key)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_bytes(b"<State/>")
    meta = write_md_meta(store, key, b"<State/>", total_steps=500)
    assert meta["state_hash"]
    loaded = load_md_state(store, key)
    assert loaded is not None
    assert loaded["total_steps"] == 500
    assert loaded["state_path"].read_bytes() == b"<State/>"
    assert loaded["state_hash"] == meta["state_hash"]


def test_md_state_key_ignores_steps_and_note():
    k1 = md_state_key("5I73", _md_protocol(steps=100), "openmm", "8.6.1")
    k2 = md_state_key("5I73", _md_protocol(steps=1000), "openmm", "8.6.1")
    assert k1 == k2  # steps is ignored, so a longer run resumes the same simulation
    k3 = md_state_key("6A93", _md_protocol(steps=100), "openmm", "8.6.1")
    assert k1 != k3  # different receptor


def test_md_state_key_sensitive_to_physics():
    hot = _md_protocol()
    hot.temperature_kelvin = 310.0
    assert md_state_key("5I73", _md_protocol(), "openmm", "8.6.1") != \
        md_state_key("5I73", hot, "openmm", "8.6.1")


def test_load_md_state_missing(tmp_path):
    store = Store(tmp_path)
    assert load_md_state(store, "nonexistent") is None


def test_load_md_state_corrupt_meta(tmp_path):
    store = Store(tmp_path)
    state_path = md_state_path(store, "corrupt")
    state_path.parent.mkdir(parents=True)
    state_path.write_bytes(b"<State/>")
    state_path.with_name("meta.json").write_text("{not json")
    loaded = load_md_state(store, "corrupt")
    assert loaded is not None
    assert loaded["total_steps"] == 0  # corrupt meta -> defaults, not a crash


def test_md_engine_continuation_tracks_cumulative_steps(tmp_path):
    engine = FakeMdEngine()
    run = engine.run_md(Path(tmp_path) / "r.pdb", "5I73", _md_protocol(steps=100),
                        resume_state=Path(tmp_path) / "prior.xml", prior_steps=400,
                        checkpoint_path=Path(tmp_path) / "out" / "final_state.xml")
    assert run.prior_steps == 400
    assert run.total_steps == 500  # 400 prior + 100 this run
    assert engine.last_resume_state is not None
    assert (Path(tmp_path) / "out" / "final_state.xml").exists()


def test_inject_resume_marks_all_tasks():
    from ddswarm.cli import _inject_resume
    from ddswarm.contracts import RunTask

    tasks = [RunTask(task_id="dock", agent="docking_minion"),
             RunTask(task_id="report", agent="report_minion")]
    out = _inject_resume(tasks)
    assert all(t.params.get("resume") is True for t in out)
    # the input list is not mutated
    assert all(t.params == {} for t in tasks)
