"""Parallel runner + parallel/cached docking: concurrency must never change
what is written (results, events, artifacts) — only how fast."""

import json
import threading
import time
from pathlib import Path

import pytest

from ddswarm.agents import Agent, DockingMinion, SelectivityMinion
from ddswarm.agents.base import AgentContext
from ddswarm.cli import DEFAULT_PIPELINE, build_agents
from ddswarm.contracts import ProjectSpec
from ddswarm.contracts.agent_result import AgentResult
from ddswarm.contracts.enums import AgentResultStatus, TaskStatus
from ddswarm.docking.base import DockingEngine, DockPose, DockProtocol, PreparedLigand, PreparedReceptor
from ddswarm.docking.parallel import CpuBudget, DockJob, dock_jobs, plan_workers
from ddswarm.docking.vina_engine import VinaEngine
from ddswarm.logging.event_writer import EventWriter
from ddswarm.logging.hashing import hash_content
from ddswarm.orchestrator import Runner, Store, TaskDefinition
from ddswarm.orchestrator.checkpoint import ComputeCache, _dock_key, dock_cache_enabled

GEFITINIB = "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4"


def _spec(project_id="p1", **kw):
    base = {"project_id": project_id, "input_type": "mixed", "objective": "x",
            "starting_materials": ["c1ccccc1"], "status": "approved"}
    base.update(kw)
    return ProjectSpec(**base)


# ---- runner: fake agents with controllable timing ----

class _Probe:
    """Shared concurrency probe: tracks how many agents run at once."""

    def __init__(self):
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.seen: dict[str, set[str]] = {}

    def enter(self):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)

    def exit(self):
        with self.lock:
            self.active -= 1


class _SleepAgent(Agent):
    """Sleeps `params['sleep']` seconds, then returns a deterministic result."""

    def __init__(self, store, probe, name="sleep_agent"):
        super().__init__(store)
        self.probe = probe
        self.name = name

    def run(self, ctx):
        self.probe.enter()
        try:
            time.sleep(float(ctx.params.get("sleep", 0)))
            if ctx.params.get("raise"):
                raise RuntimeError("boom")
            return AgentResult(project_id=ctx.project_id, agent=self.name, task_id=ctx.task_id,
                               status=AgentResultStatus.success,
                               findings=[f"{ctx.task_id} done"])
        finally:
            self.probe.exit()


class _ReaderAgent(_SleepAgent):
    """A whole-run reader: records which task results exist when it runs."""

    reads_whole_run = True

    def run(self, ctx):
        time.sleep(float(ctx.params.get("sleep", 0)))
        manifest = self.store.load_run_manifest(ctx.project_id, ctx.run_id)
        self.probe.seen[ctx.task_id] = {
            t for t in manifest.tasks
            if t != ctx.task_id and self.store.latest_attempt(ctx.project_id, ctx.run_id, t)
        }
        return AgentResult(project_id=ctx.project_id, agent=self.name, task_id=ctx.task_id,
                           status=AgentResultStatus.success)


def _runner(store, probe, workers):
    agents = {"sleep_agent": _SleepAgent(store, probe),
              "reader_agent": _ReaderAgent(store, probe, name="reader_agent")}
    return Runner(store, agents, max_workers=workers)


def _events(store, project_id, run_id):
    return [(e.get("task_id"), e["event"])
            for e in EventWriter(store.events_path(project_id, run_id)).read_all()]


def _fanout_tasks():
    # a (slow) and b, c (fast) are independent; d joins them. A sequential run
    # finishes a before b; in parallel b/c finish first — commits must not care.
    return [
        TaskDefinition(task_id="a", agent="sleep_agent", params={"sleep": 0.3}),
        TaskDefinition(task_id="b", agent="sleep_agent", params={"sleep": 0.0}),
        TaskDefinition(task_id="c", agent="sleep_agent", params={"sleep": 0.05}),
        TaskDefinition(task_id="d", agent="sleep_agent", depends_on=["a", "b", "c"]),
    ]


def test_runner_runs_independent_tasks_concurrently(tmp_path):
    store, probe = Store(tmp_path), _Probe()
    runner = _runner(store, probe, workers=4)
    m = runner.create_run(_spec(), _fanout_tasks())
    final = runner.run(m.run_id, project_id="p1")
    assert final.status.value == "completed"
    assert probe.max_active >= 2  # a, b, c overlapped


def test_runner_commits_in_topological_order(tmp_path):
    probe = _Probe()
    out = {}
    for workers in (1, 4):
        store = Store(tmp_path / f"w{workers}")
        runner = _runner(store, probe, workers)
        m = runner.create_run(_spec(), _fanout_tasks())
        runner.run(m.run_id, project_id="p1")
        out[workers] = _events(store, "p1", m.run_id)
    assert out[1] == out[4]
    assert [t for t, _e in out[4] if t] == ["a", "b", "c", "d"]  # order, not completion


def test_runner_isolates_a_crashing_task(tmp_path):
    store, probe = Store(tmp_path), _Probe()
    runner = _runner(store, probe, workers=4)
    tasks = _fanout_tasks()
    tasks[1] = TaskDefinition(task_id="b", agent="sleep_agent", params={"raise": True})
    m = runner.create_run(_spec(), tasks)
    final = runner.run(m.run_id, project_id="p1")
    assert final.status.value == "failed"
    statuses = {t: store.load_task_spec("p1", m.run_id, t).status for t in "abcd"}
    assert statuses["b"] == TaskStatus.failed
    assert statuses["a"] == statuses["c"] == statuses["d"] == TaskStatus.success


def test_whole_run_reader_is_a_barrier(tmp_path):
    store, probe = Store(tmp_path), _Probe()
    runner = _runner(store, probe, workers=4)
    tasks = [
        TaskDefinition(task_id="x", agent="sleep_agent", params={"sleep": 0.2}),
        TaskDefinition(task_id="y", agent="sleep_agent"),
        TaskDefinition(task_id="r", agent="reader_agent", depends_on=["x"], params={"sleep": 0.1}),
        TaskDefinition(task_id="z", agent="sleep_agent", depends_on=["y"]),
    ]
    m = runner.create_run(_spec(), tasks)
    assert m.tasks == ["x", "y", "r", "z"]
    runner.run(m.run_id, project_id="p1")
    # sees every earlier task (y although not a declared dep), never a later one
    assert probe.seen["r"] == {"x", "y"}
    r_done = store.load_task_spec("p1", m.run_id, "r").attempts[-1].finished_at
    z_start = store.load_task_spec("p1", m.run_id, "z").attempts[-1].started_at
    assert z_start >= r_done


def test_resume_reruns_only_unfinished_tasks_in_parallel(tmp_path):
    store, probe = Store(tmp_path), _Probe()
    runner = _runner(store, probe, workers=4)
    m = runner.create_run(_spec(), _fanout_tasks())
    runner.run(m.run_id, project_id="p1")
    # simulate a crash that left c and d unfinished
    for t in ("c", "d"):
        ts = store.load_task_spec("p1", m.run_id, t)
        ts.status = TaskStatus.running if t == "c" else TaskStatus.queued
        store.save_task_spec(ts)
    final = runner.run(m.run_id, project_id="p1")
    assert final.status.value == "completed"
    attempts = {t: len(store.load_task_spec("p1", m.run_id, t).attempts) for t in "abcd"}
    assert attempts == {"a": 1, "b": 1, "c": 2, "d": 2}


# ---- default pipeline: parallel == sequential ----

_NONDETERMINISTIC = {"timestamp", "run_id", "uri", "created_at", "completed_at",
                     "started_at", "finished_at", "pdf_uri", "pdf_hash"}


def _strip(obj):
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k not in _NONDETERMINISTIC}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    return obj


def _fingerprints(store, project_id, run_id):
    adir = store.run_dir(project_id, run_id) / "artifacts"
    return {a.name: hash_content(json.dumps(_strip(json.loads(a.read_text())), sort_keys=True))
            for a in sorted(adir.glob("*.json"))}


def test_default_pipeline_parallel_matches_sequential(tmp_path):
    out = {}
    for workers in (1, 8):
        store = Store(tmp_path / f"w{workers}")
        runner = Runner(store, build_agents(store), max_workers=workers)
        tasks = [TaskDefinition(task_id=t.task_id, agent=t.agent, depends_on=t.depends_on,
                                params=dict(t.params)) for t in DEFAULT_PIPELINE]
        spec = _spec("par", objective="inhibit EGFR", targets=["EGFR"])
        m = runner.create_run(spec, tasks)
        final = runner.run(m.run_id, project_id="par")
        out[workers] = (final.status.value, _fingerprints(store, "par", m.run_id),
                        _events(store, "par", m.run_id))
    assert out[1][0] == out[8][0] == "completed"
    assert out[1][1] and out[1][1] == out[8][1]
    assert out[1][2] == out[8][2]


# ---- parallel docking ----

class _FakeVina(DockingEngine):
    name = "fake"
    version = "1.0"

    def __init__(self, sleep=0.05, fail_on=()):
        self.sleep = sleep
        self.fail_on = set(fail_on)
        self.lock = threading.Lock()
        self.calls = 0
        self.active_cpu = 0
        self.max_active_cpu = 0
        self.cpus_seen = set()
        self.ligand_preps = 0

    def available(self):
        return True, "fake"

    def prepare_receptor(self, source, work_dir):
        return PreparedReceptor(identifier=source.identifier, content_hash=f"h-{source.identifier}",
                                pdbqt_path=Path("r.pdbqt"))

    def prepare_ligand(self, smiles, work_dir):
        with self.lock:
            self.ligand_preps += 1
        return PreparedLigand(canonical_smiles=smiles, pdbqt_path=Path("l.pdbqt"),
                              content_hash=f"l-{smiles}")

    def dock(self, receptor, ligand, protocol, work_dir):
        with self.lock:
            self.calls += 1
            self.active_cpu += protocol.cpu or 1
            self.max_active_cpu = max(self.max_active_cpu, self.active_cpu)
            self.cpus_seen.add(protocol.cpu)
        try:
            time.sleep(self.sleep * (1 + len(ligand.canonical_smiles) % 3))
            if ligand.canonical_smiles in self.fail_on:
                raise RuntimeError("vina crashed")
            work_dir.mkdir(parents=True, exist_ok=True)
            out = work_dir / "docked.pdbqt"
            aff = -5.0 - len(ligand.canonical_smiles) / 10 - len(receptor.identifier) / 100
            out.write_text(f"MODEL 1\nREMARK VINA RESULT {aff} 0 0\nENDMDL\n")
            return [DockPose(rank=1, affinity_kcal_per_mol=aff, pose_file=out,
                             pose_atoms={1: (0.0, 0.0, 0.0)})]
        finally:
            with self.lock:
                self.active_cpu -= protocol.cpu or 1


def _proto():
    return DockProtocol(box_center=(0.0, 0.0, 0.0), box_size=(20.0, 20.0, 20.0))


def _jobs(engine, tmp_path, smiles):
    rec = engine.prepare_receptor(type("S", (), {"identifier": "1ABC"})(), tmp_path)
    return [DockJob(i, "1ABC", rec, engine.prepare_ligand(s, tmp_path), _proto(), tmp_path / f"d{i}")
            for i, s in enumerate(smiles)]


def test_plan_workers_never_oversubscribes():
    assert plan_workers(1, 8) == (1, 8)        # one job gets every core
    assert plan_workers(4, 8) == (4, 2)
    assert plan_workers(20, 8) == (8, 1)
    assert plan_workers(20, 8, max_workers=2) == (2, 4)
    for n in range(1, 30):
        w, c = plan_workers(n, 12)
        assert w * c <= 12


def test_dock_jobs_parallel_ordered_isolated_and_within_budget(tmp_path):
    engine = _FakeVina(fail_on={"CCO"})
    smiles = ["C", "CC", "CCO", "CCC", "CCCC", "c1ccccc1"]
    budget = CpuBudget(4)
    res = dock_jobs(engine, ComputeCache(Store(tmp_path), False), _jobs(engine, tmp_path, smiles),
                    budget=budget)
    assert list(res) == list(range(len(smiles)))      # job order
    assert res[2] == []                                # the crashing job only
    assert all(res[i] for i in (0, 1, 3, 4, 5))
    assert engine.max_active_cpu <= 4                  # never above the CPU budget
    assert engine.cpus_seen == {1}                     # 6 jobs on 4 cpus -> 1 thread each


def test_cpu_budget_shared_across_concurrent_batches(tmp_path):
    engine = _FakeVina()
    budget = CpuBudget(3)
    cache = ComputeCache(Store(tmp_path), False)
    threads = [threading.Thread(target=dock_jobs, args=(
        engine, cache, _jobs(engine, tmp_path / str(k), ["C", "CC", "CCC"])), kwargs={"budget": budget})
        for k in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert engine.calls == 9
    assert engine.max_active_cpu <= 3


def test_dock_jobs_cache_hit_skips_engine_and_restores_pose_file(tmp_path):
    engine = _FakeVina(sleep=0)
    cache = ComputeCache(Store(tmp_path), True)
    first = dock_jobs(engine, cache, _jobs(engine, tmp_path / "run1", ["C", "CC"]))
    assert engine.calls == 2
    second = dock_jobs(engine, cache, _jobs(engine, tmp_path / "run2", ["C", "CC"]))
    assert engine.calls == 2  # all hits
    for k in first:
        assert [p.affinity_kcal_per_mol for p in first[k]] == [p.affinity_kcal_per_mol for p in second[k]]
        assert second[k][0].pose_file == tmp_path / "run2" / f"d{k}" / "docked.pdbqt"
        assert second[k][0].pose_file.read_text() == first[k][0].pose_file.read_text()


def test_failed_dock_is_not_cached(tmp_path):
    engine = _FakeVina(sleep=0, fail_on={"C"})
    cache = ComputeCache(Store(tmp_path), True)
    dock_jobs(engine, cache, _jobs(engine, tmp_path, ["C"]))
    dock_jobs(engine, cache, _jobs(engine, tmp_path, ["C"]))
    assert engine.calls == 2


# ---- cache key ----

def test_dock_key_covers_prepared_content_and_tool_versions():
    engine = _FakeVina()
    rec, lig, proto = PreparedReceptor(content_hash="r1"), PreparedLigand(content_hash="l1"), _proto()
    base = _dock_key("1ABC", "CC", proto, engine, rec, lig)
    assert base == _dock_key("1ABC", "CC", proto, engine, PreparedReceptor(content_hash="r1"), lig)
    assert base != _dock_key("1ABC", "CC", proto, engine, PreparedReceptor(content_hash="r2"), lig)
    assert base != _dock_key("1ABC", "CC", proto, engine, rec, PreparedLigand(content_hash="l2"))
    engine.prep_versions = {"meeko": "0.9"}
    assert base != _dock_key("1ABC", "CC", proto, engine, rec, lig)
    del engine.prep_versions
    # --cpu changes wall time only, never the key
    proto_cpu = DockProtocol(box_center=(0.0, 0.0, 0.0), box_size=(20.0, 20.0, 20.0), cpu=4)
    assert base == _dock_key("1ABC", "CC", proto_cpu, engine, rec, lig)


def test_dock_cache_enabled_defaults_on():
    assert dock_cache_enabled({}) is True
    assert dock_cache_enabled({"dock_cache": False}) is False
    assert dock_cache_enabled({"resume": False}) is False
    assert dock_cache_enabled({"resume": True, "dock_cache": False}) is False


def test_vina_version_probed_once(monkeypatch):
    engine = VinaEngine("/nonexistent/vina")
    calls = []
    monkeypatch.setattr(engine, "_probe_version", lambda: calls.append(1) or (True, "AutoDock Vina 1.2.7"))
    assert engine.version == "AutoDock Vina 1.2.7"
    for _ in range(5):
        _ = engine.version
    assert len(calls) == 1


# ---- prepared-receptor cache ----

_RAW_PDB = ("ATOM      1  N   ALA A   1      11.104   6.134  -6.504  1.00  0.00           N\n"
            "HETATM    2  C1  LIG A 101       1.000   2.000   3.000  1.00  0.00           C\n"
            "END\n")


def _ref_mol():
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    return Chem.RemoveHs(mol)


def test_prepared_receptor_cache_roundtrip_skips_preparation(tmp_path, monkeypatch):
    engine = VinaEngine("/nonexistent/vina")
    engine.receptor_cache_dir = tmp_path / "rcache"
    raw = tmp_path / "rec.pdb"
    raw.write_text(_RAW_PDB)
    key = engine._receptor_prep_key(_RAW_PDB)
    pdbqt = tmp_path / "prepared.pdbqt"
    pdbqt.write_text("ATOM      1  N   ALA A   1      11.104   6.134  -6.504  0.00  0.00    -0.3 N\n")
    fresh = PreparedReceptor(pdbqt_path=pdbqt, content_hash=hash_content(pdbqt.read_bytes()),
                             reference_ligand_name="LIG", reference_ligand_mol=_ref_mol(),
                             preparation_tool="pdbfixer X + openbabel-wheel Y (rigid PDBQT)",
                             note="receptor prepared and protonated")
    engine._save_prepared_receptor(fresh, key)

    # a hit must not run any preparation step
    def boom(*a, **k):
        raise AssertionError("preparation must be skipped on a cache hit")
    monkeypatch.setattr(engine, "_extract_reference_ligand", boom)
    from ddswarm.docking.base import ReceptorSource
    rec = engine.prepare_receptor(ReceptorSource(kind="file", identifier=str(raw)), tmp_path / "work")
    assert rec.pdbqt_path == tmp_path / "work" / "rec_receptor.pdbqt"
    assert rec.pdbqt_path.read_bytes() == pdbqt.read_bytes()
    assert rec.content_hash == fresh.content_hash
    assert rec.reference_ligand_name == "LIG"
    assert rec.reference_ligand_mol.GetNumAtoms() == 3
    assert rec.preparation_tool == fresh.preparation_tool
    assert rec.note == fresh.note
    assert rec.atom_count == 2  # QC still computed from the raw structure


def test_prepared_receptor_cache_rejects_tampered_entry(tmp_path):
    engine = VinaEngine("/nonexistent/vina")
    engine.receptor_cache_dir = tmp_path / "rcache"
    pdbqt = tmp_path / "p.pdbqt"
    pdbqt.write_text("ATOM\n")
    rec = PreparedReceptor(pdbqt_path=pdbqt, content_hash=hash_content(b"ATOM\n"))
    engine._save_prepared_receptor(rec, "k")
    (tmp_path / "rcache" / "k" / "receptor.pdbqt").write_text("TAMPERED\n")
    assert not engine._load_prepared_receptor(PreparedReceptor(), "k", tmp_path / "w", "x")
    (tmp_path / "rcache" / "k" / "meta.json").write_text("{not json")
    assert not engine._load_prepared_receptor(PreparedReceptor(), "k", tmp_path / "w", "x")


def test_receptor_prep_key_tracks_raw_content_and_tool_versions():
    engine = VinaEngine("/nonexistent/vina")
    k1 = engine._receptor_prep_key(_RAW_PDB)
    assert k1 == engine._receptor_prep_key(_RAW_PDB)
    assert k1 != engine._receptor_prep_key(_RAW_PDB + "REMARK changed\n")
    engine.prep_versions = {**engine.prep_versions, "pdbfixer": "99"}
    assert k1 != engine._receptor_prep_key(_RAW_PDB)


# ---- agents ----

def _sel_ctx(params):
    return AgentContext(project_id="p", run_id="r1", task_id="sel",
                        spec=_spec("p", starting_materials=["c1ccccc1", "CCO"]), params=params)


def test_selectivity_prepares_each_ligand_once_and_reuses_cache(tmp_path, monkeypatch):
    from ddswarm import agents
    engine = _FakeVina(sleep=0)
    monkeypatch.setattr(agents.selectivity, "make_engine", lambda name, vina_binary=None: engine)
    store = Store(tmp_path)
    params = {"receptors": [{"name": "SERT", "pdb_id": "5I73"}, {"name": "CHRM1", "pdb_id": "5CXV"},
                            {"name": "HRH1", "pdb_id": "3RZE"}],
              "primary": "SERT", "calibrate": True,
              "box": {"center": [0, 0, 0], "size": [20, 20, 20]}}
    res1 = SelectivityMinion(store).run(_sel_ctx(dict(params)))
    assert res1.status.value == "success"
    n_smiles = engine.ligand_preps
    calls_first = engine.calls
    assert calls_first > 0
    # each distinct SMILES (molecules + calibration ligands) prepared exactly once
    from ddswarm.chemistry.vina_calibration import CALIBRATION_SET
    distinct = {"c1ccccc1", "CCO"} | {s for r in ("SERT", "CHRM1", "HRH1")
                               for s, _k, _c in CALIBRATION_SET.get(r, [])}
    assert n_smiles == len(distinct)

    # second run against the same store: every dock (incl. calibration) is a cache hit
    ctx2 = AgentContext(project_id="p", run_id="r2", task_id="sel",
                        spec=_spec("p", starting_materials=["c1ccccc1", "CCO"]), params=dict(params))
    res2 = SelectivityMinion(store).run(ctx2)
    assert engine.calls == calls_first
    a1 = json.loads(store.artifact_path(res1.artifacts[0].uri).read_text())
    a2 = json.loads(store.artifact_path(res2.artifacts[0].uri).read_text())
    assert _strip(a1) == _strip(a2)


@pytest.mark.parametrize("workers", [1, 4])
def test_docking_agent_output_independent_of_workers(tmp_path, monkeypatch, workers):
    engine = _FakeVina(sleep=0.01)
    monkeypatch.setattr("ddswarm.agents.docking.make_engine", lambda name, vina_binary=None: engine)
    store = Store(tmp_path)
    spec = _spec(starting_materials=["C", "CC", "CCC", "CCCC", GEFITINIB])
    params = {"receptor": {"identifier": "2ITY"}, "box": {"center": [0, 0, 0], "size": [20, 20, 20]},
              "max_workers": workers, "dock_cache": False}
    res = DockingMinion(store).run(AgentContext(project_id="p1", run_id="r1", task_id="dock",
                                                spec=spec, params=params))
    art = json.loads(store.artifact_path(res.artifacts[0].uri).read_text())
    poses = [(p["molecule_id"], p["affinity_kcal_per_mol"]) for p in art["poses"]]
    assert poses == [(f"starting_material_{i}", pytest.approx(a)) for i, a in enumerate(
        [-5.0 - len(s) / 10 - 4 / 100 for s in ["C", "CC", "CCC", "CCCC", GEFITINIB]])]
