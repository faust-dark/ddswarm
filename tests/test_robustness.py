"""Robustness fixes: no-overwrite/atomic store writes, numeric version sorting,
safe receptor downloads, subprocess timeouts, engine-unavailable gaps, and the
CLI never mutating the module-level DEFAULT_PIPELINE."""

import io
import urllib.error
from pathlib import Path

import pytest

from ddswarm import cli
from ddswarm.agents import DockingMinion, SelectivityMinion
from ddswarm.agents.base import AgentContext
from ddswarm.contracts import ProjectSpec, RunRequest
from ddswarm.contracts.run_request import ReasoningConfig
from ddswarm.docking import base as dbase
from ddswarm.docking.base import ReceptorSource, download_pdb, looks_like_pdb, pdb_cache_path
from ddswarm.docking.vina_engine import VinaEngine, _file_stem, _run_tool
from ddswarm.fsutil import exclusive_write_text
from ddswarm.orchestrator import Store
from ddswarm.orchestrator.store import latest_versioned, version_of

_PDB = b"HEADER    TEST\nATOM      1  N   ALA A   1      11.104   6.134  -6.504  1.00  0.00           N\nEND\n"


def _spec(**kw):
    base = dict(project_id="p1", input_type="mixed", objective="x",
                starting_materials=["c1ccccc1"], status="approved")
    base.update(kw)
    return ProjectSpec(**base)


# ---- store: versions never overwritten, numeric ordering, atomic writes ----

def test_version_of_and_latest_versioned_numeric():
    paths = [Path(f"a.kind.v{n}.json") for n in (1, 2, 9, 10)]
    assert [version_of(p) for p in paths] == [1, 2, 9, 10]
    assert latest_versioned(paths) == Path("a.kind.v10.json")  # text sort would pick v9
    assert latest_versioned([]) is None


def test_save_artifact_never_overwrites_and_skips_gaps(tmp_path):
    store = Store(tmp_path)
    spec = _spec()
    d = store.artifacts_dir("p1", "r1")
    d.mkdir(parents=True)
    (d / "a.project_spec.v1.json").write_text("one")
    (d / "a.project_spec.v3.json").write_text("three")  # a gap at v2
    p, version, _h = store.save_artifact("p1", "r1", spec, "project_spec", "a")
    assert version == 4 and p.name == "a.project_spec.v4.json"
    assert (d / "a.project_spec.v1.json").read_text() == "one"
    assert (d / "a.project_spec.v3.json").read_text() == "three"
    assert not list(d.glob(".*.tmp"))  # no temp leftovers


def test_save_artifact_ten_versions_latest_is_v10(tmp_path):
    store = Store(tmp_path)
    for _ in range(10):
        store.save_artifact("p1", "r1", _spec(), "project_spec", "a")
    d = store.artifacts_dir("p1", "r1")
    assert latest_versioned(d.glob("a.project_spec.v*.json")).name == "a.project_spec.v10.json"


def test_exclusive_write_refuses_existing(tmp_path):
    p = tmp_path / "x.v1.json"
    exclusive_write_text(p, "first")
    with pytest.raises(FileExistsError):
        exclusive_write_text(p, "second")
    assert p.read_text() == "first"
    assert not list(tmp_path.glob(".*.tmp"))


def test_agent_result_attempt_not_overwritten(tmp_path):
    from ddswarm.contracts.agent_result import AgentResult
    store = Store(tmp_path)
    r = AgentResult(project_id="p1", agent="a", task_id="t", status="success")
    store.save_agent_result(r, "r1", 1)
    with pytest.raises(FileExistsError):
        store.save_agent_result(r, "r1", 1)
    store.save_agent_result(r, "r1", 10)
    store.save_agent_result(r, "r1", 2)
    assert store.latest_attempt("p1", "r1", "t") == 10


def test_cli_copies_highest_numeric_export_version(tmp_path):
    store = Store(tmp_path / "store")
    d = store.artifacts_dir("p1", "r1")
    d.mkdir(parents=True)
    (d / "candidates_p1.candidate_csv.v2.csv").write_text("two")
    (d / "candidates_p1.candidate_csv.v10.csv").write_text("ten")
    out = cli._copy_exports_to_reports(store, "p1", "r1", tmp_path / "reports")
    assert [p.name for p in out] == ["p1_candidates.csv"]
    assert out[0].read_text() == "ten"


# ---- CLI must not mutate DEFAULT_PIPELINE ----

def test_cli_run_does_not_mutate_default_pipeline(tmp_path):
    before = [t.model_dump() for t in cli.DEFAULT_PIPELINE]
    req = RunRequest(project=_spec(), tasks=[],
                     reasoning=ReasoningConfig(backend="hermes", command=str(tmp_path / "no-hermes")))
    f = tmp_path / "project.json"
    f.write_text(req.model_dump_json())
    cli.main(["run", str(f), "--no-iupac", "--store", str(tmp_path / "store"),
              "--reports-dir", str(tmp_path / "reports")])
    assert [t.model_dump() for t in cli.DEFAULT_PIPELINE] == before


# ---- engine unavailable is a gap, not a failure ----

@pytest.mark.parametrize("agent_cls,params", [
    (DockingMinion, {"receptor": {"pdb_id": "2ITY"}}),
    (SelectivityMinion, {}),
])
def test_engine_unavailable_is_insufficient_evidence(tmp_path, agent_cls, params):
    store = Store(tmp_path)
    params = dict(params, vina_binary=str(tmp_path / "no-vina"))
    ctx = AgentContext(project_id="p1", run_id="r1", task_id="t", spec=_spec(targets=["GABRA2"]),
                       params=params)
    result = agent_cls(store).run(ctx)
    assert result.status.value == "insufficient_evidence"
    assert not result.errors
    assert any("unavailable" in w for w in result.warnings)


# ---- receptor download ----

class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _urlopen_returning(body: bytes):
    return lambda req, timeout=None: _Resp(body)


def _urlopen_http_error(code: int):
    def f(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, code, "error", {}, io.BytesIO(b"<html>nope</html>"))
    return f


def test_looks_like_pdb():
    assert looks_like_pdb(_PDB)
    assert not looks_like_pdb(b"<html><body>404 Not Found</body></html>")
    assert not looks_like_pdb(b'{"error": "not found"}')
    assert not looks_like_pdb(b"")


def test_download_pdb_http_error_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(dbase.urllib.request, "urlopen", _urlopen_http_error(404))
    dest = tmp_path / "x.pdb"
    err = download_pdb("https://files.rcsb.org/download/XXXX.pdb", dest)
    assert err and "404" in err
    assert not dest.exists()


def test_download_pdb_rejects_non_pdb_body(tmp_path, monkeypatch):
    monkeypatch.setattr(dbase.urllib.request, "urlopen", _urlopen_returning(b"<html>404</html>"))
    dest = tmp_path / "x.pdb"
    assert download_pdb("https://example.org/x.pdb", dest)
    assert not dest.exists()


def test_download_pdb_refuses_non_http_scheme(tmp_path):
    dest = tmp_path / "x.pdb"
    err = download_pdb("file:///etc/hosts", dest)
    assert err and "scheme" in err
    assert not dest.exists()


def test_download_pdb_success(tmp_path, monkeypatch):
    monkeypatch.setattr(dbase.urllib.request, "urlopen", _urlopen_returning(_PDB))
    dest = tmp_path / "x.pdb"
    assert download_pdb("https://files.rcsb.org/download/1ABC.pdb", dest) is None
    assert dest.read_bytes() == _PDB


def test_receptor_404_is_not_cached(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    monkeypatch.setenv("DDWARM_PDB_CACHE", str(cache))
    monkeypatch.setattr(dbase.urllib.request, "urlopen", _urlopen_http_error(404))
    rec = VinaEngine(tmp_path / "no-vina").prepare_receptor(
        ReceptorSource(kind="pdb_id", identifier="9ZZZ"), tmp_path / "work")
    assert rec.pdbqt_path is None
    assert "404" in rec.note
    assert not (cache / "9ZZZ.pdb").exists()


def test_poisoned_cache_entry_is_redownloaded(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "1ABC.pdb").write_text("<html>404 Not Found</html>")  # left by the old curl bug
    monkeypatch.setenv("DDWARM_PDB_CACHE", str(cache))
    monkeypatch.setattr(dbase.urllib.request, "urlopen", _urlopen_returning(_PDB))
    VinaEngine(tmp_path / "no-vina").prepare_receptor(
        ReceptorSource(kind="pdb_id", identifier="1ABC"), tmp_path / "work")
    assert (cache / "1ABC.pdb").read_bytes() == _PDB


def test_invalid_pdb_id_not_downloaded(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("must not download")
    monkeypatch.setattr(dbase.urllib.request, "urlopen", boom)
    rec = VinaEngine(tmp_path / "no-vina").prepare_receptor(
        ReceptorSource(kind="pdb_id", identifier="../../etc/passwd"), tmp_path / "work")
    assert "invalid PDB id" in rec.note


def test_pdb_cache_path_rejects_traversal():
    with pytest.raises(ValueError):
        pdb_cache_path("../evil")
    with pytest.raises(ValueError):
        pdb_cache_path("a/b")


def test_file_stem_never_absolute():
    assert _file_stem("/abs/path/my receptor.pdb") == "my_receptor"
    assert _file_stem("2ITY") == "2ITY"
    assert _file_stem("/x/..pdb") == "receptor"


# ---- subprocess timeouts ----

def test_run_tool_times_out():
    rc, msg = _run_tool(["sleep", "5"], timeout=0.2)
    assert rc == -1 and "timed out" in msg


def test_run_tool_missing_binary():
    rc, msg = _run_tool(["/nonexistent/obabel"], timeout=5)
    assert rc == -1 and "failed to run" in msg
