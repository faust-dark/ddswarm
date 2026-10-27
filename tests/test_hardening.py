"""Hardening tests (point 4): filesystem-ID safety, CLI error handling, and
end-to-end reproducibility of a deterministic pipeline."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from ddswarm.cli import DEFAULT_PIPELINE, build_agents, main
from ddswarm.contracts import ProjectSpec, RunTask
from ddswarm.logging.hashing import hash_content
from ddswarm.orchestrator import Runner, Store, TaskDefinition

# ---- A. filesystem-ID safety (path traversal) ----


def test_store_rejects_path_traversal_ids(tmp_path):
    store = Store(tmp_path)
    for bad in ("../evil", "a/b", "a\\b", "", ".", "..", "a b", "a\x00b"):
        with pytest.raises(ValueError):
            store.project_dir(bad)


def test_store_accepts_legit_ids(tmp_path):
    store = Store(tmp_path)
    for good in ("p1", "proj_1.2-x", "egfr_hermes", "research"):
        assert store.project_dir(good).name == good


def test_contracts_reject_bad_ids():
    with pytest.raises(ValidationError):
        ProjectSpec(project_id="../evil", input_type="mixed", objective="x")
    with pytest.raises(ValidationError):
        RunTask(task_id="a/b", agent="x")
    with pytest.raises(ValidationError):
        TaskDefinition(task_id="..", agent="x")
    # legit IDs still construct
    ProjectSpec(project_id="proj_1.2-x", input_type="mixed", objective="x")
    RunTask(task_id="safety_1", agent="x")
    TaskDefinition(task_id="research", agent="researcher_minion")


# ---- B. CLI error handling ----


def test_cli_run_missing_file(tmp_path, capsys):
    assert main(["run", str(tmp_path / "nope.json")]) == 1
    assert "error:" in capsys.readouterr().err


def test_cli_run_invalid_json(tmp_path, capsys):
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    assert main(["run", str(p)]) == 1
    assert "error:" in capsys.readouterr().err


def test_cli_run_bad_project_id(tmp_path, capsys):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({
        "project": {"project_id": "../evil", "input_type": "mixed",
                    "objective": "x", "status": "approved"},
        "tasks": [],
    }))
    assert main(["run", str(p), "--store", str(tmp_path / "store")]) == 1
    assert "error:" in capsys.readouterr().err


# ---- C. end-to-end reproducibility ----


_NONDETERMINISTIC = {"timestamp", "run_id", "uri", "created_at", "completed_at",
                     "started_at", "finished_at", "pdf_uri", "pdf_hash"}


def _strip(obj):
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k not in _NONDETERMINISTIC}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    return obj


def _fingerprints(store: Store, project_id: str, run_id: str) -> dict[str, str]:
    out: dict[str, str] = {}
    adir = store.run_dir(project_id, run_id) / "artifacts"
    for a in sorted(adir.glob("*.json")):
        stripped = _strip(json.loads(a.read_text()))
        out[a.name] = hash_content(json.dumps(stripped, sort_keys=True))
    return out


def _run_default(store: Store, spec: ProjectSpec):
    runner = Runner(store, build_agents(store))
    tasks = [TaskDefinition(task_id=t.task_id, agent=t.agent,
                            depends_on=t.depends_on, params=t.params)
             for t in DEFAULT_PIPELINE]
    manifest = runner.create_run(spec, tasks)
    final = runner.run(manifest.run_id, project_id=spec.project_id)
    return manifest, final


def test_pipeline_reproducible_across_runs(tmp_path):
    def make_spec():
        return ProjectSpec(project_id="repro", input_type="mixed",
                           objective="inhibit EGFR", targets=["EGFR"],
                           starting_materials=["c1ccccc1"], status="approved")

    m1, f1 = _run_default(Store(tmp_path / "s1"), make_spec())
    m2, f2 = _run_default(Store(tmp_path / "s2"), make_spec())
    assert f1.status.value == "completed"
    assert f2.status.value == "completed"

    fp1 = _fingerprints(Store(tmp_path / "s1"), "repro", m1.run_id)
    fp2 = _fingerprints(Store(tmp_path / "s2"), "repro", m2.run_id)
    assert fp1 and fp2, "expected non-empty artifact sets"
    assert fp1 == fp2, "pipeline output is not reproducible across runs"
