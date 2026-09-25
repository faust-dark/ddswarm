"""Tests for the project memory store (cross-run, append-only)."""

from ddswarm.contracts import MemoryEntry
from ddswarm.orchestrator import Store
from ddswarm.orchestrator.memory import MemoryStore


def _entry(id, kind="finding", project_id="p"):
    return MemoryEntry(id=id, project_id=project_id, kind=kind, content=id,
                       source_agent="report_minion", run_id="r1")


def test_append_load_by_kind(tmp_path):
    store = Store(tmp_path)
    m = MemoryStore(store)
    assert m.load("p").entries == []
    m.append(_entry("e1", "finding"))
    m.append(_entry("e2", "feedback"))
    m.append(_entry("e3", "decision"))
    assert len(m.load("p").entries) == 3
    assert [e.id for e in m.by_kind("p", "feedback")] == ["e2"]


def test_append_idempotent(tmp_path):
    store = Store(tmp_path)
    m = MemoryStore(store)
    m.append(_entry("e1", "finding", "p"))
    m.append(_entry("e1", "finding", "p"))  # same id -> skipped
    assert len(m.load("p").entries) == 1


def test_memory_survives_new_store_instance(tmp_path):
    root = tmp_path
    MemoryStore(Store(root)).append(_entry("e1"))
    # a brand-new Store/MemoryStore over the same root still sees it
    assert len(MemoryStore(Store(root)).load("p").entries) == 1


def test_memory_isolated_per_project(tmp_path):
    store = Store(tmp_path)
    m = MemoryStore(store)
    m.append(_entry("e1", project_id="a"))
    m.append(_entry("e2", project_id="b"))
    assert [e.project_id for e in m.load("a").entries] == ["a"]
    assert [e.project_id for e in m.load("b").entries] == ["b"]
