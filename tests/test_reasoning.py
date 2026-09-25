"""Tests for the reasoning backend abstraction and its integration with
researcher_minion / designer_minion (offline — a fake 'hermes' script stands in
for the real subagent, so no model call is made)."""

import json
from pathlib import Path

import pytest

from ddswarm.agents import DesignerMinion, ResearcherMinion
from ddswarm.agents.base import AgentContext
from ddswarm.contracts import DesignBatch, KnowledgePack, ProjectSpec
from ddswarm.orchestrator import Store
from ddswarm.reasoning import (
    HermesSubprocessBackend,
    NullReasoningBackend,
    extract_json_list,
    make_reasoning_backend,
)


def _spec():
    return ProjectSpec(
        project_id="p", input_type="mixed", objective="inhibit EGFR",
        targets=["EGFR"], starting_materials=["c1ccccc1"], status="approved",
    )


def _fake_hermes(tmp_path: Path, payload) -> str:
    script = tmp_path / "fake_hermes.sh"
    script.write_text(f"#!/bin/sh\necho '{json.dumps(payload)}'\n")
    script.chmod(0o755)
    return str(script)


# ---- extract_json_list ----
def test_extract_json_list_variants():
    assert extract_json_list('[{"a":1}]') == [{"a": 1}]
    assert extract_json_list('prose\n[{"a":1},{"b":2}]\nmore') == [{"a": 1}, {"b": 2}]
    assert extract_json_list('```json\n[{"a":1}]\n```') == [{"a": 1}]
    assert extract_json_list('{"a":1}') == [{"a": 1}]  # bare object
    assert extract_json_list('no json here') is None
    assert extract_json_list('') is None


# ---- null backend ----
def test_null_backend():
    b = NullReasoningBackend()
    assert b.available()[0] is True
    assert b.research(_spec(), {}).entries == []
    assert b.design(_spec(), [("s", "c1ccccc1")], {}).entries == []


# ---- hermes backend (fake script) ----
def test_hermes_research(tmp_path):
    payload = [{"type": "reference", "source": "PubChem", "url": "https://x",
                "identifier": "CID 1", "observed_value": "v", "notes": "n"}]
    b = HermesSubprocessBackend(command=_fake_hermes(tmp_path, payload))
    ok, _ = b.available()
    assert ok
    res = b.research(_spec(), {})
    assert res.ok and res.entries == payload


def test_hermes_design(tmp_path):
    payload = [{"smiles": "c1ccccc1Cl", "parent_scaffold": "s",
                "transformation": "halogenation", "rationale": "r"}]
    b = HermesSubprocessBackend(command=_fake_hermes(tmp_path, payload))
    res = b.design(_spec(), [("s", "c1ccccc1")], {})
    assert res.ok and res.entries == payload


def test_hermes_unavailable():
    b = HermesSubprocessBackend(command="/nonexistent/xyz")
    ok, _ = b.available()
    assert not ok


def test_make_reasoning_backend():
    assert make_reasoning_backend("none").name == "none"
    assert make_reasoning_backend("hermes", command="hermes").name == "hermes"
    with pytest.raises(KeyError):
        make_reasoning_backend("bogus")


# ---- agent integration ----
def test_researcher_with_reasoning(tmp_path):
    payload = [{"type": "reference", "source": "PubChem", "url": "https://x",
                "identifier": "CID 1", "observed_value": "v"}]
    store = Store(tmp_path)
    ctx = AgentContext(project_id="p", run_id="r", task_id="research", spec=_spec(),
                       params={"reasoning_backend": "hermes",
                               "reasoning_command": _fake_hermes(tmp_path, payload)})
    result = ResearcherMinion(store).run(ctx)
    assert result.status.value == "success"
    kp = store.load_artifact(result.artifacts[0].uri, KnowledgePack)
    refs = [e for e in kp.entries if e.type == "reference"]
    assert len(refs) == 1
    assert refs[0].evidence_level.value == "inferred"
    assert "verify against primary source" in (refs[0].notes or "")


def test_designer_with_reasoning(tmp_path):
    payload = [{"smiles": "c1ccccc1Cl", "parent_scaffold": "starting_material",
                "transformation": "halogenation", "rationale": "r"}]
    store = Store(tmp_path)
    ctx = AgentContext(project_id="p", run_id="r", task_id="design", spec=_spec(),
                       params={"reasoning_backend": "hermes",
                               "reasoning_command": _fake_hermes(tmp_path, payload)})
    result = DesignerMinion(store).run(ctx)
    assert result.status.value == "success"
    db = store.load_artifact(result.artifacts[0].uri, DesignBatch)
    assert len(db.molecules) == 1
    assert db.molecules[0].smiles == "c1ccccc1Cl"
    assert db.molecules[0].evidence_level.value == "hypothetical"


def test_designer_backend_unavailable_falls_back(tmp_path):
    store = Store(tmp_path)
    ctx = AgentContext(project_id="p", run_id="r", task_id="design", spec=_spec(),
                       params={"reasoning_backend": "hermes",
                               "reasoning_command": "/nonexistent/xyz"})
    result = DesignerMinion(store).run(ctx)
    assert result.status.value == "success"
    db = store.load_artifact(result.artifacts[0].uri, DesignBatch)
    assert len(db.molecules) == 4  # deterministic labels over 1 scaffold
    assert any("using labels" in w for w in result.warnings)


def test_researcher_reasoning_invalid_entries_skipped(tmp_path):
    payload = [{"type": "bogus", "source": "x"}]  # invalid type literal
    store = Store(tmp_path)
    ctx = AgentContext(project_id="p", run_id="r", task_id="research", spec=_spec(),
                       params={"reasoning_backend": "hermes",
                               "reasoning_command": _fake_hermes(tmp_path, payload)})
    result = ResearcherMinion(store).run(ctx)
    assert result.status.value == "success"
    kp = store.load_artifact(result.artifacts[0].uri, KnowledgePack)
    assert all(e.type != "bogus" for e in kp.entries)
    assert any("invalid, skipped" in w for w in result.warnings)
