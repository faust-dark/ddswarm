"""Tests for the reasoning backend abstraction and its integration with
researcher_minion / designer_minion (offline — a fake 'hermes' script stands in
for the real subagent, so no model call is made)."""

import json
from pathlib import Path

import pytest

from ddswarm.agents import DesignerMinion, ResearcherMinion
from ddswarm.agents.base import AgentContext
from ddswarm.contracts import DesignBatch, KnowledgePack, KnowledgePackEntry, ProjectSpec
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
    # trailing prose after a complete array is ignored (truncated tail)
    assert extract_json_list('[{"a":1}] and then some trailing text') == [{"a": 1}]
    # the first complete array wins when several are present
    assert extract_json_list('[{"a":1}] [{"b":2}]') == [{"a": 1}]
    # brackets inside string values (SMILES) don't confuse the scanner
    assert extract_json_list('Here are candidates:\n[{"smiles":"C[N+](C)(C)C"}]') == [{"smiles": "C[N+](C)(C)C"}]
    # an empty array is a valid (empty) result, not a parse failure
    assert extract_json_list('[]') == []


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


def test_hermes_retry_on_unparseable(tmp_path):
    # first invocation returns non-JSON; the backend retries and parses the 2nd
    state = tmp_path / "state"
    script = tmp_path / "fake_hermes_flaky.sh"
    script.write_text(
        '#!/bin/sh\n'
        f'if [ ! -f "{state}" ]; then\n'
        f'  touch "{state}"\n'
        '  echo "not json at all"\n'
        'else\n'
        '  echo \'[{"smiles":"c1ccccc1Cl","parent_scaffold":"s","transformation":"t","rationale":"r"}]\'\n'
        'fi\n'
    )
    script.chmod(0o755)
    b = HermesSubprocessBackend(command=str(script))
    res = b.design(_spec(), [("s", "c1ccccc1")], {})
    assert res.ok
    assert res.entries == [{"smiles": "c1ccccc1Cl", "parent_scaffold": "s",
                            "transformation": "t", "rationale": "r"}]


def test_designer_skips_non_smiles_scaffolds(tmp_path):
    store = Store(tmp_path)
    kp = KnowledgePack(project_id="p", status="success", entries=[
        KnowledgePackEntry(id="ligand_0", type="ligand", source="x",
                           observed_value="c1ccccc1", evidence_level="inferred"),
        KnowledgePackEntry(id="ligand_1", type="ligand", source="x",
                           observed_value="Gefitinib (ZD1839); CAS 184475-35-2; not a SMILES",
                           evidence_level="inferred"),
    ])
    ctx = AgentContext(project_id="p", run_id="r", task_id="design", spec=_spec(),
                       params={}, inputs={"knowledge_pack": [kp]})
    result = DesignerMinion(store).run(ctx)
    db = store.load_artifact(result.artifacts[0].uri, DesignBatch)
    # only the valid SMILES scaffold survives -> 1 scaffold x 4 labels
    assert len(db.molecules) == 4
    assert all(m.smiles == "c1ccccc1" for m in db.molecules)


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
