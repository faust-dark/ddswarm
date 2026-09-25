"""Tests for the JSON contracts and schema export."""

import json

import pytest
from pydantic import ValidationError

from ddswarm.contracts import (
    MODELS,
    AgentResult,
    AgentResultStatus,
    EvidenceLevel,
    InputType,
    ProjectSpec,
    export_schemas,
)
from ddswarm.logging import hash_content, hash_jsonable


def test_project_spec_valid():
    spec = ProjectSpec(
        project_id="p1", input_type="target", objective="inhibit EGFR",
        targets=["EGFR"], status="approved",
    )
    assert spec.schema_version == "1.0"
    assert spec.input_type == InputType.target


def test_project_spec_rejects_empty_objective():
    with pytest.raises(ValidationError):
        ProjectSpec(project_id="p1", input_type="target", objective="")


def test_agent_result_roundtrip():
    r = AgentResult(project_id="p1", agent="researcher_minion", task_id="t1", status="success")
    r2 = AgentResult.model_validate_json(r.model_dump_json())
    assert r2.task_id == "t1"
    assert r2.status == AgentResultStatus.success


def test_evidence_level_enum():
    assert EvidenceLevel.hypothetical.value == "hypothetical"
    assert EvidenceLevel.observed.value == "observed"


def test_schema_export(tmp_path):
    written = export_schemas(tmp_path)
    assert set(written) == set(MODELS)
    for p in written.values():
        schema = json.loads(p.read_text())
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "schema_version" in schema["properties"]


def test_hash_deterministic_and_order_independent():
    assert hash_jsonable({"a": 1, "b": [1, 2, 3]}) == hash_jsonable({"b": [1, 2, 3], "a": 1})
    assert len(hash_content("abc")) == 64
