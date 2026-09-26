"""Tests for the knowledge agents (patent/clinical/regulatory).

All three delegate to a reasoning backend and have NO deterministic offline
source, so without `reasoning.backend=hermes` they return insufficient_evidence.
Delegation is tested offline with a fake 'hermes' script (same pattern as
test_reasoning.py). Every LLM entry must come back evidence_level='inferred'.
"""

import json
from pathlib import Path

from ddswarm.agents import ClinicalMinion, PatentMinion, RegulatoryMinion
from ddswarm.agents.base import AgentContext
from ddswarm.contracts import (
    ClinicalAssessment,
    PatentLandscape,
    ProjectSpec,
    RegulatoryAssessment,
)
from ddswarm.orchestrator import Store


def _fake_hermes(tmp_path: Path, payload) -> str:
    script = tmp_path / "fake_hermes.sh"
    script.write_text(f"#!/bin/sh\necho '{json.dumps(payload)}'\n")
    script.chmod(0o755)
    return str(script)


def _spec():
    return ProjectSpec(project_id="p", input_type="mixed", objective="inhibit EGFR",
                       targets=["EGFR"], starting_materials=["c1ccccc1"], status="approved")


def _run(agent_cls, store, spec, params):
    ctx = AgentContext(project_id="p", run_id="r", task_id="t", spec=spec, params=params, inputs={})
    return agent_cls(store).run(ctx)


def _load(store, result, model):
    assert result.artifacts
    return store.load_artifact(result.artifacts[0].uri, model)


# ---- patent ----
def test_patent_insufficient_without_backend(tmp_path):
    result = _run(PatentMinion, Store(tmp_path), _spec(), {})
    assert result.status.value == "insufficient_evidence"


def test_patent_delegation(tmp_path):
    store = Store(tmp_path)
    payload = [
        {"molecule_id": "m1", "patent_id": "US1234567B2", "title": "EGFR inhibitors",
         "assignee": "Acme", "status": "granted", "jurisdiction": "US",
         "relevance": "covers_candidate", "source": "Google Patents",
         "url": "https://example.com/p1", "notes": "", "evidence_level": "observed"},
        {"molecule_id": None, "patent_id": "EP9876543A1", "status": "pending",
         "relevance": "background", "source": "Espacenet"},
    ]
    params = {"reasoning_backend": "hermes",
              "reasoning_command": _fake_hermes(tmp_path, payload)}
    result = _run(PatentMinion, store, _spec(), params)
    assert result.status.value == "success"
    pl = _load(store, result, PatentLandscape)
    assert pl.summary.total_entries == 2
    assert pl.summary.flagged == 1
    assert all(e.evidence_level.value == "inferred" for e in pl.entries)  # forced, even if LLM said observed
    assert "verify against primary source" in pl.entries[0].notes


# ---- clinical ----
def test_clinical_insufficient_without_backend(tmp_path):
    result = _run(ClinicalMinion, Store(tmp_path), _spec(), {})
    assert result.status.value == "insufficient_evidence"


def test_clinical_delegation(tmp_path):
    store = Store(tmp_path)
    payload = [
        {"molecule_id": None, "entity": "osimertinib", "phase": "approved",
         "indication": "NSCLC", "status": "approved", "source": "FDA"},
        {"molecule_id": None, "entity": "gefitinib", "phase": "approved",
         "indication": "NSCLC", "status": "approved", "source": "DrugBank"},
    ]
    params = {"reasoning_backend": "hermes",
              "reasoning_command": _fake_hermes(tmp_path, payload)}
    result = _run(ClinicalMinion, store, _spec(), params)
    assert result.status.value == "success"
    ca = _load(store, result, ClinicalAssessment)
    assert ca.summary.total_entries == 2
    assert ca.summary.phase_counts["approved"] == 2
    assert all(e.evidence_level.value == "inferred" for e in ca.entries)


# ---- regulatory ----
def test_regulatory_insufficient_without_backend(tmp_path):
    result = _run(RegulatoryMinion, Store(tmp_path), _spec(), {})
    assert result.status.value == "insufficient_evidence"


def test_regulatory_delegation(tmp_path):
    store = Store(tmp_path)
    payload = [
        {"molecule_id": None, "agency": "FDA", "designation": "approved",
         "status": "approved", "source": "FDA Orange Book"},
        {"molecule_id": None, "agency": "EMA", "designation": "approved",
         "status": "approved", "source": "EMA"},
    ]
    params = {"reasoning_backend": "hermes",
              "reasoning_command": _fake_hermes(tmp_path, payload)}
    result = _run(RegulatoryMinion, store, _spec(), params)
    assert result.status.value == "success"
    ra = _load(store, result, RegulatoryAssessment)
    assert ra.summary.total_entries == 2
    assert ra.summary.agency_counts == {"FDA": 1, "EMA": 1}
    assert all(e.evidence_level.value == "inferred" for e in ra.entries)


# ---- schema export ----
def test_schema_export_includes_knowledge_models():
    from ddswarm.contracts.schema_export import MODELS
    assert "PatentLandscape" in MODELS
    assert "ClinicalAssessment" in MODELS
    assert "RegulatoryAssessment" in MODELS
