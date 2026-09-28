"""Tests for the knowledge agents (patent/clinical/regulatory).

All three delegate to a reasoning backend and have NO deterministic offline
source. Clinical/regulatory still return insufficient_evidence without
`reasoning.backend=hermes`; patent_minion now returns SUCCESS with an empty
landscape + 'patent status UNKNOWN' note (non-gating, since it sits in
DEFAULT_PIPELINE). Delegation is tested offline with a fake 'hermes' script
(same pattern as test_reasoning.py). Every LLM entry must come back
evidence_level='inferred'.
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
    RegulatoryChecklist,
)
from ddswarm.orchestrator import Store


def _fake_hermes(tmp_path: Path, payload) -> str:
    script = tmp_path / "fake_hermes.sh"
    script.write_text(f"#!/bin/sh\necho '{json.dumps(payload)}'\n")
    script.chmod(0o755)
    return str(script)


def _fake_hermes_dispatch(tmp_path: Path, assessment, checklist) -> str:
    """A fake 'hermes' that echoes the assessment payload for the regulatory
    assessment prompt and the checklist payload for the 7D checklist prompt
    (dispatched on a keyword in the --query-file)."""
    script = tmp_path / "fake_hermes_dispatch.sh"
    script.write_text(
        "#!/bin/sh\n"
        "qf=\"\"; prev=\"\"\n"
        "for a in \"$@\"; do\n"
        "  [ \"$prev\" = \"--query-file\" ] && qf=\"$a\"\n"
        "  prev=\"$a\"\n"
        "done\n"
        "if grep -q \"seven dimensions\" \"$qf\" 2>/dev/null; then\n"
        f"  echo '{json.dumps(checklist)}'\n"
        "else\n"
        f"  echo '{json.dumps(assessment)}'\n"
        "fi\n"
    )
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
def test_patent_not_assessed_without_backend(tmp_path):
    # Non-gating baseline: without a backend patent_minion returns SUCCESS with an
    # empty landscape + an explicit 'patent status UNKNOWN' note, never
    # insufficient_evidence (which would gate report/validation in DEFAULT_PIPELINE).
    store = Store(tmp_path)
    result = _run(PatentMinion, store, _spec(), {})
    assert result.status.value == "success"
    pl = _load(store, result, PatentLandscape)
    assert pl.summary.total_entries == 0
    assert pl.summary.flagged == 0
    assert pl.note is not None and "UNKNOWN" in pl.note


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


def test_regulatory_checklist_delegation(tmp_path):
    store = Store(tmp_path)
    assessment = [{"molecule_id": None, "agency": "FDA", "designation": "approved",
                   "status": "approved", "source": "FDA Orange Book"}]
    checklist = [
        {"dimension": "quality_cmc", "status": "flag",
         "summary": "no CMC data for a novel scaffold", "notes": "needs synthesis"},
        {"dimension": "clinical_efficacy", "status": "unknown",
         "summary": "no clinical efficacy data"},
        {"dimension": "regulatory_pathway", "status": "ok",
         "summary": "orphan designation plausible"},
    ]
    params = {"reasoning_backend": "hermes",
              "reasoning_command": _fake_hermes_dispatch(tmp_path, assessment, checklist)}
    result = _run(RegulatoryMinion, store, _spec(), params)
    assert result.status.value == "success"
    rc_arts = [a for a in result.artifacts if a.kind == "regulatory_checklist"]
    assert rc_arts, "expected a regulatory_checklist artifact"
    rc = store.load_artifact(rc_arts[0].uri, RegulatoryChecklist)
    assert rc.summary.total_dimensions == 3
    assert rc.summary.ok == 1
    assert rc.summary.flag == 1
    assert rc.summary.unknown == 1
    assert all(e.evidence_level.value == "inferred" for e in rc.entries)
    assert all("verify against primary source" in e.notes for e in rc.entries)


# ---- schema export ----
def test_schema_export_includes_knowledge_models():
    from ddswarm.contracts.schema_export import MODELS
    assert "PatentLandscape" in MODELS
    assert "ClinicalAssessment" in MODELS
    assert "RegulatoryAssessment" in MODELS
