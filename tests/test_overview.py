"""Tests for overview_minion (the Prompt + Overview intro of the final report).

overview_minion has a deterministic baseline (verbatim prompt + own-database
receptor facts) and ALWAYS succeeds, so it can sit in DEFAULT_PIPELINE. With a
reasoning backend it adds LLM-recalled MoA / structure / uses narratives
(evidence_level=inferred). Tested offline with a fake 'hermes' script (same
pattern as test_reasoning.py / test_knowledge.py).
"""

import json
from pathlib import Path

from ddswarm.agents import OverviewMinion
from ddswarm.agents.base import AgentContext
from ddswarm.contracts import (
    AgentResultStatus,
    OverviewReport,
    ProjectSpec,
    ProvenanceEntry,
    Report,
)
from ddswarm.orchestrator import Store
from ddswarm.reporting.pdf_render import render_report_pdf


def _spec():
    return ProjectSpec(project_id="p", input_type="mixed",
                       objective="inhibit EGFR", user_prompt="Find a selective EGFR inhibitor",
                       targets=["DRD2"], starting_materials=["c1ccccc1"], status="approved")


def _run(store, spec, params):
    ctx = AgentContext(project_id="p", run_id="r", task_id="t", spec=spec, params=params, inputs={})
    return OverviewMinion(store).run(ctx)


def _load(store, result):
    assert result.artifacts
    return store.load_artifact(result.artifacts[0].uri, OverviewReport)


def test_baseline_without_backend(tmp_path):
    store = Store(tmp_path)
    result = _run(store, _spec(), {})
    assert result.status.value == "success"  # deterministic baseline, never insufficient_evidence
    ov = _load(store, result)
    assert ov.user_prompt == "Find a selective EGFR inhibitor"
    assert ov.mechanism_of_action is None
    assert ov.chemical_structure is None
    assert ov.therapeutic_uses is None
    assert ov.own_database_facts, "own-database facts must always be present"
    assert any("DRD2" in f and "6CM4" in f for f in ov.own_database_facts)
    assert any("starting material" in f for f in ov.own_database_facts)


def test_baseline_falls_back_to_objective_when_no_user_prompt(tmp_path):
    store = Store(tmp_path)
    spec = _spec().model_copy(update={"user_prompt": None})
    result = _run(store, spec, {})
    ov = _load(store, result)
    assert ov.user_prompt == "inhibit EGFR"


def test_delegation(tmp_path):
    store = Store(tmp_path)
    payload = {
        "mechanism_of_action": "antagonizes the D2 dopamine receptor",
        "chemical_structure": "butyrophenone scaffold",
        "therapeutic_uses": "antipsychotic",
        "citations": ["Wang et al. 2018 D2 structure"],
    }
    script = tmp_path / "fake_hermes.sh"
    script.write_text(f"#!/bin/sh\necho '{json.dumps(payload)}'\n")
    script.chmod(0o755)
    params = {"reasoning_backend": "hermes", "reasoning_command": str(script)}
    result = _run(store, _spec(), params)
    assert result.status.value == "success"
    ov = _load(store, result)
    assert ov.mechanism_of_action == "antagonizes the D2 dopamine receptor"
    assert ov.chemical_structure == "butyrophenone scaffold"
    assert ov.therapeutic_uses == "antipsychotic"
    assert ov.citations == ["Wang et al. 2018 D2 structure"]
    assert ov.own_database_facts  # deterministic facts still present alongside the LLM narrative


def test_renderer_renders_prompt_and_overview(tmp_path):
    report = Report(
        project_id="p", run_id="run_1", status=AgentResultStatus.success,
        run_status="completed", executive_summary="s",
        stage_findings=[], provenance=[ProvenanceEntry(source="test")],
    )
    spec = ProjectSpec(project_id="p", input_type="mixed", objective="inhibit EGFR",
                       user_prompt="Find a selective EGFR inhibitor",
                       targets=["EGFR"], status="approved")
    ov = OverviewReport(project_id="p", status=AgentResultStatus.success,
                        user_prompt="Find a selective EGFR inhibitor",
                        mechanism_of_action="blocks the ATP pocket",
                        chemical_structure="anilinoquinazoline",
                        therapeutic_uses="NSCLC",
                        own_database_facts=["target EGFR — PDB 1M17"],
                        citations=["Cohen et al. 2003"])
    out = tmp_path / "r.pdf"
    meta = render_report_pdf(report, {"overview_report": [ov]}, spec, None, out)
    assert out.exists() and out.read_bytes().startswith(b"%PDF")
    # Prompt and Overview are the first two sections (before Reasoning and plan).
    assert meta["sections"][0] == "Prompt"
    assert meta["sections"][1] == "Overview"
    assert "Prompt" in meta["sections"] and "Overview" in meta["sections"]


def test_renderer_prompt_and_overview_without_artifact(tmp_path):
    # Even when no overview_minion ran, the Prompt + Overview sections still
    # appear (prompt from spec; overview falls back to ProjectSpec fields).
    report = Report(
        project_id="p", run_id="run_1", status=AgentResultStatus.success,
        run_status="completed", executive_summary="s",
        stage_findings=[], provenance=[ProvenanceEntry(source="test")],
    )
    spec = ProjectSpec(project_id="p", input_type="mixed", objective="inhibit EGFR",
                       user_prompt="Find a selective EGFR inhibitor",
                       mechanism_of_action="blocks ATP", indication="NSCLC",
                       existing_drugs=["gefitinib"], targets=["EGFR"], status="approved")
    out = tmp_path / "r.pdf"
    meta = render_report_pdf(report, {}, spec, None, out)
    assert meta["sections"][0] == "Prompt"
    assert meta["sections"][1] == "Overview"
