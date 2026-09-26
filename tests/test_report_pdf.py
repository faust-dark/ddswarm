"""PDF report export tests: the renderer produces a valid PDF, and the
report_pdf agent persists a ReportDocument artifact + PDF sidecar at the end of
a default-pipeline run."""

from pathlib import Path

from ddswarm.cli import DEFAULT_PIPELINE, build_agents
from ddswarm.contracts import (
    AgentResultStatus,
    EvidenceBreakdown,
    ProjectSpec,
    ProvenanceEntry,
    Report,
    ReportDocument,
    StageFinding,
)
from ddswarm.orchestrator import Runner, Store, TaskDefinition
from ddswarm.reporting.pdf_render import render_report_pdf


def test_renderer_minimal(tmp_path):
    report = Report(
        project_id="x", run_id="run_1", status=AgentResultStatus.success,
        run_status="completed", executive_summary="A minimal summary.",
        stage_findings=[StageFinding(task_id="research", agent="researcher_minion",
                                     status="success")],
        evidence_breakdown=EvidenceBreakdown(),
        gaps=[], next_steps=[], caveats=["in-silico only"],
        provenance=[ProvenanceEntry(source="test")],
    )
    spec = ProjectSpec(project_id="x", input_type="mixed", objective="obj",
                       targets=["T"], status="approved")
    out = tmp_path / "r.pdf"
    meta = render_report_pdf(report, {}, spec, None, out)
    assert out.exists()
    assert out.read_bytes().startswith(b"%PDF")
    assert meta["page_count"] >= 1
    assert "Abstract" in meta["sections"]
    assert "Conclusion" in meta["sections"]


def test_default_pipeline_exports_pdf(tmp_path):
    store = Store(tmp_path)
    spec = ProjectSpec(project_id="pdf", input_type="mixed", objective="inhibit EGFR",
                       targets=["EGFR"], starting_materials=["c1ccccc1"], status="approved")
    runner = Runner(store, build_agents(store))
    tasks = [TaskDefinition(task_id=t.task_id, agent=t.agent,
                            depends_on=t.depends_on, params=t.params)
             for t in DEFAULT_PIPELINE]
    manifest = runner.create_run(spec, tasks)
    final = runner.run(manifest.run_id, project_id=spec.project_id)
    assert final.status.value == "completed"

    adir = store.artifacts_dir("pdf", manifest.run_id)
    metas = sorted(adir.glob("report_document_pdf.report_document.v*.json"))
    assert metas, "expected a ReportDocument metadata artifact"
    doc = ReportDocument.model_validate_json(metas[-1].read_text())

    pdf = Path(doc.pdf_uri)
    assert pdf.exists(), f"PDF sidecar missing: {doc.pdf_uri}"
    assert pdf.read_bytes().startswith(b"%PDF")
    assert doc.page_count >= 1
    assert doc.figure_count >= 1
    assert doc.pdf_hash
    assert "Conclusion" in doc.sections
    assert "Evidence and confidence" in doc.sections
    assert "Identified candidate" in doc.sections
