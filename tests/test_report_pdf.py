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
from ddswarm.contracts.docking_result import DockedPose, DockingResult
from ddswarm.orchestrator import Runner, Store, TaskDefinition
from ddswarm.reporting.pdf_render import _chart_docking, render_report_pdf


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
    assert "Reasoning and plan" in meta["sections"]
    assert "Workflow schema" in meta["sections"]
    assert "Abstract" in meta["sections"]
    assert "Conclusion" in meta["sections"]


def test_narrative_reasoning_text():
    from ddswarm.contracts.report import CandidateSelection
    from ddswarm.reporting.pdf_render import _narrative_reasoning

    report = Report(
        project_id="x", run_id="run_1", status=AgentResultStatus.success,
        run_status="completed", executive_summary="s",
        ranked_candidates=[CandidateSelection(rank=1, smiles="c1ccccc1Cl", score=1.5,
                                              criteria=["QED: 0.5"])],
    )
    spec = ProjectSpec(project_id="x", input_type="mixed", objective="obj",
                       targets=["EGFR"], status="approved")
    lines = _narrative_reasoning({}, report, spec)
    joined = " ".join(lines)
    assert "EGFR" in joined
    assert "composite score" in joined
    assert "c1ccccc1Cl" in joined


def test_narrative_schema_reflects_stages():
    from ddswarm.contracts import ExecutionPlan, PlanStage
    from ddswarm.reporting.pdf_render import _narrative_schema

    plan = ExecutionPlan(project_id="x", run_id="r", generation=0, stages=[
        PlanStage(stage_id="design", agent="designer_minion",
                  role="proposes analogs", depends_on=["research"]),
    ])
    report = Report(project_id="x", run_id="r", status=AgentResultStatus.success,
                    run_status="completed", executive_summary="s")
    lines = []
    _narrative_schema(report, [plan], lines.append)
    assert any("designer_minion" in l and "research" in l for l in lines)


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
    assert "Prompt" in doc.sections
    assert "Overview" in doc.sections
    assert "Conclusion" in doc.sections
    assert "Evidence and confidence" in doc.sections
    assert "Identified candidate" in doc.sections


def test_chart_docking_one_bar_per_molecule():
    # Vina emits ~9 poses per ligand; the chart must collapse to ONE bar per
    # molecule, or a 6x9=54-bar chart overflows the page (LayoutError).
    import matplotlib.pyplot as plt

    poses = []
    for m in range(6):
        for rank in range(9):
            poses.append(DockedPose(molecule_id=f"mol_{m}", rank=rank + 1,
                                    affinity_kcal_per_mol=-7.0 + m * 0.1 + rank * 0.01))
    dr = DockingResult(project_id="p", status=AgentResultStatus.success, poses=poses)
    fig = _chart_docking([dr])
    try:
        assert fig is not None
        assert len(fig.axes[0].patches) == 6, "expected one bar per molecule"
        # the collapsed chart must be short enough to fit a page (inches)
        assert fig.get_size_inches()[1] < 4.0
    finally:
        if fig is not None:
            plt.close(fig)
