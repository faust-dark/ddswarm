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


def test_structural_modification_lines():
    from ddswarm.contracts.design_batch import DesignBatch, MoleculeDesign
    from ddswarm.reporting.pdf_render import _structural_modification_lines

    db = DesignBatch(project_id="p", status=AgentResultStatus.success, molecules=[
        MoleculeDesign(id="a", smiles="X1", transformation="cl_to_f",
                       rationale="chlorine -> fluorine halogen swap (a common bioisosteric replacement)"),
        MoleculeDesign(id="b", smiles="X2", transformation="cl_to_f",
                       rationale="chlorine -> fluorine halogen swap (a common bioisosteric replacement)"),
        MoleculeDesign(id="c", smiles="X3", transformation="8-NO2 → 8-F",
                       rationale="the nitro group drives sedation; fluorine lowers it"),
        MoleculeDesign(id="d", smiles="X4", transformation="n_demethylate",
                       rationale="minimal analog via 'n_demethylate' (label only; no new structure)"),
    ])
    lines = _structural_modification_lines([db])
    assert len(lines) == 2  # duplicate collapsed, label-only skipped
    assert any("chlorine -> fluorine halogen swap" in l for l in lines)
    assert any("8-NO2 → 8-F" in l and "sedation" in l for l in lines)
    assert not any("label only" in l for l in lines)


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
    assert "Structural modifications rationale" in doc.sections
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


def test_identified_candidate_renders_iupac_name(tmp_path):
    # The "Identified candidate" section must show the IUPAC name alongside
    # SMILES + InChIKey when one is present (fetched by default from CACTUS).
    import shutil
    import subprocess

    if shutil.which("pdftotext") is None:
        import pytest
        pytest.skip("pdftotext not available")

    from ddswarm.contracts.report import CandidateSelection

    iupac = ("9-fluoro-11,17-dihydroxy-17-(2-hydroxyacetyl)-10,13,16-trimethyl-"
             "6,7,8,11,12,14,15,16-octahydro-6H-cyclopenta[a]phenanthren-3-one")
    report = Report(
        project_id="p", run_id="run_1", status=AgentResultStatus.success,
        run_status="completed", executive_summary="s",
        identified_candidate=CandidateSelection(
            rank=1, smiles="CC1CC2C3CCC4=CC(=O)C=CC4(C)C3(F)C(O)CC2(C)C1(O)C(=O)CO",
            score=0.5, iupac_name=iupac),
        provenance=[ProvenanceEntry(source="test")],
    )
    spec = ProjectSpec(project_id="p", input_type="mixed", objective="obj",
                       targets=["X"], status="approved")
    out = tmp_path / "r.pdf"
    render_report_pdf(report, {}, spec, None, out)
    text = subprocess.run(["pdftotext", str(out), "-"], capture_output=True, text=True).stdout
    flat = "".join(text.split())  # drop all whitespace so line-wrapped names stay contiguous
    assert "IUPAC" in flat
    assert "9-fluoro-11,17-dihydroxy-17-(2-hydroxyacetyl)" in flat
