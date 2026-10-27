"""report_pdf_minion: renders the final PDF "scientific essay" for a finished run.

Reads the aggregated Report + ValidationReport + every by-kind artifact directly
from the store, then delegates the actual rendering to the deterministic
`reporting/pdf_render.py` (the tool computes; the agent only aggregates and
persists). Runs last, after validation, so the PDF includes the validation
verdict and the full run trail.

The PDF is persisted as a binary sidecar in the store (versioned, never
overwritten); the JSON `ReportDocument` metadata artifact references it and is
itself versioned/hashed like any other artifact."""

from __future__ import annotations

import tempfile
from pathlib import Path

from ddswarm import __version__

from ..contracts.agent_result import AgentResult
from ..contracts.enums import AgentResultStatus
from ..contracts.provenance import ProvenanceEntry, ToolVersion
from ..contracts.report_document import ReportDocument
from ..logging.hashing import hash_jsonable
from ..orchestrator.runner import ARTIFACT_MODELS
from ..reporting.pdf_render import render_report_pdf
from .base import Agent, AgentContext

TOOL = ToolVersion(name="ddswarm", version=__version__)


class ReportPdfMinion(Agent):
    name = "report_pdf_minion"
    reads_whole_run = True  # scans every task's result in the run manifest

    def run(self, ctx: AgentContext) -> AgentResult:
        manifest = self.store.load_run_manifest(ctx.project_id, ctx.run_id)

        by_kind: dict[str, list] = {}
        for task_id in manifest.tasks:
            if task_id == ctx.task_id:
                continue  # own result doesn't exist yet
            try:
                res = self.store.load_latest_agent_result(ctx.project_id, ctx.run_id, task_id)
            except FileNotFoundError:
                continue
            for a in res.artifacts:
                if not a.uri or a.kind not in ARTIFACT_MODELS:
                    continue
                try:
                    payload = ARTIFACT_MODELS[a.kind].model_validate_json(self.store.artifact_path(a.uri).read_text())
                except Exception:  # noqa: BLE001 — skip unreadable artifact, don't fail the run
                    continue
                by_kind.setdefault(a.kind, []).append(payload)

        reports = by_kind.get("report", [])
        if not reports:
            return AgentResult(
                project_id=ctx.project_id, agent=self.name, task_id=ctx.task_id,
                status=AgentResultStatus.partial, artifacts=[], input_refs=ctx.input_refs,
                findings=["no report artifact to render; PDF not generated"],
                tool_versions=[TOOL],
                reproducibility={"run_id": ctx.run_id},
            )
        report = reports[0]
        validation = (by_kind.get("validation_report") or [None])[0]
        spec = self.store.load_project_spec(ctx.project_id)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_pdf = Path(tmp) / "report.pdf"
            meta = render_report_pdf(report, by_kind, spec, validation, tmp_pdf)
            pdf_bytes = tmp_pdf.read_bytes()

        artifact_id = f"report_document_{ctx.project_id}"
        pdf_path, _pdf_version, pdf_hash = self.store.save_binary_artifact(
            ctx.project_id, ctx.run_id, artifact_id, "report_document", "pdf", pdf_bytes,
        )

        input_hash = hash_jsonable({
            kind: [a.model_dump(mode="json", exclude={"provenance", "run_id"}) for a in arts]
            for kind, arts in sorted(by_kind.items())
        })
        output_hash = hash_jsonable({
            "sections": meta["sections"],
            "page_count": meta["page_count"],
            "figure_count": meta["figure_count"],
        })

        doc = ReportDocument(
            project_id=ctx.project_id, run_id=ctx.run_id, status=AgentResultStatus.success,
            pdf_uri=self.store.artifact_uri(pdf_path), pdf_hash=pdf_hash,
            page_count=meta["page_count"], figure_count=meta["figure_count"],
            sections=meta["sections"],
            provenance=[ProvenanceEntry(
                source="run aggregation → PDF rendering", tool_versions=[TOOL],
                input_hash=input_hash, output_hash=output_hash,
            )],
        )
        art = self.persist_artifact(ctx, "report_document", artifact_id, doc)
        return AgentResult(
            project_id=ctx.project_id, agent=self.name, task_id=ctx.task_id,
            status=AgentResultStatus.success, artifacts=[art], input_refs=ctx.input_refs,
            findings=[f"rendered {meta['page_count']}-page PDF report "
                      f"({meta['figure_count']} figures)"],
            provenance=[ProvenanceEntry(source="PDF rendering", output_hash=art.hash,
                                        tool_versions=[TOOL])],
            tool_versions=[TOOL],
            reproducibility={"run_id": ctx.run_id, "pdf_hash": pdf_hash},
        )
