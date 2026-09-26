"""ReportDocument artifact (report_pdf_minion output).

Metadata for the exported PDF of a finished run: where the PDF lives, its
content hash (anti-tamper), how many pages/figures it contains, and the ordered
list of sections actually rendered. The PDF itself is a binary sidecar in the
store (referenced by `pdf_uri`); this JSON artifact is what the orchestrator
versions and hashes like any other artifact, so it participates in the same
provenance/validation pipeline as the rest of the run's outputs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .enums import AgentResultStatus
from .provenance import ProvenanceEntry


class ReportDocument(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    project_id: str
    run_id: str
    agent: str = "report_pdf_minion"
    status: AgentResultStatus
    pdf_uri: str
    pdf_hash: str
    page_count: int = 0
    figure_count: int = 0
    sections: list[str] = Field(default_factory=list)
    provenance: list[ProvenanceEntry] = Field(default_factory=list)
