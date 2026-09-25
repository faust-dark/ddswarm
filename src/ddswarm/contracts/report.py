"""Report artifact (report_minion output): a structured, evidence-aware
summary of one run, plus gaps, next steps, and honesty caveats."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .enums import AgentResultStatus, EvidenceLevel
from .provenance import ProvenanceEntry


class StageFinding(BaseModel):
    task_id: str
    agent: str
    status: str
    findings: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    artifact_kinds: list[str] = Field(default_factory=list)


class EvidenceBreakdown(BaseModel):
    """Counts of evidence-carrying units (entries/molecules/endpoints/poses)
    per evidence level. Coarse rollup; individual entries carry finer level."""

    observed: int = 0
    calculated: int = 0
    predicted: int = 0
    inferred: int = 0
    hypothetical: int = 0


class Report(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    project_id: str
    run_id: str
    agent: str = "report_minion"
    status: AgentResultStatus
    run_status: str
    executive_summary: str
    stage_findings: list[StageFinding] = Field(default_factory=list)
    evidence_breakdown: EvidenceBreakdown = Field(default_factory=EvidenceBreakdown)
    gaps: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    provenance: list[ProvenanceEntry] = Field(default_factory=list)
