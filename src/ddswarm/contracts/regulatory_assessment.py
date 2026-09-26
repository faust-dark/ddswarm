"""RegulatoryAssessment artifact (regulatory_minion output).

Regulatory knowledge — agency designations, approval status, and scheduling —
for the project target(s) and/or candidate molecules, delegated to a reasoning
backend (Hermes subagent). Every entry is LLM recall — evidence_level is forced
to 'inferred' and tagged 'verify against primary source'. This is a research
aid, NOT a regulatory submission or legal determination.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .enums import AgentResultStatus, EvidenceLevel
from .provenance import ProvenanceEntry


class RegulatoryFinding(BaseModel):
    id: str
    molecule_id: str | None = None
    agency: str | None = None        # FDA | EMA | WHO | DEA | ...
    designation: str | None = None   # approved | orphan drug | fast track | schedule II | ...
    status: str | None = None
    source: str | None = None
    url: str | None = None
    evidence_level: EvidenceLevel = EvidenceLevel.inferred
    notes: str | None = None


class RegulatorySummary(BaseModel):
    total_entries: int = 0
    agency_counts: dict[str, int] = Field(default_factory=dict)


class RegulatoryAssessment(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    project_id: str
    agent: str = "regulatory_minion"
    status: AgentResultStatus
    entries: list[RegulatoryFinding] = Field(default_factory=list)
    summary: RegulatorySummary = Field(default_factory=RegulatorySummary)
    note: str | None = None
    provenance: list[ProvenanceEntry] = Field(default_factory=list)
