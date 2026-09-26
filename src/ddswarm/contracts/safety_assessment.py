"""SafetyAssessment artifact (safety_minion output).

A deterministic gate over a curated, NON-EXHAUSTIVE list of controlled
substances (exact canonical-SMILES match) and dual-use/biosafety targets (name
match). A flag is an *observed* match against that list — it is NOT a legal
determination and NOT a comprehensive database."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .enums import AgentResultStatus, EvidenceLevel
from .provenance import ProvenanceEntry

SafetyVerdict = Literal["pass", "flag", "block"]


class SafetyFlag(BaseModel):
    id: str
    kind: str          # controlled_substance | dual_use | biosafety | compliance
    level: str         # info | flag | block
    subject: str       # molecule id or target name
    reason: str
    source: str
    evidence_level: EvidenceLevel = EvidenceLevel.observed


class SafetySummary(BaseModel):
    total_flags: int = 0
    block: int = 0
    flag: int = 0


class SafetyAssessment(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    project_id: str
    agent: str = "safety_minion"
    status: AgentResultStatus
    verdict: SafetyVerdict = "pass"
    flags: list[SafetyFlag] = Field(default_factory=list)
    summary: SafetySummary = Field(default_factory=SafetySummary)
    note: str | None = None
    provenance: list[ProvenanceEntry] = Field(default_factory=list)
