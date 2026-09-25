"""RetrosynthesisPlan artifact (retrosynthesis_minion output).

Per molecule: an RDKit synthetic-accessibility (SA) score and its easy/moderate/
hard classification — both *calculated*. Proposed synthesis routes are optional
and, when present, LLM-proposed (evidence_level='hypothetical'), never trusted
as demonstrated chemistry."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .enums import AgentResultStatus, EvidenceLevel
from .provenance import ProvenanceEntry

Accessibility = Literal["easy", "moderate", "hard"]


class SynthesisRoute(BaseModel):
    """An LLM-proposed synthesis route (hypothetical, not demonstrated)."""

    steps: list[str] = Field(default_factory=list)
    key_reagents: list[str] = Field(default_factory=list)
    feasibility_note: str | None = None
    evidence_level: EvidenceLevel = EvidenceLevel.hypothetical


class RetrosynthesisEntry(BaseModel):
    id: str
    canonical_smiles: str
    sa_score: float
    synthetic_accessibility: Accessibility
    routes: list[SynthesisRoute] = Field(default_factory=list)
    feasibility_note: str | None = None
    evidence_level: EvidenceLevel = EvidenceLevel.calculated


class RetrosynthesisSummary(BaseModel):
    total_molecules: int = 0
    easy: int = 0
    moderate: int = 0
    hard: int = 0
    molecules_with_routes: int = 0


class RetrosynthesisPlan(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    project_id: str
    agent: str = "retrosynthesis_minion"
    status: AgentResultStatus
    molecules: list[RetrosynthesisEntry] = Field(default_factory=list)
    summary: RetrosynthesisSummary = Field(default_factory=RetrosynthesisSummary)
    provenance: list[ProvenanceEntry] = Field(default_factory=list)
