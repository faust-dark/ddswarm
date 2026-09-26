"""ClinicalAssessment artifact (clinical_minion output).

Clinical-development knowledge for the project target(s) and/or candidate
molecules, delegated to a reasoning backend (Hermes subagent). Every entry is
LLM recall — evidence_level is forced to 'inferred' and tagged 'verify against
primary source'. This is a research aid, NOT a clinical recommendation.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .enums import AgentResultStatus, EvidenceLevel
from .provenance import ProvenanceEntry


class ClinicalFinding(BaseModel):
    id: str
    molecule_id: str | None = None
    entity: str | None = None        # drug candidate, approved drug, or target
    phase: str = "unknown"           # preclinical | phase_1 | phase_2 | phase_3 | approved | unknown
    indication: str | None = None
    status: str | None = None        # ongoing | completed | terminated | ...
    source: str | None = None
    url: str | None = None
    evidence_level: EvidenceLevel = EvidenceLevel.inferred
    notes: str | None = None


class ClinicalSummary(BaseModel):
    total_entries: int = 0
    phase_counts: dict[str, int] = Field(default_factory=dict)


class ClinicalAssessment(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    project_id: str
    agent: str = "clinical_minion"
    status: AgentResultStatus
    entries: list[ClinicalFinding] = Field(default_factory=list)
    summary: ClinicalSummary = Field(default_factory=ClinicalSummary)
    note: str | None = None
    provenance: list[ProvenanceEntry] = Field(default_factory=list)
