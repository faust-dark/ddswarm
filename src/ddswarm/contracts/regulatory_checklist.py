"""RegulatoryChecklist: a seven-dimension (7D) regulatory readiness checklist.

Unlike RegulatoryAssessment (agency/designation findings), this is a structured
readiness review across seven fixed dimensions. Each entry is LLM recall
(reasoning backend), forced to evidence_level='inferred', and tagged
'verify against primary source'. This is a research aid, NOT a regulatory
submission or legal determination.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .enums import EvidenceLevel
from .provenance import ProvenanceEntry

ChecklistDimension = Literal[
    "quality_cmc",
    "nonclinical_toxicity",
    "clinical_efficacy",
    "clinical_safety",
    "pharmacology_pkpd",
    "intellectual_property",
    "regulatory_pathway",
]

ChecklistStatus = Literal["ok", "flag", "unknown", "na"]

DIMENSION_LABELS: dict[str, str] = {
    "quality_cmc": "Quality / CMC",
    "nonclinical_toxicity": "Nonclinical toxicity",
    "clinical_efficacy": "Clinical efficacy",
    "clinical_safety": "Clinical safety",
    "pharmacology_pkpd": "Pharmacology / PK-PD",
    "intellectual_property": "Intellectual property / FTO",
    "regulatory_pathway": "Regulatory pathway",
}


class RegulatoryChecklistEntry(BaseModel):
    id: str
    dimension: ChecklistDimension
    status: ChecklistStatus = "unknown"
    summary: str = ""
    evidence_level: EvidenceLevel = EvidenceLevel.inferred
    source: str = "Hermes subagent (LLM reasoning)"
    notes: str = ""


class RegulatoryChecklistSummary(BaseModel):
    total_dimensions: int = 0
    ok: int = 0
    flag: int = 0
    unknown: int = 0
    na: int = 0


class RegulatoryChecklist(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    project_id: str
    status: str
    entries: list[RegulatoryChecklistEntry] = Field(default_factory=list)
    summary: RegulatoryChecklistSummary = Field(default_factory=RegulatoryChecklistSummary)
    provenance: list[ProvenanceEntry] = Field(default_factory=list)
