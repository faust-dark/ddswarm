"""ToxProfile artifact (tox_minion output).

Per molecule: deterministic structural toxicophore alerts (SMARTS matched,
each with endpoint + literature source). An alert is a *predicted* potential
liability from a literature rule — it is NOT experimental evidence of
toxicity, and the contract keeps that separation explicit."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .enums import AgentResultStatus, EvidenceLevel
from .provenance import ProvenanceEntry


class ToxAlert(BaseModel):
    name: str
    smarts: str
    endpoint: str
    source: str
    evidence_level: EvidenceLevel = EvidenceLevel.predicted


class MoleculeToxAssessment(BaseModel):
    id: str
    canonical_smiles: str
    alerts: list[ToxAlert] = Field(default_factory=list)
    liability_count: int = 0
    risk_note: str | None = None


class ToxSummary(BaseModel):
    total_molecules: int = 0
    molecules_with_alerts: int = 0
    total_alerts: int = 0


class ToxProfile(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    project_id: str
    agent: str = "tox_minion"
    status: AgentResultStatus
    molecules: list[MoleculeToxAssessment] = Field(default_factory=list)
    summary: ToxSummary = Field(default_factory=ToxSummary)
    provenance: list[ProvenanceEntry] = Field(default_factory=list)
