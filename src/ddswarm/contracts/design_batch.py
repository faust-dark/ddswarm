"""DesignBatch artifact (designer_minion output)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .enums import AgentResultStatus, EvidenceLevel
from .provenance import ProvenanceEntry


class MoleculeDesign(BaseModel):
    id: str
    smiles: str
    parent_scaffold: str | None = None
    transformation: str | None = None
    rationale: str | None = None
    evidence_level: EvidenceLevel = EvidenceLevel.hypothetical


class DesignBatch(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    project_id: str
    agent: str = "designer_minion"
    status: AgentResultStatus
    molecules: list[MoleculeDesign] = Field(default_factory=list)
    provenance: list[ProvenanceEntry] = Field(default_factory=list)
