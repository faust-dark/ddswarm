"""KnowledgePack artifact (researcher_minion output)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .enums import AgentResultStatus, EvidenceLevel
from .provenance import ProvenanceEntry


class KnowledgePackEntry(BaseModel):
    id: str
    type: Literal["target", "ligand", "reference", "admet", "mechanism"]
    source: str
    url: str | None = None
    identifier: str | None = None
    observed_value: str | None = None
    evidence_level: EvidenceLevel
    notes: str | None = None


class KnowledgePack(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    project_id: str
    agent: str = "researcher_minion"
    status: AgentResultStatus
    entries: list[KnowledgePackEntry] = Field(default_factory=list)
    provenance: list[ProvenanceEntry] = Field(default_factory=list)
