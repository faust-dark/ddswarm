"""Project memory contracts: durable, cross-run conclusions.

A MemoryEntry is the swarm's own conclusion about a project, written by an
agent after a run and read by agents in later runs (the feedback loop). It is
provenanced and carries an evidence_level: conclusions are *inferred* (the
swarm's reasoning), never observed data."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .enums import EvidenceLevel
from .provenance import ProvenanceEntry, utcnow

MemoryKind = Literal["finding", "decision", "lesson", "constraint", "feedback"]


class MemoryEntry(BaseModel):
    id: str
    project_id: str
    kind: MemoryKind
    content: str
    data: dict = Field(default_factory=dict)
    evidence_level: EvidenceLevel = EvidenceLevel.inferred
    source_agent: str
    run_id: str
    created_at: str = Field(default_factory=utcnow)
    provenance: list[ProvenanceEntry] = Field(default_factory=list)


class ProjectMemory(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    project_id: str
    entries: list[MemoryEntry] = Field(default_factory=list)
