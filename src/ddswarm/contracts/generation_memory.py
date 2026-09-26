"""GenerationMemory: the swarm's structured recollection of one generation.

The memory_minion runs at the end of a generation and distils the run into
best candidates, avoided scaffolds, and insights, so the next generation's
orchestrator/designer can build on what worked and avoid what did not. Written
as an artifact AND as a durable 'generation' entry in project memory (cross-run
feedback loop).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .provenance import ProvenanceEntry


class GenerationCandidate(BaseModel):
    smiles: str
    score: float | None = None
    reasons: list[str] = Field(default_factory=list)


class GenerationMemory(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    project_id: str
    run_id: str
    generation: int = 0
    summary: str = ""
    best_candidates: list[GenerationCandidate] = Field(default_factory=list)
    avoided: list[str] = Field(default_factory=list)
    insights: list[str] = Field(default_factory=list)
    converged: bool = False
    provenance: list[ProvenanceEntry] = Field(default_factory=list)
