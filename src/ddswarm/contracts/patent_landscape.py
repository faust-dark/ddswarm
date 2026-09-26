"""PatentLandscape artifact (patent_minion output).

Patent / freedom-to-operate knowledge for candidate molecules and/or the
project target, delegated to a reasoning backend (Hermes subagent). Every entry
is LLM recall — evidence_level is forced to 'inferred' and tagged 'verify
against primary source'. This is a research aid, NOT a legal freedom-to-operate
opinion and NOT a comprehensive patent search.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .enums import AgentResultStatus, EvidenceLevel
from .provenance import ProvenanceEntry


class PatentEntry(BaseModel):
    id: str
    molecule_id: str | None = None
    patent_id: str
    title: str | None = None
    assignee: str | None = None
    status: str = "unknown"          # granted | pending | expired | unknown
    jurisdiction: str | None = None  # US | EP | WO | ...
    relevance: str = "background"    # covers_candidate | related_scaffold | background
    source: str | None = None
    url: str | None = None
    evidence_level: EvidenceLevel = EvidenceLevel.inferred
    notes: str | None = None


class PatentSummary(BaseModel):
    total_entries: int = 0
    relevance_counts: dict[str, int] = Field(default_factory=dict)
    flagged: int = 0                 # entries with relevance == covers_candidate


class PatentLandscape(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    project_id: str
    agent: str = "patent_minion"
    status: AgentResultStatus
    entries: list[PatentEntry] = Field(default_factory=list)
    summary: PatentSummary = Field(default_factory=PatentSummary)
    note: str | None = None
    provenance: list[ProvenanceEntry] = Field(default_factory=list)
