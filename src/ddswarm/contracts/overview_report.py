"""OverviewReport artifact (overview_minion output).

A knowledge summary placed at the TOP of the final PDF: the verbatim user prompt
that started the research, followed by a plain-language overview of mechanism of
action, chemical structure, and therapeutic uses — synthesized from the
reasoning backend (LLM recall, evidence_level=inferred) and enriched with
deterministic facts from ddswarm's own databases (receptor registry, curated
targets), which are observed.

The agent always succeeds: without a reasoning backend it still echoes the
prompt and the deterministic own-database facts, leaving the narrative fields
empty (never fabricating a mechanism/uses it cannot recall).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .enums import AgentResultStatus
from .provenance import ProvenanceEntry


class OverviewReport(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    project_id: str
    agent: str = "overview_minion"
    status: AgentResultStatus
    user_prompt: str
    mechanism_of_action: str | None = None
    chemical_structure: str | None = None
    therapeutic_uses: str | None = None
    citations: list[str] = Field(default_factory=list)
    own_database_facts: list[str] = Field(default_factory=list)
    note: str = ""
    provenance: list[ProvenanceEntry] = Field(default_factory=list)
