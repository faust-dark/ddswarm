"""AgentResult contract (as specified in the architecture)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .enums import AgentResultStatus
from .provenance import ArtifactRef, InputRef, ProvenanceEntry, ToolVersion, utcnow


class AgentResult(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    project_id: str
    agent: str
    task_id: str
    status: AgentResultStatus
    input_refs: list[InputRef] = Field(default_factory=list)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    provenance: list[ProvenanceEntry] = Field(default_factory=list)
    created_at: str = Field(default_factory=utcnow)
    tool_versions: list[ToolVersion] = Field(default_factory=list)
    reproducibility: dict[str, Any] = Field(default_factory=dict)
