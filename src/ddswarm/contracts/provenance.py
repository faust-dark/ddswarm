"""Provenance primitives shared by every artifact."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class ToolVersion(BaseModel):
    name: str
    version: str | None = None
    model: str | None = None


class SourceRef(BaseModel):
    source: str
    url: str | None = None
    identifier: str | None = None


class ProvenanceEntry(BaseModel):
    source: str
    url: str | None = None
    identifier: str | None = None
    timestamp: str = Field(default_factory=utcnow)
    tool_versions: list[ToolVersion] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    input_hash: str | None = None
    output_hash: str | None = None
    input_exact: str | None = None
    output_exact: str | None = None


class InputRef(BaseModel):
    ref_type: str  # "artifact" | "external" | "file"
    ref_id: str | None = None
    uri: str | None = None
    hash: str | None = None


class ArtifactRef(BaseModel):
    artifact_id: str
    kind: str
    uri: str | None = None
    version: int = 1
    hash: str | None = None
