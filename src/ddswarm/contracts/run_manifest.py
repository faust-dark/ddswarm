"""RunManifest contract."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .enums import RunStatus
from .provenance import utcnow


class RunManifest(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    project_id: str
    run_id: str
    status: RunStatus = RunStatus.queued
    task_graph: dict[str, list[str]] = Field(default_factory=dict)
    tasks: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utcnow)
    completed_at: str | None = None
    log_refs: list[str] = Field(default_factory=list)
