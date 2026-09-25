"""TaskSpec contract (execution ledger for one task)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .enums import TaskStatus
from .provenance import InputRef, utcnow


class AttemptRecord(BaseModel):
    attempt: int
    started_at: str = Field(default_factory=utcnow)
    finished_at: str | None = None
    status: TaskStatus | None = None
    error: str | None = None


class TaskSpec(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    task_id: str
    parent_task_id: str | None = None
    agent: str
    project_id: str
    run_id: str
    input_refs: list[InputRef] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    status: TaskStatus = TaskStatus.queued
    retry_count: int = 0
    attempts: list[AttemptRecord] = Field(default_factory=list)
