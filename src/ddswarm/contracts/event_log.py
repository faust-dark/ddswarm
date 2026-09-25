"""EventLog contract (one JSONL line per event)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .provenance import utcnow


class EventLog(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    task_id: str | None = None
    level: str = "info"  # info | warning | error | debug
    event: str
    input_hash: str | None = None
    output_hash: str | None = None
    tool_versions: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=utcnow)
