"""RunRequest: the project.json format accepted by the `ddswarm run` CLI.

Bundles a ProjectSpec, an optional task DAG, and an optional reasoning config
so a whole run is reproducible from one versioned file."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .project_spec import ProjectSpec


class RunTask(BaseModel):
    task_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    agent: str
    depends_on: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)


class ReasoningConfig(BaseModel):
    backend: str = "hermes"         # "hermes" (standard) | "none"
    command: str | None = None     # override the hermes binary / script (default "hermes")
    max_turns: int = 30
    run_budget: int = 300          # seconds


class RunRequest(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    project: ProjectSpec
    tasks: list[RunTask] = Field(default_factory=list)
    reasoning: ReasoningConfig | None = None
