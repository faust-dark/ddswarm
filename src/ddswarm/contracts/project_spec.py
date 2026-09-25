"""ProjectSpec contract (as specified in the architecture)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .enums import InputType, ProjectStatus


class ProjectSpec(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    project_id: str
    input_type: InputType
    objective: str = Field(min_length=1)
    targets: list[str] = Field(default_factory=list)
    starting_materials: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    exclusions: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    safety_constraints: list[str] = Field(default_factory=list)
    status: ProjectStatus = ProjectStatus.draft
