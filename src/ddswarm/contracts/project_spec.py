"""ProjectSpec contract (as specified in the architecture)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .enums import InputType, ProjectStatus


class ProjectSpec(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    project_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    input_type: InputType
    objective: str = Field(min_length=1)
    targets: list[str] = Field(default_factory=list)
    starting_materials: list[str] = Field(default_factory=list)
    # ---- primary input, per the swarm workflow schema ----
    user_prompt: str | None = None
    mechanism_of_action: str | None = None
    indication: str | None = None
    existing_drugs: list[str] = Field(default_factory=list)
    chemical_scaffold: str | None = None
    adme_issue: str | None = None
    # ---- PK/PD simulation inputs (optional; used by clinical_minion) ----
    dose_mg: float | None = None
    route_of_administration: str = "oral"
    constraints: list[str] = Field(default_factory=list)
    exclusions: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    safety_constraints: list[str] = Field(default_factory=list)
    status: ProjectStatus = ProjectStatus.draft
