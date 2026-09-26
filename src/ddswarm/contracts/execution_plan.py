"""ExecutionPlan: the orchestrator's plan for one generation of the swarm.

The orchestrator_minion runs first, interprets the ProjectSpec + task graph,
and emits this plan: which stages run, in what order, how many generations,
and the convergence rule. It is the auditable record of *why* and *how* the
pipeline was assembled — the PDF report opens with this reasoning.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .provenance import ProvenanceEntry


class PlanStage(BaseModel):
    """One stage of the plan: an agent and its role in the workflow."""

    stage_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    agent: str
    role: str
    depends_on: list[str] = Field(default_factory=list)


class ConvergenceConfig(BaseModel):
    max_generations: int = 1
    stop_on_no_new_candidates: bool = True


class ExecutionPlan(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    project_id: str
    run_id: str
    generation: int = 0
    stages: list[PlanStage] = Field(default_factory=list)
    convergence: ConvergenceConfig = Field(default_factory=ConvergenceConfig)
    rationale: str = ""
    provenance: list[ProvenanceEntry] = Field(default_factory=list)
