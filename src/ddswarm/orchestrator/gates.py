"""Gating logic. Gate A (input validity) plus per-result gating."""

from __future__ import annotations

from ..contracts.enums import AgentResultStatus
from ..contracts.project_spec import ProjectSpec


class GateError(Exception):
    pass


class BlockedByGate(GateError):
    pass


def gate_a_input_validity(spec: ProjectSpec) -> list[str]:
    """Return a list of problems; empty list means the spec passes Gate A."""
    problems: list[str] = []
    if not spec.objective.strip():
        problems.append("objective is empty")
    if spec.input_type.value == "target" and not spec.targets:
        problems.append("input_type=target but no targets provided")
    if spec.input_type.value == "ligand" and not spec.starting_materials:
        problems.append("input_type=ligand but no starting_materials provided")
    if spec.status.value not in ("approved", "running"):
        problems.append(f"ProjectSpec status must be 'approved', got '{spec.status.value}'")
    return problems


def result_ok(status: AgentResultStatus) -> bool:
    """Logging helper: reads as an outright success (success/partial).

    This is NOT the gating rule — gating lives in the runner and, since
    Increment 38, only `blocked` (a safety verdict) stops downstream."""
    return status in (AgentResultStatus.success, AgentResultStatus.partial)
