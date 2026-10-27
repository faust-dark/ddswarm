"""Enumerations shared across the JSON contracts."""

from enum import StrEnum


class InputType(StrEnum):
    target = "target"
    ligand = "ligand"
    class_ = "class"
    mixed = "mixed"


class ProjectStatus(StrEnum):
    draft = "draft"
    approved = "approved"
    running = "running"
    blocked = "blocked"
    completed = "completed"


class RunStatus(StrEnum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"
    blocked = "blocked"


class TaskStatus(StrEnum):
    queued = "queued"
    running = "running"
    success = "success"
    partial = "partial"
    failed = "failed"
    blocked = "blocked"
    insufficient_evidence = "insufficient_evidence"


class AgentResultStatus(StrEnum):
    success = "success"
    partial = "partial"
    failed = "failed"
    blocked = "blocked"
    insufficient_evidence = "insufficient_evidence"


class EvidenceLevel(StrEnum):
    observed = "observed"
    calculated = "calculated"
    predicted = "predicted"
    inferred = "inferred"
    hypothetical = "hypothetical"
