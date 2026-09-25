"""Enumerations shared across the JSON contracts."""

from enum import Enum


class InputType(str, Enum):
    target = "target"
    ligand = "ligand"
    class_ = "class"
    mixed = "mixed"


class ProjectStatus(str, Enum):
    draft = "draft"
    approved = "approved"
    running = "running"
    blocked = "blocked"
    completed = "completed"


class RunStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"
    blocked = "blocked"


class TaskStatus(str, Enum):
    queued = "queued"
    running = "running"
    success = "success"
    partial = "partial"
    failed = "failed"
    blocked = "blocked"
    insufficient_evidence = "insufficient_evidence"


class AgentResultStatus(str, Enum):
    success = "success"
    partial = "partial"
    failed = "failed"
    blocked = "blocked"
    insufficient_evidence = "insufficient_evidence"


class EvidenceLevel(str, Enum):
    observed = "observed"
    calculated = "calculated"
    predicted = "predicted"
    inferred = "inferred"
    hypothetical = "hypothetical"
