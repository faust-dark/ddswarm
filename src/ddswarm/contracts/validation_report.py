"""ValidationReport artifact (validation_minion output).

A deterministic verification of a finished run: artifact schema validity,
provenance completeness, hash integrity (anti-tamper), cross-agent consistency,
evidence-level sanity, numeric sanity, and report-breakdown consistency. Each
check is pass/fail/warn; a failed check means the run's own bookkeeping is
inconsistent — it is a reproducibility audit, not a scientific judgment."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .enums import AgentResultStatus
from .provenance import ProvenanceEntry


class ValidationCheck(BaseModel):
    check_id: str
    name: str
    status: str  # pass | fail | warn
    detail: str | None = None


class ValidationSummary(BaseModel):
    total_checks: int = 0
    passed: int = 0
    failed: int = 0
    warned: int = 0


class ValidationReport(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    project_id: str
    run_id: str
    agent: str = "validation_minion"
    status: AgentResultStatus
    checks: list[ValidationCheck] = Field(default_factory=list)
    summary: ValidationSummary = Field(default_factory=ValidationSummary)
    reproducibility: dict[str, int] = Field(default_factory=dict)
    provenance: list[ProvenanceEntry] = Field(default_factory=list)
