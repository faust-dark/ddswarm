"""ScreenedBatch artifact (screener_minion output)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .enums import AgentResultStatus
from .provenance import ProvenanceEntry


class MoleculeProperties(BaseModel):
    molecular_weight: float
    logp: float
    hbd: int
    hba: int
    tpsa: float
    rotatable_bonds: int
    formal_charge: int
    ring_count: int


class MoleculeScreenResult(BaseModel):
    id: str
    input_smiles: str
    canonical_smiles: str | None = None
    valid: bool
    sanitization_error: str | None = None
    properties: MoleculeProperties | None = None
    rule_of_five_violations: list[str] = Field(default_factory=list)
    structural_alerts: list[str] = Field(default_factory=list)
    advisories: list[str] = Field(default_factory=list)
    passed: bool
    elimination_reasons: list[str] = Field(default_factory=list)


class ScreenSummary(BaseModel):
    total: int
    passed: int
    rejected: int
    invalid: int


class ScreenedBatch(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    project_id: str
    agent: str = "screener_minion"
    status: AgentResultStatus
    molecules: list[MoleculeScreenResult] = Field(default_factory=list)
    summary: ScreenSummary
    provenance: list[ProvenanceEntry] = Field(default_factory=list)
