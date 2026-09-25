"""FormulationAssessment artifact (formulation_minion output).

Deterministic, rule-based developability estimates from RDKit descriptors:
aqueous-solubility class, permeability class, BCS classification, ionizable
groups, and salt/prodrug suggestions. All *predicted* from heuristic rules —
not measured experimental properties."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .enums import AgentResultStatus, EvidenceLevel
from .provenance import ProvenanceEntry


class MoleculeFormulation(BaseModel):
    id: str
    canonical_smiles: str
    logp: float | None = None
    tpsa: float | None = None
    solubility_class: str = "unknown"    # high | moderate | low
    permeability_class: str = "unknown"  # high | moderate | low
    bcs_class: str = "unknown"           # I | II | III | IV | unknown
    ionizable_groups: list[str] = Field(default_factory=list)
    salt_suggestion: str | None = None
    prodrug_suggestion: str | None = None
    note: str | None = None
    evidence_level: EvidenceLevel = EvidenceLevel.predicted


class FormulationSummary(BaseModel):
    total_molecules: int = 0
    bcs_counts: dict[str, int] = Field(default_factory=dict)


class FormulationAssessment(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    project_id: str
    agent: str = "formulation_minion"
    status: AgentResultStatus
    molecules: list[MoleculeFormulation] = Field(default_factory=list)
    summary: FormulationSummary = Field(default_factory=FormulationSummary)
    provenance: list[ProvenanceEntry] = Field(default_factory=list)
