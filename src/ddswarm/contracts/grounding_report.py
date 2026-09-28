"""GroundingReport artifact (grounding_minion output) — external-database linkage.

Resolves each molecule against open public databases (ChEMBL, PubChem, PubMed;
DrugBank identifier via PubChem xref) and records every result as an *observed*
entry carrying an external identifier + URL + retrieval timestamp. This is the
project's first machine-readable, provenance-complete link to real external
records — the LLM-recalled knowledge entries remain *inferred*; grounded entries
are *observed*.

Network-dependent and non-deterministic across time (opt-in), so grounding is
excluded from the reproducibility hash.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .enums import AgentResultStatus, EvidenceLevel
from .provenance import ProvenanceEntry


class GroundedEntry(BaseModel):
    """One external-database record resolved for a molecule."""
    source: Literal["chembl", "pubchem", "pubmed", "drugbank"]
    external_id: str
    url: str
    title: str
    detail: str | None = None
    evidence_level: EvidenceLevel = EvidenceLevel.observed
    retrieved_at: str


class BindingAffinity(BaseModel):
    """One measured binding affinity (Ki/Kd/IC50) resolved from ChEMBL for a molecule."""
    target_chembl_id: str | None = None
    target_pref_name: str | None = None
    standard_type: Literal["Ki", "Kd", "IC50"]
    value_nm: float
    document_year: int | None = None
    evidence_level: EvidenceLevel = EvidenceLevel.observed


class MoleculeGrounding(BaseModel):
    """Per-molecule grounding result."""
    molecule_id: str
    canonical_smiles: str | None = None
    inchi_key: str | None = None
    status: Literal["grounded", "not_found", "failed"]
    entries: list[GroundedEntry] = Field(default_factory=list)
    binding_affinities: list[BindingAffinity] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class GroundingSummary(BaseModel):
    total_molecules: int = 0
    grounded: int = 0
    not_found: int = 0
    failed: int = 0
    total_entries: int = 0
    total_affinities: int = 0


class GroundingReport(BaseModel):
    schema_version: str = "1.1"
    project_id: str
    agent: str = "grounding_minion"
    status: AgentResultStatus
    molecules: list[MoleculeGrounding] = Field(default_factory=list)
    summary: GroundingSummary = Field(default_factory=GroundingSummary)
    note: str | None = None
    provenance: list[ProvenanceEntry] = Field(default_factory=list)
