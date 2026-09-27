"""HergAssessment artifact (herg_minion output) — hERG cardiotoxicity flag.

Deterministic hybrid (pharmacophore + lipophilicity) hERG liability screen, on
the same honesty standard as tox/cyp: a flag is a *predicted* liability, not a
patch-clamp assay, and potency (IC50) is not estimated.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .enums import AgentResultStatus, EvidenceLevel
from .provenance import ProvenanceEntry


class MoleculeHergProfile(BaseModel):
    """Per-molecule hERG assessment."""
    molecule_id: str
    canonical_smiles: str | None = None
    flagged: bool
    basic_amine_motifs: list[str] = Field(default_factory=list)
    aromatic_rings: int = 0
    clogp: float | None = None
    evidence_level: EvidenceLevel = EvidenceLevel.predicted
    note: str | None = None


class HergSummary(BaseModel):
    """Aggregate hERG statistics."""
    total_molecules: int = 0
    flagged: int = 0


class HergAssessment(BaseModel):
    """Artifact written by herg_minion."""
    schema_version: str = "1.0"
    project_id: str
    agent: str = "herg_minion"
    status: AgentResultStatus
    molecules: list[MoleculeHergProfile] = Field(default_factory=list)
    summary: HergSummary = Field(default_factory=HergSummary)
    note: str | None = None
    provenance: list[ProvenanceEntry] = Field(default_factory=list)
