"""DockingResult artifact (docking_minion output).

Docking is a deterministic in-silico computation: affinity scores, poses and
RMSD values are *calculated* (evidence_level = calculated), never experimental
evidence. Receptor QC and redocking validation are first-class, so a pose is
never reported without its receptor-quality and protocol context."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .enums import AgentResultStatus, EvidenceLevel
from .provenance import ProvenanceEntry


class ReceptorQCCheck(BaseModel):
    check: str          # e.g. "missing_residues", "waters_present"
    status: str         # "ok" | "issue" | "n_a"
    detail: str = ""


class ReceptorInfo(BaseModel):
    source: str                 # "RCSB PDB"
    identifier: str             # "2ITY"
    url: str | None = None
    preparation_tool: str | None = None
    preparation_status: str = "not_prepared"   # "prepared" | "not_prepared" | "failed"
    qc_status: str = "not_run"                 # "ok" | "issues" | "not_run"
    qc_checks: list[ReceptorQCCheck] = Field(default_factory=list)
    atom_count: int | None = None
    residue_count: int | None = None
    chain_count: int | None = None
    note: str | None = None


class SearchBox(BaseModel):
    center: list[float]   # [x, y, z]
    size: list[float]     # [sx, sy, sz]


class DockingProtocol(BaseModel):
    engine: str = "vina"
    engine_version: str | None = None
    search_box: SearchBox | None = None
    exhaustiveness: int | None = None
    num_modes: int | None = None
    seed: int | None = None
    note: str | None = None


class DockedPose(BaseModel):
    molecule_id: str
    rank: int
    affinity_kcal_per_mol: float | None = None
    rmsd_lower_bound: float | None = None   # Vina: rmsd l.b. vs best mode
    rmsd_upper_bound: float | None = None   # Vina: rmsd u.b. vs best mode
    pose_file: str | None = None            # path relative to the artifact
    interactions: list[str] = Field(default_factory=list)  # reserved; not computed yet


class RedockingValidation(BaseModel):
    status: str = "not_run"      # "passed" | "failed" | "not_run" | "insufficient_data"
    reference_ligand: str | None = None
    reference_source: str | None = None
    rmsd_angstrom: float | None = None
    threshold_angstrom: float = 2.0
    note: str | None = None


class DockingSummary(BaseModel):
    total_molecules: int = 0
    docked: int = 0
    failed: int = 0


class DockingResult(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    project_id: str
    agent: str = "docking_minion"
    status: AgentResultStatus
    evidence_level: EvidenceLevel = EvidenceLevel.calculated
    receptor: ReceptorInfo | None = None
    protocol: DockingProtocol | None = None
    poses: list[DockedPose] = Field(default_factory=list)
    redocking_validation: RedockingValidation | None = None
    summary: DockingSummary = Field(default_factory=DockingSummary)
    provenance: list[ProvenanceEntry] = Field(default_factory=list)
