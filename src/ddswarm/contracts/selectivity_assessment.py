"""SelectivityAssessment artifact (selectivity_minion output).

A mechanistic target-selectivity / sexual-adverse-effect assessment: each
candidate is docked against SERT, 5-HT2A and 5-HT2C; the docking affinity is
converted to a dissociation constant via the Gibbs free-energy identity
(ΔG = RT·ln Ki), a fractional receptor occupancy is computed at a fixed
reference concentration, and a selectivity index + a deterministic
sexual-ADR risk label are derived.

All values are *calculated* (docking + deterministic thermodynamics), never
experimental evidence. Vina scores are relative, not calibrated absolute free
energies, so every Ki/occupancy is a ranking estimate. Docking reports binding,
not functional sign (agonist vs antagonist) — a direct 5-HT2C engagement is
therefore flagged as of unknown sign.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .enums import AgentResultStatus, EvidenceLevel
from .provenance import ProvenanceEntry

ReceptorKind = Literal["SERT", "HTR2A", "HTR2C"]
SexualAdrRisk = Literal["class_effect", "efficacy_tradeoff", "direct_5ht2c_engagement",
                        "direct_5ht2c_antagonism", "direct_5ht2c_agonism",
                        "insufficient_data"]
Ht2Function = Literal["antagonist_like", "agonist_like", "unknown"]

CALIBRATION_SOURCE = ("NIMH PDSP Ki Database (https://pdsp.unc.edu/databases/pdsp.php); "
                      "literature Ki values — verify against the primary source")


class ReceptorAffinity(BaseModel):
    receptor: ReceptorKind
    pdb_id: str | None = None
    best_affinity_kcal_per_mol: float | None = None   # most negative (best) pose
    ki_molar: float | None = None                     # exp(dG/RT); relative estimate
    occupancy: float | None = None                    # C_ref/(C_ref+Ki) ∈ [0,1]
    evidence_level: EvidenceLevel = EvidenceLevel.calculated
    note: str | None = None


class MoleculeSelectivity(BaseModel):
    molecule_id: str
    canonical_smiles: str | None = None
    affinities: list[ReceptorAffinity] = Field(default_factory=list)
    selectivity_index_5ht2c_sert: float | None = None  # Ki_HTR2C / Ki_SERT
    selectivity_index_5ht2a_sert: float | None = None  # Ki_HTR2A / Ki_SERT
    sert_occupancy: float | None = None
    htr2c_occupancy: float | None = None
    htr2a_occupancy: float | None = None
    sexual_adr_risk: SexualAdrRisk = "insufficient_data"
    sexual_adr_rationale: str | None = None
    # functional sign of the direct 5-HT2C engagement (from documented chemotype
    # matching, NOT a functional assay) — refines direct_5ht2c_engagement into
    # direct_5ht2c_antagonism / direct_5ht2c_agonism when a chemotype matches.
    ht2c_function: Ht2Function | None = None
    ht2c_function_note: str | None = None
    evidence_level: EvidenceLevel = EvidenceLevel.calculated


class SelectivitySummary(BaseModel):
    total_molecules: int = 0
    assessed: int = 0
    insufficient_data: int = 0


class VinaCalibration(BaseModel):
    """Hybrid Vina-score calibration (offset-only, per-receptor intercept)."""
    method: str = ("offset-only per-receptor calibration: slope a = 1 (fixed), "
                   "intercept b_r = mean(ΔG_literature − Vina_score)")
    slope: float = 1.0
    intercepts: dict[str, float | None] = Field(default_factory=dict)  # receptor -> b
    n_points: dict[str, int] = Field(default_factory=dict)             # receptor -> N
    rmsd_kcal_per_mol: dict[str, float | None] = Field(default_factory=dict)
    calibration_source: str = CALIBRATION_SOURCE
    note: str | None = None


class SelectivityAssessment(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    project_id: str
    agent: str = "selectivity_minion"
    status: AgentResultStatus
    molecules: list[MoleculeSelectivity] = Field(default_factory=list)
    summary: SelectivitySummary = Field(default_factory=SelectivitySummary)
    reference_concentration_molar: float = 1e-8
    reference_molecule: str | None = None          # parent SMILES (efficacy reference)
    reference_sert_occupancy: float | None = None  # parent SERT occupancy at C_ref
    calibration: VinaCalibration | None = None     # hybrid Vina->Ki offset fit
    note: str | None = None
    provenance: list[ProvenanceEntry] = Field(default_factory=list)
