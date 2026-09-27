"""SelectivityAssessment artifact (selectivity_minion output).

A mechanistic target-selectivity assessment over a CONFIGURABLE panel of
receptors: each candidate is docked against every configured receptor, the
docking affinity is converted to a dissociation constant via the Gibbs
free-energy identity (ΔG = RT·ln Ki), a fractional receptor occupancy is
computed at a fixed reference concentration, and a general selectivity index
(Ki_off / Ki_primary) is derived for every off-target relative to the
designated primary (on-target) receptor.

An optional ADR interpretation is attached only when the panel matches a known
adverse-effect profile (e.g. the SSRI profile: SERT primary + 5-HT2C
off-target -> sexual-ADR risk). Any other panel yields pure affinity +
selectivity indices without inventing an ADR label.

All values are *calculated* (docking + deterministic thermodynamics), never
experimental evidence. Vina scores are relative, not calibrated absolute free
energies, so every Ki/occupancy is a ranking estimate. Docking reports binding,
not functional sign (agonist vs antagonist).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .enums import AgentResultStatus, EvidenceLevel
from .provenance import ProvenanceEntry

# Documented label sets of the SSRI sexual-ADR interpretation profile. The
# AdrInterpretation.label field is a free string so other profiles can add
# labels without touching this contract.
SexualAdrRisk = Literal["class_effect", "efficacy_tradeoff", "direct_5ht2c_engagement",
                        "direct_5ht2c_antagonism", "direct_5ht2c_agonism",
                        "insufficient_data"]
Ht2Function = Literal["antagonist_like", "agonist_like", "unknown"]

CALIBRATION_SOURCE = ("NIMH PDSP Ki Database (https://pdsp.unc.edu/databases/pdsp.php); "
                      "literature Ki values — verify against the primary source")

# ADR profile id of the built-in SSRI sexual-ADR interpretation (the only
# profile shipped today; more are pluggable without contract changes).
SSRI_SEXUAL_ADR_PROFILE = "sert_5ht2c_sexual_adr"


class ReceptorAffinity(BaseModel):
    receptor: str
    pdb_id: str | None = None
    best_affinity_kcal_per_mol: float | None = None   # most negative (best) pose
    ki_molar: float | None = None                     # exp(dG/RT); relative estimate
    occupancy: float | None = None                    # C_ref/(C_ref+Ki) ∈ [0,1]
    evidence_level: EvidenceLevel = EvidenceLevel.calculated
    note: str | None = None


class SelectivityIndex(BaseModel):
    """Ki_off / Ki_primary for one off-target vs the designated primary."""
    off_target: str
    primary: str
    value: float | None = None       # Ki_off / Ki_primary; larger = more selective
    evidence_level: EvidenceLevel = EvidenceLevel.calculated
    note: str | None = None


class AdrInterpretation(BaseModel):
    """Pluggable adverse-effect interpretation for a matching receptor panel."""
    profile: str                                   # e.g. "sert_5ht2c_sexual_adr"
    label: str                                     # profile-specific risk label
    rationale: str | None = None
    note: str | None = None


class MoleculeSelectivity(BaseModel):
    molecule_id: str
    canonical_smiles: str | None = None
    primary_receptor: str | None = None
    affinities: list[ReceptorAffinity] = Field(default_factory=list)
    selectivity_indices: list[SelectivityIndex] = Field(default_factory=list)
    adr: AdrInterpretation | None = None           # only when an ADR profile matches
    evidence_level: EvidenceLevel = EvidenceLevel.calculated


class SelectivitySummary(BaseModel):
    total_molecules: int = 0
    assessed: int = 0
    insufficient_data: int = 0
    primary_receptor: str | None = None


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
    schema_version: Literal["1.1"] = "1.1"
    project_id: str
    agent: str = "selectivity_minion"
    status: AgentResultStatus
    molecules: list[MoleculeSelectivity] = Field(default_factory=list)
    summary: SelectivitySummary = Field(default_factory=SelectivitySummary)
    reference_concentration_molar: float = 1e-8
    primary_receptor: str | None = None            # on-target (efficacy reference)
    reference_molecule: str | None = None          # parent SMILES (efficacy reference)
    reference_primary_occupancy: float | None = None  # parent occupancy on primary
    calibration: VinaCalibration | None = None     # hybrid Vina->Ki offset fit
    note: str | None = None
    provenance: list[ProvenanceEntry] = Field(default_factory=list)
