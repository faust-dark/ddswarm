"""Deterministic thermodynamic conversion of docking affinity -> dissociation
constant, receptor occupancy, and target-selectivity indices.

The math here is exact and deterministic (Gibbs free-energy <-> dissociation
constant identity; saturable occupancy law). The INPUTS (AutoDock Vina affinity
scores) are RELATIVE — a ranking, not a calibrated absolute free energy — so
every derived Ki/occupancy is a *relative estimate for ranking*, never an
absolute affinity. That caveat is carried on the contract, not re-explained per
molecule.
"""

from __future__ import annotations

import math

# Gas constant in kcal/(mol·K); RT at 298.15 K ~= 0.5925 kcal/mol.
GAS_CONSTANT_KCAL_PER_MOL_K = 1.987e-3
DEFAULT_TEMPERATURE_K = 298.15

# Modeling thresholds for the sexual-ADR classification (documented, citable
# as modeling choices; overridable via agent params):
#   - direct 5-HT2C engagement = drug's own 5-HT2C occupancy at the reference
#     concentration reaches this fraction (2% at 10 nM ~= Ki < 490 nM).
#   - efficacy retention = candidate SERT occupancy stays within this fraction
#     of the parent's SERT occupancy.
DIRECT_5HT2C_OCCUPANCY_THRESHOLD = 0.02
EFFICACY_RETENTION_FRACTION = 0.8


def docking_ki(affinity_kcal_per_mol: float | None,
               temperature_k: float = DEFAULT_TEMPERATURE_K) -> float | None:
    """Ki (molar) from a docking free-energy score via ΔG = RT·ln Ki.

    Ki = exp(ΔG / RT). A more negative affinity -> a smaller (tighter) Ki.
    """
    if affinity_kcal_per_mol is None:
        return None
    rt = GAS_CONSTANT_KCAL_PER_MOL_K * temperature_k
    return math.exp(affinity_kcal_per_mol / rt)


def occupancy_at(reference_concentration_molar: float | None,
                 ki_molar: float | None) -> float | None:
    """Fractional occupancy C / (C + Ki), in [0, 1].

    Saturable and monotonically increasing in C (never exceeds 1) — the
    law-of-mass-action occupancy equation.
    """
    if (reference_concentration_molar is None or ki_molar is None
            or reference_concentration_molar < 0 or ki_molar <= 0):
        return None
    c = reference_concentration_molar
    return c / (c + ki_molar)


def selectivity_index(ki_off_target_molar: float | None,
                      ki_on_target_molar: float | None) -> float | None:
    """Ki_off / Ki_on (dimensionless). Larger = more on-target selective."""
    if ki_off_target_molar is None or ki_on_target_molar is None or ki_on_target_molar <= 0:
        return None
    return ki_off_target_molar / ki_on_target_molar


def classify_sexual_adr(
    *,
    sert_occupancy: float | None,
    htr2c_occupancy: float | None,
    reference_sert_occupancy: float | None,
    direct_threshold: float = DIRECT_5HT2C_OCCUPANCY_THRESHOLD,
    retention_fraction: float = EFFICACY_RETENTION_FRACTION,
) -> tuple[str, str]:
    """Deterministic sexual-ADR risk label + rationale.

    Mechanistic basis: SSRI sexual dysfunction is mediated by elevated synaptic
    5-HT activating 5-HT2A/2C — an INDIRECT effect driven by SERT occupancy
    (the class-effect coupling). A drug's DIRECT 5-HT2C binding is a separate
    lever: it can decouple that pathway (antagonist = protective) or worsen it
    (agonist) — but docking reports binding, not functional sign, so direct
    engagement is flagged as of unknown sign.

    Returns (label, rationale). label ∈ {class_effect, efficacy_tradeoff,
    direct_5ht2c_engagement, insufficient_data}.
    """
    if htr2c_occupancy is not None and htr2c_occupancy >= direct_threshold:
        return ("direct_5ht2c_engagement",
                "direct 5-HT2C occupancy at the reference concentration reaches "
                f"{htr2c_occupancy:.0%}; docking reports binding, not functional "
                "sign, so this may reduce (antagonist) or worsen (agonist) sexual "
                "ADRs — requires a functional assay to resolve")
    if sert_occupancy is None:
        return ("insufficient_data",
                "no SERT affinity available to judge efficacy or the indirect "
                "5-HT2C class effect")
    if reference_sert_occupancy is None:
        return ("class_effect",
                "negligible direct 5-HT2C binding and no parent SERT reference; "
                "the sexual ADR remains the SERT-mediated class effect")
    if sert_occupancy >= reference_sert_occupancy * retention_fraction:
        return ("class_effect",
                "negligible direct 5-HT2C binding and SERT occupancy retained "
                f"({sert_occupancy:.0%} vs parent {reference_sert_occupancy:.0%}); "
                "the sexual ADR is the SERT-mediated class effect and is expected "
                "to persist")
    return ("efficacy_tradeoff",
            "negligible direct 5-HT2C binding but SERT occupancy is reduced "
            f"({sert_occupancy:.0%} vs parent {reference_sert_occupancy:.0%}); "
            "lower sexual ADR comes only via lower efficacy — a tradeoff, not a "
            "decoupling of efficacy from the adverse effect")
