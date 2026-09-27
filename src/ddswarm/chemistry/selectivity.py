"""Deterministic thermodynamic conversion of docking affinity -> dissociation
constant, receptor occupancy, target-selectivity indices, and a pluggable
adverse-effect (ADR) profile library.

The math here is exact and deterministic (Gibbs free-energy <-> dissociation
constant identity; saturable occupancy law). The INPUTS (AutoDock Vina affinity
scores) are RELATIVE — a ranking, not a calibrated absolute free energy — so
every derived Ki/occupancy is a *relative estimate for ranking*, never an
absolute affinity. That caveat is carried on the contract, not re-explained per
molecule.

ADR profiles are DATA + small deterministic interpreters, not code paths:
``select_adr_profiles`` matches a configured receptor panel against the built-in
profile library (the SSRI sexual-ADR profile, the NSAID COX-2/COX-1 GI-risk
profile, and a registry of off-target "antitarget" ADRs such as H1 sedation,
M1 anticholinergic burden, D2 extrapyramidal/prolactin, alpha-1 orthostatic
hypotension, 5-HT2B valvulopathy and kappa-opioid dysphoria). Every profile is
mechanistic and citable; panels that match nothing still yield pure affinity +
selectivity indices, and no ADR label is invented.
"""

from __future__ import annotations

import math

from .receptors import canonical_receptor_name

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

# NSAID GI-risk (COX) thresholds on the selectivity index SI = Ki(COX-1)/Ki(COX-2).
# COX-2-selective agents (coxibs) reach SI ~30-600x; traditional NSAIDs sit near
# or below 1 (ibuprofen ~0.6, naproxen ~0.7); SI < 1 means COX-1 is preferred.
# Documented modeling choice, not a clinical GI-risk prediction.
COX2_SELECTIVE_SI = 30.0

# Default occupancy threshold for the "antitarget" ADR profiles: half-maximal
# occupancy at the reference concentration (~Ki <= C_ref). Documented modeling
# choice; occupancy is a calibrated ranking estimate, not a measured affinity.
DEFAULT_ANTITARGET_OCCUPANCY_THRESHOLD = 0.5

# id of the SSRI sexual-ADR profile (kept for backward-compatible lookups).
SSRI_SEXUAL_ADR_PROFILE = "sert_5ht2c_sexual_adr"
COX_GI_PROFILE = "cox_gi_risk"

# Registry of off-target "antitarget" ADR profiles. Each entry maps an
# off-target receptor (canonical registry key) to a mechanistic adverse effect,
# whether the ADR is driven by receptor BLOCKADE (sign-independent: binding
# implies the ADR) or AGONISM (sign-dependent: docking reports binding, not
# function, so the label stays "engagement" until a functional assay resolves
# the sign), the occupancy threshold, and a citation. This is data — extend it
# without touching the interpreter.
ANTITARGET_PROFILES: dict[str, dict] = {
    "chrm1_anticholinergic": {
        "receptor": "CHRM1",
        "adr": "anticholinergic effects (dry mouth, blurred vision, constipation, urinary retention, cognitive impairment)",
        "sign": "blockade",
        "occupancy_threshold": DEFAULT_ANTITARGET_OCCUPANCY_THRESHOLD,
        "mechanism": "muscarinic M1 blockade",
        "citation": "Bymaster et al. 2003 (Eur J Pharmacol); Chew et al. 2008 (BMC Geriatr) — anticholinergic burden scales with muscarinic blockade",
    },
    "hrh1_sedation": {
        "receptor": "HRH1",
        "adr": "sedation / drowsiness",
        "sign": "blockade",
        "occupancy_threshold": DEFAULT_ANTITARGET_OCCUPANCY_THRESHOLD,
        "mechanism": "central histamine H1 blockade",
        "citation": "Yanai & Tashiro 2007 (Pharmacol Ther) — brain H1 occupancy predicts sedation",
    },
    "drd2_eps_prolactin": {
        "receptor": "DRD2",
        "adr": "extrapyramidal symptoms / prolactin elevation",
        "sign": "blockade",
        "occupancy_threshold": DEFAULT_ANTITARGET_OCCUPANCY_THRESHOLD,
        "mechanism": "dopamine D2 blockade",
        "citation": "Kapur & Seeman 2001 (Am J Psychiatry) — D2 occupancy predicts EPS and prolactin elevation",
    },
    "adra1a_orthostatic_hypotension": {
        "receptor": "ADRA1A",
        "adr": "orthostatic hypotension",
        "sign": "blockade",
        "occupancy_threshold": DEFAULT_ANTITARGET_OCCUPANCY_THRESHOLD,
        "mechanism": "alpha-1A adrenergic blockade",
        "citation": "alpha-1 blockade causes orthostatic hypotension (textbook alpha-1-blocker pharmacology, e.g. prazosin/terazosin)",
    },
    "adra1b_orthostatic_hypotension": {
        "receptor": "ADRA1B",
        "adr": "orthostatic hypotension",
        "sign": "blockade",
        "occupancy_threshold": DEFAULT_ANTITARGET_OCCUPANCY_THRESHOLD,
        "mechanism": "alpha-1B adrenergic blockade",
        "citation": "alpha-1 blockade causes orthostatic hypotension (textbook alpha-1-blocker pharmacology, e.g. prazosin/terazosin)",
    },
    "htr2b_valvulopathy": {
        "receptor": "HTR2B",
        "adr": "cardiac valvulopathy",
        "sign": "agonism",
        "occupancy_threshold": DEFAULT_ANTITARGET_OCCUPANCY_THRESHOLD,
        "mechanism": "5-HT2B agonism",
        "citation": "Rothman et al. 2000 (Circulation); Roth 2007 (N Engl J Med) — 5-HT2B agonism causes valvular fibrosis",
    },
    "oprk1_dysphoria": {
        "receptor": "OPRK1",
        "adr": "dysphoria / psychotomimesis",
        "sign": "agonism",
        "occupancy_threshold": DEFAULT_ANTITARGET_OCCUPANCY_THRESHOLD,
        "mechanism": "kappa-opioid agonism",
        "citation": "Pfeiffer et al. 1986 (Science) — kappa-opioid agonism produces dysphoria",
    },
}


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


def select_adr_profiles(primary: str | None, off_targets: list[str]) -> list[str]:
    """Return the ids of every ADR profile matching this receptor panel.

    Canonicalises names through the receptor registry so ``PTGS2``/``COX2``,
    ``M1``/``CHRM1``, ``5-HT2C``/``HTR2C`` etc. all resolve. A panel can match
    several profiles (e.g. a TCA-like panel with M1 + H1 + alpha-1 off-targets
    matches each of those antitarget profiles); an empty list means no built-in
    ADR profile matches and the assessment reports pure affinity + selectivity
    indices without inventing a label.
    """
    prim = canonical_receptor_name(primary) if primary else None
    offs = [canonical_receptor_name(o) for o in off_targets]

    profiles: list[str] = []
    if prim == "SERT" and "HTR2C" in offs:
        profiles.append(SSRI_SEXUAL_ADR_PROFILE)
    if prim == "COX2" and "COX1" in offs:
        profiles.append(COX_GI_PROFILE)
    for pid, prof in ANTITARGET_PROFILES.items():
        if prof["receptor"] in offs:
            profiles.append(pid)
    return profiles


def select_adr_profile(primary: str | None, off_targets: list[str]) -> str | None:
    """Backward-compatible single-profile lookup (first match, else None)."""
    profiles = select_adr_profiles(primary, off_targets)
    return profiles[0] if profiles else None


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


def classify_cox_gi(cox1_ki: float | None, cox2_ki: float | None) -> tuple[str, str]:
    """Deterministic NSAID GI-risk label + rationale.

    Mechanistic basis: COX-1 is the constitutive isoform whose prostaglandins
    maintain the gastric mucosal barrier; COX-2 is induced at inflammation
    sites. Non-selective COX-1/COX-2 inhibition therefore produces the
    anti-inflammatory effect of an NSAID while stripping mucosal protection
    (ulcer/bleed). The relevant quantity is the selectivity index
    SI = Ki(COX-1)/Ki(COX-2): larger SI = more COX-2-selective = less GI risk.

    label ∈ {cox2_preferential, non_selective_nsaid, cox1_preferring,
    insufficient_data}. The SI thresholds are documented modeling choices on a
    relative ranking, not a clinical prediction.
    """
    si = selectivity_index(cox1_ki, cox2_ki)
    if si is None:
        return ("insufficient_data",
                "no COX-1 and/or COX-2 affinity available to judge GI risk")
    if si >= COX2_SELECTIVE_SI:
        return ("cox2_preferential",
                f"COX-2 preferred {si:.0f}x over COX-1 (SI = Ki(COX-1)/Ki(COX-2) = "
                f"{si:.1f}); reduced GI risk from COX-1 sparing — note a separate "
                "cardiovascular liability of COX-2 selectivity is out of scope")
    if si >= 1.0:
        return ("non_selective_nsaid",
                f"comparable COX-1/COX-2 binding (SI = {si:.1f}); non-selective "
                "NSAID — GI ulcer/bleed risk from COX-1 inhibition")
    return ("cox1_preferring",
            f"COX-1 preferred over COX-2 (SI = {si:.2f}); elevated GI risk from "
            "strong COX-1 inhibition (and likely weak COX-2 anti-inflammatory "
            "efficacy)")


def classify_occupancy_adr(profile_id: str, occupancy: float | None) -> tuple[str, str]:
    """Deterministic label + rationale for an off-target "antitarget" ADR.

    Tiers on the off-target occupancy at the reference concentration against the
    profile's documented threshold. For sign-independent (blockade) profiles the
    label is ``liability``; for sign-dependent (agonism) profiles docking only
    proves binding, so the label is ``sign_dependent_engagement`` and the
    functional sign must be resolved separately.

    label ∈ {liability, sign_dependent_engagement, low_risk, insufficient_data}.
    """
    prof = ANTITARGET_PROFILES[profile_id]
    thr = prof["occupancy_threshold"]
    if occupancy is None:
        return ("insufficient_data",
                f"no {prof['receptor']} affinity available to judge {prof['adr']}")
    if occupancy >= thr:
        if prof["sign"] == "blockade":
            return ("liability",
                    f"{prof['receptor']} occupancy {occupancy:.0%} (>= {thr:.0%} "
                    f"threshold); {prof['mechanism']} -> {prof['adr']}. "
                    f"{prof['citation']}")
        return ("sign_dependent_engagement",
                f"{prof['receptor']} occupancy {occupancy:.0%} (>= {thr:.0%} "
                f"threshold); docking reports binding, not function, and "
                f"{prof['adr']} requires {prof['sign']} — resolve functionally. "
                f"{prof['citation']}")
    return ("low_risk",
            f"{prof['receptor']} occupancy {occupancy:.0%} below the {thr:.0%} "
            f"threshold; low {prof['adr']} risk")
