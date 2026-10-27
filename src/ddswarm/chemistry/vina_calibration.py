"""Hybrid calibration of AutoDock Vina scores against literature Ki.

Vina reports a RELATIVE docking score, systematically offset from the true
binding free energy (it underestimates affinity). This module recovers that
offset by fitting a per-receptor intercept against a curated, FIXED set of
literature Ki values, in the hybrid mechanistic/statistical spirit:

  - the MECHANISTIC model is untouched — the Gibbs identity (ΔG = RT·ln Ki) and
    the occupancy law (C/(C+Ki)) still enforce monotonicity and saturation;
  - the STATISTICAL component estimates ONE parameter per receptor — the
    intercept b_r (Vina's systematic energy-scale error) — by ordinary least
    squares, with the slope FIXED at a = 1 (no rescaling), so monotonicity is
    exact (a > 0) by construction.

This is NOT a free per-molecule fit: the calibration set is fixed, committed,
and disjoint from the candidates being evaluated (no data leakage). The
calibrated Ki remains a *calculated estimate*, tagged with the residual RMSE.
"""

from __future__ import annotations

import math

from .selectivity import DEFAULT_TEMPERATURE_K, GAS_CONSTANT_KCAL_PER_MOL_K

CALIBRATION_SOURCE = ("NIMH PDSP Ki Database (https://pdsp.unc.edu/databases/pdsp.php); "
                      "literature Ki values — verify against the primary source")

# receptor -> [(canonical SMILES, Ki molar, citation)]. Fixed and committed;
# the compounds are documented ligands of that receptor with a measured Ki.
CALIBRATION_SET: dict[str, list[tuple[str, float, str]]] = {
    "SERT": [
        ("N#Cc1ccc2c(c1)COC2(CCCN(C)C)c1ccc(F)cc1", 1.4e-9,
         "citalopram, SERT Ki = 1.4 nM (PDSP)"),
        ("CNCCC(c1ccccc1)Oc1ccc(C(F)(F)F)cc1", 0.8e-9,
         "fluoxetine, SERT Ki = 0.8 nM (PDSP)"),
        ("CN(C)CC(c1ccc(OC)cc1)C1(O)CCCCC1", 9.0e-9,
         "venlafaxine, SERT Ki = 9.0 nM (PDSP)"),
    ],
    "HTR2A": [
        ("Fc1ccc(C(=O)C2CCN(CCN3C(=O)NC(=O)c4ccccc43)CC2)cc1", 2.0e-9,
         "ketanserin, 5-HT2A Ki = 2.0 nM (PDSP)"),
        ("CN1CCN2c3ccccc3Cc3ccccc3C2C1", 3.8e-9,
         "mianserin, 5-HT2A Ki = 3.8 nM (PDSP)"),
        ("CN1CCN2c3ncccc3Cc3ccccc3C2C1", 6.9e-9,
         "mirtazapine, 5-HT2A Ki = 6.9 nM (PDSP)"),
    ],
    "HTR2C": [
        ("CN1CCN2c3ccccc3Cc3ccccc3C2C1", 1.6e-9,
         "mianserin, 5-HT2C Ki = 1.6 nM (PDSP)"),
        ("Fc1ccc(C(=O)C2CCN(CCN3C(=O)NC(=O)c4ccccc43)CC2)cc1", 148.0e-9,
         "ketanserin, 5-HT2C Ki = 148 nM (PDSP)"),
        ("CN1CCN(C2=Nc3cc(Cl)ccc3Nc3ccccc32)CC1", 8.0e-9,
         "clozapine, 5-HT2C Ki = 8.0 nM (PDSP)"),
    ],
}


def delta_g_from_ki(ki_molar: float,
                    temperature_k: float = DEFAULT_TEMPERATURE_K) -> float:
    """Experimental binding free energy (kcal/mol) from a dissociation constant.

    ΔG = RT·ln Ki. A Ki of 1 nM (1e-9 M) gives ≈ -12.3 kcal/mol at 298 K.
    """
    rt = GAS_CONSTANT_KCAL_PER_MOL_K * temperature_k
    return rt * math.log(ki_molar)


def fit_offset(pairs: list[tuple[float, float]]) -> tuple[float | None, int, float | None]:
    """Fit the per-receptor intercept b from (vina_score, experimental_ΔG) pairs.

    b = mean(ΔG_exp − Vina_score). Returns (b, n, rmsd) — the OLS intercept of a
    fixed-slope (a = 1) model, plus its residual root-mean-square deviation.
    Returns (None, 0, None) for an empty set (insufficient calibration data).
    """
    n = len(pairs)
    if n == 0:
        return (None, 0, None)
    residuals = [exp_dg - vina for vina, exp_dg in pairs]
    b = sum(residuals) / n
    rmsd = math.sqrt(sum((r - b) ** 2 for r in residuals) / n)
    return (b, n, rmsd)


def calibrated_delta_g(vina_score: float | None, b: float | None) -> float | None:
    """ΔG_calibrated = Vina_score + b (slope fixed at 1). None-safe."""
    if vina_score is None or b is None:
        return vina_score
    return vina_score + b


def calibrated_ki(vina_score: float | None, b: float | None,
                  temperature_k: float = DEFAULT_TEMPERATURE_K) -> float | None:
    """Ki from a Vina score corrected by the per-receptor offset b.

    Ki = exp((Vina_score + b) / RT). Falls back to the uncalibrated Gibbs
    identity when b is None (insufficient calibration data for that receptor).
    """
    dg = calibrated_delta_g(vina_score, b)
    if dg is None:
        return None
    rt = GAS_CONSTANT_KCAL_PER_MOL_K * temperature_k
    return math.exp(dg / rt)
