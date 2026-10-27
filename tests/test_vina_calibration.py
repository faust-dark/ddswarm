"""Tests for the hybrid Vina-score calibration (offset-only, per-receptor)."""

import math

from rdkit import Chem, RDLogger

from ddswarm.chemistry.selectivity import docking_ki, occupancy_at
from ddswarm.chemistry.vina_calibration import (
    CALIBRATION_SET,
    calibrated_delta_g,
    calibrated_ki,
    delta_g_from_ki,
    fit_offset,
)

RDLogger.DisableLog("rdApp.error")

RT = 1.987e-3 * 298.15


def test_delta_g_from_ki_identity():
    # Ki = 1 nM -> ΔG = RT·ln(1e-9) ≈ -12.3 kcal/mol
    assert math.isclose(delta_g_from_ki(1e-9), RT * math.log(1e-9), rel_tol=1e-9)
    # a tighter Ki (smaller) gives a more negative ΔG
    assert delta_g_from_ki(1e-9) < delta_g_from_ki(1e-7)


def test_fit_offset_recovers_known_constant_offset():
    # ΔG_exp = Vina - 3.0 exactly -> the offset fit must recover b = -3, RMSE = 0
    pairs = [(-7.0, -10.0), (-8.0, -11.0), (-9.0, -12.0)]
    b, n, rmsd = fit_offset(pairs)
    assert math.isclose(b, -3.0, rel_tol=1e-9)
    assert n == 3
    assert rmsd == 0.0


def test_fit_offset_empty_returns_none():
    assert fit_offset([]) == (None, 0, None)


def test_calibrated_ki_monotonic_and_fallback():
    # monotonic: tighter Vina score -> smaller Ki (slope a=1, so exact)
    assert calibrated_ki(-7.0, -3.0) < calibrated_ki(-6.0, -3.0)
    # None-safe
    assert calibrated_ki(None, -3.0) is None
    # b = None falls back to the uncalibrated Gibbs identity
    assert calibrated_ki(-7.0, None) == docking_ki(-7.0)
    # calibrated_delta_g is the raw offset shift
    assert math.isclose(calibrated_delta_g(-7.0, -3.0), -10.0, rel_tol=1e-9)


def test_calibrated_ki_recovers_antagonist_occupancy():
    # a 5-HT2C antagonist with Vina score -7 kcal/mol and a -5 kcal/mol offset
    # -> calibrated ΔG ≈ -12 -> Ki ≈ 1.6 nM -> occupancy ≈ 86% at 10 nM,
    # which crosses the direct-engagement threshold (2%).
    ki = calibrated_ki(-7.0, -5.0)
    occ = occupancy_at(1e-8, ki)
    assert occ is not None and occ > 0.5
    assert occ > 0.02  # direct-engagement threshold


def test_calibration_set_smiles_parse_and_have_ki():
    for receptor, entries in CALIBRATION_SET.items():
        assert receptor in ("SERT", "HTR2A", "HTR2C")
        for smi, ki_molar, _cit in entries:
            assert Chem.MolFromSmiles(smi) is not None, f"{receptor}: {smi}"
            assert ki_molar > 0
