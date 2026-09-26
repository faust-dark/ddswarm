"""Tests for the deterministic one-compartment PK model (chemistry/pkpd.py)."""

import math

import pytest

from ddswarm.chemistry.pkpd import DEFAULT_TIMES, resolve_cl_vd, simulate_one_compartment


def test_one_compartment_known_values():
    r = simulate_one_compartment(dose_mg=10.0, clearance_ml_min_kg=12.0,
                                 volume_of_distribution_l_kg=4.5, bioavailability=0.9)
    # ke = CL*0.06/Vd = 0.16 /h
    assert r.elimination_rate_per_hour == pytest.approx(0.16, abs=1e-9)
    # t1/2 = ln2/ke = 4.331 h
    assert r.half_life_hours == pytest.approx(math.log(2) / 0.16, abs=1e-6)
    # Cmax = F*D/(Vd*W) = 9/315
    assert r.cmax_mg_l == pytest.approx(9.0 / 315.0, abs=1e-9)
    # AUC = F*D/(CL*W*0.06) = 9/50.4
    assert r.auc_mg_h_l == pytest.approx(9.0 / 50.4, abs=1e-9)


def test_curve_monotonic_decreasing_and_half_life():
    r = simulate_one_compartment(dose_mg=10.0, clearance_ml_min_kg=12.0,
                                 volume_of_distribution_l_kg=4.5, bioavailability=0.9)
    assert r.curve[0][0] == 0.0
    conc = [c for _, c in r.curve]
    assert conc[0] == pytest.approx(r.cmax_mg_l, abs=1e-9)
    assert all(conc[i] >= conc[i + 1] for i in range(len(conc) - 1))
    # concentration halves at t1/2 (4.33 h): between the 4h and 6h samples
    c4 = dict(r.curve)[4.0]
    c6 = dict(r.curve)[6.0]
    assert c4 > r.cmax_mg_l / 2.0 > c6


def test_default_times_used():
    r = simulate_one_compartment(dose_mg=1.0, clearance_ml_min_kg=1.0,
                                 volume_of_distribution_l_kg=1.0)
    assert [t for t, _ in r.curve] == list(DEFAULT_TIMES)


def test_rejects_nonpositive_inputs():
    with pytest.raises(ValueError):
        simulate_one_compartment(dose_mg=0.0, clearance_ml_min_kg=1.0, volume_of_distribution_l_kg=1.0)
    with pytest.raises(ValueError):
        simulate_one_compartment(dose_mg=1.0, clearance_ml_min_kg=-1.0, volume_of_distribution_l_kg=1.0)
    with pytest.raises(ValueError):
        simulate_one_compartment(dose_mg=1.0, clearance_ml_min_kg=1.0, volume_of_distribution_l_kg=1.0,
                                 bioavailability=1.5)


def test_resolve_cl_vd_combinations():
    assert resolve_cl_vd(clearance_ml_min_kg=12.0, volume_of_distribution_l_kg=4.5) == (12.0, 4.5)
    # t1/2 + Vd -> recover CL = ln2/t1/2 * Vd / 0.06
    cl, vd = resolve_cl_vd(half_life_hours=4.0, volume_of_distribution_l_kg=4.5)
    assert vd == 4.5
    assert cl == pytest.approx((math.log(2) / 4.0) * 4.5 / 0.06, abs=1e-9)
    # t1/2 + CL -> recover Vd
    cl2, vd2 = resolve_cl_vd(half_life_hours=4.0, clearance_ml_min_kg=12.0)
    assert cl2 == 12.0
    assert vd2 == pytest.approx(12.0 * 0.06 / (math.log(2) / 4.0), abs=1e-9)
    # fewer than two -> None
    assert resolve_cl_vd(clearance_ml_min_kg=12.0) is None
    assert resolve_cl_vd(volume_of_distribution_l_kg=4.5) is None
    assert resolve_cl_vd(half_life_hours=4.0) is None
    assert resolve_cl_vd() is None


def test_resolve_cl_vd_rejects_nonpositive():
    assert resolve_cl_vd(half_life_hours=0.0, volume_of_distribution_l_kg=4.5) is None
    assert resolve_cl_vd(half_life_hours=-1.0, clearance_ml_min_kg=12.0) is None
