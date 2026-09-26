"""Tests for the deterministic one-compartment PK model (chemistry/pkpd.py)."""

import math

import pytest

from ddswarm.chemistry.pkpd import (
    DEFAULT_TIMES,
    emax_effect,
    fit_metrics,
    hill_effect,
    receptor_occupancy,
    resolve_cl_vd,
    simulate_effect_compartment,
    simulate_indirect_response,
    simulate_one_compartment,
    simulate_two_compartment,
)


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


def test_two_compartment_known_values():
    r = simulate_two_compartment(dose_mg=10.0, v1_l_kg=3.0, clearance_ml_min_kg=12.0,
                                 intercompartmental_clearance_ml_min_kg=10.0, v2_l_kg=7.0)
    assert r.k10_per_hour == pytest.approx(0.24, abs=1e-9)
    assert r.k12_per_hour == pytest.approx(0.2, abs=1e-9)
    assert r.k21_per_hour == pytest.approx(10 * 0.06 / 7.0, abs=1e-9)
    assert r.alpha_per_hour == pytest.approx(0.4831, abs=1e-3)
    assert r.beta_per_hour == pytest.approx(0.04258, abs=1e-4)
    assert r.terminal_half_life_hours == pytest.approx(16.28, abs=0.05)
    assert r.distribution_half_life_hours == pytest.approx(1.435, abs=0.01)
    # cmax at t=0 equals C1(0) = F*D/(V1*W)
    assert r.cmax_mg_l == pytest.approx(10.0 / (3.0 * 70.0), abs=1e-6)
    # AUC == D/CL_total (mass balance)
    assert r.auc_mg_h_l == pytest.approx(10.0 / (12.0 * 70.0 * 0.06), abs=1e-3)
    # A + B = C1(0)
    assert r.a_coefficient + r.b_coefficient == pytest.approx(r.cmax_mg_l, abs=1e-6)


def test_two_compartment_curve_matches_analytic():
    r = simulate_two_compartment(dose_mg=10.0, v1_l_kg=3.0, clearance_ml_min_kg=12.0,
                                 intercompartmental_clearance_ml_min_kg=10.0, v2_l_kg=7.0)
    import math
    for t, c in r.curve:
        expect = r.a_coefficient * math.exp(-r.alpha_per_hour * t) + \
                 r.b_coefficient * math.exp(-r.beta_per_hour * t)
        assert c == pytest.approx(expect, abs=1e-9)


def test_two_compartment_rejects_nonpositive():
    import pytest as _pytest
    with _pytest.raises(ValueError):
        simulate_two_compartment(dose_mg=0.0, v1_l_kg=3.0, clearance_ml_min_kg=12.0,
                                 intercompartmental_clearance_ml_min_kg=10.0, v2_l_kg=7.0)


def test_emax_effect():
    assert emax_effect(0.0, 0.0, 100.0, 0.02) == 0.0
    assert emax_effect(0.02, 0.0, 100.0, 0.02) == pytest.approx(50.0, abs=1e-9)
    assert emax_effect(1000.0, 0.0, 100.0, 0.02) == pytest.approx(100.0, abs=0.01)
    # baseline offset
    assert emax_effect(0.02, 10.0, 100.0, 0.02) == pytest.approx(60.0, abs=1e-9)


def test_hill_effect():
    # n=1 reduces to Emax
    assert hill_effect(0.02, 0.0, 100.0, 0.02, 1.0) == pytest.approx(50.0, abs=1e-9)
    # n=2 at C=EC50 gives half Emax
    assert hill_effect(0.02, 0.0, 100.0, 0.02, 2.0) == pytest.approx(50.0, abs=1e-9)
    assert hill_effect(0.0, 0.0, 100.0, 0.02, 2.0) == 0.0
    with pytest.raises(ValueError):
        hill_effect(0.02, 0.0, 100.0, 0.02, 0.0)


def test_fit_metrics_perfect_fit():
    m = fit_metrics([(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)], lambda t: t, n_parameters=1)
    assert m["residual_sum_of_squares"] == 0.0
    assert "perfect fit" in m["identifiability_note"]


def test_fit_metrics_aic_bic_and_identifiability():
    m = fit_metrics([(0.0, 1.0), (1.0, 1.0), (2.0, 1.0)], lambda t: 0.0, n_parameters=2)
    assert m["residual_sum_of_squares"] == pytest.approx(3.0, abs=1e-9)
    # n=3, rss=3 -> rss/n=1 -> ln(1)=0 -> aic = 2*(2+1) = 6, bic = 3*ln(3)
    assert m["aic"] == pytest.approx(6.0, abs=1e-9)
    assert m["bic"] == pytest.approx(3.0 * math.log(3.0), abs=1e-9)
    assert m["identifiability_note"].startswith("identifiable")


def test_fit_metrics_underidentified():
    m = fit_metrics([(0.0, 1.0), (1.0, 1.0)], lambda t: 0.0, n_parameters=3)
    assert "under-identified" in m["identifiability_note"]


def test_fit_metrics_empty():
    m = fit_metrics([], lambda t: 0.0, n_parameters=1)
    assert m["n_observations"] == 0
    assert "no observations" in m["identifiability_note"]


def test_receptor_occupancy():
    assert receptor_occupancy(0.0, 1.0) == 0.0
    assert receptor_occupancy(1.0, 1.0) == pytest.approx(0.5, abs=1e-12)
    assert receptor_occupancy(100.0, 1.0) == pytest.approx(100.0 / 101.0, abs=1e-12)
    with pytest.raises(ValueError):
        receptor_occupancy(1.0, 0.0)


def test_indirect_response_no_drug_stays_at_baseline():
    vals = simulate_indirect_response(lambda t: 0.0, [0.0, 1.0, 2.0, 4.0, 8.0],
                                      baseline=100.0, kout_per_hour=0.5, imax=1.0, ic50_mg_l=0.02)
    for v in vals:
        assert v == pytest.approx(100.0, abs=1e-6)


def test_indirect_response_steady_state():
    # constant high drug -> R_ss = baseline*(1 - Imax*C/(IC50+C))
    C = 10.0
    vals = simulate_indirect_response(lambda t: C, [0.0, 20.0],
                                      baseline=100.0, kout_per_hour=0.5, imax=0.8, ic50_mg_l=0.02)
    expected_ss = 100.0 * (1.0 - 0.8 * C / (0.02 + C))
    assert vals[0] == pytest.approx(100.0, abs=1e-6)
    assert vals[-1] == pytest.approx(expected_ss, abs=0.5)


def test_effect_compartment_delay():
    # constant plasma Cp -> Ce lags: Ce(t)=Cp*(1-exp(-ke0*t)); E = Emax*Ce/(EC50+Ce)
    import math
    Cp, ke0, e0, emax, ec50 = 10.0, 1.0, 0.0, 100.0, 1.0
    times = [0.0, 0.5, 10.0]
    vals = simulate_effect_compartment(lambda t: Cp, times, ke0_per_hour=ke0,
                                       e0=e0, emax=emax, ec50=ec50)
    assert vals[0] == pytest.approx(0.0, abs=1e-6)  # Ce(0)=0 -> E=0 (delay)
    ce_half = Cp * (1.0 - math.exp(-ke0 * 0.5))
    assert vals[1] == pytest.approx(emax * ce_half / (ec50 + ce_half), abs=0.5)
    ce_ss = Cp * (1.0 - math.exp(-ke0 * 10.0))
    assert vals[2] == pytest.approx(emax * ce_ss / (ec50 + ce_ss), abs=0.5)
