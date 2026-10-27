"""Deterministic one-compartment PK simulation (the PK/PD core).

This is the exact textbook one-compartment model with instant absorption — the
simplest standard approximation of oral pharmacokinetics. It is DETERMINISTIC
math (evidence_level='calculated'): given clearance, volume of distribution,
bioavailability and a dose, it computes the elimination rate, half-life, Cmax,
AUC and a concentration-time curve.

It is NOT experimental PK: it runs on predicted/recalled parameters, and must
never be presented as measured human data (the clinical_minion carries this
caveat in the artifact).
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field

LN2 = math.log(2.0)

# default sampling times (hours) for the concentration-time curve
DEFAULT_TIMES = (0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0)


@dataclass
class PkSimulationResult:
    dose_mg: float
    route: str
    clearance_ml_min_kg: float
    volume_of_distribution_l_kg: float
    bioavailability: float
    weight_kg: float
    elimination_rate_per_hour: float
    half_life_hours: float
    cmax_mg_l: float
    auc_mg_h_l: float
    curve: list[tuple[float, float]] = field(default_factory=list)


def resolve_cl_vd(
    *,
    clearance_ml_min_kg: float | None = None,
    volume_of_distribution_l_kg: float | None = None,
    half_life_hours: float | None = None,
) -> tuple[float, float] | None:
    """Resolve (clearance_ml_min_kg, volume_of_distribution_l_kg) from any two of
    {clearance, Vd, half-life}; return None when fewer than two are known.

    t1/2 = ln2 * Vd / (CL * 0.06)  ->  ke = ln2 / t1/2 = CL*0.06/Vd."""
    if clearance_ml_min_kg is not None and volume_of_distribution_l_kg is not None:
        return float(clearance_ml_min_kg), float(volume_of_distribution_l_kg)
    if half_life_hours is not None and volume_of_distribution_l_kg is not None:
        if half_life_hours <= 0 or volume_of_distribution_l_kg <= 0:
            return None
        ke = LN2 / float(half_life_hours)
        return ke * float(volume_of_distribution_l_kg) / 0.06, float(volume_of_distribution_l_kg)
    if half_life_hours is not None and clearance_ml_min_kg is not None:
        if half_life_hours <= 0 or clearance_ml_min_kg <= 0:
            return None
        ke = LN2 / float(half_life_hours)
        return float(clearance_ml_min_kg), float(clearance_ml_min_kg) * 0.06 / ke
    return None


def simulate_one_compartment(
    *,
    dose_mg: float,
    clearance_ml_min_kg: float,
    volume_of_distribution_l_kg: float,
    bioavailability: float = 1.0,
    weight_kg: float = 70.0,
    route: str = "oral",
    times: tuple[float, ...] | None = None,
) -> PkSimulationResult:
    """Run the one-compartment, instant-absorption model.

    ke = CL / Vd  (units converted so ke is 1/hour)
    t1/2 = ln2 / ke
    Cmax = F * D / (Vd * W)
    AUC = F * D / CL_total
    C(t) = Cmax * exp(-ke * t)
    """
    if dose_mg <= 0 or clearance_ml_min_kg <= 0 or volume_of_distribution_l_kg <= 0:
        raise ValueError("dose, clearance and volume of distribution must be positive")
    if not 0.0 <= bioavailability <= 1.0:
        raise ValueError("bioavailability must be in [0, 1]")

    # CL (mL/min/kg) -> L/h/kg: *60/1000 = *0.06
    ke = clearance_ml_min_kg * 0.06 / volume_of_distribution_l_kg  # 1/h
    half_life = LN2 / ke  # hours
    cmax = bioavailability * dose_mg / (volume_of_distribution_l_kg * weight_kg)  # mg/L
    total_cl_l_h = clearance_ml_min_kg * weight_kg * 0.06  # L/h
    auc = bioavailability * dose_mg / total_cl_l_h  # mg*h/L

    ts = times if times is not None else DEFAULT_TIMES
    curve = [(float(t), cmax * math.exp(-ke * t)) for t in ts]

    return PkSimulationResult(
        dose_mg=float(dose_mg),
        route=route,
        clearance_ml_min_kg=float(clearance_ml_min_kg),
        volume_of_distribution_l_kg=float(volume_of_distribution_l_kg),
        bioavailability=float(bioavailability),
        weight_kg=float(weight_kg),
        elimination_rate_per_hour=ke,
        half_life_hours=half_life,
        cmax_mg_l=cmax,
        auc_mg_h_l=auc,
        curve=curve,
    )


@dataclass
class TwoCompartmentResult:
    dose_mg: float
    route: str
    v1_l_kg: float
    v2_l_kg: float
    clearance_ml_min_kg: float
    intercompartmental_clearance_ml_min_kg: float
    bioavailability: float
    weight_kg: float
    k10_per_hour: float
    k12_per_hour: float
    k21_per_hour: float
    alpha_per_hour: float
    beta_per_hour: float
    a_coefficient: float
    b_coefficient: float
    distribution_half_life_hours: float
    terminal_half_life_hours: float
    cmax_mg_l: float
    auc_mg_h_l: float
    curve: list[tuple[float, float]] = field(default_factory=list)


def simulate_two_compartment(
    *,
    dose_mg: float,
    v1_l_kg: float,
    clearance_ml_min_kg: float,
    intercompartmental_clearance_ml_min_kg: float,
    v2_l_kg: float,
    bioavailability: float = 1.0,
    weight_kg: float = 70.0,
    route: str = "iv",
    times: tuple[float, ...] | None = None,
) -> TwoCompartmentResult:
    """Two-compartment (central + peripheral) instant-absorption model.

    Standard IV-bolus closed form: C(t) = A*exp(-alpha*t) + B*exp(-beta*t) with
    micro-constants k10 = CL/V1, k12 = Q/V1, k21 = Q/V2, and

        alpha, beta = (s +/- sqrt(s^2 - 4*k21*k10)) / 2,  s = k12 + k21 + k10
        A = C1(0) * (alpha - k21) / (alpha - beta)
        B = C1(0) * (k21 - beta) / (alpha - beta)
        C1(0) = F*D / (V1*W)

    Terminal half-life = ln2/beta; distribution half-life = ln2/alpha;
    AUC = A/alpha + B/beta."""
    if min(dose_mg, v1_l_kg, clearance_ml_min_kg,
           intercompartmental_clearance_ml_min_kg, v2_l_kg) <= 0:
        raise ValueError("dose, V1, CL, Q and V2 must be positive")
    if not 0.0 <= bioavailability <= 1.0:
        raise ValueError("bioavailability must be in [0, 1]")

    k10 = clearance_ml_min_kg * 0.06 / v1_l_kg
    k12 = intercompartmental_clearance_ml_min_kg * 0.06 / v1_l_kg
    k21 = intercompartmental_clearance_ml_min_kg * 0.06 / v2_l_kg
    s = k12 + k21 + k10
    disc = s * s - 4.0 * k21 * k10
    if disc < 0:
        raise ValueError("two-compartment rate constants yield a negative discriminant")
    alpha = (s + math.sqrt(disc)) / 2.0
    beta = (s - math.sqrt(disc)) / 2.0
    c1_0 = bioavailability * dose_mg / (v1_l_kg * weight_kg)
    a = c1_0 * (alpha - k21) / (alpha - beta)
    b = c1_0 * (k21 - beta) / (alpha - beta)
    cmax = c1_0
    auc = a / alpha + b / beta

    ts = times if times is not None else DEFAULT_TIMES
    curve = [(float(t), a * math.exp(-alpha * t) + b * math.exp(-beta * t)) for t in ts]

    return TwoCompartmentResult(
        dose_mg=float(dose_mg), route=route, v1_l_kg=float(v1_l_kg),
        v2_l_kg=float(v2_l_kg),
        clearance_ml_min_kg=float(clearance_ml_min_kg),
        intercompartmental_clearance_ml_min_kg=float(intercompartmental_clearance_ml_min_kg),
        bioavailability=float(bioavailability), weight_kg=float(weight_kg),
        k10_per_hour=k10, k12_per_hour=k12, k21_per_hour=k21,
        alpha_per_hour=alpha, beta_per_hour=beta,
        a_coefficient=a, b_coefficient=b,
        distribution_half_life_hours=LN2 / alpha,
        terminal_half_life_hours=LN2 / beta,
        cmax_mg_l=cmax, auc_mg_h_l=auc, curve=curve,
    )


def emax_effect(concentration_mg_l: float, e0: float, emax: float, ec50: float) -> float:
    """Direct Emax effect: E = E0 + Emax*C/(EC50 + C)."""
    if ec50 <= 0:
        raise ValueError("EC50 must be positive")
    return e0 + emax * concentration_mg_l / (ec50 + concentration_mg_l)


def hill_effect(concentration_mg_l: float, e0: float, emax: float, ec50: float,
                hill: float) -> float:
    """Sigmoid Emax (Hill) effect: E = E0 + Emax*C^n/(EC50^n + C^n)."""
    if ec50 <= 0:
        raise ValueError("EC50 must be positive")
    if hill <= 0:
        raise ValueError("Hill coefficient must be positive")
    c = concentration_mg_l
    cn = c ** hill
    return e0 + emax * cn / (ec50 ** hill + cn)


def fit_metrics(
    observed: list[tuple[float, float]],
    model_fn,
    n_parameters: int,
) -> dict:
    """Goodness-of-fit of a model function against observed (time, value) points.

    Returns residual sum of squares plus AIC/BIC (Gaussian-error form), which
    serve as the complexity-penalized goodness-of-fit and predictive-performance
    criteria. A parameter-identifiability note is attached: more parameters than
    observations => under-identified.

    `model_fn(t) -> value` evaluates the model at an arbitrary time."""
    n = len(observed)
    out: dict = {"n_observations": n, "n_parameters": n_parameters}
    if n == 0:
        out["residual_sum_of_squares"] = None
        out["identifiability_note"] = "no observations; identifiability not assessable"
        return out
    rss = sum((value - model_fn(t)) ** 2 for t, value in observed)
    out["residual_sum_of_squares"] = rss
    if n_parameters >= n:
        ident_note = f"under-identified: {n_parameters} parameters vs {n} observations"
    else:
        ident_note = f"identifiable: {n_parameters} parameters vs {n} observations"
    if rss <= 0:
        out["identifiability_note"] = f"perfect fit (RSS=0); {ident_note}"
        return out
    k = n_parameters + 1  # +1 for the residual variance
    out["aic"] = n * math.log(rss / n) + 2.0 * k
    out["bic"] = n * math.log(rss / n) + k * math.log(n)
    out["identifiability_note"] = ident_note
    return out


def _rk4_solve(deriv_fn, y0: float, t_end: float, dt: float = 0.1) -> tuple[list[float], list[float]]:
    """Fixed-step RK4 integration of y' = deriv_fn(t, y) from t=0 to t_end.

    Deterministic (no scipy dependency). Returns (ts, ys) including the t=0
    starting point. The step is fixed at `dt` except the final partial step.
    """
    if dt <= 0:
        raise ValueError("dt must be positive")
    if t_end < 0:
        raise ValueError("t_end must be non-negative")
    ts: list[float] = [0.0]
    ys: list[float] = [float(y0)]
    y = float(y0)
    t = 0.0
    while t < t_end - 1e-12:
        h = min(dt, t_end - t)
        k1 = deriv_fn(t, y)
        k2 = deriv_fn(t + h / 2.0, y + h / 2.0 * k1)
        k3 = deriv_fn(t + h / 2.0, y + h / 2.0 * k2)
        k4 = deriv_fn(t + h, y + h * k3)
        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += h
        ts.append(t)
        ys.append(y)
    return ts, ys


def _sample(ts: list[float], ys: list[float], times) -> list[float]:
    """Linearly interpolate ys onto the requested `times`."""
    out: list[float] = []
    for t in times:
        if t <= ts[0]:
            out.append(ys[0])
            continue
        if t >= ts[-1]:
            out.append(ys[-1])
            continue
        i = bisect.bisect_right(ts, t) - 1
        t0, t1 = ts[i], ts[i + 1]
        frac = (t - t0) / (t1 - t0)
        out.append(ys[i] + frac * (ys[i + 1] - ys[i]))
    return out


def receptor_occupancy(concentration_mg_l: float, kd_mg_l: float) -> float:
    """Fractional receptor occupancy = C / (C + Kd), in [0, 1)."""
    if kd_mg_l <= 0:
        raise ValueError("Kd must be positive")
    return concentration_mg_l / (kd_mg_l + concentration_mg_l)


def simulate_indirect_response(concentration_fn, times, *, baseline: float,
                               kout_per_hour: float, imax: float, ic50_mg_l: float,
                               dt: float = 0.1) -> list[float]:
    """Indirect-response (turnover) model, inhibition of kin (Dayneka Type I).

    dR/dt = kin * (1 - Imax*C/(IC50 + C)) - kout*R, with kin = baseline*kout and
    R(0) = baseline. Returns the response R(t) at each time in `times`.
    `concentration_fn(t)` supplies the plasma concentration at an arbitrary t.
    """
    if kout_per_hour <= 0:
        raise ValueError("kout must be positive")
    if baseline < 0:
        raise ValueError("baseline must be non-negative")
    if not (0.0 <= imax <= 1.0):
        raise ValueError("Imax must be in [0, 1]")
    if ic50_mg_l <= 0:
        raise ValueError("IC50 must be positive")
    if not times:
        return []
    kin = baseline * kout_per_hour

    def deriv(t, R):
        C = concentration_fn(t)
        return kin * (1.0 - imax * C / (ic50_mg_l + C)) - kout_per_hour * R

    ts, ys = _rk4_solve(deriv, baseline, float(times[-1]), dt=dt)
    return _sample(ts, ys, times)


def simulate_effect_compartment(concentration_fn, times, *, ke0_per_hour: float,
                                e0: float, emax: float, ec50: float,
                                dt: float = 0.1) -> list[float]:
    """Effect-compartment (link) model: dCe/dt = ke0*(Cp - Ce), E = E0 + Emax*Ce/(EC50+Ce).

    Introduces a delay between plasma concentration and effect. Ce(0) = 0.
    Returns the effect E(t) at each time in `times`.
    """
    if ke0_per_hour <= 0:
        raise ValueError("ke0 must be positive")
    if ec50 <= 0:
        raise ValueError("EC50 must be positive")
    if not times:
        return []

    def deriv(t, Ce):
        Cp = concentration_fn(t)
        return ke0_per_hour * (Cp - Ce)

    ts, ys = _rk4_solve(deriv, 0.0, float(times[-1]), dt=dt)
    ces = _sample(ts, ys, times)
    return [e0 + emax * ce / (ec50 + ce) for ce in ces]


def _rk4_solve_system(deriv_fn, y0, t_end: float, dt: float = 0.1):
    """Vector (system) fixed-step RK4. deriv_fn(t, y) -> list[float] (same length as y).

    Returns (ts, ys) where ys is a list of state-vectors, including t=0.
    """
    if dt <= 0:
        raise ValueError("dt must be positive")
    if t_end < 0:
        raise ValueError("t_end must be non-negative")
    n = len(y0)
    ts: list[float] = [0.0]
    ys: list[list[float]] = [list(y0)]
    y = list(y0)
    t = 0.0
    while t < t_end - 1e-12:
        h = min(dt, t_end - t)
        k1 = deriv_fn(t, y)
        k2 = deriv_fn(t + h / 2.0, [y[j] + h / 2.0 * k1[j] for j in range(n)])
        k3 = deriv_fn(t + h / 2.0, [y[j] + h / 2.0 * k2[j] for j in range(n)])
        k4 = deriv_fn(t + h, [y[j] + h * k3[j] for j in range(n)])
        y = [y[j] + (h / 6.0) * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j]) for j in range(n)]
        t += h
        ts.append(t)
        ys.append(list(y))
    return ts, ys


def simulate_well_stirred(concentration_fn, times, *, kp: float) -> list[float]:
    """Well-stirred tissue model: Ct(t) = Kp * C_plasma(t) (instant equilibrium)."""
    if kp <= 0:
        raise ValueError("Kp must be positive")
    return [kp * concentration_fn(t) for t in times]


def simulate_blood_flow_limited(concentration_fn, times, *, qt_ml_min_kg: float,
                                vt_l_kg: float, kp: float, dt: float = 0.1) -> list[float]:
    """Blood-flow-limited (perfusion-limited) tissue model.

    dCt/dt = (Qt/Vt) * (C_art(t) - Ct/Kp); rate constant = Qt/Vt. Returns Ct(t).
    """
    if qt_ml_min_kg <= 0 or vt_l_kg <= 0 or kp <= 0:
        raise ValueError("Qt, Vt, Kp must be positive")
    if not times:
        return []
    rate = qt_ml_min_kg * 0.06 / vt_l_kg  # 1/h

    def deriv(t, Ct):
        return rate * (concentration_fn(t) - Ct / kp)

    ts, ys = _rk4_solve(deriv, 0.0, float(times[-1]), dt=dt)
    return _sample(ts, ys, times)


def simulate_permeability_limited(concentration_fn, times, *, ps_ml_min_kg: float,
                                  vt_l_kg: float, kp: float, dt: float = 0.1) -> list[float]:
    """Permeability-limited tissue model (membrane barrier, permeability-surface PS).

    dCt/dt = (PS/Vt) * (C_plasma(t) - Ct/Kp); rate constant = PS/Vt. Returns Ct(t).
    """
    if ps_ml_min_kg <= 0 or vt_l_kg <= 0 or kp <= 0:
        raise ValueError("PS, Vt, Kp must be positive")
    if not times:
        return []
    rate = ps_ml_min_kg * 0.06 / vt_l_kg  # 1/h

    def deriv(t, Ct):
        return rate * (concentration_fn(t) - Ct / kp)

    ts, ys = _rk4_solve(deriv, 0.0, float(times[-1]), dt=dt)
    return _sample(ts, ys, times)


@dataclass
class MultiCompartmentResult:
    dose_mg: float
    route: str
    v1_l_kg: float
    clearance_ml_min_kg: float
    peripherals: list[tuple[float, float]]  # (volume_l_kg, intercompartmental clearance)
    bioavailability: float
    weight_kg: float
    cmax_mg_l: float
    auc_mg_h_l: float
    terminal_half_life_hours: float
    curve: list[tuple[float, float]]  # central C(t) at the requested times
    ts: list[float]                    # fine integration grid
    central: list[float]               # central C(t) on the fine grid


def _terminal_half_life(ts: list[float], ys: list[float]) -> float:
    """Estimate terminal t1/2 by log-linear least squares on the last 3 positive points."""
    pts = [(t, math.log(y)) for t, y in zip(ts, ys) if y > 0]
    if len(pts) < 2:
        raise ValueError("need >=2 positive concentrations to estimate terminal half-life")
    pts = pts[-3:]
    n = len(pts)
    mean_t = sum(t for t, _ in pts) / n
    mean_logc = sum(logc for _, logc in pts) / n
    num = sum((t - mean_t) * (logc - mean_logc) for t, logc in pts)
    den = sum((t - mean_t) ** 2 for t, _ in pts)
    if den == 0 or num >= 0:
        raise ValueError("concentration is not decaying; cannot estimate terminal half-life")
    slope = num / den  # negative
    return LN2 / (-slope)


def simulate_multi_compartment(dose_mg: float, v1_l_kg: float, clearance_ml_min_kg: float,
                               peripherals: list[tuple[float, float]],
                               bioavailability: float = 1.0, weight_kg: float = 70.0,
                               route: str = "oral", times=None, dt: float = 0.1):
    """Mammillary N-compartment model (central + N peripheral compartments).

    Elimination only from the central compartment; instant absorption (bolus).
    `peripherals` is a list of (volume_l_kg, intercompartmental_clearance_ml_min_kg).
    AUC uses the mass-balance identity AUC = F*D/CL_total (holds for any linear
    model with central-only elimination).
    """
    if dose_mg <= 0 or v1_l_kg <= 0 or clearance_ml_min_kg <= 0:
        raise ValueError("dose, V1, CL must be positive")
    if not peripherals:
        raise ValueError("multi-compartment requires at least one peripheral compartment")
    k10 = clearance_ml_min_kg * 0.06 / v1_l_kg
    k1i: list[float] = []
    ki1: list[float] = []
    for vi, qi in peripherals:
        if vi <= 0 or qi <= 0:
            raise ValueError("peripheral volume and intercompartmental clearance must be positive")
        k1i.append(qi * 0.06 / v1_l_kg)
        ki1.append(qi * 0.06 / vi)

    c0 = bioavailability * dose_mg / (v1_l_kg * weight_kg)

    def deriv(t, y):
        c1 = y[0]
        out = [-(k10 + sum(k1i)) * c1 + sum(ki1[j] * y[j + 1] for j in range(len(ki1)))]
        out.extend(k1i[j] * c1 - ki1[j] * y[j + 1] for j in range(len(ki1)))
        return out

    y0 = [c0] + [0.0] * len(peripherals)
    ts_out = times if times is not None else DEFAULT_TIMES
    t_end = float(max(ts_out))
    fts, fys = _rk4_solve_system(deriv, y0, t_end, dt=dt)
    central = [y[0] for y in fys]
    sampled = _sample(fts, central, ts_out)
    curve = [(float(t), float(c)) for t, c in zip(ts_out, sampled)]
    auc = bioavailability * dose_mg / (clearance_ml_min_kg * weight_kg * 0.06)
    t1_2 = _terminal_half_life(fts, central)
    return MultiCompartmentResult(
        dose_mg=float(dose_mg), route=route, v1_l_kg=float(v1_l_kg),
        clearance_ml_min_kg=float(clearance_ml_min_kg),
        peripherals=[(float(v), float(q)) for v, q in peripherals],
        bioavailability=float(bioavailability), weight_kg=float(weight_kg),
        cmax_mg_l=c0, auc_mg_h_l=auc, terminal_half_life_hours=t1_2,
        curve=curve, ts=fts, central=central,
    )
