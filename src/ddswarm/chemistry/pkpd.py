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
