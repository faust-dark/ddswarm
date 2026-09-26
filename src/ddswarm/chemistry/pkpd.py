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
