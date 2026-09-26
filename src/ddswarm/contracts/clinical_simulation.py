"""ClinicalSimulation artifact (clinical_minion PK/PD output).

Deterministic one-compartment PK/PD simulation for the candidate molecules.
The simulation is exact math (evidence_level='calculated') run on predicted or
LLM-recalled PK parameters (clearance, volume of distribution, bioavailability)
— it is a model, NOT measured human pharmacokinetics, and is tagged as such.

The simulation runs only when sufficient inputs exist (CL + Vd, or t1/2 + one
of them): otherwise the molecule is counted as insufficient_inputs. This is the
schema rule "produces simulations only when sufficient inputs exist".
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .enums import AgentResultStatus
from .provenance import ProvenanceEntry

PKSource = Literal["predicted", "inferred", "user", "derived", "not_available"]


class PKParameters(BaseModel):
    clearance_ml_min_kg: float | None = None
    volume_of_distribution_l_kg: float | None = None
    bioavailability: float | None = None
    half_life_hours: float | None = None
    source: PKSource = "not_available"
    source_ref: str | None = None       # where the parameter came from (e.g. "DrugBank")


class ConcentrationPoint(BaseModel):
    time_hours: float
    concentration_mg_l: float


class PKSimulation(BaseModel):
    molecule_id: str
    canonical_smiles: str | None = None
    dose_mg: float
    route: str = "oral"
    parameters: PKParameters = Field(default_factory=PKParameters)
    elimination_rate_per_hour: float | None = None
    half_life_hours: float | None = None
    cmax_mg_l: float | None = None
    auc_mg_h_l: float | None = None
    concentration_curve: list[ConcentrationPoint] = Field(default_factory=list)
    model: str = "one-compartment, instant absorption"
    note: str | None = None


class ClinicalSimulationSummary(BaseModel):
    total_molecules: int = 0
    simulated: int = 0
    insufficient_inputs: int = 0


class ClinicalSimulation(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    project_id: str
    agent: str = "clinical_minion"
    status: AgentResultStatus
    simulations: list[PKSimulation] = Field(default_factory=list)
    summary: ClinicalSimulationSummary = Field(default_factory=ClinicalSimulationSummary)
    note: str | None = None
    provenance: list[ProvenanceEntry] = Field(default_factory=list)
