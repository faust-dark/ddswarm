"""ClinicalSimulation artifact (clinical_minion PK/PD output).

Deterministic PK/PD models for the candidate molecules. The models are exact
math (evidence_level='calculated') run on predicted/LLM-recalled parameters
(clearance, volumes, Emax/EC50...) — a model, NOT measured human
pharmacokinetics/pharmacodynamics.

PK model complexity is selected on a two-level basis:
  1. mechanism_hint — an explicit `params.pk_model` (or PD `params.pd_model`);
  2. data_driven — goodness-of-fit / identifiability / predictive-performance
     (AIC/BIC/RSS) against supplied observed data;
otherwise the model is a labelled `default` (the simplest model justified by the
available inputs), never claimed as a data-validated selection.

Models run only when sufficient inputs exist; otherwise the molecule is counted
as `insufficient_inputs` (schema rule: "produces simulations only when
sufficient inputs exist").
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .enums import AgentResultStatus
from .provenance import ProvenanceEntry

PKSource = Literal["predicted", "inferred", "user", "derived", "not_available"]
PKModelKind = Literal["one_compartment", "two_compartment", "multi_compartment"]
TissueModelKind = Literal["well_stirred", "blood_flow_limited", "permeability_limited"]
PDModelKind = Literal["emax", "hill", "indirect_response", "effect_compartment",
                      "receptor_occupancy"]
SelectionReason = Literal["mechanism_hint", "data_driven", "default"]


class ModelSelectionMetrics(BaseModel):
    n_observations: int | None = None
    n_parameters: int | None = None
    residual_sum_of_squares: float | None = None
    aic: float | None = None
    bic: float | None = None
    identifiability_note: str | None = None


class PeripheralCompartment(BaseModel):
    volume_l_kg: float
    intercompartmental_clearance_ml_min_kg: float


class PKParameters(BaseModel):
    clearance_ml_min_kg: float | None = None
    volume_of_distribution_l_kg: float | None = None
    bioavailability: float | None = None
    half_life_hours: float | None = None
    # two-compartment (all optional)
    central_volume_l_kg: float | None = None                     # V1
    intercompartmental_clearance_ml_min_kg: float | None = None  # Q
    peripheral_volume_l_kg: float | None = None                  # V2
    # multi-compartment (N peripheral compartments beyond the central one)
    peripheral_compartments: list[PeripheralCompartment] = Field(default_factory=list)
    source: PKSource = "not_available"
    source_ref: str | None = None


class TissueParameters(BaseModel):
    tissue_partition_coefficient: float | None = None                # Kp (tissue:plasma)
    tissue_blood_flow_ml_min_kg: float | None = None                 # Qt
    tissue_volume_l_kg: float | None = None                          # Vt
    permeability_surface_product_ml_min_kg: float | None = None      # PS
    source: PKSource = "not_available"
    source_ref: str | None = None


class ConcentrationPoint(BaseModel):
    time_hours: float
    concentration_mg_l: float


class EffectPoint(BaseModel):
    time_hours: float
    effect: float


class TissueSimulation(BaseModel):
    model: TissueModelKind
    parameters: TissueParameters = Field(default_factory=TissueParameters)
    tissue_curve: list[ConcentrationPoint] = Field(default_factory=list)
    selection_reason: SelectionReason = "default"
    note: str | None = None


class PKSimulation(BaseModel):
    molecule_id: str
    canonical_smiles: str | None = None
    dose_mg: float
    route: str = "oral"
    model: PKModelKind = "one_compartment"
    parameters: PKParameters = Field(default_factory=PKParameters)
    # elimination_rate_per_hour is ke (one-compartment) or beta (two-compartment)
    elimination_rate_per_hour: float | None = None
    # half_life_hours is t1/2 (one-compartment) or the terminal t1/2 (two-compartment)
    half_life_hours: float | None = None
    alpha_per_hour: float | None = None               # two-compartment distribution rate
    distribution_half_life_hours: float | None = None  # two-compartment distribution t1/2
    cmax_mg_l: float | None = None
    auc_mg_h_l: float | None = None
    concentration_curve: list[ConcentrationPoint] = Field(default_factory=list)
    tissue: TissueSimulation | None = None   # optional organ/tissue-level model
    selection_reason: SelectionReason = "default"
    selection_metrics: ModelSelectionMetrics | None = None
    note: str | None = None


class PDParameters(BaseModel):
    baseline_effect: float | None = None       # E0 (emax/hill/effect-comp) or R0 (indirect)
    max_effect: float | None = None            # Emax
    ec50_mg_l: float | None = None             # EC50
    hill_coefficient: float | None = None      # n (Hill)
    ke0_per_hour: float | None = None          # effect-compartment equilibration rate
    kout_per_hour: float | None = None         # indirect-response turnover rate
    imax: float | None = None                  # indirect-response max inhibition [0,1]
    ic50_mg_l: float | None = None             # indirect-response IC50
    kd_mg_l: float | None = None               # receptor-occupancy dissociation constant
    source: PKSource = "not_available"
    source_ref: str | None = None


class PDSimulation(BaseModel):
    molecule_id: str
    canonical_smiles: str | None = None
    model: PDModelKind = "emax"
    parameters: PDParameters = Field(default_factory=PDParameters)
    effect_curve: list[EffectPoint] = Field(default_factory=list)
    selection_reason: SelectionReason = "default"
    selection_metrics: ModelSelectionMetrics | None = None
    note: str | None = None


class ClinicalSimulationSummary(BaseModel):
    total_molecules: int = 0
    simulated: int = 0
    insufficient_inputs: int = 0
    pd_simulated: int = 0


class ClinicalSimulation(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    project_id: str
    agent: str = "clinical_minion"
    status: AgentResultStatus
    simulations: list[PKSimulation] = Field(default_factory=list)
    pd_simulations: list[PDSimulation] = Field(default_factory=list)
    summary: ClinicalSimulationSummary = Field(default_factory=ClinicalSimulationSummary)
    note: str | None = None
    provenance: list[ProvenanceEntry] = Field(default_factory=list)
