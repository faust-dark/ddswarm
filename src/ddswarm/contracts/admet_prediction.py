"""AdmetPrediction artifact (predictor_minion output).

Every endpoint prediction carries model/version/endpoint, applicability domain,
and uncertainty as first-class fields. evidence_level is 'predicted' — a
computational score is never experimental evidence."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .enums import AgentResultStatus, EvidenceLevel
from .provenance import ProvenanceEntry

PredictionStatus = Literal["predicted", "not_available", "out_of_domain", "error"]
ADStatus = Literal["in_domain", "out_of_domain", "unknown"]
UncertaintyType = Literal["none", "confidence", "std", "interval", "not_available"]


class ApplicabilityDomain(BaseModel):
    status: ADStatus = "unknown"
    method: str | None = None
    note: str | None = None


class Uncertainty(BaseModel):
    type: UncertaintyType = "not_available"
    value: float | None = None
    note: str | None = None


class EndpointPrediction(BaseModel):
    endpoint: str
    model: str
    model_version: str
    predicted_value: float | None = None
    units: str | None = None
    predicted_label: str | None = None   # categorical rule output (e.g. "good"/"poor")
    prediction_status: PredictionStatus = "not_available"
    evidence_level: EvidenceLevel = EvidenceLevel.predicted
    applicability_domain: ApplicabilityDomain = Field(default_factory=ApplicabilityDomain)
    uncertainty: Uncertainty = Field(default_factory=Uncertainty)
    citation: str | None = None          # literature source for the rule/model


class MoleculeAdmetResult(BaseModel):
    id: str
    canonical_smiles: str
    endpoints: list[EndpointPrediction] = Field(default_factory=list)


class AdmetSummary(BaseModel):
    total_molecules: int = 0
    total_endpoints: int = 0
    predicted: int = 0
    not_available: int = 0
    out_of_domain: int = 0
    error: int = 0


class AdmetPrediction(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    project_id: str
    agent: str = "predictor_minion"
    status: AgentResultStatus
    molecules: list[MoleculeAdmetResult] = Field(default_factory=list)
    summary: AdmetSummary = Field(default_factory=AdmetSummary)
    provenance: list[ProvenanceEntry] = Field(default_factory=list)
