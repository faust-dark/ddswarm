"""Deterministic offline provider with NO model.

Every endpoint is returned as 'not_available' with a documented reason, so the
prediction contract is exercised end-to-end without fabricating any value."""

from __future__ import annotations

from ..contracts.admet_prediction import (
    ApplicabilityDomain,
    EndpointPrediction,
    Uncertainty,
)
from .base import AdmetProvider

# Illustrative endpoint set matching admet_ai 2.0 naming, so a real model can be
# dropped in later without changing the contract.
CANONICAL_ENDPOINTS = [
    "Solubility_AqSolDB",
    "Caco2_Wang",
    "BBB_Martins",
    "hERG",
    "CYP3A4_Veith",
    "CYP2D6_Veith",
    "Ames_Mutagenicity",
    "Plasma_Protein_Binding",
    "Half_Life_Obach",
    "Clearance_Hepatocyte",
]


class NullAdmetProvider(AdmetProvider):
    name = "null_admet"
    model_version = "0.0.0"
    supported_endpoints = list(CANONICAL_ENDPOINTS)

    def predict(self, canonical_smiles: str) -> list[EndpointPrediction]:
        out: list[EndpointPrediction] = []
        for ep in self.supported_endpoints:
            out.append(EndpointPrediction(
                endpoint=ep,
                model=self.name,
                model_version=self.model_version,
                predicted_value=None,
                units=None,
                prediction_status="not_available",
                applicability_domain=ApplicabilityDomain(status="unknown", note="no model configured"),
                uncertainty=Uncertainty(type="not_available", note="no model configured"),
            ))
        return out
