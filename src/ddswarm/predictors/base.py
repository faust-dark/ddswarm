"""ADMET provider abstraction. Plug a real model in later without touching
the contract or the orchestrator."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..contracts.admet_prediction import EndpointPrediction


class AdmetProvider(ABC):
    name: str
    model_version: str
    supported_endpoints: list[str]

    @abstractmethod
    def predict(self, canonical_smiles: str) -> list[EndpointPrediction]:
        """Return one EndpointPrediction per supported endpoint for the molecule."""
