"""Reasoning backend abstraction.

The deterministic swarm agents compute; the reasoning tasks (research, design)
may delegate to an LLM reasoning backend (e.g. a Hermes subagent). A backend
returns raw dicts that the agent validates against its contract — a backend
never returns contract objects directly, and a broken/garbage response is
surfaced, not trusted."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..contracts.project_spec import ProjectSpec


@dataclass
class ReasoningResult:
    entries: list[dict] = field(default_factory=list)
    ok: bool = True
    note: str = ""


class ReasoningBackend(ABC):
    name: str = ""

    @abstractmethod
    def available(self) -> tuple[bool, str]:
        """(is_available, version_or_reason)."""

    @abstractmethod
    def research(self, spec: ProjectSpec, params: dict) -> ReasoningResult:
        """Return raw dicts for KnowledgePackEntry (validated by the agent)."""

    @abstractmethod
    def design(self, spec: ProjectSpec, scaffolds: list[tuple[str, str]], params: dict) -> ReasoningResult:
        """Return raw dicts for MoleculeDesign (validated by the agent)."""

    @abstractmethod
    def synthesis_routes(self, spec: ProjectSpec, molecules: list[tuple[str, str]], params: dict) -> ReasoningResult:
        """Return raw dicts ({molecule_id, steps, key_reagents, feasibility_note})
        for proposed synthesis routes (validated by the agent)."""

    @abstractmethod
    def patent_landscape(self, spec: ProjectSpec, molecules: list[tuple[str, str]], params: dict) -> ReasoningResult:
        """Return raw dicts for PatentEntry (validated by the agent)."""

    @abstractmethod
    def clinical_assessment(self, spec: ProjectSpec, molecules: list[tuple[str, str]], params: dict) -> ReasoningResult:
        """Return raw dicts for ClinicalFinding (validated by the agent)."""

    @abstractmethod
    def regulatory_assessment(self, spec: ProjectSpec, molecules: list[tuple[str, str]], params: dict) -> ReasoningResult:
        """Return raw dicts for RegulatoryFinding (validated by the agent)."""

    @abstractmethod
    def regulatory_checklist(self, spec: ProjectSpec, molecules: list[tuple[str, str]], params: dict) -> ReasoningResult:
        """Return raw dicts ({dimension, status, summary, notes}) for a 7D
        RegulatoryChecklistEntry (validated by the agent)."""

    @abstractmethod
    def pk_parameters(self, spec: ProjectSpec, molecules: list[tuple[str, str]], params: dict) -> ReasoningResult:
        """Return raw dicts ({molecule_id, clearance_ml_min_kg, volume_of_distribution_l_kg,
        bioavailability, half_life_hours, source, notes}) of human PK parameters for
        known drugs / close analogs (validated by the agent). LLM recall — not measured data."""

    @abstractmethod
    def overview(self, spec: ProjectSpec, molecules: list[tuple[str, str]], params: dict) -> ReasoningResult:
        """Return a single raw dict ({mechanism_of_action, chemical_structure,
        therapeutic_uses, citations}) summarizing the target/scaffold for the report
        overview (validated by the agent). LLM recall — not measured data."""
