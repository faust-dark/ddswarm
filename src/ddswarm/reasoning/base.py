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
