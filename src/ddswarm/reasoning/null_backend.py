"""Null reasoning backend: no LLM, no network, deterministic. The agents fall
back to their deterministic baseline when this is active."""

from __future__ import annotations

from ..contracts.project_spec import ProjectSpec
from .base import ReasoningBackend, ReasoningResult


class NullReasoningBackend(ReasoningBackend):
    name = "none"

    def available(self) -> tuple[bool, str]:
        return True, "no reasoning backend configured"

    def research(self, spec: ProjectSpec, params: dict) -> ReasoningResult:
        return ReasoningResult(ok=True, note="no reasoning backend configured")

    def design(self, spec: ProjectSpec, scaffolds: list[tuple[str, str]], params: dict) -> ReasoningResult:
        return ReasoningResult(ok=True, note="no reasoning backend configured")

    def synthesis_routes(self, spec: ProjectSpec, molecules: list[tuple[str, str]], params: dict) -> ReasoningResult:
        return ReasoningResult(ok=True, note="no reasoning backend configured")

    def patent_landscape(self, spec: ProjectSpec, molecules: list[tuple[str, str]], params: dict) -> ReasoningResult:
        return ReasoningResult(ok=True, note="no reasoning backend configured")

    def clinical_assessment(self, spec: ProjectSpec, molecules: list[tuple[str, str]], params: dict) -> ReasoningResult:
        return ReasoningResult(ok=True, note="no reasoning backend configured")

    def regulatory_assessment(self, spec: ProjectSpec, molecules: list[tuple[str, str]], params: dict) -> ReasoningResult:
        return ReasoningResult(ok=True, note="no reasoning backend configured")

    def regulatory_checklist(self, spec: ProjectSpec, molecules: list[tuple[str, str]], params: dict) -> ReasoningResult:
        return ReasoningResult(ok=True, note="no reasoning backend configured")

    def pk_parameters(self, spec: ProjectSpec, molecules: list[tuple[str, str]], params: dict) -> ReasoningResult:
        return ReasoningResult(ok=True, note="no reasoning backend configured")

    def overview(self, spec: ProjectSpec, molecules: list[tuple[str, str]], params: dict) -> ReasoningResult:
        return ReasoningResult(ok=True, note="no reasoning backend configured")
