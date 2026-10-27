"""Reasoning backend registry."""

from __future__ import annotations

from pathlib import Path

from .base import ReasoningBackend
from .hermes_backend import HermesSubprocessBackend
from .null_backend import NullReasoningBackend


def make_reasoning_backend(
    name: str = "none",
    command: str | None = None,
    max_turns: int = 30,
    run_budget: int = 600,
    audit_dir: str | Path | None = None,
) -> ReasoningBackend:
    if name in (None, "", "none"):
        return NullReasoningBackend()
    if name == "hermes":
        return HermesSubprocessBackend(
            command=command or "hermes", max_turns=max_turns, run_budget=run_budget,
            audit_dir=audit_dir,
        )
    raise KeyError(f"unknown reasoning backend '{name}'; registered: ['none', 'hermes']")
