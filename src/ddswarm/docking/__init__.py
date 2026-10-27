"""Docking engines (Increment 3)."""

from __future__ import annotations

import os
from pathlib import Path

from .base import (
    DockingEngine,
    DockPose,
    DockProtocol,
    PreparedLigand,
    PreparedReceptor,
    QCResult,
    ReceptorSource,
)
from .vina_engine import VinaEngine

_REPO_ROOT = Path(__file__).resolve().parents[3]


def default_vina_binary() -> Path:
    return Path(os.environ.get("VINA_BINARY") or (_REPO_ROOT / "tools" / "vina"))


def make_engine(name: str, vina_binary: str | Path | None = None) -> DockingEngine:
    if name == "vina":
        return VinaEngine(binary_path=vina_binary or default_vina_binary())
    raise KeyError(f"unknown docking engine '{name}'; registered: ['vina']")


__all__ = [
    "DockPose",
    "DockProtocol",
    "DockingEngine",
    "PreparedLigand",
    "PreparedReceptor",
    "QCResult",
    "ReceptorSource",
    "VinaEngine",
    "default_vina_binary",
    "make_engine",
]
