"""MD engine abstraction (like DockingEngine): run receptor MD and return
metrics. Engines return MdRun (status + metrics), never fabricate."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..contracts.dynamics_result import MdProtocol, MdRun


class MdEngine(ABC):
    name: str = ""
    version: str | None = None

    @abstractmethod
    def run_md(self, pdb_path: Path, receptor_id: str, protocol: MdProtocol) -> MdRun:
        """Run MD on the receptor at `pdb_path` and return metrics."""


def make_engine(name: str = "openmm") -> MdEngine:
    if name in (None, "", "null", "none"):
        from .null_engine import NullMdEngine
        return NullMdEngine()
    if name == "openmm":
        from .openmm_engine import OpenMMMdEngine
        return OpenMMMdEngine()
    raise KeyError(f"unknown MD engine '{name}'")
