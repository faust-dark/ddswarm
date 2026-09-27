"""MD engine abstraction (like DockingEngine): run receptor MD and return
metrics. Engines return MdRun (status + metrics), never fabricate."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..contracts.dynamics_result import MdProtocol, MdRun

# Sidecar suffix for the minimization reference written next to a trajectory
# checkpoint State file (Increment 28). Kept in one place so the engine (writer)
# and the agent (persister) agree on the filename.
REFERENCE_SUFFIX = ".reference.npy"


class MdEngine(ABC):
    name: str = ""
    version: str | None = None

    @abstractmethod
    def run_md(self, pdb_path: Path, receptor_id: str, protocol: MdProtocol,
               resume_state: Path | None = None, prior_steps: int = 0,
               checkpoint_path: Path | None = None) -> MdRun:
        """Run MD on the receptor at `pdb_path` and return metrics.

        `resume_state` (optional) is the path to a prior OpenMM State XML; when
        given, the simulation continues from that state instead of restarting,
        and the saved minimization reference (`resume_state + REFERENCE_SUFFIX`,
        when present) is restored so the RMSD uses the same reference.
        `prior_steps` is the number of steps already simulated. `checkpoint_path`
        (optional) is where the final State is written for a future resume.
        """


def make_engine(name: str = "openmm") -> MdEngine:
    if name in (None, "", "null", "none"):
        from .null_engine import NullMdEngine
        return NullMdEngine()
    if name == "openmm":
        from .openmm_engine import OpenMMMdEngine
        return OpenMMMdEngine()
    raise KeyError(f"unknown MD engine '{name}'")
