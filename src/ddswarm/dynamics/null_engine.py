"""NullMdEngine: deterministic no-op — reports `not_run` honestly when no MD
engine is configured (mirrors NullAdmetProvider)."""

from __future__ import annotations

from pathlib import Path

from ..contracts.dynamics_result import MdProtocol, MdRun
from .base import MdEngine


class NullMdEngine(MdEngine):
    name = "null"

    def run_md(self, pdb_path: Path, receptor_id: str, protocol: MdProtocol,
               resume_state: Path | None = None, prior_steps: int = 0,
               checkpoint_path: Path | None = None) -> MdRun:
        return MdRun(
            receptor_id=receptor_id,
            status="not_run",
            stability_note="no MD engine configured (dynamics requires OpenMM + PDBFixer; "
            "ligand MD additionally requires openff-toolkit, unavailable on Python 3.14)",
        )
