"""Docking engine abstraction. The engine computes — it prepares the receptor
and ligands, runs the docking program, and returns poses with provenance. The
agent reasons over those results and builds the DockingResult artifact."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


def pdb_cache_dir() -> Path:
    """Directory for cached RCSB PDB downloads (override with ``$DDWARM_PDB_CACHE``,
    default ``~/.cache/ddswarm/pdbs`` or ``$XDG_CACHE_HOME/ddswarm/pdbs``)."""
    env = os.environ.get("DDWARM_PDB_CACHE")
    if env:
        return Path(env)
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "ddswarm" / "pdbs"


def pdb_cache_path(identifier: str) -> Path:
    """Cache path for a receptor PDB, keyed by its identifier/pdb_id."""
    return pdb_cache_dir() / f"{identifier}.pdb"


@dataclass
class ReceptorSource:
    kind: str = "pdb_id"        # "pdb_id" | "file"
    identifier: str = ""        # e.g. "2ITY"
    url: str | None = None


@dataclass
class QCResult:
    check: str
    status: str                 # "ok" | "issue" | "n_a"
    detail: str = ""


@dataclass
class PreparedReceptor:
    pdbqt_path: Path | None = None
    atom_count: int = 0
    residue_count: int = 0
    chain_count: int = 0
    preparation_tool: str = ""
    qc_checks: list[QCResult] = field(default_factory=list)
    # co-crystal ligand (if the source PDB carries one), for redocking validation
    reference_ligand_name: str | None = None
    reference_ligand_mol: object | None = None   # RDKit mol, heavy atoms
    note: str = ""


@dataclass
class PreparedLigand:
    canonical_smiles: str = ""
    pdbqt_path: Path | None = None
    smiles_idx_map: dict = field(default_factory=dict)   # smi_idx_1based -> serial
    heavy_atom_count: int = 0
    note: str = ""


@dataclass
class DockPose:
    rank: int
    affinity_kcal_per_mol: float | None = None
    rmsd_lower_bound: float | None = None
    rmsd_upper_bound: float | None = None
    pose_file: Path | None = None
    pose_atoms: dict = field(default_factory=dict)   # serial -> (x, y, z)


@dataclass
class DockProtocol:
    box_center: tuple[float, float, float] | None = None
    box_size: tuple[float, float, float] | None = None
    exhaustiveness: int = 8
    num_modes: int = 9
    seed: int = 42


def box_from_reference_mol(mol, margin: float = 11.25) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return (center, size) for a cubic search box centred on a reference
    molecule's centroid (the standard redocking setup)."""
    conf = mol.GetConformer()
    pts = conf.GetPositions()
    center = tuple(float(c) for c in pts.mean(axis=0))
    size = (margin * 2.0, margin * 2.0, margin * 2.0)
    return center, size


class DockingEngine(ABC):
    name: str = ""
    version: str = ""

    @abstractmethod
    def prepare_receptor(self, source: ReceptorSource, work_dir: Path) -> PreparedReceptor:
        """Download/prepare the receptor PDBQT, run structural QC, and extract
        any co-crystal ligand for later redocking validation."""

    @abstractmethod
    def prepare_ligand(self, smiles: str, work_dir: Path) -> PreparedLigand:
        """Generate a 3D ligand and its PDBQT (with the SMILES IDX table)."""

    @abstractmethod
    def dock(
        self,
        receptor: PreparedReceptor,
        ligand: PreparedLigand,
        protocol: DockProtocol,
        work_dir: Path,
    ) -> list[DockPose]:
        """Run the docking program and return ranked poses."""

    @abstractmethod
    def available(self) -> tuple[bool, str]:
        """Whether the engine's underlying binary is present and runnable."""
