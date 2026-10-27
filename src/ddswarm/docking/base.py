"""Docking engine abstraction. The engine computes — it prepares the receptor
and ligands, runs the docking program, and returns poses with provenance. The
agent reasons over those results and builds the DockingResult artifact."""

from __future__ import annotations

import os
import re
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from ..fsutil import atomic_write_bytes

RCSB_URL = "https://files.rcsb.org/download/{pid}.pdb"
PDB_ID_RE = re.compile(r"^[0-9A-Za-z]{4}$")
# identifiers that are safe to use as a single filename component
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
DOWNLOAD_TIMEOUT_S = 60.0
MAX_PDB_BYTES = 200 * 1024 * 1024
_UA = {"User-Agent": "ddswarm/0.1 (receptor download)"}


def is_valid_pdb_id(value: object) -> bool:
    """True for a 4-character RCSB PDB id (e.g. '2ITY')."""
    return isinstance(value, str) and bool(PDB_ID_RE.match(value))


def is_safe_name(value: object) -> bool:
    """True when ``value`` can be used as one filename component (no '/', '..')."""
    return isinstance(value, str) and bool(_SAFE_NAME_RE.match(value)) and ".." not in value


def looks_like_pdb(data: bytes) -> bool:
    """Cheap sanity check that ``data`` is a PDB-format structure, not an HTML
    error page / JSON error body: it must carry ATOM or HETATM records."""
    if not data or data.lstrip()[:1] in (b"<", b"{"):
        return False
    return (data.startswith((b"ATOM  ", b"HETATM"))
            or b"\nATOM  " in data or b"\nHETATM" in data)


def download_pdb(url: str, dest: str | Path, timeout: float = DOWNLOAD_TIMEOUT_S) -> str | None:
    """Download a PDB file to ``dest`` (atomically). Returns None on success or a
    short error message. HTTP errors (404, 5xx), non-http(s) URLs, oversized
    bodies, and non-PDB content are rejected and nothing is written."""
    scheme = urllib.parse.urlparse(url).scheme.lower()
    if scheme not in ("http", "https"):
        return f"refusing non-http(s) receptor URL scheme {scheme!r}"
    try:
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read(MAX_PDB_BYTES + 1)
    except urllib.error.HTTPError as exc:
        return f"HTTP {exc.code} for {url}"
    except Exception as exc:  # noqa: BLE001 - network errors, timeouts
        return f"download failed: {exc}"
    if len(data) > MAX_PDB_BYTES:
        return f"receptor file exceeds {MAX_PDB_BYTES} bytes"
    if not looks_like_pdb(data):
        return "downloaded content is not a PDB structure (no ATOM/HETATM records)"
    atomic_write_bytes(dest, data)
    return None


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
    """Cache path for a receptor PDB, keyed by its identifier/pdb_id.

    Raises ValueError for an identifier that could escape the cache directory."""
    if not is_safe_name(identifier):
        raise ValueError(f"invalid receptor identifier for cache path: {identifier!r}")
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
    identifier: str = ""        # receptor id / source it was prepared from
    content_hash: str = ""      # sha256 of the prepared PDBQT (docking cache key)


@dataclass
class PreparedLigand:
    canonical_smiles: str = ""
    pdbqt_path: Path | None = None
    smiles_idx_map: dict = field(default_factory=dict)   # smi_idx_1based -> serial
    heavy_atom_count: int = 0
    note: str = ""
    content_hash: str = ""      # sha256 of the prepared PDBQT (docking cache key)


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
    # CPU threads for one docking call (Vina --cpu). Set by the parallel docking
    # scheduler; None = engine default. Not part of the cache key: Vina seeds
    # each Monte-Carlo task from --seed independently of the thread count.
    cpu: int | None = None


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
