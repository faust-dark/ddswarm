"""Environment self-diagnostic for ddswarm (offline and deterministic by default).

``ddswarm doctor`` reports which compute paths will work and which will degrade,
with the exact fix for each gap. It reuses the engines' own availability probes
so the report can never disagree with what the pipeline will actually do.
"""

from __future__ import annotations

import importlib.metadata
import shutil
import sys
from pathlib import Path

from .chemistry.http import request
from .docking import default_vina_binary
from .docking.base import RCSB_URL, pdb_cache_dir
from .docking.vina_engine import VinaEngine
from .dynamics.openmm_engine import _OPENMM_AVAILABLE, _OPENMM_VERSION

_DOCKING_FIX = "pip install -e '.[docking]'"
_VINA_FIX = "put the AutoDock Vina binary at tools/vina, or set $VINA_BINARY"
_OPENMM_FIX = "pip install openmm pdbfixer"
_HERMES_FIX = "install/configure hermes, then run with --reasoning hermes"

# The packages the [docking] extra installs (the fix maps 1:1 to these names).
_DOCKING_EXTRA_PACKAGES = ("meeko", "scipy", "gemmi", "pdbfixer", "openbabel-wheel")

_CORE_DEPS = ("pydantic", "rdkit", "reportlab", "matplotlib", "mcp")


def _version_of(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def _python_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def _core_deps() -> dict[str, str]:
    return {name: _version_of(name) for name in _CORE_DEPS}


def _vina_status() -> tuple[bool, str]:
    """The AutoDock Vina binary, independent of the obabel prep tool."""
    path = default_vina_binary()
    if not path.exists():
        return False, f"not found at {path}"
    try:
        ok, version = VinaEngine(binary_path=path)._probe_version()
    except Exception as exc:  # noqa: BLE001 - probe is best-effort, never raise
        return False, f"probe failed: {exc}"
    return (ok, version if ok else "not runnable")


def _docking_prep() -> dict[str, str]:
    """Versions of the [docking] extra packages (receptor/ligand prep)."""
    return {name: _version_of(name) for name in _DOCKING_EXTRA_PACKAGES}


def _obabel_status() -> tuple[bool, str]:
    path = VinaEngine(binary_path=default_vina_binary()).obabel_path
    return (path.exists(), str(path))


def _openmm_status() -> tuple[bool, str]:
    return (_OPENMM_AVAILABLE, _OPENMM_VERSION or "unknown")


def _hermes_status() -> tuple[bool, str]:
    path = shutil.which("hermes")
    return (path is not None, path or "")


def _pdb_cache_status() -> tuple[Path, int]:
    cache_dir = pdb_cache_dir()
    count = len(list(cache_dir.glob("*.pdb"))) if cache_dir.exists() else 0
    return (cache_dir, count)


def _network_status() -> tuple[bool, str]:
    """RCSB reachability. A truncated body still proves the server responded."""
    _data, error = request(
        RCSB_URL.format(pid="6X3X"), timeout=6, retries=0,
        purpose="doctor network check", max_bytes=64,
    )
    if error is None or error.startswith("response exceeds"):
        return True, ""
    return False, error or "unreachable"


def run_doctor(network: bool = False) -> str:
    """Build the doctor report as plain text (no side effects, exit 0)."""
    lines: list[str] = ["ddswarm environment check", ""]

    lines.append(f"python         : {_python_version()} (requires >= 3.11)")
    core = _core_deps()
    lines.append("core deps      : " + ", ".join(f"{n} {v}" for n, v in core.items()))

    vina_ok, vina_msg = _vina_status()
    if vina_ok:
        lines.append(f"vina binary    : OK ({vina_msg})")
    else:
        lines.append(f"vina binary    : MISSING ({vina_msg})")
        lines.append(f"                 -> docking + selectivity degrade; fix: {_VINA_FIX}")

    prep = _docking_prep()
    missing_prep = [n for n, v in prep.items() if v == "missing"]
    if missing_prep:
        lines.append("docking prep   : MISSING " + ", ".join(missing_prep))
        lines.append(f"                 -> receptor/ligand prep degrades; fix: {_DOCKING_FIX}")
    else:
        lines.append("docking prep   : " + ", ".join(f"{n} {v}" for n, v in prep.items()))

    obabel_ok, obabel_path = _obabel_status()
    if obabel_ok:
        lines.append(f"obabel binary  : OK ({obabel_path})")
    else:
        lines.append(f"obabel binary  : MISSING ({obabel_path})")
        lines.append(f"                 -> ligand/receptor PDBQT prep degrades; fix: {_DOCKING_FIX}")

    openmm_ok, openmm_msg = _openmm_status()
    if openmm_ok:
        lines.append(f"openmm         : OK ({openmm_msg})")
    else:
        lines.append("openmm         : MISSING")
        lines.append(f"                 -> receptor MD (dynamics) degrades; fix: {_OPENMM_FIX}")

    hermes_ok, hermes_path = _hermes_status()
    if hermes_ok:
        lines.append(f"reasoning      : OK ({hermes_path})")
    else:
        lines.append("reasoning      : hermes NOT FOUND")
        lines.append("                 -> research/design/overview/retrosynth/patent/clinical/regulatory degrade; "
                     f"fix: {_HERMES_FIX}")

    cache_dir, cache_count = _pdb_cache_status()
    lines.append(f"pdb cache      : {cache_count} receptor(s) at {cache_dir}")

    if network:
        net_ok, net_msg = _network_status()
        if net_ok:
            lines.append("network (RCSB) : reachable")
        else:
            lines.append(f"network (RCSB) : unreachable ({net_msg})")
            lines.append("                 -> docking/selectivity cannot download receptors (need local PDB cache)")

    gaps: list[str] = []
    if not vina_ok:
        gaps.append("docking/selectivity")
    if missing_prep or not obabel_ok:
        gaps.append("docking prep")
    if not openmm_ok:
        gaps.append("receptor MD")
    if not hermes_ok:
        gaps.append("reasoning (knowledge agents)")
    lines.append("")
    if gaps:
        lines.append("summary        : " + ", ".join(gaps) + " will degrade")
    else:
        lines.append("summary        : all compute paths available")
    return "\n".join(lines)
