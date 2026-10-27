"""One-command setup: install the [docking] extras and fetch the AutoDock Vina binary.

Idempotent and safe: skips steps that are already satisfied, prompts before
downloading a third-party binary (override with ``--yes``), and verifies the
result by running the doctor check. Reuses the doctor's probes and the shared
HTTP client, so it can never disagree with what the pipeline will actually use.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

from .chemistry.http import request
from .docking import default_vina_binary
from .doctor import (
    _docking_prep,
    _obabel_status,
    _openmm_status,
    _vina_status,
    run_doctor,
)

_VINA_VERSION = "1.2.7"
_VINA_BASE = f"https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v{_VINA_VERSION}"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _asset_for(plat: str, machine: str) -> str:
    """Map (sys.platform, platform.machine) to the Vina release asset name."""
    m = machine.lower()
    if plat == "linux":
        if m in ("x86_64", "amd64"):
            return f"vina_{_VINA_VERSION}_linux_x86_64"
        if m in ("aarch64", "arm64"):
            return f"vina_{_VINA_VERSION}_linux_aarch64"
    elif plat == "darwin":
        if m in ("x86_64", "amd64"):
            return f"vina_{_VINA_VERSION}_mac_x86_64"
        if m in ("aarch64", "arm64"):
            return f"vina_{_VINA_VERSION}_mac_aarch64"
    elif plat == "win32":
        return f"vina_{_VINA_VERSION}_win.exe"
    return ""


def _vina_asset() -> str:
    return _asset_for(sys.platform, platform.machine())


def _needs_deps() -> bool:
    prep = _docking_prep()
    return (any(v == "missing" for v in prep.values())
            or not _obabel_status()[0]
            or not _openmm_status()[0])


def _needs_vina() -> bool:
    return not _vina_status()[0]


def _pip_install_docking() -> tuple[int, str]:
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", ".[docking]"],
            cwd=_repo_root(), capture_output=True, text=True, timeout=900,
        )
        tail = (proc.stdout or "").strip().splitlines()
        return proc.returncode, (tail[-1] if tail else "(no output)")
    except Exception as exc:  # noqa: BLE001 - report honestly, never raise
        return 1, f"pip install failed: {exc}"


def _download_vina(target: Path, asset: str) -> tuple[bool, str]:
    data, error = request(f"{_VINA_BASE}/{asset}", timeout=300, retries=2,
                          purpose="setup: download AutoDock Vina")
    if error is not None:
        return False, f"download failed: {error}"
    if not data:
        return False, "download produced no data"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        if sys.platform != "win32":
            target.chmod(0o755)
        return True, f"wrote {target}"
    except OSError as exc:
        return False, f"write failed: {exc}"


def run_setup(yes: bool = False, dry_run: bool = False) -> int:
    vina_target = default_vina_binary()
    asset = _vina_asset()
    needs_deps = _needs_deps()
    needs_vina = _needs_vina()

    plan: list[str] = []
    if needs_deps:
        plan.append("install [docking] extras (meeko, scipy, gemmi, pdbfixer, openbabel-wheel, openmm)")
    if needs_vina:
        if asset:
            plan.append(f"download AutoDock Vina {_VINA_VERSION} ({asset}) -> {vina_target}")
        else:
            plan.append(f"WARNING: no prebuilt Vina binary for {sys.platform}/{platform.machine()}; "
                        "build from source or set $VINA_BINARY")

    print("ddswarm setup")
    print()
    if not plan:
        print("nothing to do: [docking] extras and the Vina binary are already in place")
        print()
        print(run_doctor())
        return 0

    for line in plan:
        print(f"  - {line}")
    print()

    if dry_run:
        print("(dry run: no changes made)")
        return 0

    if not yes:
        answer = input("proceed? [yes/no]: ").strip().lower()
        if answer not in ("", "yes", "y"):
            print("aborted")
            return 1

    ok = True
    if needs_deps:
        print("installing [docking] extras ...")
        rc, output = _pip_install_docking()
        print(output)
        ok = ok and rc == 0
    if needs_vina and asset:
        print(f"downloading AutoDock Vina {_VINA_VERSION} ...")
        vina_ok, msg = _download_vina(vina_target, asset)
        print(msg)
        ok = ok and vina_ok

    print()
    print(run_doctor())
    return 0 if ok else 1
