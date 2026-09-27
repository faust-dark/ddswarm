"""Content-addressed compute checkpoint / resume (Increment 27).

A deterministic, opt-in result cache for expensive deterministic computations
(molecular docking, receptor MD). The cache key is a content hash of the
computation's inputs (receptor id, ligand SMILES, search box, protocol, engine),
so a cache hit reproduces the exact same result — reusing it introduces no new
non-determinism and is safe to share across runs.

Honesty contract: this is a "don't redo completed work" cache, NOT a trajectory
resumption mechanism — it skips a computation whose result is already cached; it
does not extend a partially-run simulation (OpenMM CheckpointReporter is the
right tool for that and is deliberately out of scope here).

Disabled by default (recompute clean); enable via params['resume']=true. The
cache lives at store/checkpoints/<kind>/<key>.json, outside any run dir, so it
survives across runs and is never swept by run cleanup.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..contracts.dynamics_result import MdRun
from ..docking.base import DockPose, DockProtocol, DockingEngine, PreparedLigand, PreparedReceptor
from ..logging.hashing import hash_jsonable


class ComputeCache:
    def __init__(self, store, enabled: bool):
        self._root = Path(store.root) / "checkpoints"
        self.enabled = bool(enabled)

    def _path(self, kind: str, key: str) -> Path:
        return self._root / kind / f"{key}.json"

    def get(self, kind: str, key: str) -> Any | None:
        if not self.enabled:
            return None
        p = self._path(kind, key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            return None  # corrupt entry -> treat as a miss

    def put(self, kind: str, key: str, value: Any) -> None:
        if not self.enabled:
            return
        p = self._path(kind, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value))


# ---- docking pose (de)serialization ----
# DockPose carries pose_atoms (serial -> (x,y,z)) and a work_dir-local pose_file.
# The pose_file path is not portable across runs, so only the DATA is cached:
# rank, affinity, rmsd bounds, and the atom coordinates (keys as strings, values
# as [x,y,z] lists to stay JSON-safe).

def poses_to_cache(poses: list[DockPose]) -> list[dict]:
    out = []
    for p in poses:
        out.append({
            "rank": p.rank,
            "affinity_kcal_per_mol": p.affinity_kcal_per_mol,
            "rmsd_lower_bound": p.rmsd_lower_bound,
            "rmsd_upper_bound": p.rmsd_upper_bound,
            "pose_atoms": {str(k): list(v) for k, v in (p.pose_atoms or {}).items()},
        })
    return out


def poses_from_cache(data: list) -> list[DockPose]:
    out = []
    for d in data:
        atoms = {int(k): tuple(float(x) for x in v)
                 for k, v in (d.get("pose_atoms") or {}).items()}
        out.append(DockPose(
            rank=d.get("rank", 0),
            affinity_kcal_per_mol=d.get("affinity_kcal_per_mol"),
            rmsd_lower_bound=d.get("rmsd_lower_bound"),
            rmsd_upper_bound=d.get("rmsd_upper_bound"),
            pose_file=None,  # not portable across runs
            pose_atoms=atoms,
        ))
    return out


def _dock_key(receptor_id: str, ligand_smiles: str, protocol: DockProtocol,
              engine: DockingEngine) -> str:
    return hash_jsonable({
        "receptor": receptor_id,
        "ligand": ligand_smiles,
        "box_center": list(protocol.box_center) if protocol.box_center else None,
        "box_size": list(protocol.box_size) if protocol.box_size else None,
        "exhaustiveness": protocol.exhaustiveness,
        "num_modes": protocol.num_modes,
        "seed": protocol.seed,
        "engine": engine.name,
        "engine_version": engine.version,
    })


def cached_dock(engine: DockingEngine, cache: ComputeCache, receptor_id: str,
                receptor: PreparedReceptor, ligand: PreparedLigand,
                protocol: DockProtocol, work_dir: Path) -> list[DockPose]:
    """engine.dock with opt-in content-addressed resume."""
    if not cache.enabled:
        return engine.dock(receptor, ligand, protocol, work_dir)
    key = _dock_key(receptor_id, ligand.canonical_smiles, protocol, engine)
    hit = cache.get("dock", key)
    if hit is not None:
        return poses_from_cache(hit)
    poses = engine.dock(receptor, ligand, protocol, work_dir)
    cache.put("dock", key, poses_to_cache(poses))
    return poses


def cached_run_md(engine, cache: ComputeCache, pdb_path: Path, receptor_id: str,
                  protocol) -> MdRun:
    """engine.run_md with opt-in content-addressed resume."""
    if not cache.enabled:
        return engine.run_md(pdb_path, receptor_id, protocol)
    key = hash_jsonable({
        "receptor": receptor_id,
        "protocol": protocol.model_dump(),
        "engine": engine.name,
        "engine_version": getattr(engine, "version", None),
    })
    hit = cache.get("md", key)
    if hit is not None:
        return MdRun.model_validate(hit)
    run = engine.run_md(pdb_path, receptor_id, protocol)
    cache.put("md", key, run.model_dump())
    return run
