"""Content-addressed compute checkpoint / resume (Increments 27 + 28).

Two complementary, opt-in mechanisms, both deterministic by construction (the
key is a content hash of the inputs, so a hit reproduces the exact result and
introduces no new non-determinism):

- Docking result cache (Increment 27): `cached_dock` / `ComputeCache` reuse a
  completed `engine.dock` result keyed by (receptor, ligand SMILES, box,
  protocol, engine) — "don't redo completed work".
- MD trajectory checkpoint (Increment 28): `save_md_state` / `load_md_state` /
  `md_state_key` persist the OpenMM State XML so a later run can CONTINUE the
  simulation (extend to more steps) instead of restarting — "extend a partial
  simulation". The state XML is binary and not byte-stable across OpenMM builds,
  so it is stored as a sidecar with a recorded sha256, never folded into a
  reproducibility hash of the JSON artifacts.

Both are disabled by default (recompute clean); enable via params['resume']=true.
The cache lives at store/checkpoints/, outside any run dir, so it survives
across runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..docking.base import DockPose, DockProtocol, DockingEngine, PreparedLigand, PreparedReceptor
from ..logging.hashing import hash_content, hash_jsonable


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


# ---- MD trajectory checkpoint (Increment 28) ----

_MD_STATE_IGNORE = ("steps", "note")


def md_state_key(receptor_id: str, protocol, engine_name: str,
                 engine_version: str | None) -> str:
    return hash_jsonable({
        "receptor": receptor_id,
        "protocol": {k: v for k, v in protocol.model_dump().items()
                     if k not in _MD_STATE_IGNORE},
        "engine": engine_name,
        "engine_version": engine_version,
    })


def md_state_path(store, key: str) -> Path:
    """Path to the OpenMM State XML for a checkpoint key. The engine writes the
    State (+ its `.reference.npy` minimization-reference sidecar) here directly;
    the agent only manages `meta.json` alongside it."""
    return Path(store.root) / "checkpoints" / "md_state" / key / "state.xml"


def write_md_meta(store, key: str, state_xml: bytes, total_steps: int) -> dict:
    """Write meta.json for an already-saved State; returns {total_steps, state_hash}.

    `total_steps` is the cumulative number of steps simulated (used as the next
    resume's prior_steps)."""
    d = Path(store.root) / "checkpoints" / "md_state" / key
    d.mkdir(parents=True, exist_ok=True)
    meta = {"total_steps": total_steps, "state_hash": hash_content(state_xml)}
    (d / "meta.json").write_text(json.dumps(meta))
    return meta


def load_md_state(store, key: str) -> dict | None:
    """Return {state_path, total_steps, state_hash} or None."""
    state_path = md_state_path(store, key)
    if not state_path.exists():
        return None
    meta: dict = {}
    meta_path = state_path.with_name("meta.json")
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            meta = {}
    return {
        "state_path": state_path,
        "total_steps": int(meta.get("total_steps", 0)),
        "state_hash": meta.get("state_hash"),
    }
