"""Content-addressed compute checkpoint / resume (Increments 27 + 28).

Two complementary mechanisms, both deterministic by construction (the key is a
content hash of the inputs, so a hit reproduces the exact result and introduces
no new non-determinism):

- Docking result cache (Increment 27): `cached_dock` / `ComputeCache` reuse a
  completed `engine.dock` result keyed by (receptor id + prepared-receptor
  hash, ligand SMILES + prepared-ligand hash, box, protocol, engine version,
  preparation tool versions) — "don't redo completed work". ON by default for
  docking/selectivity (see `dock_cache_enabled`); prepared receptors are cached
  alongside under checkpoints/receptor/.
- MD trajectory checkpoint (Increment 28): `save_md_state` / `load_md_state` /
  `md_state_key` persist the OpenMM State XML so a later run can CONTINUE the
  simulation (extend to more steps) instead of restarting — "extend a partial
  simulation". The state XML is binary and not byte-stable across OpenMM builds,
  so it is stored as a sidecar with a recorded sha256, never folded into a
  reproducibility hash of the JSON artifacts.

The MD checkpoint stays opt-in via params['resume']=true; the docking cache can
be turned off with params['dock_cache']=false (CLI: --no-dock-cache). The cache lives at
store/checkpoints/, outside any run dir, so it survives
across runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..docking.base import DockingEngine, DockPose, DockProtocol, PreparedLigand, PreparedReceptor
from ..fsutil import atomic_write_text
from ..logging.hashing import hash_content, hash_jsonable


class ComputeCache:
    def __init__(self, store, enabled: bool):
        self._root = Path(store.root) / "checkpoints"
        self.enabled = bool(enabled)

    @property
    def root(self) -> Path:
        return self._root

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
        atomic_write_text(self._path(kind, key), json.dumps(value))


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
              engine: DockingEngine, receptor: PreparedReceptor | None = None,
              ligand: PreparedLigand | None = None) -> str:
    """Content-addressed docking key.

    Beyond the receptor id it covers the CONTENT actually docked — the
    prepared receptor and ligand PDBQT hashes — and the preparation toolchain
    versions, so a re-prepared receptor (new PDB file, PDBFixer/Open Babel
    upgrade) or a re-prepared ligand (Meeko/RDKit upgrade) never reuses stale
    poses. `protocol.cpu` is deliberately excluded (it changes wall time, not
    the result)."""
    return hash_jsonable({
        "receptor": receptor_id,
        "receptor_pdbqt": getattr(receptor, "content_hash", "") or "",
        "ligand": ligand_smiles,
        "ligand_pdbqt": getattr(ligand, "content_hash", "") or "",
        "box_center": list(protocol.box_center) if protocol.box_center else None,
        "box_size": list(protocol.box_size) if protocol.box_size else None,
        "exhaustiveness": protocol.exhaustiveness,
        "num_modes": protocol.num_modes,
        "seed": protocol.seed,
        "engine": engine.name,
        "engine_version": engine.version,
        "prep_versions": getattr(engine, "prep_versions", None) or {},
    })


def dock_cache_enabled(params: dict) -> bool:
    """The docking result cache is ON by default (it is content-addressed and
    deterministic). Turn it off with params['dock_cache']=false; an explicit
    params['resume'] is honoured for backward compatibility."""
    if "dock_cache" in params:
        return bool(params["dock_cache"])
    if "resume" in params:
        return bool(params["resume"])
    return True


def dock_cache_lookup(cache: ComputeCache, key: str, work_dir: Path) -> list[DockPose] | None:
    """Return cached poses for ``key`` (None on a miss). When the entry carries
    the docked PDBQT text it is restored into ``work_dir`` so the poses keep a
    real pose_file, exactly like a fresh run."""
    hit = cache.get("dock", key)
    if hit is None:
        return None
    if isinstance(hit, dict):  # current format: {"poses": [...], "pose_text": str|None}
        poses = poses_from_cache(hit.get("poses") or [])
        text = hit.get("pose_text")
        if text is not None and poses:
            out = Path(work_dir) / "docked.pdbqt"
            atomic_write_text(out, text)
            for p in poses:
                p.pose_file = out
        return poses
    return poses_from_cache(hit)  # legacy list-only entries


def dock_cache_store(cache: ComputeCache, key: str, poses: list[DockPose]) -> None:
    pose_text = None
    files = {p.pose_file for p in poses if p.pose_file}
    if len(files) == 1:
        try:
            pose_text = Path(files.pop()).read_text()
        except OSError:
            pose_text = None
    cache.put("dock", key, {"poses": poses_to_cache(poses), "pose_text": pose_text})


def cached_dock(engine: DockingEngine, cache: ComputeCache, receptor_id: str,
                receptor: PreparedReceptor, ligand: PreparedLigand,
                protocol: DockProtocol, work_dir: Path) -> list[DockPose]:
    """engine.dock with content-addressed reuse."""
    if not cache.enabled:
        return engine.dock(receptor, ligand, protocol, work_dir)
    key = _dock_key(receptor_id, ligand.canonical_smiles, protocol, engine, receptor, ligand)
    hit = dock_cache_lookup(cache, key, work_dir)
    if hit is not None:
        return hit
    poses = engine.dock(receptor, ligand, protocol, work_dir)
    if poses:  # never cache a failure (timeouts / crashes are retried next run)
        dock_cache_store(cache, key, poses)
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
