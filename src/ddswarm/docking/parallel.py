"""Parallel docking of molecule x receptor pairs.

Jobs run on a thread pool: each docking call is an external Vina process, so
threads only wait on subprocesses (no GIL contention) and the prepared
receptor/ligand objects (incl. RDKit mols) need no pickling.

CPU budget: a batch of N jobs runs on W = min(N, cpus) workers and gives each
Vina call --cpu = cpus // W, so one big job still uses the whole machine and
many jobs use one thread each. A process-wide token pool (``CPU_BUDGET``) caps
the total Vina threads across concurrent batches — the docking and selectivity
agents may run at the same time under the parallel runner — so the machine is
never oversubscribed. Cache hits never take CPU tokens.

Determinism: results are returned keyed by job and consumers read them in job
(declaration) order, so completion order never leaks into an artifact. Each job
is isolated: an exception or a Vina timeout yields [] for that job only.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Hashable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from .base import DockingEngine, DockPose, DockProtocol, PreparedLigand, PreparedReceptor

if TYPE_CHECKING:
    from ..orchestrator.checkpoint import ComputeCache


def cpu_count() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


class CpuBudget:
    """Counting token pool: ``hold(n)`` blocks until n CPU tokens are free."""

    def __init__(self, total: int):
        self.total = max(1, int(total))
        self._free = self.total
        self._cond = threading.Condition()

    @contextmanager
    def hold(self, n: int):
        n = max(1, min(int(n), self.total))
        with self._cond:
            while self._free < n:
                self._cond.wait()
            self._free -= n
        try:
            yield
        finally:
            with self._cond:
                self._free += n
                self._cond.notify_all()


CPU_BUDGET = CpuBudget(cpu_count())


@dataclass
class DockJob:
    key: Hashable                 # caller's key, e.g. ("mol", receptor_name, molecule_id)
    receptor_id: str
    receptor: PreparedReceptor
    ligand: PreparedLigand
    protocol: DockProtocol
    work_dir: Path


def plan_workers(n_jobs: int, cpus: int, max_workers: int | None = None) -> tuple[int, int]:
    """(workers, vina_cpu_per_job) for a batch, never exceeding ``cpus`` threads."""
    cap = cpus if not max_workers else max(1, min(int(max_workers), cpus))
    workers = max(1, min(n_jobs, cap))
    return workers, max(1, cpus // workers)


def dock_jobs(engine: DockingEngine, cache: ComputeCache, jobs: list[DockJob],
              max_workers: int | None = None,
              budget: CpuBudget | None = None) -> dict[Hashable, list[DockPose]]:
    """Dock every job concurrently; returns {job.key: poses} ([] on failure)."""
    # imported here: ddswarm.orchestrator's package import pulls in the agents,
    # which import this module (import cycle at module load)
    from ..orchestrator.checkpoint import _dock_key, dock_cache_lookup, dock_cache_store

    if not jobs:
        return {}
    budget = budget or CPU_BUDGET
    workers, cpu = plan_workers(len(jobs), budget.total, max_workers)

    def run_one(job: DockJob) -> list[DockPose]:
        proto = replace(job.protocol, cpu=cpu)
        try:
            key = None
            if cache.enabled:
                key = _dock_key(job.receptor_id, job.ligand.canonical_smiles, proto, engine,
                                job.receptor, job.ligand)
                hit = dock_cache_lookup(cache, key, job.work_dir)
                if hit is not None:
                    return hit
            with budget.hold(cpu):
                poses = engine.dock(job.receptor, job.ligand, proto, job.work_dir)
            if key is not None and poses:
                dock_cache_store(cache, key, poses)
            return poses
        except Exception:  # noqa: BLE001 - one bad job must not sink the batch
            return []

    if workers == 1:
        return {job.key: run_one(job) for job in jobs}
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ddswarm-dock") as pool:
        results = list(pool.map(run_one, jobs))  # map preserves job order
    return {job.key: res for job, res in zip(jobs, results)}
