"""Minimal Kabsch-aligned RMSD for MD trajectories (pure numpy)."""

from __future__ import annotations

import numpy as np


def kabsch_rmsd(ref: np.ndarray, mob: np.ndarray) -> float:
    """RMSD (in the same unit as the input coords) of `mob` vs `ref` after
    optimal rigid-body superposition (Kabsch). Both are (N, 3) coordinate
    arrays; N must match."""
    ref = np.asarray(ref, dtype=float)
    mob = np.asarray(mob, dtype=float)
    ref = ref - ref.mean(axis=0)
    mob = mob - mob.mean(axis=0)
    h = ref.T @ mob
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    aligned = mob @ rot.T
    return float(np.sqrt(np.mean(np.sum((ref - aligned) ** 2, axis=1))))
