"""Deterministic synthetic-accessibility scoring (RDKit SA Score, Ertl &
Schuffenhauer 2009). The agent reasons; this computes. Pure function — no
contracts, no fabrication."""

from __future__ import annotations

from rdkit import Chem
from rdkit.Contrib.SA_Score import sascorer

# 1 = trivial to synthesize, 10 = very hard (Ertl & Schuffenhauer 2009).
EASY_MAX = 3.0
MODERATE_MAX = 6.0


def compute_sa_score(mol: Chem.Mol) -> float:
    """Synthetic accessibility score, rounded to 3 decimals."""
    return round(float(sascorer.calculateScore(mol)), 3)


def classify_accessibility(sa_score: float) -> str:
    """Map an SA score to easy/moderate/hard."""
    if sa_score <= EASY_MAX:
        return "easy"
    if sa_score <= MODERATE_MAX:
        return "moderate"
    return "hard"
