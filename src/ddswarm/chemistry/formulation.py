"""Deterministic, rule-based developability estimates from RDKit descriptors.

Everything here is a *predicted* heuristic (logP/TPSA/ionizable-group rules),
explicitly NOT measured aqueous solubility or permeability. Sources for the
rules are cited inline where they are standard (Veber TPSA<=140 A^2 for oral
bioavailability; BCS 2x2 classification)."""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import Descriptors

# SMARTS for ionizable groups (kept deliberately small and curated).
_ACIDIC_SMARTS: dict[str, str] = {
    "carboxylic acid": "[CX3](=O)[OX2H1]",
    "sulfonic acid": "[SX4](=O)(=O)[OX2H1]",
    "tetrazole": "c1nnn[nH]1",
}

# Basic nitrogen: aliphatic amines (excluding amide/sulfonamide N) plus
# pyridine-like aromatic N (nX2); pyrrole N (nX3) is correctly excluded.
_BASIC_AMINE_SMARTS = "[NX3;H0,H1,H2;!$(NC=O);!$(NS(=O)(=O))]"
_BASIC_AROMATIC_N_SMARTS = "[nX2]"


def _matches(mol: Chem.Mol, smarts: str) -> int:
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        return 0
    return len(mol.GetSubstructMatches(patt))


def detect_ionizable(mol: Chem.Mol) -> tuple[list[str], int, int]:
    """Return (labels, n_acidic, n_basic)."""
    labels: list[str] = []
    n_acidic = 0
    n_basic = 0
    for name, sma in _ACIDIC_SMARTS.items():
        c = _matches(mol, sma)
        if c:
            n_acidic += c
            labels.append(f"acidic: {name} (x{c})")
    amine = _matches(mol, _BASIC_AMINE_SMARTS)
    arom = _matches(mol, _BASIC_AROMATIC_N_SMARTS)
    if amine:
        n_basic += amine
        labels.append(f"basic: aliphatic amine (x{amine})")
    if arom:
        n_basic += arom
        labels.append(f"basic: aromatic N (x{arom})")
    return labels, n_acidic, n_basic


def _solubility_class(logp: float) -> str:
    """Crude aqueous-solubility class from logP (heuristic)."""
    if logp <= 1.0:
        return "high"
    if logp <= 5.0:
        return "moderate"
    return "low"


def _permeability_class(logp: float, tpsa: float) -> str:
    """Crude passive-permeability class from TPSA/logP (Veber TPSA<=140 A^2)."""
    if tpsa > 140.0:
        return "low"
    if tpsa < 90.0 and 0.0 <= logp <= 6.0:
        return "high"
    return "moderate"


def _bcs_class(sol: str, perm: str) -> str:
    """BCS class from the 2x2 solubility x permeability grid. `moderate` is
    treated as the favorable side (BCS solubility is binary high/low)."""
    sol_high = sol != "low"
    perm_high = perm == "high"
    if sol_high and perm_high:
        return "I"
    if not sol_high and perm_high:
        return "II"
    if sol_high and not perm_high:
        return "III"
    return "IV"


def _salt_suggestion(n_acidic: int, n_basic: int) -> str | None:
    if n_basic and not n_acidic:
        return "basic center present (amine); consider a HCl or mesylate salt if solubility-limited"
    if n_acidic and not n_basic:
        return "acidic center present (COOH/SO3H/tetrazole); consider a Na+/K+ salt if solubility-limited"
    if n_acidic and n_basic:
        return "zwitterionic (acidic + basic); internal salt possible — salt form needs case-by-case selection"
    return None


def _prodrug_suggestion(sol: str, perm: str, n_acidic: int) -> str | None:
    if perm == "low" and n_acidic:
        return "ester prodrug of the acidic group to mask polarity and improve permeability"
    if sol == "low" and n_acidic:
        return "ester prodrug of the acidic group to improve aqueous solubility"
    if sol == "low":
        return "lipophilic (logP>5); consider a prodrug bearing a solubilizing group (e.g. phosphate ester)"
    return None


def assess_formulation(mol: Chem.Mol) -> dict:
    """Deterministic developability estimate for an RDKit molecule."""
    logp = float(Descriptors.MolLogP(mol))
    tpsa = float(Descriptors.TPSA(mol))
    labels, n_acidic, n_basic = detect_ionizable(mol)
    sol = _solubility_class(logp)
    perm = _permeability_class(logp, tpsa)
    return {
        "logp": round(logp, 2),
        "tpsa": round(tpsa, 1),
        "solubility_class": sol,
        "permeability_class": perm,
        "bcs_class": _bcs_class(sol, perm),
        "ionizable_groups": labels,
        "salt_suggestion": _salt_suggestion(n_acidic, n_basic),
        "prodrug_suggestion": _prodrug_suggestion(sol, perm, n_acidic),
        "note": "rule-based estimate (logP/TPSA heuristics); not measured properties",
    }
