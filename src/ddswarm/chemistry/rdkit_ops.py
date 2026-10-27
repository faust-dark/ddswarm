"""Deterministic RDKit operations: sanitization, canonicalization, descriptors,
and structural alerts (PAINS/Brenk). Increment 1.

The agent reasons; this module computes. It returns raw computed values and
never fabricates them."""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, FilterCatalog

# The screener routinely detects invalid SMILES; silence RDKit's own stderr
# parse-error logging (MolFromSmiles still returns None, which we handle).
RDLogger.DisableLog("rdApp.error")


@dataclass
class SanitizeResult:
    ok: bool
    canonical_smiles: str | None = None
    error: str | None = None


def sanitize_and_canonicalize(smiles: str) -> SanitizeResult:
    """Sanitize a SMILES and return its RDKit canonical form. Never raises."""
    if not smiles or not smiles.strip():
        return SanitizeResult(ok=False, error="empty SMILES")
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return SanitizeResult(ok=False, error=f"could not parse SMILES {smiles!r}")
    try:
        canonical = Chem.MolToSmiles(mol)
    except Exception as exc:  # noqa: BLE001
        return SanitizeResult(ok=False, error=f"canonicalization failed: {exc}")
    return SanitizeResult(ok=True, canonical_smiles=canonical)


def compute_properties(mol: Chem.Mol) -> dict:
    """Computed molecular descriptors (rounded, deterministic)."""
    return {
        "molecular_weight": round(Descriptors.MolWt(mol), 2),
        "logp": round(Descriptors.MolLogP(mol), 2),
        "hbd": int(Descriptors.NumHDonors(mol)),
        "hba": int(Descriptors.NumHAcceptors(mol)),
        "tpsa": round(Descriptors.TPSA(mol), 2),
        "rotatable_bonds": int(Descriptors.NumRotatableBonds(mol)),
        "formal_charge": int(Chem.GetFormalCharge(mol)),
        "ring_count": int(Descriptors.RingCount(mol)),
    }


_catalogs: dict[str, FilterCatalog.FilterCatalog] = {}


def _catalog(which: str) -> FilterCatalog.FilterCatalog:
    if which not in _catalogs:
        params = FilterCatalog.FilterCatalogParams()
        catalog_const = getattr(FilterCatalog.FilterCatalogParams.FilterCatalogs, which.upper())
        params.AddCatalog(catalog_const)
        _catalogs[which] = FilterCatalog.FilterCatalog(params)
    return _catalogs[which]


def match_alerts(mol: Chem.Mol, which: str) -> list[str]:
    """Return alert descriptions (e.g. 'PAINS:...', 'BRENK:...') for catalog
    'pains' or 'brenk'. Empty list means no match."""
    catalog = _catalog(which)
    out: list[str] = []
    for m in catalog.GetMatches(mol):
        out.append(f"{which.upper()}:{m.GetDescription()}")
    return out
