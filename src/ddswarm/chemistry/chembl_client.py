"""On-demand ChEMBL literature retrieval: binding affinity + reference ligands.

Open, no-key ChEMBL REST (the same backend as the grounding client). ChEMBL is
EXCELLENT for binding affinity (Ki/Kd/IC50 — millions of records) but SPARSE for
PK/ADMET parameters. Every function returns None/[] on any failure — never a
fabricated value — and the caller records the outcome honestly.

This layer is network-dependent and non-deterministic across time, so it is
OPT-IN (like ``--grounding``): its results must not enter a reproducibility
hash unless the caller re-hashes the exact fetched values, and any downstream
score derived from them is non-reproducible by construction.

Functions
  search_target(gene)                -> ChEMBL target id (human, single protein)
  compound_affinities(ikey)          -> a molecule's binding profile: best
                                         Ki/Kd/IC50 per target, strongest first
  receptor_reference_ligands(target) -> (SMILES, Ki_molar, citation) list for
                                         fitting the Vina->Ki offset on-demand
"""

from __future__ import annotations

import json
import statistics
import time
import urllib.parse
import urllib.request

from rdkit import Chem
from rdkit.Chem.inchi import MolToInchiKey

TIMEOUT = 20.0
UA = {"User-Agent": "ddswarm/0.1 (research chembl)"}
CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"


def inchi_key(canonical_smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(canonical_smiles)
    return MolToInchiKey(mol) if mol is not None else None


def _num(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _get_json(url: str, retries: int = 2, timeout: float = TIMEOUT) -> dict | None:
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8", "replace"))
            return data if isinstance(data, dict) else None
        except Exception:
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    return None


def _resolve_molecule(ikey: str) -> dict | None:
    url = (f"{CHEMBL_BASE}/molecule.json?"
           f"molecule_structures__standard_inchi_key={urllib.parse.quote(ikey)}")
    data = _get_json(url)
    mols = (data or {}).get("molecules") or []
    if not mols:
        return None
    m = mols[0]
    return {"chembl_id": m.get("molecule_chembl_id"), "pref_name": m.get("pref_name")}


def search_target(gene: str) -> str | None:
    """Resolve a gene symbol to its human SINGLE PROTEIN ChEMBL target id.

    Returns the target_chembl_id (e.g. ``CHEMBL217`` for DRD2) or None when no
    human single-protein target matches. Best-effort: ChEMBL's gene search can
    rank orthologues first, so the Homo-sapiens/SINGLE-PROTEIN filter matters.
    """
    gene = (gene or "").strip()
    if not gene:
        return None
    data = _get_json(f"{CHEMBL_BASE}/target/search.json?q={urllib.parse.quote(gene)}&limit=20")
    targets = (data or {}).get("targets") or []
    human = [t for t in targets
             if t.get("organism") == "Homo sapiens"
             and t.get("target_type") == "SINGLE PROTEIN"]
    return human[0].get("target_chembl_id") if human else None


def compound_affinities(ikey: str, limit: int = 100) -> list[dict]:
    """A molecule's ChEMBL binding profile (best Ki/Kd/IC50 per target).

    Returns a list of dicts, strongest first, one entry per (target, type):
      {target_chembl_id, target_pref_name, standard_type, value_nm, relation,
       document_year}
    Only exact (``relation == "="``) nanomolar values are kept. Empty on any
    failure or when the molecule has no recorded binding activity.
    """
    mol = _resolve_molecule(ikey)
    if not mol:
        return []
    data = _get_json(
        f"{CHEMBL_BASE}/activity.json?molecule_chembl_id={mol['chembl_id']}"
        f"&standard_type__in=Ki,Kd,IC50&limit={limit}")
    activities = (data or {}).get("activities") or []

    best: dict[tuple[str, str], dict] = {}
    for a in activities:
        if a.get("standard_relation") != "=" or a.get("standard_units") != "nM":
            continue
        val = _num(a.get("standard_value"))
        if val is None or val <= 0:
            continue
        key = (a.get("target_chembl_id"), a.get("standard_type"))
        rec = {
            "target_chembl_id": a.get("target_chembl_id"),
            "target_pref_name": a.get("target_pref_name"),
            "standard_type": a.get("standard_type"),
            "value_nm": val,
            "relation": a.get("standard_relation"),
            "document_year": a.get("document_year"),
        }
        if key not in best or val < best[key]["value_nm"]:
            best[key] = rec
    out = sorted(best.values(), key=lambda r: r["value_nm"])
    return out


def receptor_reference_ligands(target_chembl_id: str, n: int = 5) -> list[dict]:
    """Reference ligands + measured Ki for a target, for on-demand Vina->Ki
    calibration.

    Aggregates the median Ki per molecule across assays (the same molecule can
    appear many times with different values) and returns the `n` strongest,
    sorted ascending. Each entry:
      {smiles, pref_name, ki_molar, n_assays, citation}
    Empty when the target has no exact, nanomolar Ki records with a parseable
    SMILES.
    """
    data = _get_json(
        f"{CHEMBL_BASE}/activity.json?target_chembl_id={target_chembl_id}"
        f"&standard_type=Ki&standard_units=nM&limit=500")
    activities = (data or {}).get("activities") or []

    by_mol: dict[str, list[float]] = {}
    meta: dict[str, dict] = {}
    for a in activities:
        if a.get("standard_relation") != "=":
            continue
        val = _num(a.get("standard_value"))
        smi = a.get("canonical_smiles")
        if val is None or val <= 0 or not smi:
            continue
        mid = a.get("molecule_chembl_id")
        if not mid:
            continue
        by_mol.setdefault(mid, []).append(val)
        meta.setdefault(mid, {
            "smiles": smi,
            "pref_name": a.get("molecule_pref_name"),
            "journal": a.get("document_journal"),
            "year": a.get("document_year"),
        })

    out: list[dict] = []
    for mid, vals in by_mol.items():
        ki_nm = statistics.median(vals)
        m = meta[mid]
        label = m["pref_name"] or "ChEMBL ligand"
        cite = f"{label} Ki={ki_nm:.1f} nM"
        if m["journal"]:
            cite += f" ({m['journal']}"
            if m["year"]:
                cite += f", {m['year']}"
            cite += ")"
        out.append({
            "smiles": m["smiles"],
            "ki_molar": ki_nm * 1e-9,
            "pref_name": m["pref_name"],
            "n_assays": len(vals),
            "citation": cite,
        })
    out.sort(key=lambda r: r["ki_molar"])
    return out[:n]
