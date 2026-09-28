"""Opt-in IUPAC naming via the NIH CACTUS web service (PubChem fallback).

IUPAC names are a HUMAN-READABILITY convenience only — they are never the
canonical identifier. Canonical SMILES + InChIKey remain the machine backbone
for every compute step (screening, docking, BRICS, analog generation, ADMET
descriptors). The services are network-dependent and third-party, so their
output is non-deterministic across time and MUST be marked as such in
provenance.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request

CACTUS_BASE = "https://cactus.nci.nih.gov/chemical/structure"
PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"


def _fetch_cactus_iupac(smiles: str, timeout: float) -> str | None:
    """De-novo systematic name from NIH CACTUS (generates, not a DB lookup)."""
    try:
        encoded = urllib.parse.quote(smiles, safe="")
        url = f"{CACTUS_BASE}/{encoded}/iupac_name"
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            name = resp.read().decode("utf-8", "replace").strip()
        return name or None
    except Exception:
        return None


def _fetch_pubchem_iupac(inchikey: str, timeout: float) -> str | None:
    """IUPAC name from PubChem for an ALREADY-INDEXED molecule (by InChIKey).

    Returns None for novel molecules absent from PubChem — this is a database
    lookup, not a de-novo name generator.
    """
    try:
        key = urllib.parse.quote(inchikey, safe="")
        url = f"{PUBCHEM_BASE}/inchikey/{key}/property/IUPACName/JSON"
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
        props = (data.get("PropertyTable") or {}).get("Properties") or []
        name = props[0].get("IUPACName") if props else None
        return (name or "").strip() or None
    except Exception:
        return None


def fetch_iupac_name(smiles: str, timeout: float = 15.0) -> str | None:
    """Return an IUPAC name for a SMILES, or None when no service can name it.

    Order: NIH CACTUS (de-novo generator) first, then PubChem by InChIKey (DB
    lookup for already-indexed molecules). Returns None on any error (network
    down, unknown molecule, malformed/empty response) — never a fabricated name.
    """
    name = _fetch_cactus_iupac(smiles, timeout)
    if name:
        return name
    try:
        from rdkit import Chem  # noqa: PLC0415
        mol = Chem.MolFromSmiles(smiles)
        ikey = Chem.MolToInchiKey(mol) if mol is not None else None
    except Exception:
        ikey = None
    if ikey:
        return _fetch_pubchem_iupac(ikey, timeout)
    return None
