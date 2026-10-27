"""Grounding client: resolve a molecule against open public databases.

Open, no-key sources (verified reachable):
  - ChEMBL REST  (molecule record by standard InChIKey)
  - PubChem PUG-REST (compound by InChIKey -> CID, properties, RegistryID xrefs)
  - PubMed E-utilities (literature search by molecule name)

DrugBank's direct API is license-gated (HTTP 403). Its *identifier + link* is
instead resolved via PubChem RegistryID cross-references (e.g. "DB00472"); the
record *content* still requires a DrugBank license and is noted as such.

All calls are network-dependent and non-deterministic across time. Every failure
returns None/empty — never a fabricated value — and the caller records the
outcome honestly. PubChem rate-limits (HTTP 503); _get_json retries with backoff.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from rdkit import Chem
from rdkit.Chem.inchi import MolToInchiKey
from ddswarm import __version__

TIMEOUT = 20.0
UA = {"User-Agent": f"ddswarm/{__version__} (research grounding)"}
CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"
PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"
PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def inchi_key(canonical_smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(canonical_smiles)
    return MolToInchiKey(mol) if mol is not None else None


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


def chembl_molecule(ikey: str) -> dict | None:
    url = (f"{CHEMBL_BASE}/molecule.json?"
           f"molecule_structures__standard_inchi_key={urllib.parse.quote(ikey)}")
    data = _get_json(url)
    mols = (data or {}).get("molecules") or []
    if not mols:
        return None
    m = mols[0]
    return {
        "chembl_id": m.get("molecule_chembl_id"),
        "pref_name": m.get("pref_name"),
        "max_phase": m.get("max_phase"),
        "first_approval": m.get("first_approval"),
    }


def _drugbank_xref(cid: int) -> str | None:
    url = f"{PUBCHEM_BASE}/cid/{cid}/xrefs/RegistryID/JSON"
    data = _get_json(url)
    info = ((data or {}).get("InformationList") or {}).get("Information") or [{}]
    for rid in info[0].get("RegistryID", []) or []:
        if isinstance(rid, str) and rid.startswith("DB") and rid[2:].isdigit():
            return rid
    return None


def pubchem_compound(ikey: str) -> dict | None:
    url = (f"{PUBCHEM_BASE}/inchikey/{urllib.parse.quote(ikey)}"
           f"/property/IUPACName,MolecularFormula/JSON")
    data = _get_json(url)
    props = ((data or {}).get("PropertyTable") or {}).get("Properties") or []
    if not props:
        return None
    p = props[0]
    cid = p.get("CID")
    return {
        "cid": cid,
        "iupac": p.get("IUPACName"),
        "formula": p.get("MolecularFormula"),
        "drugbank_id": _drugbank_xref(cid) if cid else None,
    }


def pubmed_ids(query: str, retmax: int = 3) -> list[str] | None:
    url = (f"{PUBMED_BASE}/esearch.fcgi?db=pubmed&retmax={retmax}"
           f"&term={urllib.parse.quote(query)}")
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            text = resp.read().decode("utf-8", "replace")
    except Exception:
        return None
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return None
    if root.find(".//ERROR") is not None:
        raise RuntimeError("NCBI service unavailable (eSearch backend error)")
    ids = [e.text for e in root.findall(".//Id") if e.text]
    return ids or None


def ground_molecule(canonical_smiles: str | None, name: str | None = None) -> dict:
    """Ground one molecule against ChEMBL + PubChem (+ PubMed if a name is known).

    Returns: {inchi_key, entries: list[dict], status, errors: list[str]} where
    status is one of "grounded" | "not_found" | "failed".
    """
    if not canonical_smiles:
        return {"inchi_key": None, "entries": [], "status": "failed",
                "errors": ["empty SMILES"]}
    ikey = inchi_key(canonical_smiles)
    if ikey is None:
        return {"inchi_key": None, "entries": [], "status": "failed",
                "errors": ["could not compute InChIKey"]}

    entries: list[dict] = []
    errors: list[str] = []
    sources_ok = 0
    effective_name = name

    try:
        cb = chembl_molecule(ikey)
        sources_ok += 1
        if cb and cb.get("chembl_id"):
            cid_ = cb["chembl_id"]
            title = f"ChEMBL {cid_}"
            if cb.get("pref_name"):
                title += f" ({cb['pref_name']})"
                effective_name = effective_name or cb["pref_name"]
            entries.append({
                "source": "chembl", "external_id": cid_,
                "url": f"https://www.ebi.ac.uk/chembl/compound_report_card/{cid_}/",
                "title": title,
                "detail": f"max_phase={cb.get('max_phase')}, first_approval={cb.get('first_approval')}",
            })
        else:
            errors.append("chembl: no record for this InChIKey")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"chembl: {exc}")

    try:
        pc = pubchem_compound(ikey)
        sources_ok += 1
        if pc and pc.get("cid"):
            cid_ = pc["cid"]
            detail = f"formula={pc.get('formula')}"
            if pc.get("iupac"):
                detail += f", IUPAC={pc['iupac']}"
                effective_name = effective_name or pc["iupac"]
            entries.append({
                "source": "pubchem", "external_id": f"CID {cid_}",
                "url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid_}",
                "title": f"PubChem CID {cid_}", "detail": detail,
            })
            if pc.get("drugbank_id"):
                db = pc["drugbank_id"]
                entries.append({
                    "source": "drugbank", "external_id": db,
                    "url": f"https://go.drugbank.com/drugs/{db}",
                    "title": f"DrugBank {db}",
                    "detail": "identifier/link via PubChem cross-reference; record content requires a DrugBank license",
                })
        else:
            errors.append("pubchem: no record for this InChIKey")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"pubchem: {exc}")

    if effective_name:
        try:
            pmids = pubmed_ids(effective_name)
            sources_ok += 1
            if pmids:
                for pmid in pmids:
                    entries.append({
                        "source": "pubmed", "external_id": f"PMID {pmid}",
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        "title": f"PubMed {pmid}",
                        "detail": f"query='{effective_name}'",
                    })
            else:
                errors.append(f"pubmed: no results for '{effective_name}'")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"pubmed: {exc}")
    else:
        errors.append("pubmed: no molecule name available for query")

    if entries:
        status = "grounded"
    elif sources_ok == 0:
        status = "failed"
    else:
        status = "not_found"
    return {"inchi_key": ikey, "entries": entries, "status": status, "errors": errors}
