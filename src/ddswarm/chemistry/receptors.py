"""Curated registry of dockable drug targets (data, not code paths).

This is a convenience lookup so a selectivity panel can name a target
(e.g. ``{"name": "DRD2"}``) and have its PDB id filled in. It is NOT a
whitelist: the selectivity_minion docks any receptor given an explicit
``pdb_id`` / ``identifier``, whether or not it appears here.

Honesty contract: a PDB id is only recorded where I can cite the structure;
entries with ``pdb_id: None`` are real targets whose reference structure I have
not yet curated — supply ``pdb_id`` explicitly for those. Every entry's
structure must be reviewed for species / conformational state / co-crystallised
ligand before its affinities are interpreted (the docking engine already runs
structural QC and extracts any co-crystal ligand for redocking validation).
"""

from __future__ import annotations

# name -> {name, gene, family, pdb_id, reference}
RECEPTOR_REGISTRY: dict[str, dict] = {
    "SERT": {
        "name": "SERT", "gene": "SLC6A4", "family": "serotonin transporter (SLC6)",
        "pdb_id": "5I73",
        "reference": "Coleman et al. 2016 (human SERT, paroxetine-bound)",
    },
    "HTR1A": {
        "name": "HTR1A", "gene": "HTR1A", "family": "5-HT1A receptor (GPCR)",
        "pdb_id": "7E2Y",
        "reference": "Xu et al. 2021 (human 5-HT1A, aripiprazole-bound)",
    },
    "HTR2A": {
        "name": "HTR2A", "gene": "HTR2A", "family": "5-HT2A receptor (GPCR)",
        "pdb_id": "6A93",
        "reference": "Kimura et al. 2019 (human 5-HT2A, zotepine-bound)",
    },
    "HTR2C": {
        "name": "HTR2C", "gene": "HTR2C", "family": "5-HT2C receptor (GPCR)",
        "pdb_id": "6BQH",
        "reference": "Peng et al. 2018 (human 5-HT2C, ritanserin-bound)",
    },
    "DRD2": {
        "name": "DRD2", "gene": "DRD2", "family": "dopamine D2 receptor (GPCR)",
        "pdb_id": "6CM4",
        "reference": "Wang et al. 2018 (human D2, risperidone-bound)",
    },
    "DRD3": {
        "name": "DRD3", "gene": "DRD3", "family": "dopamine D3 receptor (GPCR)",
        "pdb_id": "3PBL",
        "reference": "Chien et al. 2010 (human D3, eticlopride-bound)",
    },
    "CHRM1": {
        "name": "CHRM1", "gene": "CHRM1", "family": "muscarinic M1 receptor (GPCR)",
        "pdb_id": "5CXV",
        "reference": "Thal et al. 2016 (human M1, tiotropium-bound)",
    },
    "CHRM2": {
        "name": "CHRM2", "gene": "CHRM2", "family": "muscarinic M2 receptor (GPCR)",
        "pdb_id": "3UON",
        "reference": "Haga et al. 2012 (human M2, QNB-bound)",
    },
    "CHRM4": {
        "name": "CHRM4", "gene": "CHRM4", "family": "muscarinic M4 receptor (GPCR)",
        "pdb_id": "5DSG",
        "reference": "Thal et al. 2016 (human M4, tiotropium-bound)",
    },
    "NET": {
        "name": "NET", "gene": "SLC6A2", "family": "noradrenaline transporter (SLC6)",
        "pdb_id": None,
        "reference": "no single confident reference structure curated yet — supply pdb_id",
    },
    "HTR7": {
        "name": "HTR7", "gene": "HTR7", "family": "5-HT7 receptor (GPCR)",
        "pdb_id": None,
        "reference": "no single confident reference structure curated yet — supply pdb_id",
    },
    "PTGS1": {
        "name": "COX1", "gene": "PTGS1", "family": "cyclooxygenase-1 (prostaglandin synthase)",
        "pdb_id": None,
        "reference": "no single confident reference structure curated yet — supply pdb_id",
    },
    "PTGS2": {
        "name": "COX2", "gene": "PTGS2", "family": "cyclooxygenase-2 (prostaglandin synthase)",
        "pdb_id": None,
        "reference": "no single confident reference structure curated yet — supply pdb_id",
    },
    "CHRNA4B2": {
        "name": "nAChR_a4b2", "gene": "CHRNA4/CHRNB2", "family": "nicotinic α4β2 receptor (LGIC)",
        "pdb_id": None,
        "reference": "no single confident reference structure curated yet — supply pdb_id",
    },
}


def expand_receptor(config: dict) -> dict:
    """Return a receptor config with its pdb_id filled from the registry.

    Only fills a missing pdb_id/identifier/url when ``name`` is a registry key;
    never overrides an explicit value the caller supplied.
    """
    out = dict(config)
    entry = RECEPTOR_REGISTRY.get(str(out.get("name", "")))
    if entry and entry.get("pdb_id") and not out.get("pdb_id") and not out.get("identifier"):
        out["pdb_id"] = entry["pdb_id"]
    return out


def registry_entry(name: str) -> dict | None:
    return RECEPTOR_REGISTRY.get(name)
