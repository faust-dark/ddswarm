"""Curated registry of dockable drug targets (data, not code paths).

This is a convenience lookup so a selectivity panel can name a target
(e.g. ``{"name": "DRD2"}``) and have its PDB id filled in. It is NOT a
whitelist: the selectivity_minion docks any receptor given an explicit
``pdb_id`` / ``identifier``, whether or not it appears here.

Every ``pdb_id`` below was verified against the RCSB PDB data API
(entry exists, organism / ligand / resolution checked) — no id is invented.
Entries with ``pdb_id: None`` are real targets whose reference structure I have
not yet curated — supply ``pdb_id`` explicitly for those. Every entry's
structure must be reviewed for species / conformational state / co-crystallised
ligand before its affinities are interpreted (the docking engine already runs
structural QC and extracts any co-crystal ligand for redocking validation).

Receptor names are matched case-insensitively and alias-resolved via
``canonical_receptor_name`` (``PTGS2`` -> ``COX2``, ``M1`` -> ``CHRM1``,
``5-HT2C`` -> ``HTR2C``, ``mu`` -> ``OPRM1``, etc.) so a panel can name a
target several ways and still resolve to the registry entry.
"""

from __future__ import annotations

# name -> {name, gene, family, pdb_id, reference}
RECEPTOR_REGISTRY: dict[str, dict] = {
    # --- serotonin: transporter + receptors ---
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
    "HTR1B": {
        "name": "HTR1B", "gene": "HTR1B", "family": "5-HT1B receptor (GPCR)",
        "pdb_id": "4IAQ",
        "reference": "Wang et al. 2013 (human 5-HT1B, dihydroergotamine-bound)",
    },
    "HTR1D": {
        "name": "HTR1D", "gene": "HTR1D", "family": "5-HT1D receptor (GPCR)",
        "pdb_id": "7E32",
        "reference": "Xu et al. 2021 (human 5-HT1D-Gi)",
    },
    "HTR2A": {
        "name": "HTR2A", "gene": "HTR2A", "family": "5-HT2A receptor (GPCR)",
        "pdb_id": "6A93",
        "reference": "Kimura et al. 2019 (human 5-HT2A, zotepine-bound)",
    },
    "HTR2B": {
        "name": "HTR2B", "gene": "HTR2B", "family": "5-HT2B receptor (GPCR)",
        "pdb_id": "4IB4",
        "reference": "Wacker et al. 2013 (human 5-HT2B, ergotamine-bound)",
    },
    "HTR2C": {
        "name": "HTR2C", "gene": "HTR2C", "family": "5-HT2C receptor (GPCR)",
        "pdb_id": "6BQH",
        "reference": "Peng et al. 2018 (human 5-HT2C, ritanserin-bound)",
    },
    "HTR4": {
        "name": "HTR4", "gene": "HTR4", "family": "5-HT4 receptor (GPCR)",
        "pdb_id": "7XT9",
        "reference": "Huang et al. 2022 (human 5-HT4-Gs)",
    },
    "HTR6": {
        "name": "HTR6", "gene": "HTR6", "family": "5-HT6 receptor (GPCR)",
        "pdb_id": "7YS6",
        "reference": "He et al. 2023 (human 5-HT6)",
    },
    "HTR7": {
        "name": "HTR7", "gene": "HTR7", "family": "5-HT7 receptor (GPCR)",
        "pdb_id": "7XTC",
        "reference": "Huang et al. 2022 (human 5-HT7-Gs)",
    },

    # --- dopamine receptors ---
    "DRD1": {
        "name": "DRD1", "gene": "DRD1", "family": "dopamine D1 receptor (GPCR)",
        "pdb_id": "7CKW",
        "reference": "Xiao et al. 2021 (human D1, fenoldopam-bound)",
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
    "DRD4": {
        "name": "DRD4", "gene": "DRD4", "family": "dopamine D4 receptor (GPCR)",
        "pdb_id": "5WIU",
        "reference": "Wang et al. 2017 (human D4, nemonapride-bound)",
    },

    # --- adrenergic receptors ---
    "ADRA1A": {
        "name": "ADRA1A", "gene": "ADRA1A", "family": "alpha-1A adrenergic receptor (GPCR)",
        "pdb_id": "9M4Q",
        "reference": "Liu et al. 2025 (human alpha-1A, doxazosin-bound)",
    },
    "ADRA1B": {
        "name": "ADRA1B", "gene": "ADRA1B", "family": "alpha-1B adrenergic receptor (GPCR)",
        "pdb_id": "7B6W",
        "reference": "Deluigi et al. 2022 (human alpha-1B)",
    },
    "ADRA2A": {
        "name": "ADRA2A", "gene": "ADRA2A", "family": "alpha-2A adrenergic receptor (GPCR)",
        "pdb_id": "9CBL",
        "reference": "Lou et al. 2024 (human alpha-2A, epinephrine-bound)",
    },
    "ADRA2B": {
        "name": "ADRA2B", "gene": "ADRA2B", "family": "alpha-2B adrenergic receptor (GPCR)",
        "pdb_id": "6K41",
        "reference": "Yuan et al. 2020 (human alpha-2B-GoA)",
    },
    "ADRA2C": {
        "name": "ADRA2C", "gene": "ADRA2C", "family": "alpha-2C adrenergic receptor (GPCR)",
        "pdb_id": None,
        "reference": "no confident reference structure curated yet — supply pdb_id",
    },
    "ADRB1": {
        "name": "ADRB1", "gene": "ADRB1", "family": "beta-1 adrenergic receptor (GPCR)",
        "pdb_id": "2VT4",
        "reference": "Warne et al. 2008 (turkey beta-1, cyanopindolol-bound — canonical beta-1 docking template)",
    },
    "ADRB2": {
        "name": "ADRB2", "gene": "ADRB2", "family": "beta-2 adrenergic receptor (GPCR)",
        "pdb_id": "2RH1",
        "reference": "Cherezov et al. 2007 (human beta-2, carazolol-bound)",
    },
    "ADRB3": {
        "name": "ADRB3", "gene": "ADRB3", "family": "beta-3 adrenergic receptor (GPCR)",
        "pdb_id": "9IJE",
        "reference": "Zheng et al. 2024 (human beta-3, epinephrine-activated)",
    },

    # --- muscarinic acetylcholine receptors ---
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
    "CHRM3": {
        "name": "CHRM3", "gene": "CHRM3", "family": "muscarinic M3 receptor (GPCR)",
        "pdb_id": "4U15",
        "reference": "Thorsen et al. 2014 (human M3, tiotropium-bound)",
    },
    "CHRM4": {
        "name": "CHRM4", "gene": "CHRM4", "family": "muscarinic M4 receptor (GPCR)",
        "pdb_id": "5DSG",
        "reference": "Thal et al. 2016 (human M4, tiotropium-bound)",
    },
    "CHRM5": {
        "name": "CHRM5", "gene": "CHRM5", "family": "muscarinic M5 receptor (GPCR)",
        "pdb_id": "6OL9",
        "reference": "Vuckovic et al. 2019 (human M5, tiotropium-bound)",
    },

    # --- histamine receptors ---
    "HRH1": {
        "name": "HRH1", "gene": "HRH1", "family": "histamine H1 receptor (GPCR)",
        "pdb_id": "3RZE",
        "reference": "Shimamura et al. 2011 (human H1, doxepin-bound)",
    },
    "HRH2": {
        "name": "HRH2", "gene": "HRH2", "family": "histamine H2 receptor (GPCR)",
        "pdb_id": "7UL3",
        "reference": "Robertson et al. 2022 (human H2, famotidine-bound)",
    },
    "HRH3": {
        "name": "HRH3", "gene": "HRH3", "family": "histamine H3 receptor (GPCR)",
        "pdb_id": "7F61",
        "reference": "Peng et al. 2022 (human H3)",
    },
    "HRH4": {
        "name": "HRH4", "gene": "HRH4", "family": "histamine H4 receptor (GPCR)",
        "pdb_id": "9LRC",
        "reference": "Matsuzaki et al. 2025 (human H4-Gi)",
    },

    # --- opioid receptors ---
    "OPRM1": {
        "name": "OPRM1", "gene": "OPRM1", "family": "mu-opioid receptor (GPCR)",
        "pdb_id": "4DKL",
        "reference": "Manglik et al. 2012 (human mu-opioid, beta-FNA-bound)",
    },
    "OPRD1": {
        "name": "OPRD1", "gene": "OPRD1", "family": "delta-opioid receptor (GPCR)",
        "pdb_id": "4N6H",
        "reference": "Fenalti et al. 2014 (human delta-opioid, naltrindole-bound)",
    },
    "OPRK1": {
        "name": "OPRK1", "gene": "OPRK1", "family": "kappa-opioid receptor (GPCR)",
        "pdb_id": "4DJH",
        "reference": "Wu et al. 2012 (human kappa-opioid, JDTic-bound)",
    },
    "OPRL1": {
        "name": "OPRL1", "gene": "OPRL1", "family": "nociceptin/orphanin FQ receptor (GPCR)",
        "pdb_id": "4EA3",
        "reference": "Thompson et al. 2012 (human NOP, peptide antagonist)",
    },

    # --- cannabinoid receptors ---
    "CNR1": {
        "name": "CNR1", "gene": "CNR1", "family": "cannabinoid CB1 receptor (GPCR)",
        "pdb_id": "5U09",
        "reference": "Shao et al. 2016 (human CB1, AM6538-bound)",
    },
    "CNR2": {
        "name": "CNR2", "gene": "CNR2", "family": "cannabinoid CB2 receptor (GPCR)",
        "pdb_id": "6KPC",
        "reference": "Hua et al. 2020 (human CB2, agonist-bound)",
    },

    # --- adenosine receptors ---
    "ADORA1": {
        "name": "ADORA1", "gene": "ADORA1", "family": "adenosine A1 receptor (GPCR)",
        "pdb_id": "5UEN",
        "reference": "Glukhova et al. 2017 (human A1)",
    },
    "ADORA2A": {
        "name": "ADORA2A", "gene": "ADORA2A", "family": "adenosine A2A receptor (GPCR)",
        "pdb_id": "4EIY",
        "reference": "Liu et al. 2012 (human A2A, ZM241385-bound)",
    },
    "ADORA2B": {
        "name": "ADORA2B", "gene": "ADORA2B", "family": "adenosine A2B receptor (GPCR)",
        "pdb_id": "7XY6",
        "reference": "Chen et al. 2022 (human A2B, agonist-bound)",
    },
    "ADORA3": {
        "name": "ADORA3", "gene": "ADORA3", "family": "adenosine A3 receptor (GPCR)",
        "pdb_id": "9EHS",
        "reference": "Zhang et al. 2025 (human A3)",
    },

    # --- melatonin receptors ---
    "MTNR1A": {
        "name": "MTNR1A", "gene": "MTNR1A", "family": "melatonin MT1 receptor (GPCR)",
        "pdb_id": "6ME2",
        "reference": "Stauch et al. 2019 (human MT1)",
    },
    "MTNR1B": {
        "name": "MTNR1B", "gene": "MTNR1B", "family": "melatonin MT2 receptor (GPCR)",
        "pdb_id": "6ME6",
        "reference": "Johansson et al. 2019 (human MT2)",
    },

    # --- other GPCRs ---
    "GLP1R": {
        "name": "GLP1R", "gene": "GLP1R", "family": "glucagon-like peptide-1 receptor (GPCR)",
        "pdb_id": "7LCJ",
        "reference": "Zhang et al. 2021 (human GLP-1R, PF-06882961-bound)",
    },
    "CCR5": {
        "name": "CCR5", "gene": "CCR5", "family": "CC chemokine receptor 5 (GPCR)",
        "pdb_id": "4MBS",
        "reference": "Tan et al. 2013 (human CCR5, maraviroc-bound)",
    },
    "MC4R": {
        "name": "MC4R", "gene": "MC4R", "family": "melanocortin-4 receptor (GPCR)",
        "pdb_id": "6W25",
        "reference": "Yu et al. 2020 (human MC4R, SHU9119-bound)",
    },

    # --- monoamine transporters ---
    "NET": {
        "name": "NET", "gene": "SLC6A2", "family": "noradrenaline transporter (SLC6)",
        "pdb_id": "8ZOY",
        "reference": "Song et al. 2024 (human NET, norepinephrine-bound)",
    },
    "DAT": {
        "name": "DAT", "gene": "SLC6A3", "family": "dopamine transporter (SLC6)",
        "pdb_id": "8Y2D",
        "reference": "Li et al. 2024 (human DAT, dopamine-bound)",
    },

    # --- enzymes ---
    "COX1": {
        "name": "COX1", "gene": "PTGS1", "family": "cyclooxygenase-1 (prostaglandin synthase)",
        "pdb_id": "6Y3C",
        "reference": "Miciaccia et al. 2021 (human COX-1)",
    },
    "COX2": {
        "name": "COX2", "gene": "PTGS2", "family": "cyclooxygenase-2 (prostaglandin synthase)",
        "pdb_id": "3LN1",
        "reference": "Wang et al. 2010 (human COX-2, celecoxib-bound)",
    },
    "PDE5A": {
        "name": "PDE5A", "gene": "PDE5A", "family": "phosphodiesterase-5 (enzyme)",
        "pdb_id": "1UDT",
        "reference": "Sung et al. 2003 (human PDE5, sildenafil-bound)",
    },
    "ACE": {
        "name": "ACE", "gene": "ACE", "family": "angiotensin-converting enzyme (enzyme)",
        "pdb_id": "1O86",
        "reference": "Natesh et al. 2003 (human ACE, lisinopril-bound)",
    },
    "HMGCR": {
        "name": "HMGCR", "gene": "HMGCR", "family": "HMG-CoA reductase (enzyme)",
        "pdb_id": "1DQ9",
        "reference": "Istvan et al. 2000 (human HMG-CoA reductase, atorvastatin-bound)",
    },
    "MAOA": {
        "name": "MAOA", "gene": "MAOA", "family": "monoamine oxidase A (enzyme)",
        "pdb_id": "2Z5X",
        "reference": "Son et al. 2008 (human MAO-A, harmine-bound)",
    },
    "MAOB": {
        "name": "MAOB", "gene": "MAOB", "family": "monoamine oxidase B (enzyme)",
        "pdb_id": "2V5Z",
        "reference": "Binda et al. 2007 (human MAO-B, safinamide-bound)",
    },
    "ACHE": {
        "name": "ACHE", "gene": "ACHE", "family": "acetylcholinesterase (enzyme)",
        "pdb_id": "4EY7",
        "reference": "Cheung et al. 2012 (human AChE, donepezil-bound)",
    },
    "DPP4": {
        "name": "DPP4", "gene": "DPP4", "family": "dipeptidyl peptidase-4 (enzyme)",
        "pdb_id": "1X70",
        "reference": "Kim et al. 2005 (human DPP4, inhibitor-bound)",
    },

    # --- ion channels ---
    "KCNH2": {
        "name": "KCNH2", "gene": "KCNH2", "family": "hERG (Kv11.1) voltage-gated potassium channel",
        "pdb_id": "7CN1",
        "reference": "Asai et al. 2021 (human hERG, K+-bound)",
    },
    "SCN5A": {
        "name": "SCN5A", "gene": "SCN5A", "family": "voltage-gated sodium channel Nav1.5",
        "pdb_id": "9V3S",
        "reference": "Liu et al. 2026 (human Nav1.5, quinidine-bound)",
    },
    "CACNA1C": {
        "name": "CACNA1C", "gene": "CACNA1C", "family": "voltage-gated calcium channel Cav1.2",
        "pdb_id": "8WEA",
        "reference": "Gao et al. 2023 (human Cav1.2, pinaverium-bound)",
    },
    "CHRNA4B2": {
        "name": "nAChR_a4b2", "gene": "CHRNA4/CHRNB2", "family": "nicotinic alpha4beta2 receptor (LGIC)",
        "pdb_id": None,
        "reference": "no single confident reference structure curated yet — supply pdb_id",
    },
}

# Greek-letter transliterations so alpha/beta/mu/delta/kappa spellings resolve.
_GREEK = {"Α": "ALPHA", "Β": "BETA", "Μ": "MU", "Δ": "DELTA", "Κ": "KAPPA"}

# Explicit alias table: normalized alias -> canonical registry key. The
# registry's own keys, names and genes are resolved automatically below; this
# table adds the common shorthand names (M1, CB1, 5-HT2C, mu, H1, ...).
_RECEPTOR_ALIASES: dict[str, str] = {
    # serotonin
    "5HT1A": "HTR1A", "5HT1B": "HTR1B", "5HT1D": "HTR1D",
    "5HT2A": "HTR2A", "5HT2B": "HTR2B", "5HT2C": "HTR2C",
    "5HT4": "HTR4", "5HT6": "HTR6", "5HT7": "HTR7",
    # dopamine
    "D1": "DRD1", "D2": "DRD2", "D3": "DRD3", "D4": "DRD4", "D5": "DRD5",
    # muscarinic
    "M1": "CHRM1", "M2": "CHRM2", "M3": "CHRM3", "M4": "CHRM4", "M5": "CHRM5",
    # adrenergic (note A2A == adenosine, not alpha-2A: adrenergic uses ALPHA/BETA)
    "ALPHA1A": "ADRA1A", "ALPHA1B": "ADRA1B",
    "ALPHA2A": "ADRA2A", "ALPHA2B": "ADRA2B", "ALPHA2C": "ADRA2C",
    "BETA1": "ADRB1", "BETA2": "ADRB2", "BETA3": "ADRB3",
    # histamine
    "H1": "HRH1", "H2": "HRH2", "H3": "HRH3", "H4": "HRH4",
    # opioid
    "MU": "OPRM1", "MOR": "OPRM1",
    "DELTA": "OPRD1", "DOR": "OPRD1",
    "KAPPA": "OPRK1", "KOR": "OPRK1",
    "NOP": "OPRL1", "ORL1": "OPRL1",
    # cannabinoid
    "CB1": "CNR1", "CB2": "CNR2",
    # adenosine
    "A1": "ADORA1", "A2A": "ADORA2A", "A2B": "ADORA2B", "A3": "ADORA3",
    # COX
    "PTGS1": "COX1", "PTGS2": "COX2",
    # channels
    "HERG": "KCNH2", "KV111": "KCNH2",
    "NAV15": "SCN5A", "CAV12": "CACNA1C",
    # melatonin
    "MT1": "MTNR1A", "MT2": "MTNR1B",
}


def _normalize(name: str) -> str:
    """Uppercase, transliterate Greek, drop non-alphanumerics."""
    s = str(name).strip().upper()
    for greek, latin in _GREEK.items():
        s = s.replace(greek, latin)
    return "".join(ch for ch in s if ch.isalnum())


# Build the full alias map once: explicit aliases + every registry key, display
# name and gene symbol (normalized) resolving to its canonical key.
_ALIAS_MAP: dict[str, str] = dict(_RECEPTOR_ALIASES)
for _key, _entry in RECEPTOR_REGISTRY.items():
    for _token in (_key, _entry.get("name", ""), _entry.get("gene", "")):
        if _token:
            _ALIAS_MAP.setdefault(_normalize(_token), _key)


def canonical_receptor_name(name: str) -> str:
    """Resolve a receptor name/alias to the canonical registry key.

    Case-insensitive; tolerant of ``5-HT2C``/``5HT2C``, ``alpha1A``/``alpha-1A``,
    ``mu``/``MOR``, ``PTGS2``/``COX2``, ``M1``/``CHRM1``, etc. Unknown names are
    returned normalized (uppercase, punctuation-stripped) so matching never
    fabricates a mapping — it just won't match a registry key.
    """
    return _ALIAS_MAP.get(_normalize(name), _normalize(name))


def expand_receptor(config: dict) -> dict:
    """Return a receptor config with its pdb_id filled from the registry.

    Resolves the name through ``canonical_receptor_name`` (so ``PTGS2`` fills
    ``COX2``'s pdb_id, etc.) and only fills a missing pdb_id/identifier/url;
    never overrides an explicit value the caller supplied.
    """
    out = dict(config)
    key = canonical_receptor_name(str(out.get("name", "")))
    entry = RECEPTOR_REGISTRY.get(key)
    if entry and entry.get("pdb_id") and not out.get("pdb_id") and not out.get("identifier"):
        out["pdb_id"] = entry["pdb_id"]
    return out


def registry_entry(name: str) -> dict | None:
    """Return the registry entry for a canonical name (alias-resolved)."""
    return RECEPTOR_REGISTRY.get(canonical_receptor_name(name))
