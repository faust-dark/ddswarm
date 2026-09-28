"""Deterministic functional-sign classification for GPCR/transporter ligands.

Docking reports BINDING, not function. This module resolves the FUNCTIONAL SIGN
(agonist vs antagonist) of a molecule that binds a receptor, by matching it
against a curated, cited library of DOCUMENTED chemotypes — the same honesty
contract as the tox/CYP libraries: a SMARTS match is a deterministic fact (the
molecule contains a documented chemotype), but the label is
"antagonist-like"/"agonist-like", a structural similarity to a known class, NOT
a functional-assay verdict. The library is small and non-exhaustive; "unknown"
means "no documented chemotype matched", not "definitely neither".

Receptors covered (canonical keys): "D2", "H1", "M1", "alpha1A", "mu_opioid",
"kappa_opioid". (5-HT2A/5-HT2C functional sign stays in ht2_function.py — the
SSRI-profile-specific library that predates this general one.)

Each SMARTS was derived from the named reference compound(s) and verified
empirically with RDKit against BOTH the class reference and a negative control
before shipping.
"""

from __future__ import annotations

from typing import Any

from rdkit import Chem

# receptor -> {"antagonist"|"agonist": [(slug, SMARTS, citation), ...]}
FUNCTIONAL_SIGN_LIBRARY: dict[str, dict[str, list[tuple[str, str, str]]]] = {
    "D2": {
        "antagonist": [
            ("butyrophenone",
             "[cR1]C(=O)CCCN",
             "butyrophenone (haloperidol/droperidol) — D2 antagonist; Seeman 1992"),
            ("phenothiazine",
             "c1ccc2Sc3ccccc3Nc2c1",
             "phenothiazine (chlorpromazine/fluphenazine) — D2 antagonist; Seeman 1992"),
            ("ortho_methoxybenzamide",
             "COc1ccccc1C(=O)N",
             "2-methoxybenzamide (sulpiride/amisulpride) — selective D2/D3 antagonist; Caley & Weber 1995"),
        ],
        "agonist": [
            ("aminotetralin",
             "C1Cc2ccccc2CC1N",
             "2-aminotetralin (5-OH-DPAT/7-OH-DPAT) — D2/D3 agonist; Seiler & Markstein 1984"),
        ],
    },
    "H1": {
        "antagonist": [
            ("diaryl_ether",
             "O(C(c1ccccc1)c1ccccc1)CCN",
             "benzhydryl-ether ethanolamine (diphenhydramine/doxylamine) — H1 antagonist; Hill et al. 1997"),
            ("phenothiazine",
             "c1ccc2Sc3ccccc3Nc2c1",
             "phenothiazine (promethazine) — H1 antagonist; Hill et al. 1997"),
            ("benzhydryl_piperazine",
             "C(c1ccccc1)(c1ccccc1)N1CCNCC1",
             "benzhydryl-piperazine (cyclizine/meclizine/cetirizine) — H1 antagonist; Hill et al. 1997"),
        ],
    },
    "M1": {
        "antagonist": [
            ("tropane_antimuscarinic",
             "CN1C2CCC1CC(C2)O",
             "tropane 3-oxyester/ether (atropine/scopolamine/benztropine) — muscarinic M1 antagonist; Wess 1996"),
        ],
    },
    "alpha1A": {
        "antagonist": [
            ("quinazoline",
             "c1ccc2c(c1)ncnc2",
             "4-amino-6,7-dimethoxyquinazoline (prazosin/terazosin/doxazosin) — alpha1 antagonist; Hieble & Ruffolo 1991"),
        ],
    },
    "mu_opioid": {
        "agonist": [
            ("phenylpiperidine",
             "c1ccc(cc1)C1CCNCC1",
             "4-phenylpiperidine (meperidine/fentanyl) — mu-opioid agonist; Casy & Parfitt 1986"),
            ("morphinan",
             "[CH3]N1CCC23C=CC4C1Cc1ccccc1-c1cccc(c12)OC43",
             "N-methyl morphinan (morphine/oxycodone/hydromorphone) — mu-opioid agonist; Casy & Parfitt 1986"),
        ],
        "antagonist": [
            ("n_allyl_morphinan",
             "C=CCN1CCC23C=CC4C1Cc1ccccc1-c1cccc(c12)OC43",
             "N-allyl morphinan (naloxone) — mu-opioid antagonist; Casy & Parfitt 1986"),
            ("n_cyclopropylmethyl_morphinan",
             "C1CC1CN1CCC23C=CC4C1Cc1ccccc1-c1cccc(c12)OC43",
             "N-cyclopropylmethyl morphinan (naltrexone/nalmefene) — mu-opioid antagonist; Casy & Parfitt 1986"),
        ],
    },
    "kappa_opioid": {
        "agonist": [
            ("arylacetamide",
             "O=C(Cc1ccc(Cl)cc1)N(C)C1CCCCC1",
             "2-(3,4-dichlorophenyl)acetamide (U-50488/U-69593) — kappa-opioid agonist; Szmuszkovicz 1999"),
        ],
    },
}


def _compile(motifs: list[tuple[str, str, str]]) -> list[tuple[str, Any, str]]:
    return [(slug, Chem.MolFromSmarts(smarts), citation) for slug, smarts, citation in motifs]


_COMPILED: dict[str, dict[str, list[tuple[str, Any, str]]]] = {
    receptor: {sign: _compile(motifs) for sign, motifs in signs.items()}
    for receptor, signs in FUNCTIONAL_SIGN_LIBRARY.items()
}


def classify_function(smiles: str | None, receptor: str):
    """Classify the functional sign of a ligand for one receptor by chemotype.

    Returns (sign, matched_motifs, citations) where sign ∈ {"antagonist_like",
    "agonist_like", "unknown"}. An unknown receptor key returns ("unknown", [],
    []). A molecule matching BOTH an antagonist and an agonist motif is
    "unknown" (ambiguous), with the matches listed.
    """
    signs = _COMPILED.get(receptor)
    if signs is None:
        return ("unknown", [], [])
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return ("unknown", [], [])

    antag = [(slug, cit) for slug, patt, cit in signs.get("antagonist", [])
             if mol.HasSubstructMatch(patt)]
    agon = [(slug, cit) for slug, patt, cit in signs.get("agonist", [])
            if mol.HasSubstructMatch(patt)]

    if antag and not agon:
        return ("antagonist_like", [s for s, _ in antag], [c for _, c in antag])
    if agon and not antag:
        return ("agonist_like", [s for s, _ in agon], [c for _, c in agon])
    matched = antag + agon
    return ("unknown", [s for s, _ in matched], [c for _, c in matched])
