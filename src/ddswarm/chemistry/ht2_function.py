"""Deterministic functional-sign classification for 5-HT2A/5-HT2C ligands.

Docking reports BINDING, not function. This module resolves the FUNCTIONAL
SIGN (agonist vs antagonist) of a molecule that binds 5-HT2C/5-HT2A by
matching it against a curated, cited library of DOCUMENTED chemotypes.

Honesty contract (identical to the tox toxicophore library): a SMARTS match is
a deterministic fact — the molecule contains a documented chemotype — but the
label is "antagonist-like"/"agonist-like", a structural similarity to a known
class, NOT a functional-assay verdict. The library is deliberately small and
non-exhaustive; an "unknown" result means "no documented chemotype matched",
not "definitely neither".
"""

from __future__ import annotations

from typing import Any

from rdkit import Chem

# (slug, SMARTS, citation). SMARTS verified against the named reference
# compound(s); each is specific to one documented 5-HT2A/2C class.
ANTAGONIST_MOTIFS: list[tuple[str, str, str]] = [
    ("aroylpiperidine_5ht2a",
     "[cR1]C(=O)C1CCNCC1",
     "4-aroyl-piperidine class (ketanserin/ritanserin; 5-HT2A antagonist) — Leysen et al. 1981"),
    ("tetracyclic_dibenzo",
     "N1CCN2c3ccccc3Cc3ccccc3C2C1",
     "mianserin tetracyclic piperazino-azepine (5-HT2A/2C antagonist) — Nickolson & Wieringa 1981"),
    ("tetracyclic_benzopyrido",
     "N1CCN2c3ncccc3Cc3ccccc3C2C1",
     "mirtazapine tetracyclic piperazino-azepine (5-HT2A/2C antagonist) — de Boer 1996"),
    ("dibenzodiazepine",
     "C1=Nc2ccccc2Nc2ccccc21",
     "clozapine/olanzapine dibenzodiazepine class (5-HT2A/2C antagonist) — Meltzer 1994"),
]

AGONIST_MOTIFS: list[tuple[str, str, str]] = [
    ("tryptamine",
     "c1ccc2[nH]cc(CCN)c2c1",
     "tryptamine/indole-ethylamine core (5-HT2A/2C agonist chemotype) — Nichols 2004"),
    ("dimethoxyphenethylamine",
     "COc1cc(OC)ccc1CCN",
     "2,5-dimethoxy-phenethylamine core (DOI/mescaline; 5-HT2A/2C agonist) — Nichols 2004"),
]


def _compile(motifs: list[tuple[str, str, str]]) -> list[tuple[str, Any, str]]:
    return [(slug, Chem.MolFromSmarts(smarts), citation) for slug, smarts, citation in motifs]


_ANTAGONIST = _compile(ANTAGONIST_MOTIFS)
_AGONIST = _compile(AGONIST_MOTIFS)


def classify_ht2_function(smiles: str | None):
    """Classify the functional sign of a 5-HT2 ligand by documented chemotype.

    Returns (sign, matched_motifs, citations) where sign ∈ {"antagonist_like",
    "agonist_like", "unknown"} and matched_motifs/citations list every motif
    that matched (possibly empty). A molecule matching BOTH an antagonist and
    an agonist motif is reported "unknown" (ambiguous), with the matches listed.
    """
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return ("unknown", [], [])

    antag = [(slug, cit) for slug, patt, cit in _ANTAGONIST if mol.HasSubstructMatch(patt)]
    agon = [(slug, cit) for slug, patt, cit in _AGONIST if mol.HasSubstructMatch(patt)]

    if antag and not agon:
        return ("antagonist_like", [s for s, _ in antag], [c for _, c in antag])
    if agon and not antag:
        return ("agonist_like", [s for s, _ in agon], [c for _, c in agon])
    # both matched (ambiguous) or neither matched
    matched = antag + agon
    return ("unknown", [s for s, _ in matched], [c for _, c in matched])
