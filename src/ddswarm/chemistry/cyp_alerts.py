"""Deterministic CYP450 drug-drug-interaction (DDI) flags.

Matches molecules against a curated, cited library of structural chemotypes for
CYP inhibition and CYP3A4 induction (PXR-mediated). Same honesty standard as the
tox toxicophore library: a SMARTS match is a deterministic FACT (the molecule
contains a documented chemotype), but the label is "potential inhibitor/inducer"
— a structural alert, NOT a functional enzyme assay.

The library is small and non-exhaustive. Induction is a weak 2D-structure signal
(PXR/CAR/AhR-driven, poorly captured by SMARTS), so induction alerts carry a
harder caveat. Enzymes covered by at least one clean chemotype: CYP3A4, 2C9,
2C19, 2D6, 1A2, 2A6, 2B6, 2C8, 2E1. Rifampicin (the strongest clinical PXR
inducer) and St. John's wort (hyperforin) are intentionally omitted — their
macrolide-ansamycin / phloroglucinol scaffolds have no clean 2D SMARTS.
"""

from __future__ import annotations

from typing import Any

from rdkit import Chem

# (enzyme, slug, SMARTS, citation). SMARTS verified against the named reference
# compound(s); each is specific to one documented class.
INHIBITION_ALERTS: list[tuple[str, str, str, str]] = [
    ("CYP3A4", "azole_imidazole", "n1ccnc1",
     "imidazole azole (clotrimazole/ketoconazole/miconazole) — CYP3A4 heme-iron coordination; Greenblatt & von Moltke"),
    ("CYP3A4", "azole_triazole", "c1ncnn1",
     "1,2,4-triazole azole (fluconazole/itraconazole/voriconazole) — CYP3A4 heme-iron "
     "coordination; Greenblatt & von Moltke"),
    ("CYP2C9", "aryl_sulfonamide", "S(=O)(=O)N",
     "aryl sulfonamide/sulfonylurea (sulfamethoxazole/celecoxib/sulfaphenazole) — CYP2C9; Rettie & Jones 2005"),
    ("CYP2D6", "aryloxy_alkylamine", "[cR1]O[CH2,CH1][CH2][CH2][N]",
     "aryloxy-alkylamine with basic N (fluoxetine/venlafaxine/duloxetine) — CYP2D6 pharmacophore; Deodhar et al."),
    ("CYP2C19", "benzimidazole", "c1ccc2[nH]cnc2c1",
     "benzimidazole proton-pump inhibitor (omeprazole/lansoprazole) — CYP2C19; Li et al. 2004"),
    ("CYP1A2", "purine_xanthine", "c1ncnc2ncnc12",
     "purine/methylxanthine (caffeine/theophylline) — CYP1A2 substrate/inhibitor; Kot & Daniel 2008"),
    ("CYP2A6", "coumarin", "O=c1ccc2ccccc2o1",
     "coumarin (probe substrate) / furanocoumarin methoxsalen mechanism-based inactivator — "
     "CYP2A6; Pelkonen et al. 2000"),
    ("CYP2B6", "thienopyridine", "c1cc2c(s1)CCNC2",
     "thienopyridine (clopidogrel/ticlopidine/prasugrel) mechanism-based inhibitor — CYP2B6; Richter et al. 2004"),
    ("CYP2C8", "diaminopyrimidine", "Nc1nc(N)ncc1",
     "2,4-diaminopyrimidine (trimethoprim) — CYP2C8 inhibitor; Wen et al. 2002"),
    ("CYP2E1", "pyrazole", "c1cn[nH]c1",
     "pyrazole (fomepizole / 4-methylpyrazole) — CYP2E1 inhibitor; Blomme et al."),
    ("CYP2D6", "methylenedioxybenzene", "C1Oc2ccccc2O1",
     "1,3-benzodioxole methylenedioxy (paroxetine/MDMA) — CYP2D6 mechanism-based inhibitor; Bertelsen et al. 2003"),
]

INDUCTION_ALERTS: list[tuple[str, str, str, str]] = [
    ("CYP3A4", "barbiturate", "O=C1NC(=O)NC(=O)C1",
     "barbiturate (phenobarbital) — PXR/CAR-mediated CYP3A4 induction; Lehmann et al. 1998"),
    ("CYP3A4", "hydantoin", "O=C1NC(=O)CN1",
     "hydantoin (phenytoin) — PXR/CAR-mediated CYP3A4 induction; Lehmann et al. 1998"),
    ("CYP3A4", "dibenzoazepine", "N1C(=O)C=Cc2ccccc2-c2ccccc21",
     "dibenzoazepine (carbamazepine) — PXR-mediated CYP3A4 induction; Bertilsson et al."),
    ("CYP3A4", "thiazolidinedione", "O=C1NC(=O)SC1",
     "thiazolidinedione (rosiglitazone/pioglitazone) — PXR-mediated CYP3A4 induction; Sahi et al. 2003"),
]


def _compile(alerts: list[tuple[str, str, str, str]]) -> list[tuple[str, str, Any, str]]:
    return [(enzyme, slug, Chem.MolFromSmarts(smarts), citation)
            for enzyme, slug, smarts, citation in alerts]


_INHIBITION = _compile(INHIBITION_ALERTS)
_INDUCTION = _compile(INDUCTION_ALERTS)


def assess_cyp(smiles: str | None) -> list[tuple[str, str, str, str]]:
    """Return matched DDI alerts as (enzyme, effect, slug, citation).

    effect ∈ {"inhibition", "induction"}. Empty list = no documented chemotype
    matched (not "clean"). None/invalid SMILES -> [].
    """
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return []
    hits: list[tuple[str, str, str, str]] = []
    for enzyme, slug, patt, citation in _INHIBITION:
        if mol.HasSubstructMatch(patt):
            hits.append((enzyme, "inhibition", slug, citation))
    for enzyme, slug, patt, citation in _INDUCTION:
        if mol.HasSubstructMatch(patt):
            hits.append((enzyme, "induction", slug, citation))
    return hits
