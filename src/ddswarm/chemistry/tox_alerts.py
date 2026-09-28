"""Curated deterministic toxicophore structural alerts.

A sourced set (non-exhaustive) of literature structural-alert rules: each is a
substructure historically associated with a toxicity endpoint. A match is a
*predicted* potential liability, not experimental evidence — the agent reports
it as such.

SMARTS validated against: nitrobenzene, dimethylnitrosamine, oxirane, aziridine,
acrolein, chloroethane, 1,4-benzoquinone, aniline, azobenzene, ethyl azide,
isoniazid, methyl isocyanate, allyl isothiocyanate, acetyl chloride, phenacyl
bromide, methyl mesylate, nitrosobenzene, vinyl chloride, beta-propiolactone,
catechol, glyoxal; gefitinib matches none.
"""

from __future__ import annotations

from rdkit import Chem

TOX_ALERTS: list[dict] = [
    {
        "name": "nitro group",
        "smarts": "[N+](=O)[O-]",
        "endpoint": "mutagenicity (potential)",
        "source": "Ashby & Tennant (1988) nitro mutagenicity structural alert",
    },
    {
        "name": "N-nitrosamine",
        "smarts": "[#7][#7]=[#8]",
        "endpoint": "carcinogenicity (potential)",
        "source": "ICH M7 N-nitrosamine structural class",
    },
    {
        "name": "epoxide",
        "smarts": "C1OC1",
        "endpoint": "mutagenicity (potential, alkylating)",
        "source": "epoxide DNA-alkylating structural alert",
    },
    {
        "name": "aziridine",
        "smarts": "C1NC1",
        "endpoint": "mutagenicity (potential, alkylating)",
        "source": "aziridine DNA-alkylating structural alert",
    },
    {
        "name": "Michael acceptor",
        "smarts": "[CX3]=[CX3][CX3]=[OX1]",
        "endpoint": "reactivity (protein alkylation)",
        "source": "alpha,beta-unsaturated carbonyl reactivity alert (broad)",
    },
    {
        "name": "alkyl halide",
        "smarts": "[Cl,Br,I][CX4;H1,H2]",
        "endpoint": "mutagenicity (potential, alkylating)",
        "source": "primary/secondary alkyl halide alkylating alert",
    },
    {
        "name": "quinone",
        "smarts": "C1(=O)C=CC(=O)C=C1",
        "endpoint": "oxidative stress (redox cycling)",
        "source": "1,4-quinone redox-cycling structural alert",
    },
    {
        "name": "aromatic primary amine (aniline)",
        "smarts": "[NX3;H2][c]",
        "endpoint": "mutagenicity (potential, nitrenium via metabolic activation)",
        "source": "Benigni & Bossa (2008) aromatic amine mutagenicity",
    },
    {
        "name": "aromatic azo",
        "smarts": "[#6][#7]=[#7][#6]",
        "endpoint": "carcinogenicity (potential, reductive cleavage)",
        "source": "Benigni & Bossa (2008) azo-dye carcinogenicity",
    },
    {
        "name": "azide",
        "smarts": "[#7]=[#7+]=[#7-]",
        "endpoint": "reactivity (potential, nitrogen release)",
        "source": "organic azide reactive-toxicophore alert",
    },
    {
        "name": "hydrazine / hydrazide",
        "smarts": "[#7][#7]",
        "endpoint": "hepatotoxicity (potential)",
        "source": "hydrazine/hydrazide toxicophore (isoniazid); Nelson et al.",
    },
    {
        "name": "isocyanate",
        "smarts": "[#7X2]=[#6X2]=[#8X1]",
        "endpoint": "respiratory sensitization / protein reactivity",
        "source": "isocyanate electrophile reactive-toxicophore",
    },
    {
        "name": "isothiocyanate",
        "smarts": "[#7X2]=[#6X2]=[#16X1]",
        "endpoint": "protein reactivity (potential)",
        "source": "isothiocyanate electrophile reactive-toxicophore",
    },
    {
        "name": "acyl halide",
        "smarts": "[#6X3](=[#8X1])[F,Cl,Br,I]",
        "endpoint": "reactivity (acylation)",
        "source": "acyl halide electrophile reactive-toxicophore",
    },
    {
        "name": "alpha,beta-unsaturated aldehyde",
        "smarts": "[#6X3]=[#6X3][#6X3H1]=[#8X1]",
        "endpoint": "mutagenicity / reactivity (potential)",
        "source": "acrolein-type alpha,beta-unsaturated aldehyde alert",
    },
    {
        "name": "alpha-halo carbonyl",
        "smarts": "[Cl,Br,I][#6X4][#6X3](=[#8X1])",
        "endpoint": "mutagenicity (potential, alkylating)",
        "source": "alpha-halo ketone/ester (phenacyl-type) alkylating alert",
    },
    {
        "name": "sulfonate ester",
        "smarts": "[#6][SX4](=[#8X1])(=[#8X1])[#8X2][#6]",
        "endpoint": "mutagenicity (potential, alkylating)",
        "source": "ICH M7 alkyl sulfonate ester (mesylate/tosylate) class",
    },
    {
        "name": "C-nitroso",
        "smarts": "[#6][#7X2]=[#8X1]",
        "endpoint": "reactivity / methemoglobinemia (potential)",
        "source": "C-nitroso reactive-toxicophore",
    },
    {
        "name": "vinyl halide",
        "smarts": "[Cl,Br,I][#6X3]=[#6X3]",
        "endpoint": "carcinogenicity (potential, epoxide via oxidation)",
        "source": "vinyl chloride/halide carcinogenicity (IARC)",
    },
    {
        "name": "beta-lactone",
        "smarts": "O=C1CCO1",
        "endpoint": "mutagenicity (potential, alkylating)",
        "source": "beta-propiolactone alkylating structural alert",
    },
    {
        "name": "catechol / ortho-diphenol",
        "smarts": "[OH]c1c([OH])cccc1",
        "endpoint": "hepatotoxicity (potential, quinone redox cycling)",
        "source": "catechol ortho-diphenol redox-cycling alert",
    },
    {
        "name": "1,2-dicarbonyl (glyoxal)",
        "smarts": "[#8X1]=[#6X3][#6X3]=[#8X1]",
        "endpoint": "mutagenicity (potential)",
        "source": "glyoxal/1,2-dicarbonyl mutagenicity alert",
    },
]

_patterns: dict[str, Chem.Mol] = {}


def _pattern(smarts: str) -> Chem.Mol | None:
    if smarts not in _patterns:
        _patterns[smarts] = Chem.MolFromSmarts(smarts)
    return _patterns[smarts]


def match_tox_alerts(mol: Chem.Mol) -> list[dict]:
    """Return the curated alerts matched by `mol` (name/smarts/endpoint/source),
    or an empty list. Never raises for a valid molecule."""
    out: list[dict] = []
    for alert in TOX_ALERTS:
        pat = _pattern(alert["smarts"])
        if pat is not None and mol.HasSubstructMatch(pat):
            out.append(dict(alert))
    return out
