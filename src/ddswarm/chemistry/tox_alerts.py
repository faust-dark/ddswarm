"""Curated deterministic toxicophore structural alerts.

A SMALL, sourced starting set (not exhaustive). Each alert is a literature
structural-alert rule: a substructure historically associated with a toxicity
endpoint. A match is a *predicted* potential liability, not experimental
evidence — the agent reports it as such.

SMARTS validated against: nitrobenzene, dimethylnitrosamine, oxirane, aziridine,
acrolein, chloroethane, 1,4-benzoquinone; gefitinib matches none."""

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
