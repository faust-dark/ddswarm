"""Curated, non-exhaustive safety-gating lists for safety_minion.

Controlled substances are matched by EXACT canonical SMILES; dual-use/biosafety
targets by case-insensitive name. These are small illustrative lists — a policy
aid, NOT a legal determination and NOT a comprehensive database."""

from __future__ import annotations

from .rdkit_ops import sanitize_and_canonicalize

CONTROLLED_SUBSTANCES: list[dict] = [
    {
        "name": "fentanyl",
        "smiles": "CCC(=O)N(c1ccccc1)c1ccc(CCN2CCCCC2)cc1",
        "schedule": "US Schedule II",
        "source": "UN Single Convention on Narcotic Drugs",
    },
    {
        "name": "methamphetamine",
        "smiles": "CNC(C)Cc1ccccc1",
        "schedule": "US Schedule II",
        "source": "UN Convention on Psychotropic Substances",
    },
    {
        "name": "MDMA",
        "smiles": "COc1ccc(CC(C)NC)cc1",
        "schedule": "US Schedule I",
        "source": "UN Convention on Psychotropic Substances",
    },
]

DUAL_USE_TARGETS: list[dict] = [
    {"name": "botulinum toxin", "category": "biotoxin", "source": "Australia Group control list"},
    {"name": "ricin", "category": "biotoxin", "source": "Australia Group control list"},
    {"name": "anthrax", "category": "select-agent pathogen", "source": "US select-agent list"},
    {"name": "variola", "category": "select-agent pathogen (smallpox)", "source": "US select-agent list"},
    {"name": "ebola", "category": "select-agent pathogen", "source": "US select-agent list"},
    {"name": "sarin", "category": "chemical-weapon agent", "source": "CWC Schedule 1"},
    {"name": "sulfur mustard", "category": "chemical-weapon agent", "source": "CWC Schedule 1"},
]

# Canonicalize the list SMILES once, so matching is exact-canonical.
_CONTROLLED_CANONICAL: dict[str, str | None] = {
    e["name"]: (lambda s: s.canonical_smiles if s.ok else None)(sanitize_and_canonicalize(e["smiles"]))
    for e in CONTROLLED_SUBSTANCES
}


def match_controlled_substance(canonical_smiles: str) -> list[dict]:
    """Controlled-substance entries whose canonical SMILES equals the input."""
    out: list[dict] = []
    for e in CONTROLLED_SUBSTANCES:
        if _CONTROLLED_CANONICAL.get(e["name"]) == canonical_smiles:
            out.append(e)
    return out


def match_dual_use_target(target_name: str) -> list[dict]:
    """Dual-use/biosafety entries whose name appears in the target (case-insensitive)."""
    tl = target_name.lower()
    return [e for e in DUAL_USE_TARGETS if e["name"] in tl]
