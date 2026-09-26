"""Deterministic analog generation from a parent molecule using a fixed,
explainable library of medicinal-chemistry transformations (RDKit reactions).

This is the *discovery* half of ddswarm: given a starting molecule it produces
genuinely NEW candidate structures (real SMILES), each tagged with the
transformation applied and a plain-language rationale. Pure RDKit, offline, no
randomness — the same input yields the same ordered analog list, so the whole
run stays reproducible.

The agent reasons; this module computes. It returns real structures or an empty
list (never fabricates a molecule the tool didn't produce)."""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem

from .rdkit_ops import sanitize_and_canonicalize

RDLogger.DisableLog("rdApp.error")


@dataclass(frozen=True)
class GeneratedAnalog:
    smiles: str            # canonical SMILES of the NEW molecule
    transformation: str    # transformation slug, e.g. "cl_to_f"
    rationale: str         # plain-language description of the change


# Ordered library of single-step transformations. Each is a medicinal-chemistry
# operation expressed as a reaction SMARTS: (slug, reaction SMARTS, rationale).
# Order is significant — it is the deterministic enumeration order.
_TRANSFORMATIONS: list[tuple[str, str, str]] = [
    ("cl_to_f", "[Cl:1]>>[F:1]",
     "chlorine -> fluorine halogen swap (a common bioisosteric replacement)"),
    ("cl_to_br", "[Cl:1]>>[Br:1]",
     "chlorine -> bromine halogen swap"),
    ("cl_to_i", "[Cl:1]>>[I:1]",
     "chlorine -> iodine halogen swap"),
    ("methyl_to_ethyl", "[CH3:1]>>[CH2:1]C",
     "methyl -> ethyl homologation (extends a substituent by one carbon)"),
    ("n_demethylate", "[N:1][CH3:2]>>[N:1][H:2]",
     "N-methyl -> N-H demethylation (removes a nitrogen methyl)"),
]

_RXNS: dict[str, object] = {
    slug: AllChem.ReactionFromSmarts(smarts) for slug, smarts, _ in _TRANSFORMATIONS
}


def generate_analogs(canonical_smiles: str, max_analogs: int = 20) -> list[GeneratedAnalog]:
    """Generate new analog molecules from one parent.

    Returns analogs in a deterministic order (transformation-library order, then
    first product per transformation). Each analog is a canonical SMILES
    distinct from the input and from every other analog. Never raises; returns
    an empty list when the parent is unparseable or has no transformable
    substructure."""
    parent = sanitize_and_canonicalize(canonical_smiles)
    if not parent.ok or not parent.canonical_smiles:
        return []
    mol = Chem.MolFromSmiles(parent.canonical_smiles)
    if mol is None:
        return []

    out: list[GeneratedAnalog] = []
    seen: set[str] = {parent.canonical_smiles}
    for slug, _smarts, rationale in _TRANSFORMATIONS:
        if len(out) >= max_analogs:
            break
        prods = _RXNS[slug].RunReactants((mol,))
        if not prods:
            continue
        for prod in prods[0]:
            try:
                Chem.SanitizeMol(prod)
                prod = Chem.RemoveHs(prod)  # normalize explicit H (e.g. N-demethylation)
                canonical = Chem.MolToSmiles(prod)
            except Exception:  # noqa: BLE001
                continue
            if not canonical or canonical in seen:
                continue
            seen.add(canonical)
            out.append(GeneratedAnalog(smiles=canonical, transformation=slug,
                                       rationale=rationale))
            if len(out) >= max_analogs:
                break
    return out
