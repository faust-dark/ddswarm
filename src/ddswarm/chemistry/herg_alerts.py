"""Deterministic hERG (Kv11.1) cardiotoxicity-liability flag.

Hybrid mechanistic + descriptor rule, on the same honesty standard as tox and
cyp: a match is a *predicted* liability, never a patch-clamp / functional assay.

Mechanistic pharmacophore (the hERG pore): a protonatable basic nitrogen that
coordinates the K+ selectivity filter, flanked by >=2 aromatic rings that engage
the Y652/F656 aromatic cage. Descriptor gate: hERG blockers must be lipophilic
enough to partition into the membrane (cLogP). A molecule matches only if ALL
three hold — this is what separates classic blockers (terfenadine, cisapride,
haloperidol) from atenolol/pravastatin (too polar / no basic amine) and warfarin
(lipophilic and aromatic but not basic).

Potency (IC50) is NOT estimated here — a 2D pharmacophore cannot rank affinity.
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

# Protonatable basic amines (pKa >= ~7 at physiological pH). Explicitly excludes
# amides, sulfonamides, anilines (pKa ~5), and nitro/nitrile nitrogens, which are
# not protonated at pH 7.4 and therefore do not engage the selectivity filter.
BASIC_AMINE_SMARTS: list[tuple[str, str, str]] = [
    ("tertiary_amine",
     "[N;H0;+0;v3;!$(N[C,c]=[O,S]);!$(N=[O,S,N]);!$(N#*)]",
     "Aronov 2005 (Drug Discov Today) hERG pharmacophore; Mitcheson et al. 2000 (PNAS) T623 basic-N contact"),
    ("secondary_amine",
     "[N;H1;+0;v3;!$(N[C,c]=[O,S]);!$(N=[O,S,N])]",
     "Aronov 2005; Mitcheson et al. 2000"),
    ("piperidine", "N1CCCCC1",
     "Aronov 2005 (cationic center motif)"),
    ("piperazine", "N1CCNCC1",
     "Aronov 2005 (cationic center motif)"),
    ("imidazole", "n1ccnc1",
     "Sanguinetti & Tristani-Firouzi 2006 (Nature) hERG blockers"),
]

# Documented hERG-binding SCAFFOLDS (structural classes strongly associated with
# Kv11.1 blockade). These do NOT set the flag on their own — the descriptor
# pharmacophore (basic amine + >=2 aromatics + cLogP) is the mechanistic gate —
# they ENRICH the reason with the specific class when a molecule both matches a
# chemotype AND clears the gate. A chemotype match is a structural fact, not a
# functional assay.
HERG_CHEMOTYPES: list[tuple[str, str, str]] = [
    ("arylpiperazine",
     "c1ccc(N2CCNCC2)cc1",
     "N-aryl-piperazine class (ziprasidone/trazodone) — hERG-binding scaffold; Aronov 2005"),
    ("phenothiazine",
     "c1ccc2Sc3ccccc3Nc2c1",
     "phenothiazine tricyclic (chlorpromazine/thioridazine) — classic hERG blocker; Thomas et al. 2006"),
    ("butyrophenone",
     "[cR1]C(=O)CCCN",
     "butyrophenone (haloperidol/droperidol) — hERG blocker; Suessbrich et al. 1997"),
    ("methanesulfonanilide",
     "CS(=O)(=O)N[c]",
     "methanesulfonanilide (dofetilide/sotalol) — class III antiarrhythmic hERG blocker; Mitcheson et al. 2000"),
]

# Pharmacophore thresholds.
AROMATIC_RING_MIN = 2
CLOGP_MIN = 3.0


def assess_herg(smiles: str | None) -> dict:
    """Return a hERG-liability assessment dict for one molecule.

    Keys: flagged (bool), basic_amine_motifs (list[str]), aromatic_rings (int),
    clogp (float | None), chemotype_matches (list[str]), reason (str).
    """
    empty = {
        "flagged": False,
        "basic_amine_motifs": [],
        "aromatic_rings": 0,
        "clogp": None,
        "chemotype_matches": [],
        "reason": "invalid or empty SMILES",
    }
    if not smiles:
        return empty
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return empty

    motifs = [slug for slug, smarts, _cit in BASIC_AMINE_SMARTS
              if mol.HasSubstructMatch(Chem.MolFromSmarts(smarts))]
    chemotypes = [slug for slug, smarts, _cit in HERG_CHEMOTYPES
                  if mol.HasSubstructMatch(Chem.MolFromSmarts(smarts))]
    arom = rdMolDescriptors.CalcNumAromaticRings(mol)
    logp = Descriptors.MolLogP(mol)

    flagged = bool(motifs) and arom >= AROMATIC_RING_MIN and logp >= CLOGP_MIN

    if not motifs:
        reason = "no protonatable basic amine (not an hERG pharmacophore)"
    elif arom < AROMATIC_RING_MIN:
        reason = f"only {arom} aromatic ring(s) (< {AROMATIC_RING_MIN}); basic amine present but no aromatic cage reach"
    elif logp < CLOGP_MIN:
        reason = f"cLogP {logp:.2f} below the lipophilicity gate ({CLOGP_MIN}); too polar to reach the channel"
    else:
        class_hint = f"; known hERG scaffold ({', '.join(chemotypes)})" if chemotypes else ""
        reason = (f"basic amine ({', '.join(motifs)}) + {arom} aromatic rings + "
                  f"cLogP {logp:.2f} — predicted potential hERG liability (not a functional assay)"
                  f"{class_hint}")

    return {
        "flagged": flagged,
        "basic_amine_motifs": motifs,
        "aromatic_rings": arom,
        "clogp": round(logp, 2),
        "chemotype_matches": chemotypes,
        "reason": reason,
    }
