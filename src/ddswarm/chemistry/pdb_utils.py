"""Pure-Python PDB structural analysis for receptor QC (Increment 3).

The agent reasons; this module computes. It parses a PDB file's records and
reports objective structural facts — atom/residue/chain counts, waters,
heterogens, hydrogens, missing residues (SEQRES vs observed ATOM) and residues
missing backbone atoms — without any external dependency. It never edits or
'fixes' the structure; that is the receptor-preparation step's job."""

from __future__ import annotations

from dataclasses import dataclass, field


# Standard amino-acid residue names (3-letter). Non-standard residues and
# ligands are 'heterogens'.
_STD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "SEC", "PYL", "MSE", "ASX", "GLX", "UNK",
}
_WATER_NAMES = {"HOH", "WAT", "H2O", "DOD", "SOL"}
_BACKBONE = {"N", "CA", "C", "O"}


@dataclass
class PDBStats:
    atom_count: int = 0
    hetatm_count: int = 0
    residue_count: int = 0
    chain_ids: list[str] = field(default_factory=list)
    water_count: int = 0
    hydrogen_count: int = 0
    hetero_residues: list[str] = field(default_factory=list)      # non-water HETATM residue names
    missing_residues: list[str] = field(default_factory=list)     # "chain A: 120-124" etc.
    residues_missing_backbone: list[str] = field(default_factory=list)
    seqres_residue_count: int = 0


def analyze_pdb(text: str) -> PDBStats:
    """Parse ATOM/HETATM/SEQRES records and return objective structural facts.

    Missing residues are detected by comparing SEQRES (the full intended
    sequence) to observed ATOM residues per chain. Without SEQRES, a fallback
    gap heuristic on residue numbering is used. Neither is a proof of a defect;
    both are reported as observed facts."""
    stats = PDBStats()

    # --- observed atoms/residues ---
    # chain -> ordered list of (residue_number, insertion_code, residue_name)
    observed: dict[str, list[tuple[int, str, str]]] = {}
    # chain -> set of (resnum, icode) for observed residues
    observed_res: dict[str, set[tuple[int, str]]] = {}
    # (chain, resnum, icode) -> set of atom names present
    residue_atoms: dict[tuple[str, int, str], set[str]] = {}
    # SEQRES chain -> list of residue names (intended sequence)
    seqres: dict[str, list[str]] = {}
    seen_het: set[str] = set()

    for line in text.splitlines():
        if line.startswith("SEQRES"):
            chain = line[11:12].strip() or " "
            resnames = line[19:].split()
            seqres.setdefault(chain, []).extend(resnames)
            continue
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        is_het = line.startswith("HETATM")
        atom_name = line[12:16].strip()
        res_name = line[17:20].strip()
        chain = line[21:22].strip() or " "
        resnum = int(line[22:26])
        icode = line[26:27].strip() or " "
        element = (line[76:78].strip() or atom_name[:1]).upper()

        stats.atom_count += 1
        if is_het:
            stats.hetatm_count += 1
        if element == "H":
            stats.hydrogen_count += 1

        if is_het:
            if res_name in _WATER_NAMES:
                stats.water_count += 1
            elif res_name not in seen_het:
                seen_het.add(res_name)
                stats.hetero_residues.append(res_name)
            continue  # heterogens don't contribute to polymer residue accounting

        key = (resnum, icode)
        observed.setdefault(chain, []).append((resnum, icode, res_name))
        observed_res.setdefault(chain, set()).add(key)
        residue_atoms.setdefault((chain, resnum, icode), set()).add(atom_name)

    # --- chains / residues ---
    stats.chain_ids = sorted(observed.keys())
    stats.residue_count = sum(len(v) for v in observed_res.values())
    if not stats.chain_ids and seqres:
        stats.chain_ids = sorted(seqres.keys())
    stats.seqres_residue_count = sum(len(v) for v in seqres.values())

    # --- missing residues (SEQRES vs observed) ---
    if seqres:
        for chain in seqres:
            obs_keys = observed_res.get(chain, set())
            if not obs_keys:
                stats.missing_residues.append(f"chain {chain}: no observed residues (all {len(seqres[chain])} missing)")
                continue
            observed_nums = sorted({r[0] for r in obs_keys})
            n_seq = len(seqres[chain])
            # map observed residue numbers (sorted) onto the SEQRES positions by
            # order, then flag stretches where numbering is discontinuous.
            obs_ordered = observed.get(chain, [])
            # dedupe preserving order by (resnum, icode)
            seen: set[tuple[int, str]] = set()
            ordered_res = []
            for r in obs_ordered:
                if r[:2] not in seen:
                    seen.add(r[:2])
                    ordered_res.append(r)
            prev = None
            for i, (resnum, icode, _name) in enumerate(ordered_res):
                if prev is not None and resnum != prev + 1:
                    # a numbering gap => possibly missing residues
                    lo, hi = prev + 1, resnum - 1
                    stats.missing_residues.append(f"chain {chain}: {lo}-{hi}")
                prev = resnum
            # crude size mismatch (only meaningful if numbering is contiguous)
            if n_seq and len(observed_res[chain]) < n_seq and not stats.missing_residues:
                stats.missing_residues.append(
                    f"chain {chain}: SEQRES {n_seq} residues vs {len(observed_res[chain])} observed"
                )
    else:
        # fallback: contiguous-numbering gap heuristic
        for chain, residues in observed.items():
            seen: set[tuple[int, str]] = set()
            ordered = []
            for r in residues:
                if r[:2] not in seen:
                    seen.add(r[:2])
                    ordered.append(r)
            prev = None
            for resnum, _icode, _name in ordered:
                if prev is not None and resnum != prev + 1:
                    stats.missing_residues.append(f"chain {chain}: {prev + 1}-{resnum - 1}")
                prev = resnum

    # --- residues missing backbone atoms ---
    for (chain, resnum, icode), atoms in residue_atoms.items():
        missing = sorted(_BACKBONE - atoms)
        if missing:
            stats.residues_missing_backbone.append(
                f"chain {chain} res {resnum}{icode.strip()}: missing {','.join(missing)}"
            )

    return stats
