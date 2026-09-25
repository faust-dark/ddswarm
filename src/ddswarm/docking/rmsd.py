"""PDBQT parsing + docking-pose RMSD (Increment 3).

Vina preserves the input ligand's atom serial numbers in its output, and Meeko
emits a 'REMARK SMILES IDX' table mapping SMILES atom indices (1-based) to
PDBQT atom serials. We use that table to place the docked pose's coordinates
onto a SMILES-derived molecule (correct connectivity), then compare against a
reference (co-crystal) molecule with CalcRMS, which does its own symmetry-aware
substructure matching. Heavy atoms only — no hydrogens — to avoid the
combinatorial blow-up CalcRMS warns about."""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import rdMolAlign


def parse_smiles_idx(pdbqt_text: str) -> dict[int, int]:
    """Return {smiles_idx_1based: pdbqt_atom_serial} from REMARK SMILES IDX lines."""
    mapping: dict[int, int] = {}
    for line in pdbqt_text.splitlines():
        if line.startswith("REMARK SMILES IDX"):
            tokens = [int(t) for t in line.split()[3:]]
            for i in range(0, len(tokens), 2):
                mapping[tokens[i]] = tokens[i + 1]
    return mapping


def parse_model_atoms(pdbqt_text: str, model: int = 1) -> dict[int, tuple[float, float, float]]:
    """Return {atom_serial: (x, y, z)} for ATOM/HETATM records in MODEL <model>."""
    atoms: dict[int, tuple[float, float, float]] = {}
    in_model = False
    for line in pdbqt_text.splitlines():
        if line.startswith("MODEL"):
            in_model = (line.split()[1] == str(model))
            continue
        if line.startswith("ENDMDL"):
            in_model = False
            continue
        if in_model and (line.startswith("ATOM") or line.startswith("HETATM")):
            serial = int(line[6:11])
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            atoms[serial] = (x, y, z)
    return atoms


def parse_affinity_modes(pdbqt_text: str) -> list[tuple[int, float, float, float]]:
    """Return [(rank, affinity, rmsd_lb, rmsd_ub), ...] from REMARK VINA RESULT lines."""
    out: list[tuple[int, float, float, float]] = []
    rank = 0
    for line in pdbqt_text.splitlines():
        if line.startswith("REMARK VINA RESULT:"):
            rank += 1
            vals = line.split(":")[1].split()
            aff, lb, ub = float(vals[0]), float(vals[1]), float(vals[2])
            out.append((rank, aff, lb, ub))
    return out


def pose_mol_from_coords(
    smiles: str, smiles_idx_map: dict[int, int], pose_atoms: dict[int, tuple[float, float, float]]
) -> Chem.Mol | None:
    """Build an RDKit mol (SMILES connectivity) with a conformer holding the
    docked pose coordinates, mapped from PDBQT serials via the SMILES IDX table."""
    smi_mol = Chem.MolFromSmiles(smiles)
    if smi_mol is None:
        return None
    pose_mol = Chem.Mol(smi_mol)
    conf = Chem.Conformer(smi_mol.GetNumAtoms())
    try:
        for rd_idx in range(smi_mol.GetNumAtoms()):
            serial = smiles_idx_map[rd_idx + 1]
            conf.SetAtomPosition(rd_idx, pose_atoms[serial])
    except (KeyError, ValueError):
        return None
    pose_mol.AddConformer(conf)
    return pose_mol


def heavy_atom_rmsd(ref_mol: Chem.Mol, prb_mol: Chem.Mol) -> float | None:
    """Symmetry-aware RMSD between two conformers of the same molecule.

    Uses rdMolAlign.CalcRMS 'in place' (no pre-alignment), which performs its
    own substructure match — the right tool for docked-pose vs co-crystal RMSD.
    Returns None if the two molecules are not isomorphic."""
    try:
        return float(rdMolAlign.CalcRMS(prb_mol, ref_mol))
    except RuntimeError:
        return None
