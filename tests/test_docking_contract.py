"""Tests for docking_minion, receptor QC, PDBQT parsing and RMSD (Increment 3).

Everything here is offline: synthetic PDBs, synthetic PDBQT, and a fake engine.
The real Vina binary is exercised separately (test_vina_engine.py) and in the demo."""

from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

from ddswarm.agents import DockingMinion
from ddswarm.agents.base import AgentContext
from ddswarm.chemistry.pdb_utils import analyze_pdb
from ddswarm.contracts import DockingResult, ProjectSpec
from ddswarm.docking.base import (
    DockPose,
    PreparedLigand,
    PreparedReceptor,
    QCResult,
)
from ddswarm.docking.rmsd import (
    heavy_atom_rmsd,
    parse_affinity_modes,
    parse_model_atoms,
    parse_smiles_idx,
    pose_mol_from_coords,
)
from ddswarm.orchestrator import Store

GEFITINIB = "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4"


# ---- PDB record builders (fixed-column format) ----
def _atom(serial, name, resname, chain, resnum, x, y, z, element="C", het=False):
    rec = "HETATM" if het else "ATOM  "
    return (
        f"{rec}{serial:5d} {name:<4s} {resname:>3s} {chain}{resnum:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}          {element:>2s}"
    )


def _gly(serial, chain, resnum):
    """A complete glycine residue (N, CA, C, O)."""
    return "\n".join([
        _atom(serial, "N", "GLY", chain, resnum, 0.0, 0.0, 0.0, "N"),
        _atom(serial + 1, "CA", "GLY", chain, resnum, 1.0, 0.0, 0.0, "C"),
        _atom(serial + 2, "C", "GLY", chain, resnum, 2.0, 0.0, 0.0, "C"),
        _atom(serial + 3, "O", "GLY", chain, resnum, 3.0, 0.0, 0.0, "O"),
    ])


# ---- receptor QC ----
def test_analyze_pdb_counts_and_flags():
    pdb = "\n".join([
        _gly(1, "A", 1),
        _atom(5, "O", "HOH", "A", 100, 0, 0, 0, "O", het=True),   # water
        _atom(6, "C", "IRE", "A", 101, 0, 0, 0, "C", het=True),   # ligand heterogen
    ])
    stats = analyze_pdb(pdb)
    assert stats.water_count == 1
    assert stats.hetero_residues == ["IRE"]
    assert stats.hydrogen_count == 0
    assert stats.residue_count == 1
    assert stats.chain_ids == ["A"]


def test_analyze_pdb_missing_residue_gap():
    pdb = "\n".join([
        _atom(1, "N", "GLY", "A", 1, 0, 0, 0, "N"),
        _atom(2, "N", "GLY", "A", 3, 0, 0, 0, "N"),  # gap: residue 2 missing
    ])
    stats = analyze_pdb(pdb)
    assert stats.missing_residues == ["chain A: 2-2"]


def test_analyze_pdb_missing_backbone():
    pdb = _atom(1, "CA", "GLY", "A", 1, 0, 0, 0, "C")  # only CA, no N/C/O
    stats = analyze_pdb(pdb)
    assert any("res 1" in m and "N" in m for m in stats.residues_missing_backbone)


def test_analyze_pdb_seqres_missing():
    pdb = "\n".join([
        "SEQRES   1 A  4  GLY GLY GLY GLY",
        _gly(1, "A", 1),
        _gly(5, "A", 2),
    ])
    stats = analyze_pdb(pdb)
    # SEQRES declares 4, observed 2 -> flagged
    assert stats.seqres_residue_count == 4
    assert any("SEQRES 4" in m or "2 observed" in m for m in stats.missing_residues)


# ---- PDBQT parsing ----
def test_parse_smiles_idx():
    pdbqt = "REMARK SMILES IDX 22 1 21 2 3 3\n"
    assert parse_smiles_idx(pdbqt) == {22: 1, 21: 2, 3: 3}


def test_parse_model_atoms():
    pdbqt = (
        "MODEL 1\n"
        "ATOM      1  C   UNL     1       1.000   2.000   3.000  1.00  0.00    +0.000 C \n"
        "ENDMDL\n"
        "MODEL 2\n"
        "ATOM      1  C   UNL     1       9.000   9.000   9.000  1.00  0.00    +0.000 C \n"
        "ENDMDL\n"
    )
    assert parse_model_atoms(pdbqt, model=1) == {1: (1.0, 2.0, 3.0)}
    assert parse_model_atoms(pdbqt, model=2) == {1: (9.0, 9.0, 9.0)}


def test_parse_affinity_modes():
    pdbqt = "REMARK VINA RESULT:    -7.920      0.000      0.000\nREMARK VINA RESULT:    -7.100      2.465      5.696\n"
    assert parse_affinity_modes(pdbqt) == [(1, -7.92, 0.0, 0.0), (2, -7.1, 2.465, 5.696)]


# ---- RMSD ----
def test_heavy_atom_rmsd_identical_is_zero():
    mol = Chem.MolFromSmiles("c1ccccc1")
    AllChem.EmbedMolecule(mol, randomSeed=1)
    ref = Chem.Mol(mol)
    prb = Chem.Mol(mol)
    r = heavy_atom_rmsd(ref, prb)
    assert r is not None and abs(r) < 1e-6


def test_heavy_atom_rmsd_translation():
    mol = Chem.MolFromSmiles("c1ccccc1")
    AllChem.EmbedMolecule(mol, randomSeed=1)
    ref = Chem.Mol(mol)
    prb = Chem.Mol(mol)
    conf = prb.GetConformer()
    for i in range(prb.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (pos.x + 5.0, pos.y, pos.z))
    r = heavy_atom_rmsd(ref, prb)
    assert r is not None and abs(r - 5.0) < 0.01


def test_heavy_atom_rmsd_non_isomorphic_is_none():
    a = Chem.MolFromSmiles("c1ccccc1")
    b = Chem.MolFromSmiles("CCO")
    AllChem.EmbedMolecule(a, randomSeed=1)
    AllChem.EmbedMolecule(b, randomSeed=1)
    assert heavy_atom_rmsd(a, b) is None


def test_pose_mol_from_coords_mapping():
    pdbqt_smiles_idx = {i + 1: i + 1 for i in range(6)}  # benzene: identity map
    atoms = {i + 1: (float(i), 0.0, 0.0) for i in range(6)}
    mol = pose_mol_from_coords("c1ccccc1", pdbqt_smiles_idx, atoms)
    assert mol is not None and mol.GetNumAtoms() == 6


# ---- docking_minion with a fake engine ----
class _FakeEngine:
    name = "vina"
    version = "fake-1.0"

    def available(self):
        return (True, "fake-1.0")

    def prepare_receptor(self, source, work_dir):
        return PreparedReceptor(
            pdbqt_path=Path("receptor.pdbqt"), atom_count=10, residue_count=2, chain_count=1,
            preparation_tool="fake", qc_checks=[QCResult("waters_present", "ok", "0 water")],
        )

    def prepare_ligand(self, smiles, work_dir):
        return PreparedLigand(canonical_smiles=smiles, pdbqt_path=Path("ligand.pdbqt"),
                              smiles_idx_map={i + 1: i + 1 for i in range(6)}, heavy_atom_count=6)

    def dock(self, receptor, ligand, protocol, work_dir):
        return [DockPose(rank=1, affinity_kcal_per_mol=-9.0, pose_file=Path("docked.pdbqt"),
                         pose_atoms={})]


def _run_docking(store, monkeypatch, params):
    monkeypatch.setattr("ddswarm.agents.docking.make_engine", lambda name, vina_binary=None: _FakeEngine())
    agent = DockingMinion(store)
    spec = ProjectSpec(project_id="p1", input_type="mixed", objective="dock",
                       starting_materials=[GEFITINIB], status="approved")
    ctx = AgentContext(project_id="p1", run_id="r1", task_id="dock", spec=spec, params=params)
    return agent.run(ctx)


def test_docking_no_receptor_insufficient(tmp_path, monkeypatch):
    store = Store(tmp_path)
    result = _run_docking(store, monkeypatch, {})
    assert result.status.value == "insufficient_evidence"


def test_docking_no_box_insufficient(tmp_path, monkeypatch):
    store = Store(tmp_path)
    result = _run_docking(store, monkeypatch, {"receptor": {"identifier": "2ITY"}})
    # fake receptor has no reference ligand, and no box given -> insufficient
    assert result.status.value == "insufficient_evidence"


def test_docking_success_with_fake_engine(tmp_path, monkeypatch):
    store = Store(tmp_path)
    params = {
        "receptor": {"identifier": "2ITY", "kind": "pdb_id"},
        "box": {"center": [0.0, 0.0, 0.0], "size": [10.0, 10.0, 10.0]},
    }
    result = _run_docking(store, monkeypatch, params)
    assert result.status.value == "success"
    assert result.artifacts
    batch = store.load_artifact(result.artifacts[0].uri, DockingResult)
    assert batch.summary.total_molecules == 1
    assert batch.summary.docked == 1
    assert len(batch.poses) == 1
    assert batch.poses[0].affinity_kcal_per_mol == -9.0
    assert batch.receptor.identifier == "2ITY"
    assert batch.protocol.engine_version == "fake-1.0"
