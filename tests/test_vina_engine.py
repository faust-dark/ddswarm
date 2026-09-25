"""Real AutoDock Vina engine smoke tests (skip unless the binary is present).

These exercise the actual docking path — ligand prep (Meeko), receptor
conversion (Open Babel), and a Vina run — on a tiny synthetic peptide, so they
need no network. The full receptor-preparation path (PDBFixer + a real PDB) is
exercised by scripts/demo_docking.py."""

import subprocess
from pathlib import Path

import pytest

from ddswarm.docking import default_vina_binary, make_engine
from ddswarm.docking.base import DockProtocol, PreparedReceptor

GEFITINIB = "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4"

_VINA = default_vina_binary()
pytestmark = pytest.mark.skipif(not _VINA.exists(), reason="vina binary not present")


def _atom(serial, name, resname, chain, resnum, x, y, z, element="C"):
    return (
        f"ATOM  {serial:5d} {name:<4s} {resname:>3s} {chain}{resnum:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}          {element:>2s}"
    )


def _peptide_pdb_text() -> str:
    lines = []
    serial = 1
    for r in range(1, 6):
        for name, dx in [("N", -0.5), ("CA", 0.0), ("C", 0.5), ("O", 1.0)]:
            x = (r - 1) * 3.8 + dx
            lines.append(_atom(serial, name, "GLY", "A", r, x, 0.0, 0.0, name[0] if name[0] in "NCO" else "C"))
            serial += 1
    return "\n".join(lines) + "\nEND\n"


def test_engine_available_reports_version():
    engine = make_engine("vina")
    ok, msg = engine.available()
    assert ok, msg
    assert "v1.2" in msg


def test_prepare_ligand_gefitinib(tmp_path):
    engine = make_engine("vina")
    ligand = engine.prepare_ligand(GEFITINIB, tmp_path)
    assert ligand.pdbqt_path and ligand.pdbqt_path.exists()
    assert ligand.heavy_atom_count == 31          # gefitinib: 31 heavy atoms
    assert ligand.canonical_smiles


def test_vina_dock_smoke(tmp_path):
    engine = make_engine("vina")
    ok, _ = engine.available()
    assert ok

    pdb = tmp_path / "peptide.pdb"
    pdb.write_text(_peptide_pdb_text())
    recpdbqt = tmp_path / "peptide.pdbqt"
    r = subprocess.run([str(engine.obabel_path), str(pdb), "-O", str(recpdbqt), "-xr"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr

    receptor = PreparedReceptor(pdbqt_path=recpdbqt)
    ligand = engine.prepare_ligand("c1ccccc1", tmp_path)
    assert ligand.pdbqt_path

    protocol = DockProtocol(
        box_center=(5.0, 0.0, 0.0), box_size=(20.0, 20.0, 20.0),
        exhaustiveness=4, num_modes=3, seed=42,
    )
    poses = engine.dock(receptor, ligand, protocol, tmp_path)
    assert poses, "expected at least one docked pose"
    assert poses[0].rank == 1
    assert poses[0].affinity_kcal_per_mol is not None
    assert poses[0].pose_atoms  # coordinates parsed for RMSD
