"""AutoDock Vina docking engine (real, Increment 3).

Receptor: raw PDB -> PDBFixer (clean, add H) -> Open Babel (-xr) -> rigid PDBQT.
Ligand:   SMILES -> Meeko -> PDBQT with a 'REMARK SMILES IDX' table.
Docking:  the AutoDock Vina binary (subprocess). Poses/affinities are parsed
          straight from Vina's output; nothing is invented."""

from __future__ import annotations

import importlib.metadata
import subprocess
import sys
from pathlib import Path

from ..chemistry.pdb_utils import analyze_pdb
from .base import (
    DockPose,
    DockProtocol,
    DockingEngine,
    PreparedLigand,
    PreparedReceptor,
    QCResult,
    ReceptorSource,
)
from .rmsd import parse_affinity_modes, parse_model_atoms, parse_smiles_idx

_RCSB_URL = "https://files.rcsb.org/download/{pid}.pdb"


def _pkg_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "?"


class VinaEngine(DockingEngine):
    name = "vina"

    def __init__(self, binary_path: str | Path, obabel_path: str | Path | None = None):
        self.binary_path = Path(binary_path)
        self.obabel_path = Path(obabel_path) if obabel_path else Path(sys.executable).parent / "obabel"

    @property
    def version(self) -> str:
        ok, msg = self._probe_version()
        return msg if ok else "unknown"

    def _probe_version(self) -> tuple[bool, str]:
        try:
            r = subprocess.run(
                [str(self.binary_path), "--version"], capture_output=True, text=True, timeout=30
            )
            line = (r.stdout or "").strip().splitlines()[0] if (r.stdout or "").strip() else ""
            return (r.returncode == 0), line
        except (OSError, subprocess.SubprocessError):
            return False, ""

    def available(self) -> tuple[bool, str]:
        if not self.binary_path.exists():
            return False, f"vina binary not found at {self.binary_path}"
        ok, ver = self._probe_version()
        if not ok:
            return False, "vina binary not runnable"
        if not self.obabel_path.exists():
            return False, f"obabel not found at {self.obabel_path}"
        return True, ver

    # ---- receptor ----
    def prepare_receptor(self, source: ReceptorSource, work_dir: Path) -> PreparedReceptor:
        work_dir.mkdir(parents=True, exist_ok=True)
        rec = PreparedReceptor()

        url = source.url or (_RCSB_URL.format(pid=source.identifier) if source.kind == "pdb_id" else None)
        if source.kind == "file":
            raw_path = Path(source.identifier)
            raw_text = raw_path.read_text()
        else:
            if url is None:
                rec.note = f"no receptor URL for source kind '{source.kind}'"
                return rec
            raw_path = work_dir / f"{source.identifier}_raw.pdb"
            r = subprocess.run(["curl", "-sL", url, "-o", str(raw_path)], capture_output=True, text=True)
            if r.returncode != 0 or not raw_path.exists() or raw_path.stat().st_size == 0:
                rec.note = f"could not download receptor ({r.stderr.strip() or 'empty file'})"
                return rec
            raw_text = raw_path.read_text()

        # structural QC on the raw structure
        stats = analyze_pdb(raw_text)
        rec.qc_checks = self._qc_checks(stats)
        rec.atom_count = stats.atom_count
        rec.residue_count = stats.residue_count
        rec.chain_count = len(stats.chain_ids)

        # co-crystal ligand (largest non-water heterogen) for redocking
        ref_name, ref_mol = self._extract_reference_ligand(raw_text, work_dir)
        rec.reference_ligand_name = ref_name
        rec.reference_ligand_mol = ref_mol

        # clean + protonate, then rigid PDBQT
        try:
            from pdbfixer import PDBFixer  # noqa: PLC0415
            from openmm.app import PDBFile  # noqa: PLC0415

            fixer = PDBFixer(filename=str(raw_path))
            fixer.findMissingResidues()
            fixer.findNonstandardResidues()
            fixer.replaceNonstandardResidues()
            fixer.removeHeterogens(keepWater=False)
            fixer.findMissingAtoms()
            fixer.addMissingAtoms()
            fixer.addMissingHydrogens(7.0)
            clean_pdb = work_dir / f"{source.identifier}_receptor.pdb"
            with open(clean_pdb, "w") as fh:
                PDBFile.writeFile(fixer.topology, fixer.positions, fh)
        except Exception as exc:  # noqa: BLE001
            rec.preparation_tool = "pdbfixer failed"
            rec.note = f"receptor preparation failed: {exc}"
            return rec

        recpdbqt = work_dir / f"{source.identifier}_receptor.pdbqt"
        r = subprocess.run(
            [str(self.obabel_path), str(clean_pdb), "-O", str(recpdbqt), "-xr"],
            capture_output=True, text=True,
        )
        if r.returncode != 0 or not recpdbqt.exists():
            rec.preparation_tool = "pdbfixer + openbabel"
            rec.note = f"PDB->PDBQT conversion failed: {r.stderr.strip()}"
            return rec

        rec.pdbqt_path = recpdbqt
        rec.preparation_tool = (
            f"pdbfixer {_pkg_version('pdbfixer')} + openbabel-wheel {_pkg_version('openbabel-wheel')} "
            f"(rigid PDBQT)"
        )
        rec.note = "receptor prepared and protonated"
        return rec

    @staticmethod
    def _qc_checks(stats) -> list[QCResult]:
        return [
            QCResult("missing_residues",
                     "issue" if stats.missing_residues else "ok",
                     "; ".join(stats.missing_residues) or "no missing residues detected"),
            QCResult("residues_missing_backbone_atoms",
                     "issue" if stats.residues_missing_backbone else "ok",
                     "; ".join(stats.residues_missing_backbone[:5]) or "backbone complete"),
            QCResult("waters_present",
                     "issue" if stats.water_count else "ok",
                     f"{stats.water_count} water molecule(s)"),
            QCResult("heterogens_present",
                     "issue" if stats.hetero_residues else "ok",
                     ", ".join(stats.hetero_residues) or "no heterogens"),
            QCResult("hydrogens_present",
                     "issue" if stats.hydrogen_count == 0 else "ok",
                     f"{stats.hydrogen_count} hydrogen atom(s)"),
        ]

    def _extract_reference_ligand(self, raw_text: str, work_dir: Path):
        """Extract the largest non-water HETATM residue as an RDKit mol."""
        try:
            from rdkit import Chem  # noqa: PLC0415
        except ImportError:
            return None, None

        residues: dict[tuple[str, str], list[str]] = {}
        for line in raw_text.splitlines():
            if not line.startswith("HETATM"):
                continue
            resname = line[17:20].strip()
            if not resname or resname in {"HOH", "WAT", "H2O", "DOD", "SOL"}:
                continue
            chain = line[21:22].strip() or " "
            residues.setdefault((resname, chain), []).append(line)

        if not residues:
            return None, None
        (name, _chain), lines = max(residues.items(), key=lambda kv: len(kv[1]))
        het_pdb = work_dir / f"reference_{name}.pdb"
        het_pdb.write_text("\n".join(lines) + "\nEND\n")
        sdf = work_dir / f"reference_{name}.sdf"
        r = subprocess.run(
            [str(self.obabel_path), str(het_pdb), "-O", str(sdf)], capture_output=True, text=True
        )
        if r.returncode != 0 or not sdf.exists():
            return name, None
        mol = Chem.MolFromMolFile(str(sdf), removeHs=True, sanitize=True)
        if mol is None or mol.GetNumAtoms() == 0:
            return name, None
        return name, mol

    # ---- ligand ----
    def prepare_ligand(self, smiles: str, work_dir: Path) -> PreparedLigand:
        work_dir.mkdir(parents=True, exist_ok=True)
        lig = PreparedLigand()
        try:
            from meeko import MoleculePreparation, PDBQTWriterLegacy  # noqa: PLC0415
            from rdkit import Chem  # noqa: PLC0415
            from rdkit.Chem import AllChem  # noqa: PLC0415
        except ImportError as exc:
            lig.note = f"ligand prep unavailable: {exc}"
            return lig

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            lig.note = f"could not parse SMILES {smiles!r}"
            return lig
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
            lig.note = "3D embedding failed"
            return lig
        AllChem.MMFFOptimizeMolecule(mol)
        try:
            setups = MoleculePreparation().prepare(mol)
            pdbqt, ok, err = PDBQTWriterLegacy.write_string(setups[0])
        except Exception as exc:  # noqa: BLE001
            lig.note = f"Meeko preparation failed: {exc}"
            return lig
        if not ok:
            lig.note = f"Meeko write failed: {err}"
            return lig

        smiles_line = next(
            (l.split("SMILES ", 1)[1] for l in pdbqt.splitlines() if l.startswith("REMARK SMILES ")),
            None,
        )
        lig.canonical_smiles = smiles_line or ""
        lig.smiles_idx_map = parse_smiles_idx(pdbqt)
        lig.heavy_atom_count = len(lig.smiles_idx_map)
        lig.pdbqt_path = work_dir / "ligand.pdbqt"
        lig.pdbqt_path.write_text(pdbqt)
        return lig

    # ---- docking ----
    def dock(
        self,
        receptor: PreparedReceptor,
        ligand: PreparedLigand,
        protocol: DockProtocol,
        work_dir: Path,
    ) -> list[DockPose]:
        if not receptor.pdbqt_path or not ligand.pdbqt_path:
            return []
        if protocol.box_center is None or protocol.box_size is None:
            return []

        work_dir.mkdir(parents=True, exist_ok=True)
        out = work_dir / "docked.pdbqt"
        cmd = [
            str(self.binary_path),
            "--receptor", str(receptor.pdbqt_path),
            "--ligand", str(ligand.pdbqt_path),
            "--center_x", f"{protocol.box_center[0]:.3f}",
            "--center_y", f"{protocol.box_center[1]:.3f}",
            "--center_z", f"{protocol.box_center[2]:.3f}",
            "--size_x", f"{protocol.box_size[0]:.3f}",
            "--size_y", f"{protocol.box_size[1]:.3f}",
            "--size_z", f"{protocol.box_size[2]:.3f}",
            "--exhaustiveness", str(protocol.exhaustiveness),
            "--num_modes", str(protocol.num_modes),
            "--seed", str(protocol.seed),
            "--out", str(out),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not out.exists():
            return []

        text = out.read_text()
        affinities = parse_affinity_modes(text)
        poses: list[DockPose] = []
        for rank, aff, lb, ub in affinities:
            poses.append(DockPose(
                rank=rank,
                affinity_kcal_per_mol=aff,
                rmsd_lower_bound=lb,
                rmsd_upper_bound=ub,
                pose_file=out,
                pose_atoms=parse_model_atoms(text, model=rank),
            ))
        return poses
