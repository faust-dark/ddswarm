"""AutoDock Vina docking engine (real, Increment 3).

Receptor: raw PDB -> PDBFixer (clean, add H) -> Open Babel (-xr) -> rigid PDBQT.
Ligand:   SMILES -> Meeko -> PDBQT with a 'REMARK SMILES IDX' table.
Docking:  the AutoDock Vina binary (subprocess). Poses/affinities are parsed
          straight from Vina's output; nothing is invented."""

from __future__ import annotations

import importlib.metadata
import json
import re
import subprocess
import sys
from functools import cached_property
from pathlib import Path

from ..chemistry.pdb_utils import analyze_pdb
from ..fsutil import atomic_write_bytes, atomic_write_text
from ..logging.hashing import hash_content, hash_jsonable
from .base import (
    RCSB_URL,
    DockingEngine,
    DockPose,
    DockProtocol,
    PreparedLigand,
    PreparedReceptor,
    QCResult,
    ReceptorSource,
    download_pdb,
    is_safe_name,
    is_valid_pdb_id,
    looks_like_pdb,
    pdb_cache_path,
)
from .rmsd import parse_affinity_modes, parse_model_atoms, parse_smiles_idx

_RCSB_URL = RCSB_URL
# Subprocess wall-clock limits (seconds): a hung converter or a pathological
# docking search must fail the molecule, never hang the run.
OBABEL_TIMEOUT_S = 300
VINA_TIMEOUT_S = 1800
# Bump when the receptor-preparation recipe below changes (PDBFixer steps, pH,
# Open Babel flags, reference-ligand selection) so cached preparations made by
# an older recipe are never reused.
_RECEPTOR_RECIPE_VERSION = 1
_PREP_PACKAGES = ("pdbfixer", "openmm", "openbabel-wheel", "meeko", "rdkit")


def _run_tool(cmd: list[str], timeout: float) -> tuple[int, str]:
    """Run an external tool with a timeout -> (returncode, stderr). A timeout or
    OS error is reported as returncode -1 with a message, never raised."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stderr or ""
    except subprocess.TimeoutExpired:
        return -1, f"{Path(cmd[0]).name} timed out after {timeout:g}s"
    except (OSError, subprocess.SubprocessError) as exc:
        return -1, f"{Path(cmd[0]).name} failed to run: {exc}"


def _file_stem(identifier: str) -> str:
    """Filesystem-safe stem for work-dir filenames derived from a receptor id or
    a local file path (an absolute path must never be joined onto work_dir).

    A name with no meaningful stem — empty or only dots/underscores after
    stripping a trailing extension — falls back to ``"receptor"``. Dotfile
    handling is explicit here (``Path.stem`` changed dotfile behaviour in 3.14).
    """
    name = Path(identifier).name
    base, dot, _ext = name.rpartition(".")
    if not dot:
        base = name  # no extension at all
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", base).lstrip("._")
    return stem or "receptor"

# Heterogens that are NOT the co-crystal drug: water, monatomic ions/metals,
# carbohydrates (glycans), and common crystallisation buffers / lipids /
# detergents / cryoprotectants. Excluded from reference-ligand selection so a
# glycosylated receptor (e.g. GABA-A pentamers) does not pick a mannose/NAG
# glycan chain as the redocking ligand instead of the drug molecule.
_NON_DRUG_HETATM = {
    # water
    "HOH", "WAT", "H2O", "DOD", "SOL", "TIP", "TIP3",
    # monatomic ions / metals
    "NA", "CL", "MG", "CA", "K", "ZN", "MN", "FE", "CU", "CO", "NI", "CD", "HG",
    "BR", "CS", "RB", "SR", "BA", "LI", "F", "IOD",
    # carbohydrates / glycans
    "NAG", "NGA", "NDG", "BMA", "MAN", "FUC", "GAL", "GLC", "GLA", "SIA", "XYS",
    "XYP", "FUL", "RAM", "RIB", "BDP", "GLO", "MUR", "A2G", "BGC", "BGLC", "BMA",
    # lipids / detergents / buffers / cryoprotectants
    "LMT", "DDM", "CHS", "OLC", "POV", "PEE", "LMU", "LDA", "PLC", "PE", "PC",
    "GOL", "EDO", "PEG", "PGE", "TRS", "MES", "EPE", "SO4", "PO4", "ACT", "EOH",
    "DMS", "MPD", "HEZ", "SPM", "SPD", "CIT", "FMT", "ACY", "NO3",
}


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
        # content-addressed cache of prepared receptors (PDBFixer + Open Babel
        # output); set by the agent when the compute cache is enabled
        self.receptor_cache_dir: Path | None = None

    @cached_property
    def version(self) -> str:
        # probed once per engine instance: the docking cache key reads this for
        # every molecule, and shelling out to `vina --version` each time is waste
        ok, msg = self._probe_version()
        return msg if ok else "unknown"

    @cached_property
    def prep_versions(self) -> dict[str, str]:
        """Versions of the preparation toolchain (part of every cache key)."""
        return {name: _pkg_version(name) for name in _PREP_PACKAGES}

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

        if source.kind == "pdb_id" and not is_valid_pdb_id(source.identifier):
            rec.note = f"invalid PDB id {source.identifier!r} (expected 4 alphanumeric characters)"
            return rec
        url = source.url or (_RCSB_URL.format(pid=source.identifier) if source.kind == "pdb_id" else None)
        stem = _file_stem(source.identifier)
        if source.kind == "file":
            raw_path = Path(source.identifier)
            raw_text = raw_path.read_text()
        else:
            if url is None:
                rec.note = f"no receptor URL for source kind '{source.kind}'"
                return rec
            # only plain RCSB downloads are cached, keyed by their validated PDB id
            # (a custom URL must not populate or read the shared cache)
            cache_path = (pdb_cache_path(source.identifier)
                          if source.kind == "pdb_id" and not source.url and is_safe_name(source.identifier)
                          else None)
            if (cache_path is not None and cache_path.exists()
                    and looks_like_pdb(cache_path.read_bytes())):
                raw_path = cache_path  # already downloaded once
            else:
                raw_path = work_dir / f"{stem}_raw.pdb"
                err = download_pdb(url, raw_path)
                if err:
                    rec.note = f"could not download receptor ({err})"
                    return rec
                if cache_path is not None:
                    try:
                        atomic_write_bytes(cache_path, raw_path.read_bytes())
                    except OSError:
                        pass  # cache is best-effort, never blocks docking
            raw_text = raw_path.read_text()

        rec.identifier = source.identifier

        # structural QC on the raw structure (cheap; always recomputed)
        stats = analyze_pdb(raw_text)
        rec.qc_checks = self._qc_checks(stats)
        rec.atom_count = stats.atom_count
        rec.residue_count = stats.residue_count
        rec.chain_count = len(stats.chain_ids)

        # the expensive part (reference-ligand extraction, PDBFixer, Open Babel)
        # is content-addressed on the raw structure + recipe + tool versions
        prep_key = self._receptor_prep_key(raw_text)
        if self._load_prepared_receptor(rec, prep_key, work_dir, stem):
            return rec

        # co-crystal ligand (largest non-water heterogen) for redocking
        ref_name, ref_mol = self._extract_reference_ligand(raw_text, work_dir)
        rec.reference_ligand_name = ref_name
        rec.reference_ligand_mol = ref_mol

        # clean + protonate, then rigid PDBQT
        try:
            from openmm.app import PDBFile  # noqa: PLC0415
            from pdbfixer import PDBFixer  # noqa: PLC0415

            fixer = PDBFixer(filename=str(raw_path))
            fixer.findMissingResidues()
            fixer.findNonstandardResidues()
            fixer.replaceNonstandardResidues()
            fixer.removeHeterogens(keepWater=False)
            fixer.findMissingAtoms()
            fixer.addMissingAtoms()
            fixer.addMissingHydrogens(7.0)
            clean_pdb = work_dir / f"{stem}_receptor.pdb"
            with open(clean_pdb, "w") as fh:
                PDBFile.writeFile(fixer.topology, fixer.positions, fh)
        except Exception as exc:  # noqa: BLE001
            rec.preparation_tool = "pdbfixer failed"
            rec.note = f"receptor preparation failed: {exc}"
            return rec

        recpdbqt = work_dir / f"{stem}_receptor.pdbqt"
        rc, stderr = _run_tool(
            [str(self.obabel_path), str(clean_pdb), "-O", str(recpdbqt), "-xr"], OBABEL_TIMEOUT_S,
        )
        if rc != 0 or not recpdbqt.exists():
            rec.preparation_tool = "pdbfixer + openbabel"
            rec.note = f"PDB->PDBQT conversion failed: {stderr.strip()}"
            return rec

        rec.pdbqt_path = recpdbqt
        rec.content_hash = hash_content(recpdbqt.read_bytes())
        rec.preparation_tool = (
            f"pdbfixer {_pkg_version('pdbfixer')} + openbabel-wheel {_pkg_version('openbabel-wheel')} "
            f"(rigid PDBQT)"
        )
        rec.note = "receptor prepared and protonated"
        self._save_prepared_receptor(rec, prep_key)
        return rec

    # ---- prepared-receptor cache ----
    def _receptor_prep_key(self, raw_text: str) -> str:
        return hash_jsonable({
            "raw_pdb_sha256": hash_content(raw_text),
            "recipe": _RECEPTOR_RECIPE_VERSION,
            "ph": 7.0,
            "obabel_flags": ["-xr"],
            "prep_versions": self.prep_versions,
        })

    def _load_prepared_receptor(self, rec: PreparedReceptor, key: str, work_dir: Path,
                                stem: str) -> bool:
        """Fill ``rec`` from a cached preparation. A hit is transparent: the
        PDBQT is copied into this run's work_dir and every field matches a fresh
        preparation, so outputs are identical with or without the cache."""
        if self.receptor_cache_dir is None:
            return False
        d = self.receptor_cache_dir / key
        try:
            meta = json.loads((d / "meta.json").read_text())
            pdbqt = (d / "receptor.pdbqt").read_bytes()
        except (OSError, json.JSONDecodeError):
            return False  # miss, or a partial / corrupt entry
        if hash_content(pdbqt) != meta.get("pdbqt_sha256"):
            return False
        ref_mol = None
        if meta.get("reference_molblock"):
            try:
                from rdkit import Chem  # noqa: PLC0415
                ref_mol = Chem.MolFromMolBlock(meta["reference_molblock"], removeHs=True, sanitize=True)
            except ImportError:
                return False
            if ref_mol is None:
                return False
        out = work_dir / f"{stem}_receptor.pdbqt"
        atomic_write_bytes(out, pdbqt)
        rec.pdbqt_path = out
        rec.content_hash = meta["pdbqt_sha256"]
        rec.reference_ligand_name = meta.get("reference_ligand_name")
        rec.reference_ligand_mol = ref_mol
        rec.preparation_tool = meta.get("preparation_tool", "")
        rec.note = meta.get("note", "receptor prepared and protonated")
        return True

    def _save_prepared_receptor(self, rec: PreparedReceptor, key: str) -> None:
        if self.receptor_cache_dir is None or rec.pdbqt_path is None:
            return
        molblock = None
        if rec.reference_ligand_mol is not None:
            from rdkit import Chem  # noqa: PLC0415
            molblock = Chem.MolToMolBlock(rec.reference_ligand_mol)
        d = self.receptor_cache_dir / key
        try:
            atomic_write_bytes(d / "receptor.pdbqt", rec.pdbqt_path.read_bytes())
            # meta.json last: its presence marks a complete entry
            atomic_write_text(d / "meta.json", json.dumps({
                "pdbqt_sha256": rec.content_hash,
                "reference_ligand_name": rec.reference_ligand_name,
                "reference_molblock": molblock,
                "preparation_tool": rec.preparation_tool,
                "note": rec.note,
            }))
        except OSError:
            pass  # the cache is best-effort, never blocks docking

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
        """Extract the largest non-water, non-glycan HETATM residue as an RDKit mol.

        Grouping is per INDIVIDUAL residue (chain + sequence number + insertion
        code), not per (resname, chain) — a glycan chain of many same-named
        sugar residues must not merge into one pseudo-ligand that outsizes the
        drug, and sugars/ions/buffers are excluded outright (``_NON_DRUG_HETATM``).
        """
        try:
            from rdkit import Chem  # noqa: PLC0415
        except ImportError:
            return None, None

        residues: dict[tuple[str, str, str, str], list[str]] = {}
        for line in raw_text.splitlines():
            if not line.startswith("HETATM"):
                continue
            resname = line[17:20].strip()
            if not resname or resname in _NON_DRUG_HETATM:
                continue
            chain = line[21:22].strip() or " "
            resseq = line[22:26].strip()
            icode = line[26:27].strip() or " "
            residues.setdefault((resname, chain, resseq, icode), []).append(line)

        if not residues:
            return None, None
        (name, _chain, _resseq, _icode), lines = max(residues.items(), key=lambda kv: len(kv[1]))
        het_pdb = work_dir / f"reference_{name}.pdb"
        het_pdb.write_text("\n".join(lines) + "\nEND\n")
        sdf = work_dir / f"reference_{name}.sdf"
        rc, _stderr = _run_tool([str(self.obabel_path), str(het_pdb), "-O", str(sdf)], OBABEL_TIMEOUT_S)
        if rc != 0 or not sdf.exists():
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
            (line.split("SMILES ", 1)[1] for line in pdbqt.splitlines() if line.startswith("REMARK SMILES ")),
            None,
        )
        lig.canonical_smiles = smiles_line or ""
        lig.smiles_idx_map = parse_smiles_idx(pdbqt)
        lig.heavy_atom_count = len(lig.smiles_idx_map)
        lig.pdbqt_path = work_dir / "ligand.pdbqt"
        lig.pdbqt_path.write_text(pdbqt)
        lig.content_hash = hash_content(pdbqt)
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
        if protocol.cpu:
            cmd += ["--cpu", str(protocol.cpu)]
        rc, _stderr = _run_tool(cmd, VINA_TIMEOUT_S)
        if rc != 0 or not out.exists():
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
