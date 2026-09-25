"""OpenMMMdEngine: real receptor MD (PDBFixer prep + amber14 + OBC2 GBSA
implicit solvent, Langevin NVT, fixed seed). Reports backbone RMSD (stability)
and potential-energy mean/std (fluctuation) — never binding free energies.

OpenMM + PDBFixer are optional (the `[docking]` extra); import is lazy so the
module never hard-crashes when they are absent."""

from __future__ import annotations

from pathlib import Path

from ..contracts.dynamics_result import MdProtocol, MdRun
from .base import MdEngine
from .rmsd import kabsch_rmsd

try:
    from openmm import LangevinIntegrator, Platform
    from openmm.app import ForceField, NoCutoff, Simulation
    from openmm.unit import femtosecond, kelvin, kilojoule_per_mole, nanometer, picosecond
    from pdbfixer import PDBFixer

    _OPENMM_AVAILABLE = True
    try:
        import openmm as _mm
        _OPENMM_VERSION = _mm.version.full_version
    except Exception:  # noqa: BLE001 — version string is cosmetic
        _OPENMM_VERSION = None
except ImportError:
    _OPENMM_AVAILABLE = False
    _OPENMM_VERSION = None

_BACKBONE_ATOMS = ("N", "CA", "C")


class OpenMMMdEngine(MdEngine):
    name = "openmm"
    version = _OPENMM_VERSION

    def run_md(self, pdb_path: Path, receptor_id: str, protocol: MdProtocol) -> MdRun:
        if not _OPENMM_AVAILABLE:
            return MdRun(
                receptor_id=receptor_id,
                status="failed",
                error="OpenMM/PDBFixer not installed (pip install 'ddswarm[docking]')",
            )
        pdb_path = Path(pdb_path)
        if not pdb_path.exists():
            return MdRun(receptor_id=receptor_id, status="failed", error=f"PDB not found: {pdb_path}")

        try:
            return self._run(pdb_path, receptor_id, protocol)
        except Exception as exc:  # noqa: BLE001 — report honestly, never fabricate
            return MdRun(receptor_id=receptor_id, status="failed", error=f"{type(exc).__name__}: {exc}")

    def _run(self, pdb_path: Path, receptor_id: str, protocol: MdProtocol) -> MdRun:
        # 1. PDBFixer prep (same receptor QC as docking_minion)
        fixer = PDBFixer(filename=str(pdb_path))
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)
        fixer.removeHeterogens(keepWater=False)

        # 2. amber14 + OBC2 GBSA, Langevin NVT
        ff = ForceField("amber14-all.xml", "implicit/obc2.xml")
        system = ff.createSystem(fixer.topology, nonbondedMethod=NoCutoff)
        integrator = LangevinIntegrator(
            protocol.temperature_kelvin * kelvin,
            1 / picosecond,
            protocol.timestep_femtoseconds * femtosecond,
        )
        integrator.setRandomNumberSeed(protocol.seed)
        sim = Simulation(fixer.topology, system, integrator, Platform.getPlatformByName("CPU"))
        sim.context.setPositions(fixer.positions)
        sim.minimizeEnergy(maxIterations=protocol.minimize_max_iterations)

        # 3. reference (minimized) backbone coordinates
        state = sim.context.getState(getPositions=True)
        ref = state.getPositions(asNumpy=True).value_in_unit(nanometer)
        backbone = [a.index for a in fixer.topology.atoms() if a.name in _BACKBONE_ATOMS]
        if not backbone:
            return MdRun(receptor_id=receptor_id, status="failed", error="no backbone (N/CA/C) atoms found")

        # 4. short NVT trajectory
        energies = []
        rmsds = []
        steps = max(1, protocol.steps)
        sample_every = max(1, steps // 40) if steps >= 40 else 1
        remaining = steps
        while remaining > 0:
            chunk = min(sample_every, remaining)
            sim.step(chunk)
            remaining -= chunk
            st = sim.context.getState(getEnergy=True, getPositions=True)
            energies.append(st.getPotentialEnergy().value_in_unit(kilojoule_per_mole))
            pos = st.getPositions(asNumpy=True).value_in_unit(nanometer)
            rmsds.append(kabsch_rmsd(ref[backbone], pos[backbone]) * 10.0)  # nm -> A

        import numpy as np

        e = np.array(energies)
        r = np.array(rmsds)
        rmsd_mean = float(r.mean())
        rmsd_max = float(r.max())

        stability_note = (
            f"backbone RMSD {rmsd_mean:.2f} A (max {rmsd_max:.2f} A) over a "
            f"{steps * protocol.timestep_femtoseconds / 1000:.1f} ps relaxation; "
            "short snapshot, not a converged stability assessment"
        )

        from ..contracts.dynamics_result import MdMetrics

        return MdRun(
            receptor_id=receptor_id,
            status="success",
            metrics=MdMetrics(
                backbone_rmsd_mean_angstrom=round(rmsd_mean, 3),
                backbone_rmsd_max_angstrom=round(rmsd_max, 3),
                potential_energy_mean_kj_per_mol=round(float(e.mean()), 1),
                potential_energy_std_kj_per_mol=round(float(e.std()), 1),
            ),
            stability_note=stability_note,
        )
