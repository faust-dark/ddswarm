"""OpenMMMdEngine: real receptor MD (PDBFixer prep + amber14 + OBC2 GBSA
implicit solvent, Langevin NVT, fixed seed). Reports backbone RMSD (stability)
and potential-energy mean/std (fluctuation) — never binding free energies.

Supports trajectory checkpoint/resume (Increment 28): the final OpenMM State is
written to `checkpoint_path` when given, and a prior state can be loaded via
`resume_state` so a longer simulation continues from where a previous run
stopped instead of restarting. `loadState` restores positions, velocities and
the integrator RNG, so the continuation is deterministic GIVEN the checkpoint.
The minimization reference (backbone coords) is saved alongside the State and
restored on resume, so the RMSD is always measured against the same minimized
structure across the whole extended trajectory. NOTE: the base MD itself is
chaotic and not bit-reproducible across *fresh* runs (minimizeEnergy + thread
scheduling differ by ~1e-1 kJ/mol and diverge exponentially) — a pre-existing
property of MD, not a regression introduced by resume.

OpenMM + PDBFixer are optional (the `[docking]` extra); import is lazy so the
module never hard-crashes when they are absent."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..contracts.dynamics_result import MdProtocol, MdRun
from .base import MdEngine, REFERENCE_SUFFIX
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


def _save_reference(checkpoint_path: Path, ref) -> None:
    np.save(str(checkpoint_path) + REFERENCE_SUFFIX, ref)


def _load_reference(state_path: Path):
    p = Path(str(state_path) + REFERENCE_SUFFIX)
    return np.load(p) if p.exists() else None


class OpenMMMdEngine(MdEngine):
    name = "openmm"
    version = _OPENMM_VERSION

    def run_md(self, pdb_path: Path, receptor_id: str, protocol: MdProtocol,
               resume_state: Path | None = None, prior_steps: int = 0,
               checkpoint_path: Path | None = None) -> MdRun:
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
            return self._run(pdb_path, receptor_id, protocol, resume_state, prior_steps, checkpoint_path)
        except Exception as exc:  # noqa: BLE001 — report honestly, never fabricate
            return MdRun(receptor_id=receptor_id, status="failed",
                         error=f"{type(exc).__name__}: {exc}",
                         prior_steps=prior_steps, total_steps=prior_steps)

    def _run(self, pdb_path: Path, receptor_id: str, protocol: MdProtocol,
             resume_state: Path | None, prior_steps: int,
             checkpoint_path: Path | None) -> MdRun:
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

        # 3. backbone atom indices (deterministic: PDBFixer prep is reproducible)
        backbone = [a.index for a in fixer.topology.atoms() if a.name in _BACKBONE_ATOMS]
        if not backbone:
            return MdRun(receptor_id=receptor_id, status="failed",
                         error="no backbone (N/CA/C) atoms found",
                         prior_steps=prior_steps, total_steps=prior_steps)

        # 4. minimization reference (backbone coords). On resume we restore the
        #    SAVED reference so the RMSD is measured against the same minimized
        #    structure across the whole extended trajectory; otherwise we use the
        #    freshly-minimized coords.
        state = sim.context.getState(getPositions=True)
        ref = state.getPositions(asNumpy=True).value_in_unit(nanometer)
        if resume_state is not None:
            sim.loadState(str(resume_state))
            saved_ref = _load_reference(resume_state)
            if saved_ref is not None:
                ref = saved_ref

        # 5. short NVT trajectory
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

        # 6. save the final State + minimization reference for a future resume
        if checkpoint_path is not None:
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            sim.saveState(str(checkpoint_path))
            _save_reference(checkpoint_path, ref)

        e = np.array(energies)
        r = np.array(rmsds)
        rmsd_mean = float(r.mean())
        rmsd_max = float(r.max())

        total_steps = prior_steps + steps
        stability_note = (
            f"backbone RMSD {rmsd_mean:.2f} A (max {rmsd_max:.2f} A) over a "
            f"{steps * protocol.timestep_femtoseconds / 1000:.1f} ps relaxation; "
            "short snapshot, not a converged stability assessment"
        )
        if prior_steps:
            stability_note += (
                f"; continued from a {prior_steps * protocol.timestep_femtoseconds / 1000:.1f} ps "
                f"checkpoint (cumulative {total_steps * protocol.timestep_femtoseconds / 1000:.1f} ps); "
                "metrics reported over the continued segment only"
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
            prior_steps=prior_steps,
            total_steps=total_steps,
        )
