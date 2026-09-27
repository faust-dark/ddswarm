"""DynamicsResult artifact (dynamics_minion output).

Receptor molecular dynamics (OpenMM + amber14 + GBSA implicit solvent): a
short relaxation trajectory, NOT production MD. Backbone RMSD (stability) and
potential-energy mean/std (fluctuation) are *calculated*; a 2 ps snapshot is
never interpreted as a converged stability assessment."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .enums import AgentResultStatus, EvidenceLevel
from .provenance import ProvenanceEntry

MdRunStatus = Literal["success", "failed", "not_run"]


class MdProtocol(BaseModel):
    engine: str = "openmm"
    engine_version: str | None = None
    forcefield: str = "amber14 + OBC2 GBSA implicit solvent"
    integrator: str = "Langevin NVT"
    temperature_kelvin: float = 300.0
    timestep_femtoseconds: float = 2.0
    steps: int = 1000
    seed: int = 42
    minimize_max_iterations: int = 200
    note: str = "short relaxation trajectory; NOT production MD"


class MdMetrics(BaseModel):
    backbone_rmsd_mean_angstrom: float | None = None
    backbone_rmsd_max_angstrom: float | None = None
    potential_energy_mean_kj_per_mol: float | None = None
    potential_energy_std_kj_per_mol: float | None = None


class MdRun(BaseModel):
    receptor_id: str
    status: MdRunStatus = "not_run"
    error: str | None = None
    metrics: MdMetrics = Field(default_factory=MdMetrics)
    stability_note: str | None = None
    prior_steps: int = 0          # steps simulated in prior runs (loaded from a checkpoint)
    total_steps: int = 0          # prior_steps + this run's steps (cumulative simulation length)
    checkpoint_uri: str | None = None   # store URI of the saved final state (for a future resume)
    checkpoint_hash: str | None = None  # anti-tamper hash of the saved state XML
    evidence_level: EvidenceLevel = EvidenceLevel.calculated


class DynamicsSummary(BaseModel):
    total_runs: int = 0
    succeeded: int = 0
    failed: int = 0


class DynamicsResult(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    project_id: str
    agent: str = "dynamics_minion"
    status: AgentResultStatus
    protocol: MdProtocol = Field(default_factory=MdProtocol)
    runs: list[MdRun] = Field(default_factory=list)
    summary: DynamicsSummary = Field(default_factory=DynamicsSummary)
    provenance: list[ProvenanceEntry] = Field(default_factory=list)
