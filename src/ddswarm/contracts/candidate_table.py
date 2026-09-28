"""CandidateTable artifact (export_minion output) — machine-readable candidate export.

One flat row per screened molecule with every computed/predicted score and a
deterministic go/no-go verdict. The verdict reuses the pipeline's existing gates
— it invents no new thresholds:

  no-go   eliminated at screening (PAINS/Ro5/invalid) OR a safety block flag
  review  passed screening but has >=1 advisory liability (tox, CYP, hERG,
          Brenk, BCS IV)
  go      passed screening, no safety flags, no advisories

The verdict is a ranking/decision aid, not an experimental claim.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .enums import AgentResultStatus
from .provenance import ProvenanceEntry

Verdict = Literal["go", "review", "no-go"]


class CandidateRow(BaseModel):
    rank: int | None = None          # None for screening rejects
    molecule_id: str
    input_smiles: str
    canonical_smiles: str | None = None
    screening_status: Literal["passed", "rejected", "invalid"]
    molecular_weight: float | None = None
    logp: float | None = None
    hbd: int | None = None
    hba: int | None = None
    tpsa: float | None = None
    qed: float | None = None
    ro5_violations: int = 0
    bcs_class: str | None = None
    sa_score: float | None = None
    pains_alerts: int = 0
    brenk_alerts: int = 0
    tox_alerts: int = 0
    cyp_flags: list[str] = Field(default_factory=list)
    herg_flag: bool = False
    safety_flags: int = 0
    verdict: Verdict
    reasons: list[str] = Field(default_factory=list)
    # IUPAC readability label (NIH CACTUS, network-fetched) for the ranked
    # survivors — None for screening rejects. Non-deterministic, so it is
    # excluded from the export's reproducibility hash.
    iupac_name: str | None = None


class ExportSummary(BaseModel):
    total_molecules: int = 0
    go: int = 0
    review: int = 0
    no_go: int = 0


class CandidateTable(BaseModel):
    schema_version: str = "1.0"
    project_id: str
    agent: str = "export_minion"
    status: AgentResultStatus
    rows: list[CandidateRow] = Field(default_factory=list)
    summary: ExportSummary = Field(default_factory=ExportSummary)
    note: str | None = None
    provenance: list[ProvenanceEntry] = Field(default_factory=list)
