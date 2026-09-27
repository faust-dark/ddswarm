"""CypAssessment artifact (cyp_minion output) — CYP450 DDI flags.

Deterministic structural-alert screening for CYP inhibition and CYP3A4
induction, on the same honesty standard as the tox toxicophore library: a
SMARTS match is a fact (the molecule contains a documented chemotype) but the
label is "potential inhibitor/inducer" — predicted, never a functional assay.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .enums import AgentResultStatus, EvidenceLevel
from .provenance import ProvenanceEntry

CypEnzyme = Literal["CYP3A4", "CYP2D6", "CYP2C9", "CYP2C19", "CYP1A2"]
CypEffect = Literal["inhibition", "induction"]


class CypLiability(BaseModel):
    enzyme: CypEnzyme
    effect: CypEffect
    alert_slug: str
    citation: str
    evidence_level: EvidenceLevel = EvidenceLevel.predicted
    note: str | None = None


class MoleculeCypProfile(BaseModel):
    molecule_id: str
    canonical_smiles: str | None = None
    liabilities: list[CypLiability] = Field(default_factory=list)
    evidence_level: EvidenceLevel = EvidenceLevel.predicted
    note: str | None = None


class CypSummary(BaseModel):
    total_molecules: int = 0
    flagged_inhibition: int = 0    # molecules with >=1 inhibition alert
    flagged_induction: int = 0     # molecules with >=1 induction alert
    total_alerts: int = 0


class CypAssessment(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    project_id: str
    agent: str = "cyp_minion"
    status: AgentResultStatus
    molecules: list[MoleculeCypProfile] = Field(default_factory=list)
    summary: CypSummary = Field(default_factory=CypSummary)
    note: str | None = None
    provenance: list[ProvenanceEntry] = Field(default_factory=list)
