"""Export JSON Schemas for every contract (single source of truth)."""

from __future__ import annotations

import json
from pathlib import Path

from .admet_prediction import AdmetPrediction
from .agent_result import AgentResult
from .clinical_assessment import ClinicalAssessment
from .clinical_simulation import ClinicalSimulation
from .design_batch import DesignBatch
from .docking_result import DockingResult
from .dynamics_result import DynamicsResult
from .event_log import EventLog
from .execution_plan import ExecutionPlan
from .formulation_assessment import FormulationAssessment
from .generation_memory import GenerationMemory
from .knowledge_pack import KnowledgePack
from .memory import ProjectMemory
from .patent_landscape import PatentLandscape
from .project_spec import ProjectSpec
from .regulatory_assessment import RegulatoryAssessment
from .regulatory_checklist import RegulatoryChecklist
from .report import Report
from .report_document import ReportDocument
from .run_manifest import RunManifest
from .run_request import RunRequest
from .retrosynthesis_plan import RetrosynthesisPlan
from .cyp_assessment import CypAssessment
from .herg_assessment import HergAssessment
from .grounding_report import GroundingReport
from .safety_assessment import SafetyAssessment
from .screened_batch import ScreenedBatch
from .selectivity_assessment import SelectivityAssessment
from .task_spec import TaskSpec
from .tox_profile import ToxProfile
from .validation_report import ValidationReport

MODELS: dict[str, type] = {
    "AdmetPrediction": AdmetPrediction,
    "ProjectSpec": ProjectSpec,
    "AgentResult": AgentResult,
    "RunManifest": RunManifest,
    "TaskSpec": TaskSpec,
    "EventLog": EventLog,
    "KnowledgePack": KnowledgePack,
    "DesignBatch": DesignBatch,
    "ScreenedBatch": ScreenedBatch,
    "DockingResult": DockingResult,
    "DynamicsResult": DynamicsResult,
    "FormulationAssessment": FormulationAssessment,
    "ExecutionPlan": ExecutionPlan,
    "GenerationMemory": GenerationMemory,
    "RunRequest": RunRequest,
    "RetrosynthesisPlan": RetrosynthesisPlan,
    "CypAssessment": CypAssessment,
    "HergAssessment": HergAssessment,
    "GroundingReport": GroundingReport,
    "ToxProfile": ToxProfile,
    "Report": Report,
    "ReportDocument": ReportDocument,
    "ProjectMemory": ProjectMemory,
    "PatentLandscape": PatentLandscape,
    "ClinicalAssessment": ClinicalAssessment,
    "ClinicalSimulation": ClinicalSimulation,
    "RegulatoryAssessment": RegulatoryAssessment,
    "RegulatoryChecklist": RegulatoryChecklist,
    "SafetyAssessment": SafetyAssessment,
    "SelectivityAssessment": SelectivityAssessment,
    "ValidationReport": ValidationReport,
}


def export_schemas(out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, model in MODELS.items():
        schema = model.model_json_schema()
        p = out / f"{name}.schema.json"
        p.write_text(json.dumps(schema, indent=2))
        written[name] = p
    return written
