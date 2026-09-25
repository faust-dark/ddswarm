"""Export JSON Schemas for every contract (single source of truth)."""

from __future__ import annotations

import json
from pathlib import Path

from .admet_prediction import AdmetPrediction
from .agent_result import AgentResult
from .design_batch import DesignBatch
from .docking_result import DockingResult
from .dynamics_result import DynamicsResult
from .event_log import EventLog
from .formulation_assessment import FormulationAssessment
from .knowledge_pack import KnowledgePack
from .memory import ProjectMemory
from .project_spec import ProjectSpec
from .report import Report
from .run_manifest import RunManifest
from .run_request import RunRequest
from .retrosynthesis_plan import RetrosynthesisPlan
from .screened_batch import ScreenedBatch
from .task_spec import TaskSpec
from .tox_profile import ToxProfile

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
    "RunRequest": RunRequest,
    "RetrosynthesisPlan": RetrosynthesisPlan,
    "ToxProfile": ToxProfile,
    "Report": Report,
    "ProjectMemory": ProjectMemory,
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
