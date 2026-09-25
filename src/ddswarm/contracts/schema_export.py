"""Export JSON Schemas for every contract (single source of truth)."""

from __future__ import annotations

import json
from pathlib import Path

from .agent_result import AgentResult
from .design_batch import DesignBatch
from .event_log import EventLog
from .knowledge_pack import KnowledgePack
from .project_spec import ProjectSpec
from .run_manifest import RunManifest
from .task_spec import TaskSpec

MODELS: dict[str, type] = {
    "ProjectSpec": ProjectSpec,
    "AgentResult": AgentResult,
    "RunManifest": RunManifest,
    "TaskSpec": TaskSpec,
    "EventLog": EventLog,
    "KnowledgePack": KnowledgePack,
    "DesignBatch": DesignBatch,
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
