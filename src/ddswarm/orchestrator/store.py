"""Versioned on-disk store for project specs, run manifests, task ledgers,
agent results, and artifacts. Nothing is overwritten without a new version."""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel

from ..contracts.agent_result import AgentResult
from ..contracts.project_spec import ProjectSpec
from ..contracts.run_manifest import RunManifest
from ..contracts.task_spec import TaskSpec
from ..logging.hashing import hash_content

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _check_id(value: str, what: str) -> None:
    """Reject identifiers that could escape the store root via a filesystem
    path (empty, slashes, '..', control chars)."""
    if not isinstance(value, str) or not _ID_RE.match(value):
        raise ValueError(f"invalid {what} for filesystem path: {value!r}")


class Store:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    # ---- paths ----
    def project_dir(self, project_id: str) -> Path:
        _check_id(project_id, "project_id")
        return self.root / project_id

    def project_spec_path(self, project_id: str) -> Path:
        return self.project_dir(project_id) / "project_spec.json"

    def run_dir(self, project_id: str, run_id: str) -> Path:
        _check_id(run_id, "run_id")
        return self.project_dir(project_id) / "runs" / run_id

    def run_manifest_path(self, project_id: str, run_id: str) -> Path:
        return self.run_dir(project_id, run_id) / "run_manifest.json"

    def tasks_dir(self, project_id: str, run_id: str) -> Path:
        return self.run_dir(project_id, run_id) / "tasks"

    def artifacts_dir(self, project_id: str, run_id: str) -> Path:
        return self.run_dir(project_id, run_id) / "artifacts"

    def events_path(self, project_id: str, run_id: str) -> Path:
        return self.run_dir(project_id, run_id) / "events.jsonl"

    # ---- project spec ----
    def save_project_spec(self, spec: ProjectSpec) -> Path:
        p = self.project_spec_path(spec.project_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(spec.model_dump_json(indent=2))
        return p

    def load_project_spec(self, project_id: str) -> ProjectSpec:
        return ProjectSpec.model_validate_json(self.project_spec_path(project_id).read_text())

    # ---- run manifest ----
    def save_run_manifest(self, manifest: RunManifest) -> Path:
        p = self.run_manifest_path(manifest.project_id, manifest.run_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(manifest.model_dump_json(indent=2))
        return p

    def load_run_manifest(self, project_id: str, run_id: str) -> RunManifest:
        return RunManifest.model_validate_json(self.run_manifest_path(project_id, run_id).read_text())

    # ---- task ledger ----
    def save_task_spec(self, spec: TaskSpec) -> Path:
        _check_id(spec.task_id, "task_id")
        p = self.tasks_dir(spec.project_id, spec.run_id) / f"{spec.task_id}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(spec.model_dump_json(indent=2))
        return p

    def load_task_spec(self, project_id: str, run_id: str, task_id: str) -> TaskSpec:
        _check_id(task_id, "task_id")
        p = self.tasks_dir(project_id, run_id) / f"{task_id}.json"
        return TaskSpec.model_validate_json(p.read_text())

    # ---- agent results (versioned per attempt) ----
    def save_agent_result(self, result: AgentResult, run_id: str, attempt: int) -> Path:
        _check_id(result.task_id, "task_id")
        p = self.tasks_dir(result.project_id, run_id) / f"{result.task_id}.result.v{attempt}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(result.model_dump_json(indent=2))
        return p

    def load_agent_result(self, project_id: str, run_id: str, task_id: str, attempt: int) -> AgentResult:
        _check_id(task_id, "task_id")
        p = self.tasks_dir(project_id, run_id) / f"{task_id}.result.v{attempt}.json"
        return AgentResult.model_validate_json(p.read_text())

    def latest_attempt(self, project_id: str, run_id: str, task_id: str) -> int:
        _check_id(task_id, "task_id")
        d = self.tasks_dir(project_id, run_id)
        cands = list(d.glob(f"{task_id}.result.v*.json"))
        if not cands:
            return 0
        return max(int(c.stem.rsplit("v", 1)[1]) for c in cands)

    def load_latest_agent_result(self, project_id: str, run_id: str, task_id: str) -> AgentResult:
        n = self.latest_attempt(project_id, run_id, task_id)
        if n == 0:
            raise FileNotFoundError(f"no result for task {task_id}")
        return self.load_agent_result(project_id, run_id, task_id, n)

    # ---- artifacts (auto-versioned, never overwritten) ----
    def save_artifact(
        self, project_id: str, run_id: str, artifact: BaseModel, kind: str, artifact_id: str
    ) -> tuple[Path, int, str]:
        _check_id(artifact_id, "artifact_id")
        _check_id(kind, "kind")
        d = self.artifacts_dir(project_id, run_id)
        d.mkdir(parents=True, exist_ok=True)
        existing = list(d.glob(f"{artifact_id}.{kind}.v*.json"))
        version = len(existing) + 1
        p = d / f"{artifact_id}.{kind}.v{version}.json"
        content = artifact.model_dump_json(indent=2)
        p.write_text(content)
        h = hash_content(content)
        return p, version, h

    def load_artifact(self, uri: str, model: type[BaseModel]) -> BaseModel:
        return model.model_validate_json(Path(uri).read_text())

    def save_binary_artifact(
        self, project_id: str, run_id: str, artifact_id: str, kind: str, suffix: str, data: bytes
    ) -> tuple[Path, int, str]:
        """Persist a non-JSON artifact (e.g. a PDF) next to the JSON artifacts,
        versioned the same way and never overwritten. `suffix` is the file
        extension without a leading dot (e.g. 'pdf')."""
        _check_id(artifact_id, "artifact_id")
        _check_id(kind, "kind")
        d = self.artifacts_dir(project_id, run_id)
        d.mkdir(parents=True, exist_ok=True)
        existing = list(d.glob(f"{artifact_id}.{kind}.v*.{suffix}"))
        version = len(existing) + 1
        p = d / f"{artifact_id}.{kind}.v{version}.{suffix}"
        p.write_bytes(data)
        h = hash_content(data)
        return p, version, h
