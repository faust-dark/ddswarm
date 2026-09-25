"""Orchestrator: state machine, task graph execution, gating, retry, versioning."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..agents.base import Agent, AgentContext
from ..contracts.admet_prediction import AdmetPrediction
from ..contracts.agent_result import AgentResult
from ..contracts.design_batch import DesignBatch
from ..contracts.docking_result import DockingResult
from ..contracts.dynamics_result import DynamicsResult
from ..contracts.enums import AgentResultStatus, ProjectStatus, RunStatus, TaskStatus
from ..contracts.event_log import EventLog
from ..contracts.formulation_assessment import FormulationAssessment
from ..contracts.knowledge_pack import KnowledgePack
from ..contracts.project_spec import ProjectSpec
from ..contracts.retrosynthesis_plan import RetrosynthesisPlan
from ..contracts.report import Report
from ..contracts.screened_batch import ScreenedBatch
from ..contracts.provenance import InputRef, utcnow
from ..contracts.run_manifest import RunManifest
from ..contracts.task_spec import AttemptRecord, TaskSpec
from ..contracts.tox_profile import ToxProfile
from ..logging.event_writer import EventWriter
from .gates import BlockedByGate, gate_a_input_validity, result_ok
from .store import Store

ARTIFACT_MODELS: dict[str, type] = {
    "admet_prediction": AdmetPrediction,
    "docking_result": DockingResult,
    "dynamics_result": DynamicsResult,
    "formulation_assessment": FormulationAssessment,
    "knowledge_pack": KnowledgePack,
    "design_batch": DesignBatch,
    "screened_batch": ScreenedBatch,
    "retrosynthesis_plan": RetrosynthesisPlan,
    "tox_profile": ToxProfile,
    "report": Report,
}


class TaskDefinition(BaseModel):
    task_id: str
    agent: str
    depends_on: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    parent_task_id: str | None = None


class Runner:
    def __init__(self, store: Store, agents: dict[str, Agent]):
        self.store = store
        self.agents = agents

    @staticmethod
    def new_run_id() -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        return f"run_{stamp}_{uuid.uuid4().hex[:6]}"

    def _topo_order(self, task_graph: dict[str, list[str]]) -> list[str]:
        indeg = {t: 0 for t in task_graph}
        for t, deps in task_graph.items():
            for d in deps:
                if d not in indeg:
                    raise ValueError(f"dependency '{d}' of task '{t}' is not a declared task")
                indeg[t] += 1
        q = [t for t, d in indeg.items() if d == 0]
        order: list[str] = []
        while q:
            t = q.pop(0)
            order.append(t)
            for other, deps in task_graph.items():
                if t in deps:
                    indeg[other] -= 1
                    if indeg[other] == 0:
                        q.append(other)
        if len(order) != len(task_graph):
            raise ValueError("task graph contains a cycle")
        return order

    def _log(self, project_id: str, run_id: str, event: EventLog) -> None:
        EventWriter(self.store.events_path(project_id, run_id)).write(event)

    def create_run(
        self, spec: ProjectSpec, tasks: list[TaskDefinition], run_id: str | None = None
    ) -> RunManifest:
        problems = gate_a_input_validity(spec)
        if problems:
            raise BlockedByGate("Gate A (input validity) failed: " + "; ".join(problems))
        run_id = run_id or self.new_run_id()

        task_graph: dict[str, list[str]] = {}
        for t in tasks:
            if t.agent not in self.agents:
                raise ValueError(f"no agent registered for '{t.agent}'")
            task_graph[t.task_id] = list(t.depends_on)
        order = self._topo_order(task_graph)  # validates acyclic + known deps

        spec.status = ProjectStatus.running
        self.store.save_project_spec(spec)

        manifest = RunManifest(
            project_id=spec.project_id, run_id=run_id, task_graph=task_graph, tasks=order
        )
        self.store.save_run_manifest(manifest)
        for t in tasks:
            self.store.save_task_spec(TaskSpec(
                task_id=t.task_id,
                parent_task_id=t.parent_task_id,
                agent=t.agent,
                project_id=spec.project_id,
                run_id=run_id,
                params=t.params,
                status=TaskStatus.queued,
            ))
        self._log(spec.project_id, run_id, EventLog(run_id=run_id, level="info", event="run_created"))
        return manifest

    def run(self, run_id: str, project_id: str | None = None) -> RunManifest:
        if project_id is None:
            manifest = self._find_manifest(run_id)
            project_id = manifest.project_id
        else:
            manifest = self.store.load_run_manifest(project_id, run_id)

        spec = self.store.load_project_spec(project_id)
        manifest.status = RunStatus.running
        self.store.save_run_manifest(manifest)
        order = self._topo_order(manifest.task_graph)

        task_status: dict[str, TaskStatus] = {
            t: self.store.load_task_spec(project_id, run_id, t).status for t in order
        }

        for task_id in order:
            ts = self.store.load_task_spec(project_id, run_id, task_id)
            deps = manifest.task_graph.get(task_id, [])

            if ts.status in (TaskStatus.success, TaskStatus.partial):
                continue  # already completed (resume)
            if ts.status in (TaskStatus.failed, TaskStatus.blocked, TaskStatus.insufficient_evidence):
                continue  # already terminal-failed

            dep_failed = any(
                task_status.get(d) in (TaskStatus.failed, TaskStatus.blocked, TaskStatus.insufficient_evidence)
                for d in deps
            )
            if dep_failed:
                ts.status = TaskStatus.blocked
                self.store.save_task_spec(ts)
                task_status[task_id] = TaskStatus.blocked
                self._log(project_id, run_id, EventLog(
                    run_id=run_id, task_id=task_id, level="warning",
                    event="skipped: upstream dependency did not succeed",
                ))
                continue

            ts.status = TaskStatus.running
            ts.attempts.append(AttemptRecord(attempt=len(ts.attempts) + 1, started_at=utcnow()))
            self.store.save_task_spec(ts)

            ctx = self._build_context(project_id, run_id, task_id, spec, ts, deps)
            agent = self.agents[ts.agent]
            try:
                result = agent.run(ctx)
            except Exception as exc:  # noqa: BLE001
                result = AgentResult(
                    project_id=project_id, agent=ts.agent, task_id=task_id,
                    status=AgentResultStatus.failed,
                    errors=[f"{type(exc).__name__}: {exc}"],
                    reproducibility={"reason": "unhandled exception in agent"},
                )

            attempt = len(ts.attempts)
            self.store.save_agent_result(result, run_id, attempt)
            ts.status = TaskStatus(result.status.value)
            ts.attempts[-1].finished_at = utcnow()
            ts.attempts[-1].status = ts.status
            if result.errors:
                ts.attempts[-1].error = "; ".join(result.errors)
            self.store.save_task_spec(ts)
            task_status[task_id] = ts.status

            level = "info" if result_ok(result.status) else "warning"
            self._log(project_id, run_id, EventLog(
                run_id=run_id, task_id=task_id, level=level,
                event=f"task completed: {result.status.value}",
            ))

        manifest = self.store.load_run_manifest(project_id, run_id)
        manifest.status = self._final_status([task_status[t] for t in order])
        manifest.completed_at = utcnow()
        manifest.log_refs = [str(self.store.events_path(project_id, run_id))]
        self.store.save_run_manifest(manifest)
        return manifest

    def retry_task(self, project_id: str, run_id: str, task_id: str) -> AgentResult:
        ts = self.store.load_task_spec(project_id, run_id, task_id)
        if ts.status == TaskStatus.queued:
            raise ValueError(f"task {task_id} has not run yet; nothing to retry")
        spec = self.store.load_project_spec(project_id)
        manifest = self.store.load_run_manifest(project_id, run_id)

        ts.status = TaskStatus.running
        ts.retry_count += 1
        ts.attempts.append(AttemptRecord(attempt=len(ts.attempts) + 1, started_at=utcnow()))
        self.store.save_task_spec(ts)

        deps = manifest.task_graph.get(task_id, [])
        ctx = self._build_context(project_id, run_id, task_id, spec, ts, deps)
        agent = self.agents[ts.agent]
        try:
            result = agent.run(ctx)
        except Exception as exc:  # noqa: BLE001
            result = AgentResult(
                project_id=project_id, agent=ts.agent, task_id=task_id,
                status=AgentResultStatus.failed,
                errors=[f"{type(exc).__name__}: {exc}"],
            )

        attempt = len(ts.attempts)
        self.store.save_agent_result(result, run_id, attempt)
        ts.status = TaskStatus(result.status.value)
        ts.attempts[-1].finished_at = utcnow()
        ts.attempts[-1].status = ts.status
        self.store.save_task_spec(ts)
        self._log(project_id, run_id, EventLog(
            run_id=run_id, task_id=task_id, level="info",
            event=f"retry {ts.retry_count}: {result.status.value}",
        ))
        return result

    def _build_context(
        self, project_id: str, run_id: str, task_id: str, spec: ProjectSpec, ts: TaskSpec, deps: list[str]
    ) -> AgentContext:
        inputs: dict[str, list[Any]] = {}
        refs = []
        for d in deps:
            r = self._load_upstream(project_id, run_id, d)
            if r is None:
                continue
            for art in r.artifacts:
                if art.uri and art.kind in ARTIFACT_MODELS:
                    payload = ARTIFACT_MODELS[art.kind].model_validate_json(Path(art.uri).read_text())
                    inputs.setdefault(art.kind, []).append(payload)
                refs.append(InputRef(ref_type="artifact", ref_id=art.artifact_id, uri=art.uri, hash=art.hash))
        return AgentContext(
            project_id=project_id, run_id=run_id, task_id=task_id,
            spec=spec, params=ts.params, inputs=inputs, input_refs=refs,
        )

    def _load_upstream(self, project_id: str, run_id: str, task_id: str) -> AgentResult | None:
        n = self.store.latest_attempt(project_id, run_id, task_id)
        if n == 0:
            return None
        return self.store.load_agent_result(project_id, run_id, task_id, n)

    def _final_status(self, statuses: list[TaskStatus]) -> RunStatus:
        if any(s == TaskStatus.failed for s in statuses):
            return RunStatus.failed
        if any(s in (TaskStatus.blocked, TaskStatus.insufficient_evidence) for s in statuses):
            return RunStatus.blocked
        return RunStatus.completed

    def _find_manifest(self, run_id: str) -> RunManifest:
        for proj_dir in self.store.root.iterdir():
            if not proj_dir.is_dir():
                continue
            mpath = proj_dir / "runs" / run_id / "run_manifest.json"
            if mpath.exists():
                return RunManifest.model_validate_json(mpath.read_text())
        raise FileNotFoundError(f"no manifest found for run_id {run_id}")
