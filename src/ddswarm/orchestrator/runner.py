"""Orchestrator: state machine, task graph execution, gating, retry, versioning."""

from __future__ import annotations

import os
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..agents.base import Agent, AgentContext
from ..contracts.admet_prediction import AdmetPrediction
from ..contracts.agent_result import AgentResult
from ..contracts.clinical_assessment import ClinicalAssessment
from ..contracts.clinical_simulation import ClinicalSimulation
from ..contracts.design_batch import DesignBatch
from ..contracts.docking_result import DockingResult
from ..contracts.dynamics_result import DynamicsResult
from ..contracts.enums import AgentResultStatus, ProjectStatus, RunStatus, TaskStatus
from ..contracts.event_log import EventLog
from ..contracts.execution_plan import ExecutionPlan
from ..contracts.formulation_assessment import FormulationAssessment
from ..contracts.generation_memory import GenerationMemory
from ..contracts.knowledge_pack import KnowledgePack
from ..contracts.overview_report import OverviewReport
from ..contracts.patent_landscape import PatentLandscape
from ..contracts.project_spec import ProjectSpec
from ..contracts.regulatory_assessment import RegulatoryAssessment
from ..contracts.regulatory_checklist import RegulatoryChecklist
from ..contracts.retrosynthesis_plan import RetrosynthesisPlan
from ..contracts.report import Report
from ..contracts.report_document import ReportDocument
from ..contracts.cyp_assessment import CypAssessment
from ..contracts.candidate_table import CandidateTable
from ..contracts.herg_assessment import HergAssessment
from ..contracts.grounding_report import GroundingReport
from ..contracts.safety_assessment import SafetyAssessment
from ..contracts.screened_batch import ScreenedBatch
from ..contracts.selectivity_assessment import SelectivityAssessment
from ..contracts.provenance import InputRef, utcnow
from ..contracts.run_manifest import RunManifest
from ..contracts.task_spec import AttemptRecord, TaskSpec
from ..contracts.tox_profile import ToxProfile
from ..contracts.validation_report import ValidationReport
from ..logging import get_logger
from ..logging.event_writer import EventWriter
from .gates import BlockedByGate, gate_a_input_validity, result_ok
from .store import Store

logger = get_logger("runner")

ARTIFACT_MODELS: dict[str, type] = {
    "admet_prediction": AdmetPrediction,
    "docking_result": DockingResult,
    "dynamics_result": DynamicsResult,
    "formulation_assessment": FormulationAssessment,
    "knowledge_pack": KnowledgePack,
    "overview_report": OverviewReport,
    "design_batch": DesignBatch,
    "screened_batch": ScreenedBatch,
    "retrosynthesis_plan": RetrosynthesisPlan,
    "cyp_assessment": CypAssessment,
    "candidate_table": CandidateTable,
    "herg_assessment": HergAssessment,
    "grounding_report": GroundingReport,
    "tox_profile": ToxProfile,
    "report": Report,
    "report_document": ReportDocument,
    "patent_landscape": PatentLandscape,
    "clinical_assessment": ClinicalAssessment,
    "clinical_simulation": ClinicalSimulation,
    "regulatory_assessment": RegulatoryAssessment,
    "regulatory_checklist": RegulatoryChecklist,
    "safety_assessment": SafetyAssessment,
    "selectivity_assessment": SelectivityAssessment,
    "validation_report": ValidationReport,
    "execution_plan": ExecutionPlan,
    "generation_memory": GenerationMemory,
}


class TaskDefinition(BaseModel):
    task_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    agent: str
    depends_on: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    parent_task_id: str | None = None


def default_max_workers() -> int:
    return max(1, min(8, os.cpu_count() or 1))


class Runner:
    """Executes a run's task graph.

    Ready tasks (all dependencies committed) run concurrently on a thread pool —
    agents are I/O-bound or shell out to external tools (Vina, Open Babel, the
    reasoning CLI, HTTP), so threads give the overlap without pickling agent
    state. Determinism is kept by COMMITTING in topological order: a finished
    task's AgentResult, ledger update and event are written only once every
    task before it in the order has been committed, so the store and the event
    log read exactly as in a sequential run, whatever the completion order.

    Agents that read the whole run from the store (``reads_whole_run``: report,
    export, validation, PDF) are barriers: they start only after every earlier
    task is committed, and no later task starts before they commit — so they
    see exactly the prefix a sequential run would show them.

    ``max_workers=1`` reproduces the sequential runner exactly."""

    def __init__(self, store: Store, agents: dict[str, Agent], max_workers: int | None = None):
        self.store = store
        self.agents = agents
        self.max_workers = max(1, int(max_workers)) if max_workers else default_max_workers()

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
        logger.info("run %s: %d tasks, %d workers", run_id, len(order), self.max_workers)

        specs: dict[str, TaskSpec] = {
            t: self.store.load_task_spec(project_id, run_id, t) for t in order
        }
        task_status: dict[str, TaskStatus] = {t: specs[t].status for t in order}
        deps_of = self._effective_deps(manifest, order, specs)
        terminal = {TaskStatus.success, TaskStatus.partial, TaskStatus.failed,
                    TaskStatus.blocked, TaskStatus.insufficient_evidence}
        # already terminal on entry (resume) -> committed, never re-run
        committed: set[str] = {t for t in order if task_status[t] in terminal}
        started: set[str] = set()
        finished: dict[str, tuple[AgentResult, Any]] = {}  # done, awaiting in-order commit
        running: dict[Future, str] = {}

        def blocked_upstream(task_id: str) -> bool:
            return any(task_status.get(d) == TaskStatus.blocked
                       for d in manifest.task_graph.get(task_id, []))

        with ThreadPoolExecutor(max_workers=self.max_workers,
                                thread_name_prefix="ddswarm-task") as pool:
            i = 0
            while i < len(order):
                # 1) commit every finished task at the head of the order
                while i < len(order):
                    task_id = order[i]
                    if task_id in committed:
                        i += 1
                    elif task_id in finished:
                        result, finished_at = finished.pop(task_id)
                        self._commit(project_id, run_id, specs[task_id], result, finished_at)
                        task_status[task_id] = specs[task_id].status
                        committed.add(task_id)
                        i += 1
                    elif task_id not in started and blocked_upstream(task_id):
                        # deps precede it in the order, so they are all committed here
                        ts = specs[task_id]
                        ts.status = TaskStatus.blocked
                        self.store.save_task_spec(ts)
                        task_status[task_id] = TaskStatus.blocked
                        committed.add(task_id)
                        self._log(project_id, run_id, EventLog(
                            run_id=run_id, task_id=task_id, level="warning",
                            event="skipped: upstream dependency did not succeed",
                        ))
                        i += 1
                    else:
                        break
                if i >= len(order):
                    break

                # 2) start every ready task, in order
                for task_id in order[i:]:
                    if (task_id in started or task_id in committed
                            or not deps_of[task_id] <= committed or blocked_upstream(task_id)):
                        continue
                    started.add(task_id)
                    running[pool.submit(self._execute, project_id, run_id, task_id, spec,
                                        specs[task_id], manifest)] = task_id

                # 3) wait for at least one task to finish
                if not running:  # cannot happen: order[i]'s deps all precede it
                    raise RuntimeError(f"scheduler stalled at task {order[i]!r}")
                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for fut in done:
                    finished[running.pop(fut)] = fut.result()

        manifest = self.store.load_run_manifest(project_id, run_id)
        manifest.status = self._final_status([task_status[t] for t in order])
        manifest.completed_at = utcnow()
        manifest.log_refs = [str(self.store.events_path(project_id, run_id))]
        self.store.save_run_manifest(manifest)
        return manifest

    def _effective_deps(self, manifest: RunManifest, order: list[str],
                        specs: dict[str, TaskSpec]) -> dict[str, set[str]]:
        """Declared deps plus the barrier edges for whole-run readers."""
        deps = {t: set(manifest.task_graph.get(t, [])) for t in order}
        last_barrier: str | None = None
        for idx, task_id in enumerate(order):
            if last_barrier is not None:
                deps[task_id].add(last_barrier)  # nothing overtakes a barrier
            agent = self.agents.get(specs[task_id].agent)
            if getattr(agent, "reads_whole_run", False):
                deps[task_id].update(order[:idx])
                last_barrier = task_id
        return deps

    def _execute(self, project_id: str, run_id: str, task_id: str, spec: ProjectSpec,
                 ts: TaskSpec, manifest: RunManifest) -> tuple[AgentResult, Any]:
        """Run one task (on a worker thread). Only the attempt-start ledger write
        happens here; the result is committed later, in order, by run()."""
        ts.status = TaskStatus.running
        ts.attempts.append(AttemptRecord(attempt=len(ts.attempts) + 1, started_at=utcnow()))
        self.store.save_task_spec(ts)
        logger.info("task %s (%s) started", task_id, ts.agent)
        deps = manifest.task_graph.get(task_id, [])
        try:
            ctx = self._build_context(project_id, run_id, task_id, spec, ts, deps)
            result = self.agents[ts.agent].run(ctx)
        except Exception as exc:  # noqa: BLE001
            result = AgentResult(
                project_id=project_id, agent=ts.agent, task_id=task_id,
                status=AgentResultStatus.failed,
                errors=[f"{type(exc).__name__}: {exc}"],
                reproducibility={"reason": "unhandled exception in agent"},
            )
        return result, utcnow()

    def _commit(self, project_id: str, run_id: str, ts: TaskSpec, result: AgentResult,
                finished_at: Any) -> None:
        attempt = len(ts.attempts)
        self.store.save_agent_result(result, run_id, attempt)
        ts.status = TaskStatus(result.status.value)
        ts.attempts[-1].finished_at = finished_at
        ts.attempts[-1].status = ts.status
        if result.errors:
            ts.attempts[-1].error = "; ".join(result.errors)
        self.store.save_task_spec(ts)
        logger.info("task %s completed: %s", ts.task_id, result.status.value)
        level = "info" if result_ok(result.status) else "warning"
        self._log(project_id, run_id, EventLog(
            run_id=run_id, task_id=ts.task_id, level=level,
            event=f"task completed: {result.status.value}",
        ))

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
                    payload = ARTIFACT_MODELS[art.kind].model_validate_json(self.store.artifact_path(art.uri).read_text())
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
        # Non-blocking since Increment 38: only a safety `blocked` verdict gates
        # the run. failed (a crashed agent) still marks the run failed but does
        # not stop downstream; insufficient_evidence is a reported gap, not a
        # blocker — the run completes and the report lists what is missing.
        if any(s == TaskStatus.blocked for s in statuses):
            return RunStatus.blocked
        if any(s == TaskStatus.failed for s in statuses):
            return RunStatus.failed
        return RunStatus.completed

    def _find_manifest(self, run_id: str) -> RunManifest:
        for proj_dir in self.store.root.iterdir():
            if not proj_dir.is_dir():
                continue
            mpath = proj_dir / "runs" / run_id / "run_manifest.json"
            if mpath.exists():
                return RunManifest.model_validate_json(mpath.read_text())
        raise FileNotFoundError(f"no manifest found for run_id {run_id}")
