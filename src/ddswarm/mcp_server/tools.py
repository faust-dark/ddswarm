"""Thin stateless adapters: each MCP tool maps to one ddswarm agent.

Every adapter builds a throwaway Store, runs the agent against a minimal
ProjectSpec derived from the tool inputs, and returns a JSON dict embedding the
full artifact payload(s) — so an MCP client never has to read the temp store.

The swarm stays deterministic and platform-agnostic here: the reasoning backend
is forced to "none" (no embedded LLM — the MCP client *is* the reasoning layer).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from ..agents import DesignerMinion, ScreenerMinion
from ..agents.base import AgentContext
from ..contracts.enums import InputType, ProjectStatus
from ..contracts.project_spec import ProjectSpec
from ..contracts.report import Report
from ..orchestrator.runner import ARTIFACT_MODELS
from ..orchestrator.store import Store

_PROJECT_ID = "mcp"


def _fresh_store() -> Store:
    """A throwaway store per tool call (stateless tools)."""
    return Store(Path(tempfile.mkdtemp(prefix="ddswarm_mcp_")))


def _spec(**overrides: Any) -> ProjectSpec:
    base: dict[str, Any] = {
        "project_id": _PROJECT_ID,
        "input_type": InputType.mixed,
        "objective": "ddswarm MCP tool call",
        "status": ProjectStatus.approved,
    }
    base.update(overrides)
    return ProjectSpec(**base)


def _ctx(spec: ProjectSpec, params: dict[str, Any]) -> AgentContext:
    return AgentContext(
        project_id=spec.project_id, run_id="mcp", task_id="tool",
        spec=spec, params=params,
    )


def _result_payload(result: Any) -> dict[str, Any]:
    """Normalize an AgentResult into a JSON dict with full artifact payloads."""
    payload: dict[str, Any] = {
        "status": result.status.value,
        "findings": result.findings,
        "warnings": result.warnings,
        "errors": result.errors,
        "artifacts": {},
    }
    for art in result.artifacts:
        if art.uri and art.kind in ARTIFACT_MODELS:
            data = ARTIFACT_MODELS[art.kind].model_validate_json(Path(art.uri).read_text())
            payload["artifacts"][art.kind] = data.model_dump()
    return payload


def design_analogs(starting_materials: list[str]) -> dict[str, Any]:
    """Generate new analog molecules from starting SMILES (deterministic)."""
    store = _fresh_store()
    spec = _spec(starting_materials=starting_materials)
    result = DesignerMinion(store).run(_ctx(spec, {"reasoning_backend": "none"}))
    return _result_payload(result)


def screen_molecules(smiles: list[str], filters: dict[str, Any] | None = None) -> dict[str, Any]:
    """Screen molecules against hard + advisory filters (deterministic)."""
    store = _fresh_store()
    spec = _spec()
    params: dict[str, Any] = {
        "reasoning_backend": "none",
        "molecules": [{"id": f"mol_{i}", "smiles": s} for i, s in enumerate(smiles)],
    }
    if filters:
        params["filters"] = filters
    result = ScreenerMinion(store).run(_ctx(spec, params))
    return _result_payload(result)


def run_pipeline(project: dict[str, Any]) -> dict[str, Any]:
    """Run the full deterministic 18-agent pipeline from a ProjectSpec dict.

    Docking and target-selectivity resolve their receptor(s) from the project
    targets and degrade non-blocking when unavailable (no PDB / no network).
    """
    from ..cli import DEFAULT_PIPELINE, build_agents
    from ..orchestrator import Runner, TaskDefinition
    from ..orchestrator.gates import BlockedByGate

    store = _fresh_store()
    try:
        spec = ProjectSpec.model_validate(project)
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "errors": [f"invalid ProjectSpec: {exc}"], "report": None}

    if spec.status == ProjectStatus.draft:
        # an MCP tool call is itself the approval gate
        spec.status = ProjectStatus.approved

    try:
        runner = Runner(store, build_agents(store))
        task_defs = [TaskDefinition(task_id=t.task_id, agent=t.agent,
                                    depends_on=t.depends_on, params=t.params)
                     for t in DEFAULT_PIPELINE]
        manifest = runner.create_run(spec, task_defs)
    except BlockedByGate as exc:
        return {"status": "blocked", "errors": [str(exc)], "report": None}

    final = runner.run(manifest.run_id, project_id=spec.project_id)
    report = _load_artifact(store, spec.project_id, manifest.run_id, "report", "report", Report)
    return {
        "status": final.status.value,
        "run_id": manifest.run_id,
        "project_id": spec.project_id,
        "report": report,
    }


def _load_artifact(
    store: Store, project_id: str, run_id: str, task_id: str, kind: str, model: Any,
) -> dict[str, Any] | None:
    n = store.latest_attempt(project_id, run_id, task_id)
    if n == 0:
        return None
    res = store.load_agent_result(project_id, run_id, task_id, n)
    for art in res.artifacts:
        if art.kind == kind and art.uri:
            return model.model_validate_json(Path(art.uri).read_text()).model_dump()
    return None
