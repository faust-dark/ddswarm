"""Tests for orchestrator_minion (deterministic ExecutionPlan derivation)."""

from ddswarm.agents import OrchestratorMinion
from ddswarm.agents.base import AgentContext
from ddswarm.contracts import ExecutionPlan, ProjectSpec
from ddswarm.orchestrator import Store


def _spec():
    return ProjectSpec(project_id="p", input_type="mixed", objective="inhibit EGFR",
                       targets=["EGFR"], starting_materials=["c1ccccc1"], status="approved",
                       mechanism_of_action="EGFR kinase inhibition", indication="NSCLC")


def _run(store, params):
    ctx = AgentContext(project_id="p", run_id="r", task_id="orchestrator", spec=_spec(),
                       params=params, inputs={})
    return OrchestratorMinion(store).run(ctx)


def test_orchestrator_produces_plan(tmp_path):
    store = Store(tmp_path)
    tasks = [
        {"task_id": "research", "agent": "researcher_minion", "depends_on": []},
        {"task_id": "design", "agent": "designer_minion", "depends_on": ["research"]},
        {"task_id": "screen", "agent": "screener_minion", "depends_on": ["design"]},
    ]
    res = _run(store, {"tasks": tasks, "generation": 1, "iterations": 3})
    assert res.status.value == "success"
    plan = store.load_artifact(res.artifacts[0].uri, ExecutionPlan)
    assert plan.generation == 1
    assert plan.convergence.max_generations == 3
    assert len(plan.stages) == 3
    assert plan.stages[0].agent == "researcher_minion"
    assert plan.stages[0].role  # role from AGENT_ROLES
    assert plan.stages[1].depends_on == ["research"]
    assert "EGFR" in plan.rationale
    assert "NSCLC" in plan.rationale


def test_orchestrator_excludes_self_from_stages(tmp_path):
    store = Store(tmp_path)
    tasks = [
        {"task_id": "orchestrator", "agent": "orchestrator_minion", "depends_on": []},
        {"task_id": "design", "agent": "designer_minion", "depends_on": ["orchestrator"]},
    ]
    res = _run(store, {"tasks": tasks})
    plan = store.load_artifact(res.artifacts[0].uri, ExecutionPlan)
    assert [s.agent for s in plan.stages] == ["designer_minion"]
