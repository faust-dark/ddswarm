"""Tests for the orchestrator: happy path, gating, idempotent retry, resume."""

import pytest

from ddswarm.agents import DesignerMinion, ResearcherMinion, Agent
from ddswarm.contracts import ProjectSpec, RunStatus
from ddswarm.contracts.agent_result import AgentResult
from ddswarm.contracts.enums import AgentResultStatus
from ddswarm.orchestrator import BlockedByGate, Runner, Store, TaskDefinition

GEFITINIB_SMILES = "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4"


def make_agents(store):
    return {"researcher_minion": ResearcherMinion(store), "designer_minion": DesignerMinion(store)}


def make_spec(project_id="p1", **kw):
    base = dict(
        project_id=project_id,
        input_type="mixed",
        objective="identify a developable EGFR inhibitor candidate",
        targets=["EGFR"],
        starting_materials=[GEFITINIB_SMILES],
        status="approved",
    )
    base.update(kw)
    return ProjectSpec(**base)


def make_tasks(with_seed=True):
    research_params = {}
    if with_seed:
        research_params = {"seed_entries": [{
            "id": "ref_gefitinib",
            "type": "reference",
            "source": "PubChem",
            "url": "https://pubchem.ncbi.nlm.nih.gov/compound/123631",
            "identifier": "CID 123631",
            "observed_value": "gefitinib: approved reversible EGFR tyrosine kinase inhibitor (FDA 2003; NSCLC first-line 2015)",
            "evidence_level": "observed",
            "notes": "reference ligand for scaffold design",
        }]}
    return [
        TaskDefinition(task_id="research", agent="researcher_minion", params=research_params),
        TaskDefinition(task_id="design", agent="designer_minion", depends_on=["research"]),
    ]


def test_happy_path(tmp_path):
    store = Store(tmp_path)
    runner = Runner(store, make_agents(store))
    manifest = runner.create_run(make_spec(), make_tasks())
    assert manifest.status == RunStatus.queued
    final = runner.run(manifest.run_id, project_id="p1")
    assert final.status == RunStatus.completed

    arts = list((tmp_path / "p1" / "runs" / manifest.run_id / "artifacts").glob("*"))
    assert any("knowledge_pack" in a.name for a in arts)
    assert any("design_batch" in a.name for a in arts)

    events = (tmp_path / "p1" / "runs" / manifest.run_id / "events.jsonl").read_text().splitlines()
    assert len(events) >= 3  # run_created + 2 task completions

    result = store.load_latest_agent_result("p1", manifest.run_id, "design")
    assert result.status.value == "success"


def test_gate_a_rejects_draft(tmp_path):
    store = Store(tmp_path)
    runner = Runner(store, make_agents(store))
    with pytest.raises(BlockedByGate):
        runner.create_run(make_spec(status="draft"), make_tasks())


def test_gate_a_rejects_empty_objective(tmp_path):
    store = Store(tmp_path)
    runner = Runner(store, make_agents(store))
    with pytest.raises(BlockedByGate):
        runner.create_run(make_spec(objective="   "), make_tasks())


class FailingAgent(Agent):
    name = "researcher_minion"

    def run(self, ctx):
        raise RuntimeError("boom")


class SafetyBlockingAgent(Agent):
    name = "safety_minion"

    def run(self, ctx):
        return AgentResult(project_id=ctx.project_id, agent=self.name, task_id=ctx.task_id,
                           status=AgentResultStatus.blocked)


def test_failure_marks_run_failed_but_downstream_runs(tmp_path):
    store = Store(tmp_path)
    runner = Runner(store, {"researcher_minion": FailingAgent(store), "designer_minion": DesignerMinion(store)})
    manifest = runner.create_run(make_spec(), make_tasks())
    final = runner.run(manifest.run_id, project_id="p1")
    assert final.status == RunStatus.failed  # a crash still marks the run failed
    ts_design = store.load_task_spec("p1", manifest.run_id, "design")
    assert ts_design.status.value == "success"  # but downstream is NOT gated


def test_insufficient_evidence_does_not_block_run(tmp_path):
    store = Store(tmp_path)
    runner = Runner(store, make_agents(store))
    spec = make_spec(starting_materials=[], targets=[])
    manifest = runner.create_run(spec, make_tasks(with_seed=False))
    final = runner.run(manifest.run_id, project_id="p1")
    assert final.status == RunStatus.completed  # insufficient_evidence is a gap, not a block
    res = store.load_latest_agent_result("p1", manifest.run_id, "research")
    assert res.status.value == "insufficient_evidence"


def test_safety_block_still_gates_downstream(tmp_path):
    # The ONE remaining hard gate: a safety `blocked` verdict must still stop
    # every downstream task and mark the run blocked.
    store = Store(tmp_path)
    runner = Runner(store, {"safety_minion": SafetyBlockingAgent(store),
                            "researcher_minion": ResearcherMinion(store),
                            "designer_minion": DesignerMinion(store)})
    tasks = [
        TaskDefinition(task_id="safety", agent="safety_minion"),
        TaskDefinition(task_id="research", agent="researcher_minion", depends_on=["safety"]),
        TaskDefinition(task_id="design", agent="designer_minion", depends_on=["research"]),
    ]
    manifest = runner.create_run(make_spec(), tasks)
    final = runner.run(manifest.run_id, project_id="p1")
    assert final.status == RunStatus.blocked
    assert store.load_task_spec("p1", manifest.run_id, "research").status.value == "blocked"
    assert store.load_task_spec("p1", manifest.run_id, "design").status.value == "blocked"


def test_retry_is_versioned(tmp_path):
    store = Store(tmp_path)
    runner = Runner(store, make_agents(store))
    manifest = runner.create_run(make_spec(), make_tasks())
    runner.run(manifest.run_id, project_id="p1")

    runner.retry_task("p1", manifest.run_id, "research")
    ts = store.load_task_spec("p1", manifest.run_id, "research")
    assert ts.retry_count == 1

    v1 = store.load_agent_result("p1", manifest.run_id, "research", 1)
    v2 = store.load_agent_result("p1", manifest.run_id, "research", 2)
    assert v1.created_at != v2.created_at

    kps = sorted((tmp_path / "p1" / "runs" / manifest.run_id / "artifacts").glob("kp_p1.knowledge_pack.v*.json"))
    assert len(kps) == 2


def test_resume_skips_completed(tmp_path):
    store = Store(tmp_path)
    runner = Runner(store, make_agents(store))
    manifest = runner.create_run(make_spec(), make_tasks())
    runner.run(manifest.run_id, project_id="p1")
    # re-run should be a no-op: both tasks already success
    final = runner.run(manifest.run_id, project_id="p1")
    assert final.status == RunStatus.completed
    ts = store.load_task_spec("p1", manifest.run_id, "research")
    assert len(ts.attempts) == 1  # no new attempt on resume
