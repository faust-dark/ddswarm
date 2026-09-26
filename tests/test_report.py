"""Tests for report_minion (run aggregation -> Report + memory) and the
feedback loop (researcher/designer consuming project memory)."""

from ddswarm.agents import (
    DesignerMinion,
    PredictorMinion,
    ReportMinion,
    ResearcherMinion,
    RetrosynthesisMinion,
    ScreenerMinion,
    ToxMinion,
)
from ddswarm.agents.base import AgentContext
from ddswarm.contracts import (
    DesignBatch,
    KnowledgePack,
    MemoryEntry,
    ProjectSpec,
    Report,
)
from ddswarm.orchestrator import Runner, Store, TaskDefinition
from ddswarm.orchestrator.memory import MemoryStore


def _agents(store):
    return {
        "researcher_minion": ResearcherMinion(store),
        "designer_minion": DesignerMinion(store),
        "screener_minion": ScreenerMinion(store),
        "predictor_minion": PredictorMinion(store),
        "retrosynthesis_minion": RetrosynthesisMinion(store),
        "tox_minion": ToxMinion(store),
        "report_minion": ReportMinion(store),
    }


def _tasks():
    return [
        TaskDefinition(task_id="research", agent="researcher_minion", params={"seed_entries": []}),
        TaskDefinition(task_id="design", agent="designer_minion", depends_on=["research"]),
        TaskDefinition(task_id="screen", agent="screener_minion", depends_on=["design"]),
        TaskDefinition(task_id="predict", agent="predictor_minion", depends_on=["screen"],
                       params={"providers": ["null_admet"]}),
        TaskDefinition(task_id="retrosynth", agent="retrosynthesis_minion", depends_on=["screen"]),
        TaskDefinition(task_id="tox", agent="tox_minion", depends_on=["screen"]),
        TaskDefinition(task_id="report", agent="report_minion",
                       depends_on=["predict", "retrosynth", "tox"]),
    ]


def _spec():
    return ProjectSpec(project_id="p", input_type="mixed", objective="inhibit X",
                       targets=["X"], starting_materials=["c1ccccc1"], status="approved")


def test_full_run_produces_report_and_memory(tmp_path):
    store = Store(tmp_path)
    runner = Runner(store, _agents(store))
    manifest = runner.create_run(_spec(), _tasks())
    final = runner.run(manifest.run_id, project_id="p")
    assert final.status.value == "completed"

    res = store.load_latest_agent_result("p", manifest.run_id, "report")
    report = store.load_artifact(res.artifacts[0].uri, Report)
    assert report.run_status == "completed"
    assert report.executive_summary.startswith("Run summary:")
    assert report.evidence_breakdown.calculated > 0
    assert report.evidence_breakdown.hypothetical > 0  # design labels
    assert "no docking performed this run" in report.gaps
    assert any("no ADMET endpoints predicted" in g for g in report.gaps)

    mem = MemoryStore(store).load("p")
    kinds = {e.kind for e in mem.entries}
    assert {"finding", "feedback"} <= kinds
    fb = next(e for e in mem.entries if e.kind == "feedback")
    assert "c1ccccc1" in fb.data["survivors"]


def test_designer_consumes_feedback(tmp_path):
    store = Store(tmp_path)
    MemoryStore(store).append(MemoryEntry(
        id="feedback_r0", project_id="p", kind="feedback", content="x",
        data={"survivors": ["c1ccccc1Cl"], "gaps": []},
        source_agent="report_minion", run_id="r0",
    ))
    ctx = AgentContext(project_id="p", run_id="r1", task_id="design", spec=_spec(), params={})
    result = DesignerMinion(store).run(ctx)
    assert result.status.value == "success"
    db = store.load_artifact(result.artifacts[0].uri, DesignBatch)
    # benzene scaffold has no transformable handles; the chlorobenzene survivor
    # yields 3 halogen-swap analogs (Cl->F/Br/I).
    assert len(db.molecules) == 3
    assert all(m.parent_scaffold == "prior_r0_0" for m in db.molecules)
    assert {m.transformation for m in db.molecules} == {"cl_to_f", "cl_to_br", "cl_to_i"}
    assert any("feedback loop" in w for w in result.warnings)


def test_designer_feedback_no_survivors_warns(tmp_path):
    store = Store(tmp_path)
    MemoryStore(store).append(MemoryEntry(
        id="feedback_r0", project_id="p", kind="feedback", content="x",
        data={"survivors": [], "gaps": ["0 molecules passed screening"]},
        source_agent="report_minion", run_id="r0",
    ))
    ctx = AgentContext(project_id="p", run_id="r1", task_id="design", spec=_spec(), params={})
    result = DesignerMinion(store).run(ctx)
    assert result.status.value == "success"
    assert any("no survivors" in w for w in result.warnings)


def test_researcher_consumes_finding(tmp_path):
    store = Store(tmp_path)
    MemoryStore(store).append(MemoryEntry(
        id="finding_r0", project_id="p", kind="finding", content="prior run: 4 passed",
        data={}, source_agent="report_minion", run_id="r0",
    ))
    ctx = AgentContext(project_id="p", run_id="r1", task_id="research",
                       spec=_spec(), params={"seed_entries": []})
    result = ResearcherMinion(store).run(ctx)
    kp = store.load_artifact(result.artifacts[0].uri, KnowledgePack)
    mem_entries = [e for e in kp.entries if e.source == "project memory"]
    assert len(mem_entries) == 1
    assert mem_entries[0].evidence_level.value == "inferred"
