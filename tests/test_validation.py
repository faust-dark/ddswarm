"""Tests for validation_minion: reproducibility + consistency checks over a run."""

from ddswarm.agents import (
    DesignerMinion,
    ReportMinion,
    ResearcherMinion,
    ScreenerMinion,
    ValidationMinion,
)
from ddswarm.agents.base import AgentContext
from ddswarm.contracts import ProjectSpec, ValidationReport
from ddswarm.orchestrator import Runner, Store, TaskDefinition


def _tasks():
    return [
        TaskDefinition(task_id="research", agent="researcher_minion", params={"seed_entries": []}),
        TaskDefinition(task_id="design", agent="designer_minion", depends_on=["research"]),
        TaskDefinition(task_id="screen", agent="screener_minion", depends_on=["design"]),
        TaskDefinition(task_id="report", agent="report_minion", depends_on=["screen"]),
        TaskDefinition(task_id="validation", agent="validation_minion", depends_on=["report"]),
    ]


def _agents(store):
    return {
        "researcher_minion": ResearcherMinion(store),
        "designer_minion": DesignerMinion(store),
        "screener_minion": ScreenerMinion(store),
        "report_minion": ReportMinion(store),
        "validation_minion": ValidationMinion(store),
    }


def _spec():
    return ProjectSpec(project_id="p", input_type="mixed", objective="inhibit EGFR",
                       targets=["EGFR"], starting_materials=["c1ccccc1"], status="approved")


def _run_pipeline(tmp_path):
    store = Store(tmp_path)
    runner = Runner(store, _agents(store))
    manifest = runner.create_run(_spec(), _tasks())
    final = runner.run(manifest.run_id, project_id="p")
    return store, final


def _load_report(store, result):
    assert result.artifacts
    return store.load_artifact(result.artifacts[0].uri, ValidationReport)


def test_validation_all_checks_pass(tmp_path):
    store, final = _run_pipeline(tmp_path)
    assert final.status.value == "completed"
    res = store.load_latest_agent_result("p", final.run_id, "validation")
    vr = _load_report(store, res)
    assert vr.summary.total_checks == 7
    assert vr.summary.failed == 0
    assert vr.status.value == "success"
    assert vr.reproducibility["n_artifacts_checked"] >= 4


def test_validation_detects_tamper(tmp_path):
    store, final = _run_pipeline(tmp_path)
    # tamper one artifact file on disk
    files = list(store.artifacts_dir("p", final.run_id).glob("*.json"))
    assert files
    files[0].write_text(files[0].read_text() + "\n// tampered")

    ctx = AgentContext(project_id="p", run_id=final.run_id, task_id="validation",
                       spec=store.load_project_spec("p"), params={}, inputs={})
    result = ValidationMinion(store).run(ctx)
    vr = _load_report(store, result)
    hashes = [c for c in vr.checks if c.check_id == "hash_integrity"]
    assert hashes and hashes[0].status == "fail"
    assert result.status.value == "partial"


def test_validation_empty_store_warns(tmp_path):
    store = Store(tmp_path)
    runner = Runner(store, _agents(store))
    manifest = runner.create_run(_spec(), _tasks())
    # do not run the pipeline; validation sees no artifacts
    ctx = AgentContext(project_id="p", run_id=manifest.run_id, task_id="validation",
                       spec=store.load_project_spec("p"), params={}, inputs={})
    result = ValidationMinion(store).run(ctx)
    vr = _load_report(store, result)
    assert vr.summary.total_checks == 1
    assert vr.checks[0].check_id == "no_artifacts"
    assert vr.checks[0].status == "warn"
