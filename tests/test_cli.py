"""CLI tests: run a pipeline from project.json (null reasoning), init, runs, report."""

from pathlib import Path

from ddswarm import cli
from ddswarm.contracts import ProjectSpec, RunRequest


def _write_project(tmp_path: Path, project_id="p1", smiles="c1ccccc1") -> Path:
    spec = ProjectSpec(
        project_id=project_id, input_type="mixed", objective="inhibit X",
        targets=["X"], starting_materials=[smiles], status="approved",
    )
    req = RunRequest(project=spec, tasks=[])
    f = tmp_path / "project.json"
    f.write_text(req.model_dump_json(indent=2))
    return f


def test_run_default_pipeline(tmp_path):
    f = _write_project(tmp_path)
    store = tmp_path / "store"
    rc = cli.main(["run", str(f), "--store", str(store)])
    assert rc == 0
    runs = list((store / "p1" / "runs").iterdir())
    assert len(runs) == 1
    artifacts = list((runs[0] / "artifacts").glob("*.json"))
    kinds = {a.name.split(".")[1] for a in artifacts}
    assert {"knowledge_pack", "design_batch", "screened_batch", "admet_prediction",
            "retrosynthesis_plan", "tox_profile", "report"} <= kinds


def test_init_writes_project(tmp_path):
    out = tmp_path / "p.json"
    rc = cli.main(["init", str(out), "--project-id", "x1", "--objective", "inhibit Y",
                   "--targets", "T1", "--smiles", "c1ccccc1"])
    assert rc == 0
    req = RunRequest.model_validate_json(out.read_text())
    assert req.project.project_id == "x1"
    assert len(req.tasks) == 11  # safety, research, design, screen, predict, retrosynth, tox, formulation, report, validation, report_pdf
    assert req.reasoning is None


def test_runs_and_report(tmp_path):
    f = _write_project(tmp_path, project_id="p2")
    store = tmp_path / "store"
    assert cli.main(["run", str(f), "--store", str(store)]) == 0
    assert cli.main(["runs", "p2", "--store", str(store)]) == 0
    assert cli.main(["report", "p2", "--store", str(store)]) == 0
    assert cli.main(["memory", "p2", "--store", str(store)]) == 0


def test_inject_reasoning():
    from ddswarm.cli import _inject_reasoning
    from ddswarm.contracts import ReasoningConfig, RunTask

    def fresh():
        return [
            RunTask(task_id="research", agent="researcher_minion"),
            RunTask(task_id="design", agent="designer_minion"),
            RunTask(task_id="screen", agent="screener_minion"),
            RunTask(task_id="patent", agent="patent_minion"),
            RunTask(task_id="clinical", agent="clinical_minion"),
            RunTask(task_id="regulatory", agent="regulatory_minion"),
        ]

    out = _inject_reasoning(fresh(), ReasoningConfig(backend="hermes", command="hermes",
                                                     max_turns=10, run_budget=100))
    assert out[0].params["reasoning_backend"] == "hermes"
    assert out[0].params["reasoning_max_turns"] == 10
    assert out[1].params["reasoning_backend"] == "hermes"
    assert out[2].params == {}  # non-reasoning agents untouched
    assert out[3].params["reasoning_backend"] == "hermes"  # patent
    assert out[4].params["reasoning_backend"] == "hermes"  # clinical
    assert out[5].params["reasoning_backend"] == "hermes"  # regulatory

    out2 = _inject_reasoning(fresh(), ReasoningConfig(backend="none"))
    assert all(t.params == {} for t in out2)
