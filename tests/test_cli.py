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
    reports = tmp_path / "reports"
    rc = cli.main(["run", str(f), "--reasoning", "none", "--no-iupac", "--store", str(store),
                   "--reports-dir", str(reports)])
    assert rc == 0
    runs = list((store / "p1" / "runs").iterdir())
    assert len(runs) == 1
    artifacts = list((runs[0] / "artifacts").glob("*.json"))
    kinds = {a.name.split(".")[1] for a in artifacts}
    assert {"knowledge_pack", "design_batch", "screened_batch", "admet_prediction",
            "retrosynthesis_plan", "tox_profile", "report", "report_document"} <= kinds
    # the exported PDF is copied into the reports dir
    pdf = reports / "p1.pdf"
    assert pdf.exists()
    assert pdf.read_bytes().startswith(b"%PDF")


def test_init_writes_project(tmp_path):
    out = tmp_path / "p.json"
    rc = cli.main(["init", str(out), "--project-id", "x1", "--objective", "inhibit Y",
                   "--targets", "T1", "--smiles", "c1ccccc1"])
    assert rc == 0
    req = RunRequest.model_validate_json(out.read_text())
    assert req.project.project_id == "x1"
    # safety, research, overview, design, screen, docking, selectivity, predict, retrosynth,
    # tox, cyp, herg, formulation, patent, report, export, validation, report_pdf
    assert len(req.tasks) == 18
    assert req.reasoning is None


def test_init_interactive_wizard(tmp_path, monkeypatch, capsys):
    out = tmp_path / "proj.json"
    answers = [
        "x1",                                  # project_id
        "inhibit Y",                           # objective
        "SERT",                                # known target
        "PTGS2",                               # alias -> COX2
        "not-a-target",                        # free-text target
        "",                                    # targets done
        "C1=CC=CC=C1",                         # benzene (non-canonical -> canonicalized)
        "not-a-smiles",                        # invalid -> reprompt
        "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21",  # diazepam
        "",                                    # SMILES done
        "",                                    # reasoning (default none)
        "no",                                  # iupac off
        "3",                                   # iterations
    ]
    monkeypatch.setattr("builtins.input", lambda *_: answers.pop(0))
    rc = cli.main(["init", str(out)])
    assert rc == 0
    req = RunRequest.model_validate_json(out.read_text())
    assert req.project.project_id == "x1"
    assert req.project.objective == "inhibit Y"
    assert req.project.targets == ["SERT", "COX2", "not-a-target"]
    # benzene is canonicalized; the invalid SMILES is dropped
    assert req.project.starting_materials == ["c1ccccc1", "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21"]
    assert req.reasoning is None
    assert len(req.tasks) == 18
    captured = capsys.readouterr().out
    assert "next: ddswarm run" in captured
    assert "--no-iupac" in captured
    assert "--iterations 3" in captured


def test_init_interactive_invalid_project_id_reprompts(tmp_path, monkeypatch):
    out = tmp_path / "p.json"
    answers = ["../evil", "good-id", "obj", "", "", "", "", ""]
    monkeypatch.setattr("builtins.input", lambda *_: answers.pop(0))
    rc = cli.main(["init", str(out)])
    assert rc == 0
    req = RunRequest.model_validate_json(out.read_text())
    assert req.project.project_id == "good-id"


def test_init_interactive_hermes_reasoning(tmp_path, monkeypatch):
    out = tmp_path / "p.json"
    answers = ["p1", "obj", "", "", "hermes", "yes", "1"]
    monkeypatch.setattr("builtins.input", lambda *_: answers.pop(0))
    rc = cli.main(["init", str(out)])
    assert rc == 0
    req = RunRequest.model_validate_json(out.read_text())
    assert req.reasoning is not None
    assert req.reasoning.backend == "hermes"


def test_runs_and_report(tmp_path):
    f = _write_project(tmp_path, project_id="p2")
    store = tmp_path / "store"
    reports = tmp_path / "reports"
    assert cli.main(["run", str(f), "--reasoning", "none", "--no-iupac", "--store", str(store),
                     "--reports-dir", str(reports)]) == 0
    assert cli.main(["runs", "p2", "--store", str(store)]) == 0
    assert cli.main(["report", "p2", "--store", str(store), "--reports-dir", str(reports)]) == 0
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


def test_reasoning_config_defaults_to_none():
    # platform-agnostic skeleton: no LLM backend unless explicitly opted in
    from ddswarm.contracts import ReasoningConfig
    assert ReasoningConfig().backend == "none"


def test_run_iterations_two_generations(tmp_path):
    f = _write_project(tmp_path, project_id="piter")
    store = tmp_path / "store"
    reports = tmp_path / "reports"
    rc = cli.main(["run", str(f), "--reasoning", "none", "--no-iupac", "--iterations", "2",
                   "--store", str(store), "--reports-dir", str(reports)])
    assert rc == 0
    runs = list((store / "piter" / "runs").iterdir())
    assert len(runs) == 2  # two generations = two runs
    # the last generation produced the orchestration + memory artifacts
    last = sorted(runs)[-1]
    artifacts = list((last / "artifacts").glob("*.json"))
    kinds = {a.name.split(".")[1] for a in artifacts}
    assert "execution_plan" in kinds
    assert "generation_memory" in kinds
    assert "report_document" in kinds


def test_receptors_list_and_lookup(capsys):
    assert cli.main(["receptors", "list"]) == 0
    out = capsys.readouterr().out
    assert "SERT" in out and "5I73" in out
    assert "65 target(s) shown (63 with pdb_id, 2 without)" in out

    assert cli.main(["receptors", "list", "--family", "enzyme"]) == 0
    out = capsys.readouterr().out
    assert "ACE" in out
    assert "SERT" not in out  # transporters are filtered out

    # alias resolution: PTGS2 -> COX2, M1 -> CHRM1
    assert cli.main(["receptors", "lookup", "PTGS2"]) == 0
    out = capsys.readouterr().out
    assert "COX2" in out and "3LN1" in out

    assert cli.main(["receptors", "lookup", "M1"]) == 0
    out = capsys.readouterr().out
    assert "CHRM1" in out and "5CXV" in out

    assert cli.main(["receptors", "lookup", "DOES_NOT_EXIST"]) == 0
    assert "not found" in capsys.readouterr().out


def test_receptors_adr_and_match(capsys):
    assert cli.main(["receptors", "adr"]) == 0
    out = capsys.readouterr().out
    assert "sert_5ht2c_sexual_adr" in out
    assert "cox_gi_risk" in out
    assert "chrm1_anticholinergic" in out
    assert "hrh1_sedation" in out

    assert cli.main(["receptors", "match", "--primary", "COX2", "--off-targets", "COX1"]) == 0
    assert "cox_gi_risk" in capsys.readouterr().out

    assert cli.main(["receptors", "match", "--primary", "SERT",
                     "--off-targets", "M1,H1,ADRA1A"]) == 0
    out = capsys.readouterr().out
    assert "chrm1_anticholinergic" in out
    assert "hrh1_sedation" in out
    assert "adra1a_orthostatic_hypotension" in out

    assert cli.main(["receptors", "match", "--primary", "DRD2", "--off-targets", "HTR2A"]) == 0
    assert "no built-in ADR profile" in capsys.readouterr().out
