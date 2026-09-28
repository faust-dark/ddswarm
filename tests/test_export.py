"""Tests for export_minion + the go/no-go verdict + CSV/SDF writers."""

from ddswarm import cli
from ddswarm.agents.export import ExportMinion, _verdict
from ddswarm.contracts.candidate_table import CandidateRow


def test_verdict_rejected_is_no_go():
    v, r = _verdict("rejected", 0, 0, [], False, 0, None, ["PAINS", "Ro5"])
    assert v == "no-go"
    assert any("PAINS" in x for x in r)


def test_verdict_safety_block_is_no_go():
    v, r = _verdict("passed", 1, 0, [], False, 0, "I", [])
    assert v == "no-go"
    assert any("safety" in x for x in r)


def test_verdict_advisories_are_review():
    v, r = _verdict("passed", 0, 1, ["CYP2D6 inhibition"], True, 1, "IV", [])
    assert v == "review"
    assert len(r) >= 4  # tox + cyp + hERG + Brenk + BCS IV


def test_verdict_clean_is_go():
    v, r = _verdict("passed", 0, 0, [], False, 0, "I", [])
    assert v == "go"
    assert r == []


def test_to_csv_headers_and_row():
    rows = [CandidateRow(
        rank=1, molecule_id="m0", input_smiles="CC", canonical_smiles="CC",
        screening_status="passed", verdict="go", qed=0.8,
    )]
    out = ExportMinion._to_csv(rows)
    assert "molecule_id" in out
    assert "m0" in out
    assert "verdict" in out
    assert "iupac_name" in out


def test_to_csv_includes_iupac_name():
    rows = [CandidateRow(
        rank=1, molecule_id="m0", input_smiles="CC", canonical_smiles="CC",
        screening_status="passed", verdict="go",
        iupac_name="ethane",
    )]
    out = ExportMinion._to_csv(rows)
    assert "ethane" in out


def test_to_sdf_writes_molecule_and_props():
    rows = [CandidateRow(
        rank=1, molecule_id="m0", input_smiles="CC", canonical_smiles="CC",
        screening_status="passed", verdict="go", qed=0.8,
    )]
    data = ExportMinion._to_sdf(rows)
    assert b"molecule_id" in data
    assert b"m0" in data
    assert b"verdict" in data


def test_default_pipeline_has_export_after_report():
    ids = [t.task_id for t in cli.DEFAULT_PIPELINE]
    assert "export" in ids
    assert ids.index("export") == ids.index("report") + 1
    report_pdf = next(t for t in cli.DEFAULT_PIPELINE if t.task_id == "report_pdf")
    assert "export" in report_pdf.depends_on


def test_export_propagates_iupac_from_report(tmp_path, monkeypatch):
    from ddswarm.contracts import ProjectSpec
    from ddswarm.contracts.candidate_table import CandidateTable
    from ddswarm.orchestrator import Runner, Store, TaskDefinition

    monkeypatch.setattr("ddswarm.agents.report.fetch_iupac_name",
                        lambda smi: f"IUPAC({smi})")
    store = Store(tmp_path)
    runner = Runner(store, cli.build_agents(store))
    tasks = []
    for t in cli.DEFAULT_PIPELINE:
        params = dict(t.params or {})
        if t.task_id == "report":
            params["iupac"] = True
        tasks.append(TaskDefinition(task_id=t.task_id, agent=t.agent,
                                    depends_on=t.depends_on, params=params))
    spec = ProjectSpec(project_id="p", input_type="mixed", objective="inhibit X",
                       targets=["X"], starting_materials=["c1ccccc1"], status="approved")
    manifest = runner.create_run(spec, tasks)
    final = runner.run(manifest.run_id, project_id="p")
    assert final.status.value == "completed"

    exp = store.load_latest_agent_result("p", manifest.run_id, "export")
    table = store.load_artifact(exp.artifacts[0].uri, CandidateTable)
    ranked_iupacs = [r.iupac_name for r in table.rows if r.rank is not None]
    assert ranked_iupacs, "expected ranked survivors to carry IUPAC names"
    assert all(n and n.startswith("IUPAC(") for n in ranked_iupacs)
