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
