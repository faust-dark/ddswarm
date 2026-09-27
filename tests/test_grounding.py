"""Tests for grounding_minion + the external-DB client (network mocked).

Real network calls are non-deterministic, so client source functions are
monkeypatched with canned responses to test the assembly logic; the InChIKey
computation and the agent's no-molecule path run against real RDKit.
"""

import pytest

from ddswarm import cli
from ddswarm.agents import GroundingMinion
from ddswarm.agents.base import AgentContext
from ddswarm.contracts import ProjectSpec, RunTask
from ddswarm.grounding import client as gc
from ddswarm.orchestrator import Store

FLUOXETINE = "CNCCC(c1ccccc1)Oc1ccc(C(F)(F)F)cc1"


def test_inchi_key():
    assert gc.inchi_key(FLUOXETINE) == "RTHCYVBBDHJXIQ-UHFFFAOYSA-N"


def test_ground_molecule_empty_smiles_fails():
    r = gc.ground_molecule("")
    assert r["status"] == "failed"


def test_ground_molecule_assembles_entries(monkeypatch):
    monkeypatch.setattr(gc, "chembl_molecule", lambda ikey: {
        "chembl_id": "CHEMBL41", "pref_name": "fluoxetine",
        "max_phase": 4.0, "first_approval": 1987,
    })
    monkeypatch.setattr(gc, "pubchem_compound", lambda ikey: {
        "cid": 3386, "iupac": "N-methyl-3-phenyl-3-[4-(trifluoromethyl)phenoxy]propan-1-amine",
        "formula": "C17H18F3NO", "drugbank_id": "DB00472",
    })
    monkeypatch.setattr(gc, "pubmed_ids", lambda q, retmax=3: ["12345", "67890"])

    r = gc.ground_molecule(FLUOXETINE)
    assert r["status"] == "grounded"
    sources = [e["source"] for e in r["entries"]]
    assert sources.count("chembl") == 1
    assert sources.count("pubchem") == 1
    assert sources.count("drugbank") == 1
    assert sources.count("pubmed") == 2
    db = next(e for e in r["entries"] if e["source"] == "drugbank")
    assert db["external_id"] == "DB00472"
    assert "go.drugbank.com" in db["url"]


def test_ground_molecule_not_found_when_sources_empty(monkeypatch):
    monkeypatch.setattr(gc, "chembl_molecule", lambda ikey: None)
    monkeypatch.setattr(gc, "pubchem_compound", lambda ikey: None)
    monkeypatch.setattr(gc, "pubmed_ids", lambda q, retmax=3: None)
    r = gc.ground_molecule(FLUOXETINE, name="fluoxetine")
    assert r["status"] == "not_found"
    assert r["entries"] == []


def test_agent_no_molecules_insufficient(tmp_path):
    store = Store(tmp_path)
    spec = ProjectSpec(project_id="p", input_type="mixed", objective="o", status="approved")
    ctx = AgentContext(project_id="p", run_id="r1", task_id="grounding",
                       spec=spec, params={}, inputs={})
    res = GroundingMinion(store).run(ctx)
    assert res.status.value == "insufficient_evidence"


def test_inject_grounding_inserts_task_and_wires_report():
    tasks = [
        RunTask(task_id="screen", agent="screener_minion", depends_on=[]),
        RunTask(task_id="report", agent="report_minion", depends_on=["screen"]),
    ]
    out = cli._inject_grounding(tasks)
    assert [t.task_id for t in out] == ["screen", "grounding", "report"]
    grounding = next(t for t in out if t.task_id == "grounding")
    assert grounding.agent == "grounding_minion"
    report = next(t for t in out if t.task_id == "report")
    assert "grounding" in report.depends_on


def test_default_pipeline_has_no_grounding():
    ids = [t.task_id for t in cli.DEFAULT_PIPELINE]
    assert "grounding" not in ids
    assert len(ids) == 13
