"""Integration test: research -> design -> screen."""

from ddswarm.agents import DesignerMinion, ResearcherMinion, ScreenerMinion
from ddswarm.contracts import ProjectSpec, RunStatus, ScreenedBatch
from ddswarm.orchestrator import Runner, Store, TaskDefinition

GEFITINIB = "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4"


def _agents(store):
    return {
        "researcher_minion": ResearcherMinion(store),
        "designer_minion": DesignerMinion(store),
        "screener_minion": ScreenerMinion(store),
    }


def test_research_design_screen(tmp_path):
    store = Store(tmp_path)
    runner = Runner(store, _agents(store))
    spec = ProjectSpec(
        project_id="p1", input_type="mixed", objective="inhibit EGFR",
        targets=["EGFR"], starting_materials=[GEFITINIB], status="approved",
    )
    tasks = [
        TaskDefinition(task_id="research", agent="researcher_minion", params={"seed_entries": [{
            "id": "ref", "type": "reference", "source": "PubChem",
            "url": "https://pubchem.ncbi.nlm.nih.gov/compound/123631",
            "identifier": "CID 123631", "observed_value": "gefitinib: approved reversible EGFR TKI",
            "evidence_level": "observed",
        }]}),
        TaskDefinition(task_id="design", agent="designer_minion", depends_on=["research"]),
        TaskDefinition(task_id="screen", agent="screener_minion", depends_on=["design"]),
    ]
    manifest = runner.create_run(spec, tasks)
    final = runner.run(manifest.run_id, project_id="p1")
    assert final.status == RunStatus.completed

    res = store.load_latest_agent_result("p1", manifest.run_id, "screen")
    assert res.status.value == "success"
    batch = store.load_artifact(res.artifacts[0].uri, ScreenedBatch)
    # designer emits 4 labeled designs over the gefitinib scaffold; all are valid
    # and all are Brenk-flagged (Aliphatic_long_chain) under the default filters.
    assert batch.summary.total == 4
    assert all(m.valid for m in batch.molecules)
    assert batch.summary.rejected == 4
    assert batch.summary.passed == 0
