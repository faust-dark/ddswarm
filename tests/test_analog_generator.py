"""Offline analog generation tests: the designer now produces genuinely NEW
molecules deterministically (RDKit transformation library), not label-only
copies of the input."""

from rdkit import Chem

from ddswarm.agents import DesignerMinion, ResearcherMinion, ScreenerMinion
from ddswarm.chemistry.analog_generator import generate_analogs
from ddswarm.contracts import DesignBatch, ProjectSpec, RunStatus, ScreenedBatch
from ddswarm.orchestrator import Runner, Store, TaskDefinition

ALPRAZOLAM = "Cc1nnc2n1-c1ccc(Cl)cc1C(c1ccccc1)=NC2"


def _canonical(smi: str) -> str:
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi))


def test_generate_analogs_produces_new_molecules():
    parent = _canonical(ALPRAZOLAM)
    analogs = generate_analogs(ALPRAZOLAM)
    assert analogs, "alprazolam has chlorine + methyl handles; expect >= 1 analog"
    smis = [a.smiles for a in analogs]
    assert all(s != parent for s in smis), "analogs must be new structures, not the parent"
    assert len(smis) == len(set(smis)), "analogs must be deduplicated"
    assert all(Chem.MolFromSmiles(s) is not None for s in smis), "every analog must parse"
    assert all(a.transformation and a.rationale for a in analogs), "each analog needs a tag + rationale"
    # deterministic: same input -> same ordered output
    assert [a.smiles for a in generate_analogs(ALPRAZOLAM)] == smis


def test_generate_analogs_no_handles_returns_empty():
    assert generate_analogs("c1ccccc1") == []  # benzene: no Cl/methyl/N-methyl
    assert generate_analogs("not-a-smiles") == []


def test_designer_offline_generates_real_analogs(tmp_path):
    store = Store(tmp_path)
    runner = Runner(store, {
        "researcher_minion": ResearcherMinion(store),
        "designer_minion": DesignerMinion(store),
        "screener_minion": ScreenerMinion(store),
    })
    spec = ProjectSpec(
        project_id="p1", input_type="mixed", objective="characterize and design analogs",
        targets=["GABAA"], starting_materials=[ALPRAZOLAM], status="approved",
    )
    tasks = [
        TaskDefinition(task_id="research", agent="researcher_minion"),
        TaskDefinition(task_id="design", agent="designer_minion", depends_on=["research"]),
        TaskDefinition(task_id="screen", agent="screener_minion", depends_on=["design"]),
    ]
    manifest = runner.create_run(spec, tasks)
    final = runner.run(manifest.run_id, project_id="p1")
    assert final.status == RunStatus.completed

    dres = store.load_latest_agent_result("p1", manifest.run_id, "design")
    db = store.load_artifact(dres.artifacts[0].uri, DesignBatch)
    parent = _canonical(ALPRAZOLAM)
    assert db.molecules, "designer must emit candidate designs"
    # every design is a genuinely new structure, not a copy of the parent
    assert all(_canonical(m.smiles) != parent for m in db.molecules)
    # transformations are real slugs, not the old label vocabulary
    assert all(m.transformation in {"cl_to_f", "cl_to_br", "cl_to_i",
                                    "methyl_to_ethyl", "n_demethylate"}
               for m in db.molecules)

    sres = store.load_latest_agent_result("p1", manifest.run_id, "screen")
    sb = store.load_artifact(sres.artifacts[0].uri, ScreenedBatch)
    assert sb.summary.passed >= 1, "at least one analog should pass screening"
