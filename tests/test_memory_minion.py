"""Tests for memory_minion (deterministic GenerationMemory aggregation)."""

from ddswarm.agents import MemoryMinion
from ddswarm.agents.base import AgentContext
from ddswarm.contracts import (
    AgentResultStatus,
    GenerationMemory,
    ProjectSpec,
    Report,
    ScreenSummary,
    ScreenedBatch,
    MoleculeScreenResult,
)
from ddswarm.contracts.report import CandidateSelection
from ddswarm.orchestrator import Store
from ddswarm.orchestrator.memory import MemoryStore


def _spec():
    return ProjectSpec(project_id="p", input_type="mixed", objective="inhibit EGFR",
                       targets=["EGFR"], starting_materials=["c1ccccc1"], status="approved")


def _report():
    return Report(
        project_id="p", run_id="r", status=AgentResultStatus.success, run_status="completed",
        executive_summary="x", ranked_candidates=[
            CandidateSelection(rank=1, smiles="c1ccccc1Cl", score=1.2,
                               criteria=["QED: 0.5"]),
            CandidateSelection(rank=2, smiles="c1ccccc1F", score=0.9,
                               criteria=["QED: 0.4"]),
        ],
        gaps=["no docking performed this run"],
    )


def _screened():
    return ScreenedBatch(
        project_id="p", status=AgentResultStatus.success,
        summary=ScreenSummary(total=2, passed=1, rejected=1, invalid=0),
        molecules=[
            MoleculeScreenResult(id="a", input_smiles="c1ccccc1Cl", canonical_smiles="c1ccccc1Cl",
                                 valid=True, passed=True),
            MoleculeScreenResult(id="b", input_smiles="c1ccccc1I", canonical_smiles="c1ccccc1I",
                                 valid=True, passed=False),
        ],
    )


def test_memory_aggregates(tmp_path):
    store = Store(tmp_path)
    ctx = AgentContext(project_id="p", run_id="r", task_id="memory", spec=_spec(),
                       params={"generation": 2},
                       inputs={"report": [_report()], "screened_batch": [_screened()]})
    res = MemoryMinion(store).run(ctx)
    assert res.status.value == "success"
    mem = store.load_artifact(res.artifacts[0].uri, GenerationMemory)
    assert mem.generation == 2
    assert [c.smiles for c in mem.best_candidates] == ["c1ccccc1Cl", "c1ccccc1F"]
    assert "c1ccccc1I" in mem.avoided  # rejected
    assert "c1ccccc1Cl" not in mem.avoided  # survivor not flagged as avoid
    assert any("no docking" in i for i in mem.insights)

    # durable 'generation' entry written to project memory for the next generation
    entries = MemoryStore(store).by_kind("p", "generation")
    assert len(entries) == 1
    assert entries[0].data["generation"] == 2
    assert len(entries[0].data["best_candidates"]) == 2


def test_memory_no_survivors_converged(tmp_path):
    store = Store(tmp_path)
    ctx = AgentContext(project_id="p", run_id="r", task_id="memory", spec=_spec(),
                       params={}, inputs={"report": [
                           Report(project_id="p", run_id="r", status=AgentResultStatus.success,
                                  run_status="completed", executive_summary="x",
                                  ranked_candidates=[]),
                       ]})
    res = MemoryMinion(store).run(ctx)
    mem = store.load_artifact(res.artifacts[0].uri, GenerationMemory)
    assert mem.best_candidates == []
    assert mem.converged is True
