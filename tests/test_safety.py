"""Tests for safety_minion + the curated controlled-substance/dual-use lists."""

from ddswarm.agents import SafetyMinion
from ddswarm.agents.base import AgentContext
from ddswarm.chemistry.rdkit_ops import sanitize_and_canonicalize
from ddswarm.chemistry.safety import match_controlled_substance, match_dual_use_target
from ddswarm.contracts import ProjectSpec, SafetyAssessment
from ddswarm.orchestrator import Store

FENTANYL = "CCC(=O)N(c1ccccc1)c1ccc(CCN2CCCCC2)cc1"


def _spec(targets=None, materials=None):
    return ProjectSpec(project_id="p", input_type="mixed", objective="inhibit X",
                       targets=targets or ["EGFR"],
                       starting_materials=materials or ["c1ccccc1"], status="approved")


def _run(store, spec, params=None):
    ctx = AgentContext(project_id="p", run_id="r", task_id="safety", spec=spec,
                       params=params or {}, inputs={})
    return SafetyMinion(store).run(ctx)


def _loaded(store, result):
    assert result.artifacts
    return store.load_artifact(result.artifacts[0].uri, SafetyAssessment)


# ---- chemistry ----
def test_controlled_substance_match():
    f = sanitize_and_canonicalize(FENTANYL).canonical_smiles
    assert [e["name"] for e in match_controlled_substance(f)] == ["fentanyl"]
    assert match_controlled_substance(sanitize_and_canonicalize("c1ccccc1").canonical_smiles) == []


def test_dual_use_target_match():
    assert [e["name"] for e in match_dual_use_target("inhibit ricin A chain")] == ["ricin"]
    assert match_dual_use_target("EGFR kinase") == []


# ---- agent ----
def test_safety_pass(tmp_path):
    store = Store(tmp_path)
    result = _run(store, _spec())
    assert result.status.value == "success"
    sa = _loaded(store, result)
    assert sa.verdict == "pass"
    assert sa.flags == []


def test_safety_blocks_fentanyl(tmp_path):
    store = Store(tmp_path)
    result = _run(store, _spec(materials=[FENTANYL]))
    assert result.status.value == "blocked"
    sa = _loaded(store, result)
    assert sa.verdict == "block"
    assert sa.summary.block == 1
    assert sa.flags[0].kind == "controlled_substance"
    assert sa.flags[0].evidence_level.value == "observed"


def test_safety_blocks_dual_use_target(tmp_path):
    store = Store(tmp_path)
    result = _run(store, _spec(targets=["ricin A chain"]))
    assert result.status.value == "blocked"
    sa = _loaded(store, result)
    assert sa.verdict == "block"
    assert sa.flags[0].kind == "dual_use"
