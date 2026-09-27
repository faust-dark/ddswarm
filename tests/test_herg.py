"""Tests for herg_minion + the hERG pharmacophore/lipophilicity rule."""

from ddswarm.agents import HergMinion
from ddswarm.agents.base import AgentContext
from ddswarm.chemistry.herg_alerts import assess_herg
from ddswarm.contracts import ProjectSpec
from ddswarm.orchestrator import Store

# Classic hERG blockers (potent, must flag).
BLOCKERS = {
    "terfenadine": "CC(C)(C)c1ccc(cc1)C(O)CCCN1CCC(CC1)C(O)(c1ccccc1)c1ccccc1",
    "haloperidol": "OC1(CCN(CCCC(=O)c2ccc(F)cc2)CC1)c1ccc(Cl)cc1",
    "cisapride": "COc1cc(cc(c1OC)Cl)C(=O)NC1CCN(CCCOc2ccc(F)cc2)CC1",
    "droperidol": "O=C1Nc2ccccc2N1C1CCN(CC1)CCCC(=O)c1ccc(F)cc1",
    "astemizole": "COc1ccc(N2CCC(CC2)Nc2nc3ccccc3[nH]2)cc1",
    "pimozide": "Fc1ccc(cc1)C1CCN(CC1)CCCCNC(=O)Nc1ccccc1N1CCN(CC1)c1ccccc1F",
    "quinidine": "COc1ccc2nccc([C@@H](O)[C@H]3CC4CCN3C[C@@H]4C=C)c2c1",
}

# SSRIs — modest hERG liability; the coarse pharmacophore correctly flags them
# (they are basic lipophilic amines with 2 aromatics), which is NOT a false
# positive for the class, only a non-ranking of potency.
MODERATE = {
    "fluoxetine": "CNCCC(c1ccccc1)Oc1ccc(C(F)(F)F)cc1",
    "escitalopram": "CN(C)CCC[C@]1(c2ccc(F)cc2)OCc2cc(C#N)ccc21",
    "citalopram": "CN(C)CCCC1(c2ccc(F)cc2)OCc2cc(C#N)ccc21",
}

# Not hERG blockers under the rule (missing basic amine, <2 aromatics, or too polar).
NEGATIVES = {
    "atenolol": "CC(C)NCC(O)c1ccc(OCC(N)=O)cc1",
    "pravastatin": "CCC(C)C(O)(CC(O)C1CC(O)CC2=CC(OC(=O)NC(CC(O)C(O)C1C2(C)C)C(=O)O)CC(C)C)C(=O)O",
    "warfarin": "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",
}


def test_classic_blockers_flag():
    for name, smi in BLOCKERS.items():
        assert assess_herg(smi)["flagged"], f"{name} must flag hERG"


def test_ssris_flag_as_modest_liability():
    for name, smi in MODERATE.items():
        assert assess_herg(smi)["flagged"], f"{name} must flag (modest hERG liability)"


def test_negatives_clear():
    for name, smi in NEGATIVES.items():
        assert not assess_herg(smi)["flagged"], f"{name} must clear hERG"


def test_warfarin_discriminates_on_basic_amine():
    # 3 aromatic rings + cLogP 3.61 but no basic amine -> clear (the key test).
    call = assess_herg(NEGATIVES["warfarin"])
    assert call["aromatic_rings"] >= 2 and call["clogp"] >= 3.0
    assert call["basic_amine_motifs"] == [] and not call["flagged"]


def test_none_and_invalid_clear():
    assert not assess_herg(None)["flagged"]
    assert not assess_herg("")["flagged"]
    assert not assess_herg("not-a-smiles")["flagged"]


def test_agent_no_molecules_insufficient(tmp_path):
    store = Store(tmp_path)
    spec = ProjectSpec(project_id="p", input_type="mixed", objective="o", status="approved")
    ctx = AgentContext(project_id="p", run_id="r1", task_id="herg", spec=spec, params={}, inputs={})
    res = HergMinion(store).run(ctx)
    assert res.status.value == "insufficient_evidence"
