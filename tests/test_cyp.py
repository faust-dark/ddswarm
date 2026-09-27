"""Tests for cyp_minion + the CYP DDI chemotype library."""

from ddswarm.agents import CypMinion
from ddswarm.agents.base import AgentContext
from ddswarm.chemistry.cyp_alerts import assess_cyp
from ddswarm.contracts import ProjectSpec
from ddswarm.orchestrator import Store

INHIBITORS = {
    "fluconazole": ("CYP3A4", "OCc1nncn1C(CN1C=NC=N1)c1ccc(F)cc1"),
    "clotrimazole": ("CYP3A4", "Clc1ccccc1C(c1ccccc1)(c1ccccc1)n1ccnc1"),
    "sulfamethoxazole": ("CYP2C9", "Cc1cc(no1)NS(=O)(=O)c1ccc(N)cc1"),
    "celecoxib": ("CYP2C9", "Cc1ccc(cc1)c1cc(C(F)(F)F)nn1-c1ccc(S(N)(=O)=O)cc1"),
    "fluoxetine": ("CYP2D6", "CNCCC(c1ccccc1)Oc1ccc(C(F)(F)F)cc1"),
    "omeprazole": ("CYP2C19", "COc1ccc2nc([nH]c2c1)S(=O)Cc1ncc(C)c(OC)c1C"),
    "caffeine": ("CYP1A2", "Cn1c(=O)c2c(ncn2C)n(C)c1=O"),
}

INDUCERS = {
    "phenobarbital": ("CYP3A4", "CCC1(C(=O)NC(=O)NC1=O)c1ccccc1"),
    "phenytoin": ("CYP3A4", "O=C1NC(=O)C(c2ccccc2)(c2ccccc2)N1"),
    "carbamazepine": ("CYP3A4", "O=C1Nc2ccccc2-c2ccccc2C=C1"),
}

NEGATIVES = {
    "escitalopram": "CN(C)CCC[C@]1(c2ccc(F)cc2)OCc2cc(C#N)ccc21",
    "n_ethyl": "CCN(C)CCC[C@]1(c2ccc(F)cc2)OCc2cc(C#N)ccc21",
    "n_demethyl": "CNCCC[C@]1(c2ccc(F)cc2)OCc2cc(C#N)ccc21",
}


def test_known_inhibitors_flag_correct_enzyme():
    for name, (enzyme, smi) in INHIBITORS.items():
        hits = assess_cyp(smi)
        assert any(e == enzyme and ef == "inhibition" for e, ef, _s, _c in hits), \
            f"{name}: expected {enzyme} inhibition, got {hits}"


def test_known_inducers_flag_correct_enzyme():
    for name, (enzyme, smi) in INDUCERS.items():
        hits = assess_cyp(smi)
        assert any(e == enzyme and ef == "induction" for e, ef, _s, _c in hits), \
            f"{name}: expected {enzyme} induction, got {hits}"


def test_escitalopram_and_analogs_have_no_alerts():
    for name, smi in NEGATIVES.items():
        assert assess_cyp(smi) == [], f"{name}: got {assess_cyp(smi)}"


def test_none_and_invalid_are_empty():
    assert assess_cyp(None) == []
    assert assess_cyp("") == []
    assert assess_cyp("not-a-smiles") == []


def test_agent_no_molecules_insufficient(tmp_path):
    store = Store(tmp_path)
    spec = ProjectSpec(project_id="p", input_type="mixed", objective="o", status="approved")
    ctx = AgentContext(project_id="p", run_id="r1", task_id="cyp", spec=spec, params={}, inputs={})
    res = CypMinion(store).run(ctx)
    assert res.status.value == "insufficient_evidence"
