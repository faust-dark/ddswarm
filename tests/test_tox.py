"""Tests for tox_minion + the curated toxicophore alert library (offline)."""

from rdkit import Chem

from ddswarm.agents import ToxMinion
from ddswarm.agents.base import AgentContext
from ddswarm.chemistry.tox_alerts import TOX_ALERTS, match_tox_alerts
from ddswarm.contracts import ProjectSpec, ToxProfile
from ddswarm.orchestrator import Store

GEFITINIB = "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4"
NITROBENZENE = "c1ccccc1[N+](=O)[O-]"


def _spec():
    return ProjectSpec(project_id="p", input_type="mixed", objective="inhibit EGFR",
                       targets=["EGFR"], starting_materials=[GEFITINIB], status="approved")


def _run(store, molecules):
    ctx = AgentContext(project_id="p", run_id="r", task_id="tox", spec=_spec(),
                       params={"molecules": molecules})
    return ToxMinion(store).run(ctx)


def _profile(store, result):
    assert result.artifacts
    return store.load_artifact(result.artifacts[0].uri, ToxProfile)


# ---- alert library ----
def test_library_has_sources():
    assert len(TOX_ALERTS) == 22
    for a in TOX_ALERTS:
        assert a["name"] and a["smarts"] and a["endpoint"] and a["source"]


def test_nitrobenzene_flags_nitro():
    alerts = match_tox_alerts(Chem.MolFromSmiles(NITROBENZENE))
    names = [a["name"] for a in alerts]
    assert "nitro group" in names


def test_new_toxicophores_flag_reference_compounds():
    cases = {
        "aromatic primary amine (aniline)": "Nc1ccccc1",
        "aromatic azo": "c1ccc(-N=N-c2ccccc2)cc1",
        "azide": "[N-]=[N+]=Nc1ccccc1",
        "hydrazine / hydrazide": "NNC(=O)c1ccncc1",  # isoniazid
        "isocyanate": "CN=C=O",
        "isothiocyanate": "C=CCN=C=S",
        "acyl halide": "CC(=O)Cl",
        "alpha-halo carbonyl": "BrCC(=O)c1ccccc1",  # phenacyl bromide
        "sulfonate ester": "COS(=O)(=O)C",  # methyl mesylate
        "C-nitroso": "O=Nc1ccccc1",
        "vinyl halide": "C=CCl",
        "catechol / ortho-diphenol": "Oc1ccccc1O",
        "1,2-dicarbonyl (glyoxal)": "O=CC=O",
    }
    for name, smi in cases.items():
        alerts = match_tox_alerts(Chem.MolFromSmiles(smi))
        names = [a["name"] for a in alerts]
        assert name in names, f"{name} not flagged for {smi}"


def test_gefitinib_no_alerts():
    assert match_tox_alerts(Chem.MolFromSmiles(GEFITINIB)) == []


# ---- agent ----
def test_tox_flags_nitro(tmp_path):
    store = Store(tmp_path)
    result = _run(store, [{"id": "m0", "smiles": NITROBENZENE}])
    assert result.status.value == "success"
    profile = _profile(store, result)
    assert profile.summary.molecules_with_alerts == 1
    m = profile.molecules[0]
    assert m.liability_count >= 1
    assert all(a.evidence_level.value == "predicted" for a in m.alerts)
    assert "NOT experimental" in (m.risk_note or "")


def test_tox_clean_gefitinib(tmp_path):
    store = Store(tmp_path)
    result = _run(store, [{"id": "m0", "smiles": GEFITINIB}])
    profile = _profile(store, result)
    assert profile.summary.total_alerts == 0
    assert profile.molecules[0].alerts == []


def test_tox_no_molecules_insufficient(tmp_path):
    store = Store(tmp_path)
    spec = ProjectSpec(project_id="p", input_type="mixed", objective="x", status="approved")
    ctx = AgentContext(project_id="p", run_id="r", task_id="tox", spec=spec,
                       params={"molecules": []}, inputs={})
    result = ToxMinion(store).run(ctx)
    assert result.status.value == "insufficient_evidence"
