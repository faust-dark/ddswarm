"""MCP server tests: tools register and respond over an in-process MCP client."""

import json

import anyio

from ddswarm.mcp_server import tools
from ddswarm.mcp_server.server import server

_ALL_TOOLS = {
    "design_analogs", "screen_molecules", "predict_admet", "retrosynthesis",
    "toxicity_alerts", "cyp_alerts", "herg_flag", "formulation_assess",
    "safety_check", "pkpd_simulate", "dock_molecule", "selectivity_profile",
    "receptor_list", "receptor_lookup", "adr_profiles", "run_pipeline",
}


def _call_result_dict(res) -> dict:
    """Extract the tool's dict return from an mcp CallToolResult."""
    sc = getattr(res, "structured_content", None)
    if isinstance(sc, dict):
        return sc
    for block in getattr(res, "content", []):
        text = getattr(block, "text", None)
        if text:
            return json.loads(text)
    return {}


def test_tools_are_registered():
    async def run():
        result = await server.list_tools()  # server-side returns a plain list
        names = {t.name for t in result}
        assert _ALL_TOOLS <= names
    anyio.run(run)


# ---- design / screening ----

def test_design_analogs_tool():
    out = tools.design_analogs(["Clc1ccccc1"])
    assert out["status"] == "success"
    db = out["artifacts"]["design_batch"]
    # chlorobenzene -> F/Br/I halogen-swap analogs
    assert any(m["transformation"].startswith("cl_") for m in db["molecules"])


def test_screen_molecules_tool():
    out = tools.screen_molecules(["c1ccccc1", "O=C1CSC(=S)N1", "CC(C"])
    sb = out["artifacts"]["screened_batch"]
    assert sb["summary"]["total"] == 3
    assert sb["summary"]["passed"] == 1      # benzene
    assert sb["summary"]["rejected"] == 1    # rhodanine (PAINS)
    assert sb["summary"]["invalid"] == 1     # broken SMILES


# ---- ADMET / structure / toxicology / formulation / safety ----

def test_predict_admet_tool():
    out = tools.predict_admet(["c1ccccc1"])
    assert out["status"] == "success"
    ap = out["artifacts"]["admet_prediction"]
    assert ap["summary"]["total_molecules"] == 1


def test_retrosynthesis_tool():
    out = tools.retrosynthesis(["Clc1ccccc1"])
    assert out["status"] == "success"
    assert "retrosynthesis_plan" in out["artifacts"]


def test_toxicity_alerts_tool():
    out = tools.toxicity_alerts(["c1ccccc1"])
    assert out["status"] == "success"
    assert "tox_profile" in out["artifacts"]


def test_cyp_alerts_tool():
    out = tools.cyp_alerts(["c1ccccc1"])
    assert out["status"] == "success"
    assert "cyp_assessment" in out["artifacts"]


def test_herg_flag_tool():
    out = tools.herg_flag(["c1ccccc1"])
    assert out["status"] == "success"
    assert "herg_assessment" in out["artifacts"]


def test_formulation_assess_tool():
    out = tools.formulation_assess(["c1ccccc1"])
    assert out["status"] == "success"
    assert "formulation_assessment" in out["artifacts"]


def test_safety_check_tool():
    out = tools.safety_check(["c1ccccc1"])
    assert out["status"] == "success"
    assert out["artifacts"]["safety_assessment"]["verdict"] == "pass"


def test_pkpd_simulate_tool():
    out = tools.pkpd_simulate(["c1ccccc1"], pk_parameters=[{
        "molecule_id": "starting_material_0",
        "clearance_ml_min_kg": 12.0, "volume_of_distribution_l_kg": 4.5,
        "bioavailability": 0.9,
    }])
    assert out["status"] == "success"
    cs = out["artifacts"]["clinical_simulation"]
    assert cs["summary"]["simulated"] == 1


# ---- docking / selectivity (download-gated) ----

def test_dock_molecule_offline_gap():
    out = tools.dock_molecule(["c1ccccc1"], target="GABRA1", allow_download=False)
    assert out["status"] == "insufficient_evidence"
    assert any("local PDB" in w for w in out["warnings"])


def test_dock_molecule_requires_receptor():
    out = tools.dock_molecule(["c1ccccc1"])
    assert out["status"] == "insufficient_evidence"
    assert any("no receptor" in w for w in out["warnings"])


def test_selectivity_offline_gap():
    out = tools.selectivity_profile(["c1ccccc1"], primary="GABRA1",
                                    allow_download=False)
    assert out["status"] == "insufficient_evidence"
    assert any("local PDB" in w for w in out["warnings"])


# ---- registry / ADR data ----

def test_receptor_list_tool():
    out = tools.receptor_list()
    recs = out["artifacts"]["receptors"]
    assert len(recs) >= 30
    names = {r["name"] for r in recs}
    assert "GABRA1" in names
    assert all("pdb_id" in r for r in recs)


def test_receptor_list_family_filter():
    out = tools.receptor_list(family="GABA")
    recs = out["artifacts"]["receptors"]
    assert recs and all("GABA" in r["family"] for r in recs)


def test_receptor_lookup_tool():
    out = tools.receptor_lookup("GABRA1")
    assert out["status"] == "success"
    assert out["artifacts"]["receptor"]["gene"] == "GABRA1"


def test_receptor_lookup_unknown():
    out = tools.receptor_lookup("NOT_A_RECEPTOR")
    assert out["status"] == "insufficient_evidence"


def test_adr_profiles_tool():
    out = tools.adr_profiles()
    assert out["status"] == "success"
    assert "special_adr_profiles" in out["artifacts"]
    assert "antitarget_profiles" in out["artifacts"]


# ---- aggregate pipeline ----

def test_run_pipeline_tool():
    out = tools.run_pipeline({
        "project_id": "mcptest", "input_type": "mixed", "objective": "test run",
        "starting_materials": ["c1ccccc1"], "targets": [], "status": "approved",
    })
    assert out["status"] == "completed"
    assert out["report"] is not None
    assert "executive_summary" in out["report"]


# ---- in-process MCP round-trip ----

def test_mcp_roundtrip_call_tool():
    async def run():
        from mcp.client.client import Client
        async with Client(server) as client:
            names = {t.name for t in (await client.list_tools()).tools}
            assert _ALL_TOOLS <= names

            res = await client.call_tool(
                "design_analogs", {"starting_materials": ["Clc1ccccc1"]})
            data = _call_result_dict(res)
            assert data["status"] == "success"
            assert "design_batch" in data["artifacts"]

            res2 = await client.call_tool("receptor_lookup", {"name": "GABRA1"})
            data2 = _call_result_dict(res2)
            assert data2["artifacts"]["receptor"]["gene"] == "GABRA1"
    anyio.run(run)
