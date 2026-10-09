"""MCP server tests: tools register and respond over an in-process MCP client."""

import json

import anyio

from ddswarm.mcp_server import tools
from ddswarm.mcp_server.server import server


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
        assert {"design_analogs", "screen_molecules", "run_pipeline"} <= names
    anyio.run(run)


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


def test_run_pipeline_tool():
    out = tools.run_pipeline({
        "project_id": "mcptest", "input_type": "mixed", "objective": "test run",
        "starting_materials": ["c1ccccc1"], "targets": [], "status": "approved",
    })
    assert out["status"] == "completed"
    assert out["report"] is not None
    assert "executive_summary" in out["report"]


def test_mcp_roundtrip_call_tool():
    async def run():
        from mcp.client.client import Client
        async with Client(server) as client:
            names = {t.name for t in (await client.list_tools()).tools}
            assert "design_analogs" in names
            res = await client.call_tool(
                "design_analogs", {"starting_materials": ["Clc1ccccc1"]})
            data = _call_result_dict(res)
            assert data["status"] == "success"
            assert "design_batch" in data["artifacts"]
    anyio.run(run)
