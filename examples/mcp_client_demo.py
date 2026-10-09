"""Drive ddswarm from any MCP client — minimal stdio example.

ddswarm is exposed as an MCP server (16 tools). This script shows the two ways
a calling agent (OpenMinion, Google ADK, Hermes, Claude, …) can use it:

  A) granular tools      — receptor_lookup -> design_analogs -> screen_molecules
                           (deterministic, runs in seconds)
  B) one-shot run_pipeline — the full 18-agent pipeline from a ProjectSpec dict
                           (~4 min offline; longer with real docking/PDB download)

Usage:

    pip install mcp
    python -u examples/mcp_client_demo.py      # -u = live output

The server binary is resolved from $DDWARM_MCP, defaulting to ``ddswarm-mcp``
(assumed on PATH, e.g. after ``pip install -e .``).
"""
import asyncio
import json
import os

from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

DDWARM = os.environ.get("DDWARM_MCP", "ddswarm-mcp")
DIAZEPAM = "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21"


def p(*a):
    print(*a, flush=True)


def payload(result) -> dict:
    """Unpack a CallToolResult into a plain dict (join text blocks -> JSON)."""
    text = "\n".join(getattr(b, "text", "") for b in result.content).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}


async def main():
    params = StdioServerParameters(command=DDWARM, args=[])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 0. discover the tools
            tools = await session.list_tools()
            names = [t.name for t in tools.tools]
            p(f"[0] tools/list -> {len(names)} tools")
            p("    " + ", ".join(names))

            # A) granular (fast, deterministic)
            r = payload(await session.call_tool("receptor_lookup", {"name": "GABRA2"}))
            rec = r["artifacts"]["receptor"]
            p(f"\n[A1] receptor_lookup(GABRA2) -> {rec['gene']} (PDB {rec.get('pdb_id')})")

            r = payload(await session.call_tool("design_analogs", {"starting_materials": [DIAZEPAM]}))
            mols = r["artifacts"]["design_batch"]["molecules"]
            p(f"[A2] design_analogs -> {r['status']}, {len(mols)} analog(s):")
            for m in mols[:3]:
                p("     ", m["smiles"])

            smis = [DIAZEPAM] + [m["smiles"] for m in mols[:2]]
            r = payload(await session.call_tool("screen_molecules", {"smiles": smis}))
            p(f"[A3] screen_molecules -> {r['status']}, summary={r['artifacts']['screened_batch']['summary']}")

            # B) one-shot pipeline (ligand-only: docking/selectivity degrade to gaps)
            project = {
                "project_id": "demo",
                "objective": "characterize diazepam analogs",
                "targets": [],
                "starting_materials": [DIAZEPAM],
                "input_type": "ligand",
                "status": "approved",
            }
            p("\n[B] run_pipeline (full 18-agent run, ~4 min offline)...")
            r = payload(await session.call_tool("run_pipeline", {"project": project}))
            rep = r.get("report") or {}
            p(f"[B] run_pipeline -> status={r.get('status')}, run_id={r.get('run_id')}")
            p("    summary:", (rep.get("executive_summary") or "")[:140])
            top = rep.get("identified_candidate") or (rep.get("ranked_candidates") or [None])[0]
            if top:
                p("    identified candidate:", top.get("smiles"), "|", top.get("name") or top.get("identifier"))
            p(f"    ranked_candidates: {len(rep.get('ranked_candidates') or [])}")


if __name__ == "__main__":
    asyncio.run(main())
