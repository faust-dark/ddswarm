"""ddswarm MCP server: the drug-discovery swarm exposed as MCP tools.

Any MCP-capable agentic stack (Google ADK/AG2, Hermes, OpenMinion, Claude, ...)
can connect over stdio and drive the deterministic drug-discovery pipeline.
"""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer

from . import tools

server = MCPServer(
    name="ddswarm",
    title="ddswarm — deterministic drug-discovery swarm",
    description=(
        "Reproducible, traceable, verifiable multi-agent drug-discovery tools: "
        "analog design, molecular screening, and a full end-to-end pipeline "
        "(design -> screen -> docking -> selectivity -> ADMET -> PK/PD -> "
        "toxicology -> retrosynthesis -> report). Deterministic by default; no "
        "LLM or harness dependency."
    ),
    version="0.1.0",
)


@server.tool()
def design_analogs(starting_materials: list[str]) -> dict[str, Any]:
    """Generate new analog molecules from starting SMILES via the deterministic
    RDKit transformation library (halogen swap, homologation, N-demethylation).
    Returns the design batch: candidate SMILES with transformation + rationale."""
    return tools.design_analogs(starting_materials)


@server.tool()
def screen_molecules(smiles: list[str], filters: dict[str, Any] | None = None) -> dict[str, Any]:
    """Screen molecules against hard filters (validity, rule-of-five, PAINS) and
    advisory checks (TPSA, rotatable bonds, formal charge, Brenk). Returns
    per-molecule properties, alerts, and pass/reject verdicts."""
    return tools.screen_molecules(smiles, filters)


@server.tool()
def run_pipeline(project: dict[str, Any]) -> dict[str, Any]:
    """Run the full deterministic 18-agent pipeline from a ProjectSpec dict
    (project_id, objective, targets, starting_materials, ...). Returns the run
    status plus the structured report (ranked candidates, identified candidate,
    gaps). Docking/selectivity resolve their receptor from the targets and
    degrade non-blocking when unavailable."""
    return tools.run_pipeline(project)


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
