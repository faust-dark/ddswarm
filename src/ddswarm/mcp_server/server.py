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
        "analog design, molecular screening, docking, selectivity, ADMET, PK/PD, "
        "toxicology, retrosynthesis, safety, formulation, and a full end-to-end "
        "pipeline. Deterministic by default; no LLM or harness dependency."
    ),
    version="0.1.0",
)


# ---- design / screening ----

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


# ---- ADMET / structure / toxicology / formulation / safety ----

@server.tool()
def predict_admet(smiles: list[str], providers: list[str] | None = None) -> dict[str, Any]:
    """Predict ADMET endpoints (solubility, logP, QED, Ro5, Veber, Egan) with the
    deterministic rule_admet provider. Returns per-molecule endpoint predictions."""
    return tools.predict_admet(smiles, providers)


@server.tool()
def retrosynthesis(smiles: list[str]) -> dict[str, Any]:
    """Propose retrosynthetic disconnections (synthetic accessibility + BRICS).
    Returns disconnection trees with synthon SMILES."""
    return tools.retrosynthesis(smiles)


@server.tool()
def toxicity_alerts(smiles: list[str]) -> dict[str, Any]:
    """Flag toxicophore alerts (deterministic SMARTS). Returns per-molecule
    matched alerts with severity."""
    return tools.toxicity_alerts(smiles)


@server.tool()
def cyp_alerts(smiles: list[str]) -> dict[str, Any]:
    """Flag CYP-interaction alerts (deterministic SMARTS). Returns per-molecule
    CYP liabilities."""
    return tools.cyp_alerts(smiles)


@server.tool()
def herg_flag(smiles: list[str]) -> dict[str, Any]:
    """Flag hERG (cardiotoxicity) liability (deterministic SMARTS). Returns
    per-molecule hERG risk flags."""
    return tools.herg_flag(smiles)


@server.tool()
def formulation_assess(smiles: list[str]) -> dict[str, Any]:
    """Assess formulation/biopharmaceutics (BCS class, solubility-driven risks)."""
    return tools.formulation_assess(smiles)


@server.tool()
def safety_check(smiles: list[str], targets: list[str] | None = None) -> dict[str, Any]:
    """Run the hard safety gate (controlled-substance + dual-use target matching,
    exact canonical-SMILES). Returns a pass/flag/block verdict. This is the only
    hard gate in the pipeline."""
    return tools.safety_check(smiles, targets)


@server.tool()
def pkpd_simulate(
    smiles: list[str],
    pk_parameters: list[dict[str, Any]] | None = None,
    dose_mg: float | None = None,
    route: str = "oral",
) -> dict[str, Any]:
    """Run the deterministic PK/PD simulation (1/2/multi-compartment RK4 with
    mechanistic PD). Pass pk_parameters per molecule (clearance + Vd, or t1/2 +
    one of them); otherwise they are predicted/inferred."""
    return tools.pkpd_simulate(smiles, pk_parameters, dose_mg, route)


# ---- docking / selectivity (receptor-driven, download-gated) ----

@server.tool()
def dock_molecule(
    smiles: list[str],
    receptor: dict[str, Any] | None = None,
    target: str | None = None,
    allow_download: bool = True,
    box: dict[str, Any] | None = None,
    exhaustiveness: int = 8,
    num_modes: int = 9,
    seed: int = 42,
) -> dict[str, Any]:
    """Dock molecules (Vina) against a receptor. Receptor is either an explicit
    source ({kind: pdb_id/file, identifier/pdb_id}) or a registry 'target' name.
    With allow_download=false, a pdb_id that needs downloading returns
    insufficient_evidence (pass a local PDB file instead)."""
    return tools.dock_molecule(smiles, receptor, target, allow_download, box,
                               exhaustiveness, num_modes, seed)


@server.tool()
def selectivity_profile(
    smiles: list[str],
    primary: str,
    off_targets: list[str] | None = None,
    receptors: list[dict[str, Any]] | None = None,
    allow_download: bool = True,
    reference_concentration_molar: float = 1e-8,
) -> dict[str, Any]:
    """Compute selectivity/ADR profile (docking -> Ki -> occupancy -> selectivity
    index + ADR library). 'primary' is the on-target; 'off_targets' the
    antitargets. With allow_download=false, receptors needing download return
    insufficient_evidence."""
    return tools.selectivity_profile(smiles, primary, off_targets, receptors,
                                     allow_download, reference_concentration_molar)


# ---- registry / ADR data ----

@server.tool()
def receptor_list(family: str | None = None) -> dict[str, Any]:
    """List the built-in receptor registry (~61 RCSB-verified entries). Optionally
    filter by family substring (e.g. 'GABA', 'serotonin', 'dopamine')."""
    return tools.receptor_list(family)


@server.tool()
def receptor_lookup(name: str) -> dict[str, Any]:
    """Resolve a receptor name/alias (e.g. 'GABRA1', 'PTGS2', '5-HT2C') to its
    registry entry (gene, PDB id, reference)."""
    return tools.receptor_lookup(name)


@server.tool()
def adr_profiles() -> dict[str, Any]:
    """Return the built-in ADR profiles (special profiles + antitarget profiles)
    used by the selectivity agent for adverse-effect classification."""
    return tools.adr_profiles()


# ---- aggregate pipeline ----

@server.tool()
def run_pipeline(project: dict[str, Any], iupac: bool = False) -> dict[str, Any]:
    """Run the full deterministic 18-agent pipeline from a ProjectSpec dict
    (project_id, objective, targets, starting_materials, ...). Returns the run
    status plus the structured report (ranked candidates, identified candidate,
    gaps). Docking/selectivity resolve their receptor from the targets and
    degrade non-blocking when unavailable. Set iupac=true to also fetch IUPAC
    names for the ranked candidates (network; ~15s each; labels only)."""
    return tools.run_pipeline(project, iupac)


def main() -> None:
    """Entry point: serve the swarm over stdio (default) or HTTP.

    stdio is the universal MCP transport; streamable-http exposes the same 16
    tools over a remote HTTP endpoint (e.g. for a hosted/shared instance).
    """
    import argparse
    import sys

    p = argparse.ArgumentParser(
        prog="ddswarm-mcp",
        description="ddswarm MCP server (deterministic drug-discovery tools).",
    )
    p.add_argument(
        "--transport", choices=["stdio", "sse", "streamable-http"], default="stdio",
        help="transport to serve (default: stdio)",
    )
    p.add_argument("--host", default="127.0.0.1", help="bind host (sse/streamable-http)")
    p.add_argument("--port", type=int, default=8000, help="bind port (sse/streamable-http)")
    p.add_argument(
        "--path", default=None,
        help="endpoint path (default: /mcp for streamable-http, /sse for sse)",
    )
    p.add_argument(
        "--stateless-http", action="store_true",
        help="streamable-http: no session tracking (each request is self-contained)",
    )
    args = p.parse_args()

    if args.transport != "stdio":
        # remote callers: registry names / PDB ids only (no local files or URLs)
        tools.set_network_mode(True)
        if args.host not in ("127.0.0.1", "localhost", "::1"):
            print(f"ddswarm-mcp: WARNING: serving on {args.host} without authentication; "
                  "put it behind an authenticating reverse proxy", file=sys.stderr)

    if args.transport == "stdio":
        server.run(transport="stdio")
    elif args.transport == "sse":
        kw: dict[str, object] = {"host": args.host, "port": args.port}
        if args.path:
            kw["sse_path"] = args.path
        server.run(transport="sse", **kw)
    else:
        kw = {"host": args.host, "port": args.port}
        if args.path:
            kw["streamable_http_path"] = args.path
        if args.stateless_http:
            kw["stateless_http"] = True
        server.run(transport="streamable-http", **kw)


if __name__ == "__main__":
    main()
