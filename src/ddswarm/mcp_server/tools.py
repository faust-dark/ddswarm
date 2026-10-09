"""Thin stateless adapters: each MCP tool maps to one ddswarm agent.

Every adapter builds a throwaway Store, runs the agent against a minimal
ProjectSpec derived from the tool inputs, and returns a JSON dict embedding the
full artifact payload(s) — so an MCP client never has to read the temp store.

The swarm stays deterministic and platform-agnostic here: the reasoning backend
is forced to "none" (no embedded LLM — the MCP client *is* the reasoning layer).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from ..agents import (
    ClinicalMinion,
    CypMinion,
    DesignerMinion,
    DockingMinion,
    FormulationMinion,
    HergMinion,
    PredictorMinion,
    RetrosynthesisMinion,
    SafetyMinion,
    ScreenerMinion,
    SelectivityMinion,
    ToxMinion,
)
from ..agents.base import AgentContext
from ..chemistry.receptors import RECEPTOR_REGISTRY, registry_entry
from ..chemistry.selectivity import ANTITARGET_PROFILES, SPECIAL_ADR_PROFILES
from ..contracts.enums import InputType, ProjectStatus
from ..contracts.project_spec import ProjectSpec
from ..contracts.report import Report
from ..docking.base import pdb_cache_path
from ..orchestrator.runner import ARTIFACT_MODELS
from ..orchestrator.store import Store

_PROJECT_ID = "mcp"
_OFFLINE_HINT = (
    "receptor unavailable offline — provide a local PDB "
    '("receptor": {"kind": "file", "identifier": "/path/to/receptor.pdb"}) '
    "or set allow_download=true"
)


def _fresh_store() -> Store:
    """A throwaway store per tool call (stateless tools)."""
    return Store(Path(tempfile.mkdtemp(prefix="ddswarm_mcp_")))


def _spec(**overrides: Any) -> ProjectSpec:
    base: dict[str, Any] = {
        "project_id": _PROJECT_ID,
        "input_type": InputType.mixed,
        "objective": "ddswarm MCP tool call",
        "status": ProjectStatus.approved,
    }
    base.update(overrides)
    return ProjectSpec(**base)


def _ctx(spec: ProjectSpec, params: dict[str, Any]) -> AgentContext:
    params = dict(params)
    params.setdefault("reasoning_backend", "none")
    return AgentContext(
        project_id=spec.project_id, run_id="mcp", task_id="tool",
        spec=spec, params=params,
    )


def _result_payload(result: Any) -> dict[str, Any]:
    """Normalize an AgentResult into a JSON dict with full artifact payloads."""
    payload: dict[str, Any] = {
        "status": result.status.value,
        "findings": result.findings,
        "warnings": result.warnings,
        "errors": result.errors,
        "artifacts": {},
    }
    for art in result.artifacts:
        if art.uri and art.kind in ARTIFACT_MODELS:
            data = ARTIFACT_MODELS[art.kind].model_validate_json(Path(art.uri).read_text())
            payload["artifacts"][art.kind] = data.model_dump()
    return payload


def _gap(message: str) -> dict[str, Any]:
    return {
        "status": "insufficient_evidence",
        "findings": [], "warnings": [message], "errors": [],
        "artifacts": {},
    }


def _run(agent_cls: Any, smiles: list[str], params: dict[str, Any] | None = None,
         targets: list[str] | None = None) -> dict[str, Any]:
    """Run one agent against ``smiles`` (passed as starting_materials, the
    universal molecule fallback for every agent) and return its payload."""
    try:
        store = _fresh_store()
        spec = _spec(starting_materials=list(smiles), targets=list(targets or []))
        result = agent_cls(store).run(_ctx(spec, params or {}))
        return _result_payload(result)
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "findings": [], "warnings": [],
                "errors": [f"{agent_cls.__name__} raised: {exc}"], "artifacts": {}}


# ---- design / screening ----

def design_analogs(starting_materials: list[str]) -> dict[str, Any]:
    """Generate new analog molecules from starting SMILES (deterministic)."""
    return _run(DesignerMinion, starting_materials)


def screen_molecules(smiles: list[str], filters: dict[str, Any] | None = None) -> dict[str, Any]:
    """Screen molecules against hard + advisory filters (deterministic)."""
    return _run(ScreenerMinion, smiles, {"filters": filters} if filters else None)


# ---- ADMET / structure / toxicology / formulation / safety ----

def predict_admet(smiles: list[str], providers: list[str] | None = None) -> dict[str, Any]:
    """Predict ADMET endpoints (default: the deterministic rule_admet provider)."""
    return _run(PredictorMinion, smiles, {"providers": providers or ["rule_admet"]})


def retrosynthesis(smiles: list[str]) -> dict[str, Any]:
    """Propose retrosynthetic disconnections (SA + BRICS, deterministic)."""
    return _run(RetrosynthesisMinion, smiles)


def toxicity_alerts(smiles: list[str]) -> dict[str, Any]:
    """Flag toxicophore alerts (deterministic SMARTS)."""
    return _run(ToxMinion, smiles)


def cyp_alerts(smiles: list[str]) -> dict[str, Any]:
    """Flag CYP-interaction alerts (deterministic SMARTS)."""
    return _run(CypMinion, smiles)


def herg_flag(smiles: list[str]) -> dict[str, Any]:
    """Flag hERG liability (deterministic SMARTS)."""
    return _run(HergMinion, smiles)


def formulation_assess(smiles: list[str]) -> dict[str, Any]:
    """Assess formulation/biopharmaceutics (BCS etc.)."""
    return _run(FormulationMinion, smiles)


def safety_check(smiles: list[str], targets: list[str] | None = None) -> dict[str, Any]:
    """Run the hard safety gate (controlled substances + dual-use targets)."""
    return _run(SafetyMinion, smiles, targets=targets)


def pkpd_simulate(smiles: list[str], pk_parameters: list[dict[str, Any]] | None = None,
                  dose_mg: float | None = None, route: str = "oral") -> dict[str, Any]:
    """Run the deterministic PK/PD simulation (RK4, mechanistic PD)."""
    params: dict[str, Any] = {"route": route}
    if pk_parameters:
        params["pk_parameters"] = pk_parameters
    if dose_mg is not None:
        params["dose_mg"] = dose_mg
    return _run(ClinicalMinion, smiles, params)


# ---- docking / selectivity (receptor-driven, download-gated) ----

def _receptor_identifier(rc: dict[str, Any]) -> str:
    """Resolve the cache key (pdb_id) for a receptor config, or '' if unknown."""
    ident = rc.get("identifier") or rc.get("pdb_id")
    if ident:
        return str(ident)
    name = str(rc.get("name", ""))
    entry = registry_entry(name) if name else None
    return str(entry.get("pdb_id", "")) if entry else ""


def _offline_ok(rc: dict[str, Any]) -> bool:
    """True when a receptor needs no download: a local file, an already-cached
    PDB, or no resolvable PDB at all (downstream reports no_receptor, not a
    download gap)."""
    if rc.get("kind", "pdb_id") == "file":
        return True
    ident = _receptor_identifier(rc)
    return not ident or pdb_cache_path(ident).exists()


def dock_molecule(smiles: list[str], receptor: dict[str, Any] | None = None,
                  target: str | None = None, allow_download: bool = True,
                  box: dict[str, Any] | None = None,
                  exhaustiveness: int = 8, num_modes: int = 9,
                  seed: int = 42) -> dict[str, Any]:
    """Dock molecules against a receptor (registry name or explicit source)."""
    if receptor is None and target is None:
        return _gap("no receptor configured — pass 'receptor' ({kind, identifier/pdb_id}) "
                    "or 'target' (registry name)")
    receptor_cfg = receptor or {}
    check_rc = receptor_cfg if receptor is not None else {"name": target}
    if not allow_download and not _offline_ok(check_rc):
        return _gap(_OFFLINE_HINT)
    params: dict[str, Any] = {
        "exhaustiveness": exhaustiveness, "num_modes": num_modes, "seed": seed,
    }
    if box:
        params["box"] = box
    if receptor is not None:
        params["receptor"] = receptor_cfg
    targets: list[str] = []
    if receptor is None:
        assert target is not None  # guarded by the early return above
        targets = [target]
    return _run(DockingMinion, smiles, params, targets=targets)


def selectivity_profile(smiles: list[str], primary: str,
                        off_targets: list[str] | None = None,
                        receptors: list[dict[str, Any]] | None = None,
                        allow_download: bool = True,
                        reference_concentration_molar: float = 1e-8) -> dict[str, Any]:
    """Compute selectivity/ADR profile (docking -> Ki -> occupancy -> SI)."""
    if receptors is None:
        receptors = [{"name": primary}] + [{"name": o} for o in (off_targets or [])]
    if not allow_download and not all(_offline_ok(rc) for rc in receptors):
        return _gap(_OFFLINE_HINT)
    params: dict[str, Any] = {
        "receptors": receptors,
        "primary": primary,
        "reference_concentration_molar": reference_concentration_molar,
    }
    return _run(SelectivityMinion, smiles, params, targets=[primary])


# ---- registry / ADR data (deterministic, no agent) ----

def receptor_list(family: str | None = None) -> dict[str, Any]:
    """List the built-in receptor registry (RCSB-verified PDB entries)."""
    rows = []
    for _name, e in sorted(RECEPTOR_REGISTRY.items()):
        if family and family.lower() not in str(e.get("family", "")).lower():
            continue
        rows.append({
            "name": e["name"], "gene": e.get("gene"), "family": e.get("family"),
            "pdb_id": e.get("pdb_id"), "reference": e.get("reference"),
        })
    return {"status": "success", "findings": [f"{len(rows)} receptor(s)"],
            "warnings": [], "errors": [], "artifacts": {"receptors": rows}}


def receptor_lookup(name: str) -> dict[str, Any]:
    """Resolve a receptor name/alias to its registry entry."""
    entry = registry_entry(name)
    if entry is None:
        return _gap(f"unknown receptor {name!r} — not in the registry")
    return {"status": "success", "findings": [], "warnings": [], "errors": [],
            "artifacts": {"receptor": dict(entry)}}


def adr_profiles() -> dict[str, Any]:
    """Return the built-in ADR profiles (special + antitarget)."""
    return {"status": "success", "findings": [], "warnings": [], "errors": [],
            "artifacts": {
                "special_adr_profiles": SPECIAL_ADR_PROFILES,
                "antitarget_profiles": ANTITARGET_PROFILES,
            }}


# ---- aggregate pipeline ----

def run_pipeline(project: dict[str, Any]) -> dict[str, Any]:
    """Run the full deterministic 18-agent pipeline from a ProjectSpec dict.

    Docking and target-selectivity resolve their receptor(s) from the project
    targets and degrade non-blocking when unavailable (no PDB / no network).
    """
    from ..cli import DEFAULT_PIPELINE, build_agents
    from ..orchestrator import Runner, TaskDefinition
    from ..orchestrator.gates import BlockedByGate

    store = _fresh_store()
    try:
        spec = ProjectSpec.model_validate(project)
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "errors": [f"invalid ProjectSpec: {exc}"], "report": None}

    if spec.status == ProjectStatus.draft:
        # an MCP tool call is itself the approval gate
        spec.status = ProjectStatus.approved

    try:
        runner = Runner(store, build_agents(store))
        task_defs = [TaskDefinition(task_id=t.task_id, agent=t.agent,
                                    depends_on=t.depends_on, params=t.params)
                     for t in DEFAULT_PIPELINE]
        manifest = runner.create_run(spec, task_defs)
    except BlockedByGate as exc:
        return {"status": "blocked", "errors": [str(exc)], "report": None}

    final = runner.run(manifest.run_id, project_id=spec.project_id)
    report = _load_artifact(store, spec.project_id, manifest.run_id, "report", "report", Report)
    return {
        "status": final.status.value,
        "run_id": manifest.run_id,
        "project_id": spec.project_id,
        "report": report,
    }


def _load_artifact(
    store: Store, project_id: str, run_id: str, task_id: str, kind: str, model: Any,
) -> dict[str, Any] | None:
    n = store.latest_attempt(project_id, run_id, task_id)
    if n == 0:
        return None
    res = store.load_agent_result(project_id, run_id, task_id, n)
    for art in res.artifacts:
        if art.kind == kind and art.uri:
            return model.model_validate_json(Path(art.uri).read_text()).model_dump()
    return None
