"""Thin stateless adapters: each MCP tool maps to one ddswarm agent.

Every adapter builds a throwaway Store, runs the agent against a minimal
ProjectSpec derived from the tool inputs, and returns a JSON dict embedding the
full artifact payload(s) — so an MCP client never has to read the temp store.

The swarm stays deterministic and platform-agnostic here: the reasoning backend
is forced to "none" (no embedded LLM — the MCP client *is* the reasoning layer).
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
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
from ..docking.base import is_valid_pdb_id, looks_like_pdb, pdb_cache_path
from ..orchestrator.runner import ARTIFACT_MODELS
from ..orchestrator.store import Store

_PROJECT_ID = "mcp"
_OFFLINE_HINT = (
    "receptor unavailable offline — provide a local PDB "
    '("receptor": {"kind": "file", "identifier": "/path/to/receptor.pdb"}) '
    "or set allow_download=true"
)


# ---- safety limits (the server may be reachable over HTTP) ----
MAX_MOLECULES = 200          # SMILES per granular (non-docking) tool call
MAX_DOCK_MOLECULES = 20      # SMILES per docking / selectivity call
MAX_SMILES_LEN = 1000        # characters per SMILES string
MAX_RECEPTORS = 12           # receptors per selectivity panel
MAX_EXHAUSTIVENESS = 32
MAX_NUM_MODES = 20
_RECEPTOR_KINDS = ("pdb_id", "file")

# Set by the server entry point when serving over a network transport
# (sse / streamable-http). Remote callers must not be able to make the server
# read local files or fetch arbitrary URLs, so only registry names and
# validated RCSB PDB ids are accepted as receptor sources in that mode.
NETWORK_MODE = False


def set_network_mode(enabled: bool) -> None:
    global NETWORK_MODE
    NETWORK_MODE = bool(enabled)


@contextmanager
def _fresh_store() -> Iterator[Store]:
    """A throwaway store per tool call (stateless tools), deleted on exit.

    Every payload is read back into memory before the context closes, so the
    client never needs the on-disk store."""
    with tempfile.TemporaryDirectory(prefix="ddswarm_mcp_") as d:
        yield Store(Path(d))


def _error(message: str) -> dict[str, Any]:
    return {"status": "failed", "findings": [], "warnings": [], "errors": [message],
            "artifacts": {}}


def _check_smiles(smiles: Any, limit: int) -> str | None:
    """Return an error message when the SMILES list is malformed or too large."""
    if not isinstance(smiles, list) or not all(isinstance(x, str) for x in smiles):
        return "smiles must be a list of strings"
    if len(smiles) > limit:
        return f"too many molecules: {len(smiles)} > {limit} per call"
    if any(len(x) > MAX_SMILES_LEN for x in smiles):
        return f"SMILES longer than {MAX_SMILES_LEN} characters"
    return None


def _check_receptor(rc: Any) -> str | None:
    """Validate one receptor config; return an error message or None.

    Over a network transport only registry names / PDB ids are allowed (no
    local file paths, no caller-supplied URLs). PDB ids must be 4 alphanumerics."""
    if not isinstance(rc, dict):
        return "receptor must be an object"
    kind = rc.get("kind", "pdb_id")
    if kind not in _RECEPTOR_KINDS:
        return f"unsupported receptor kind {kind!r} (expected one of {list(_RECEPTOR_KINDS)})"
    if NETWORK_MODE:
        if kind == "file" or rc.get("path"):
            return "local file receptors are disabled over a network transport; use a registry name or PDB id"
        if rc.get("url"):
            return "receptor URLs are disabled over a network transport; use a registry name or PDB id"
    if kind == "pdb_id":
        for key in ("pdb_id", "identifier"):
            if rc.get(key) is not None and not is_valid_pdb_id(rc.get(key)):
                return f"invalid PDB id {rc.get(key)!r} (expected 4 alphanumeric characters)"
    elif not rc.get("identifier") and not rc.get("path"):
        return "file receptor needs an 'identifier' (local PDB path)"
    if rc.get("name") is not None and not isinstance(rc.get("name"), str):
        return "receptor name must be a string"
    return None


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
    err = _check_smiles(smiles, MAX_MOLECULES)
    if err:
        return _error(err)
    try:
        with _fresh_store() as store:
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
    if not ident:
        return True
    if not is_valid_pdb_id(ident):
        return False
    p = pdb_cache_path(ident)
    # only a real cached structure counts (a stale error page must not)
    return p.exists() and looks_like_pdb(p.read_bytes())


def dock_molecule(smiles: list[str], receptor: dict[str, Any] | None = None,
                  target: str | None = None, allow_download: bool = True,
                  box: dict[str, Any] | None = None,
                  exhaustiveness: int = 8, num_modes: int = 9,
                  seed: int = 42) -> dict[str, Any]:
    """Dock molecules against a receptor (registry name or explicit source)."""
    if receptor is None and target is None:
        return _gap("no receptor configured — pass 'receptor' ({kind, identifier/pdb_id}) "
                    "or 'target' (registry name)")
    err = _check_smiles(smiles, MAX_DOCK_MOLECULES)
    if err:
        return _error(err)
    if receptor is not None:
        err = _check_receptor(receptor)
        if err:
            return _error(err)
    if not 1 <= int(exhaustiveness) <= MAX_EXHAUSTIVENESS:
        return _error(f"exhaustiveness must be between 1 and {MAX_EXHAUSTIVENESS}")
    if not 1 <= int(num_modes) <= MAX_NUM_MODES:
        return _error(f"num_modes must be between 1 and {MAX_NUM_MODES}")
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
    err = _check_smiles(smiles, MAX_DOCK_MOLECULES)
    if err:
        return _error(err)
    if receptors is None:
        receptors = [{"name": primary}] + [{"name": o} for o in (off_targets or [])]
    if not isinstance(receptors, list) or len(receptors) > MAX_RECEPTORS:
        return _error(f"receptors must be a list of at most {MAX_RECEPTORS} entries")
    for rc in receptors:
        err = _check_receptor(rc)
        if err:
            return _error(err)
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

def _build_task_defs(iupac: bool) -> list[Any]:
    """Build the default 18-task pipeline DAG; ``iupac=True`` opts the
    report_minion into the IUPAC name fetch (network readability labels)."""
    from ..cli import DEFAULT_PIPELINE
    from ..orchestrator import TaskDefinition

    task_defs: list[Any] = []
    for t in DEFAULT_PIPELINE:
        params = dict(t.params or {})
        if iupac and t.agent == "report_minion":
            params["iupac"] = True
        task_defs.append(TaskDefinition(task_id=t.task_id, agent=t.agent,
                                        depends_on=t.depends_on, params=params))
    return task_defs


def run_pipeline(project: dict[str, Any], iupac: bool = False) -> dict[str, Any]:
    """Run the full deterministic 18-agent pipeline from a ProjectSpec dict.

    Docking and target-selectivity resolve their receptor(s) from the project
    targets and degrade non-blocking when unavailable (no PDB / no network).

    With ``iupac=True`` the report agent fetches IUPAC names for the ranked
    candidates (NIH CACTUS -> PubChem fallback, ~15s timeout each); names are
    readability labels only and never affect the scores or reproducibility
    hashes. Defaults to False (offline-first).
    """
    from ..cli import build_agents
    from ..orchestrator import Runner
    from ..orchestrator.gates import BlockedByGate

    try:
        spec = ProjectSpec.model_validate(project)
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "errors": [f"invalid ProjectSpec: {exc}"], "report": None}
    err = _check_smiles(list(spec.starting_materials), MAX_DOCK_MOLECULES)
    if err:
        return {"status": "failed", "errors": [f"starting_materials: {err}"], "report": None}

    if spec.status == ProjectStatus.draft:
        # an MCP tool call is itself the approval gate
        spec.status = ProjectStatus.approved

    with _fresh_store() as store:
        try:
            runner = Runner(store, build_agents(store))
            manifest = runner.create_run(spec, _build_task_defs(iupac))
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
