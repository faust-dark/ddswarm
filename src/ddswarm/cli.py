"""ddswarm CLI: run a pipeline from a project.json file and inspect the store.

    ddswarm run project.json [--reasoning none|hermes] [--store DIR]
    ddswarm runs [project_id] [--store DIR]
    ddswarm report project_id [run_id] [--store DIR]
    ddswarm init project.json --project-id ID --objective "..." [--targets ...] [--smiles ...]
    ddswarm receptors {list|lookup|adr|match} [...]

project.json is a RunRequest: {project: ProjectSpec, tasks: [...], reasoning: {...}}.
If `tasks` is omitted a default pipeline runs (research -> design -> screen -> predict);
docking is opt-in because it needs the Vina binary + network."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .agents import (
    ClinicalMinion,
    DesignerMinion,
    DockingMinion,
    DynamicsMinion,
    FormulationMinion,
    MemoryMinion,
    OrchestratorMinion,
    OverviewMinion,
    PatentMinion,
    PredictorMinion,
    RegulatoryMinion,
    ReportMinion,
    ReportPdfMinion,
    ResearcherMinion,
    RetrosynthesisMinion,
    SafetyMinion,
    ScreenerMinion,
    SelectivityMinion,
    ToxMinion,
    ValidationMinion,
    CypMinion,
    HergMinion,
    GroundingMinion,
    ExportMinion,
)
from .contracts import (
    GenerationMemory, ProjectSpec, ReportDocument, RunRequest, ReasoningConfig, RunTask, export_schemas,
)
from pydantic import ValidationError

from .chemistry.receptors import RECEPTOR_REGISTRY, canonical_receptor_name
from .chemistry.selectivity import ANTITARGET_PROFILES, SPECIAL_ADR_PROFILES, select_adr_profiles
from .orchestrator import Runner, Store, TaskDefinition
from .orchestrator.gates import BlockedByGate
from .orchestrator.memory import MemoryStore
from .orchestrator.store import latest_versioned

DEFAULT_PIPELINE = [
    RunTask(task_id="safety", agent="safety_minion"),
    RunTask(task_id="research", agent="researcher_minion", params={"seed_entries": []}),
    RunTask(task_id="overview", agent="overview_minion", depends_on=["research"]),
    RunTask(task_id="design", agent="designer_minion", depends_on=["research", "safety"]),
    RunTask(task_id="screen", agent="screener_minion", depends_on=["design"]),
    RunTask(task_id="docking", agent="docking_minion", depends_on=["screen"]),
    RunTask(task_id="selectivity", agent="selectivity_minion", depends_on=["screen"]),
    RunTask(task_id="predict", agent="predictor_minion", depends_on=["screen"],
            params={"providers": ["rule_admet"]}),
    RunTask(task_id="retrosynth", agent="retrosynthesis_minion", depends_on=["screen"]),
    RunTask(task_id="tox", agent="tox_minion", depends_on=["screen"]),
    RunTask(task_id="cyp", agent="cyp_minion", depends_on=["screen"]),
    RunTask(task_id="herg", agent="herg_minion", depends_on=["screen"]),
    RunTask(task_id="formulation", agent="formulation_minion", depends_on=["screen"]),
    RunTask(task_id="patent", agent="patent_minion", depends_on=["screen"]),
    RunTask(task_id="report", agent="report_minion", depends_on=["overview", "predict", "retrosynth", "tox", "cyp", "herg", "formulation", "patent", "docking", "selectivity"]),
    RunTask(task_id="export", agent="export_minion", depends_on=["report"]),
    RunTask(task_id="validation", agent="validation_minion", depends_on=["report"]),
    RunTask(task_id="report_pdf", agent="report_pdf_minion", depends_on=["validation", "export"]),
]


def _default_store() -> Path:
    return Path(os.environ.get("DDSWARM_STORE", "./store"))


def _default_reports_dir() -> Path:
    root = Path(__file__).resolve().parent.parent.parent / "reports"
    return Path(os.environ.get("DDSWARM_REPORTS_DIR", str(root)))


def build_agents(store: Store) -> dict:
    return {
        "researcher_minion": ResearcherMinion(store),
        "designer_minion": DesignerMinion(store),
        "screener_minion": ScreenerMinion(store),
        "predictor_minion": PredictorMinion(store),
        "docking_minion": DockingMinion(store),
        "dynamics_minion": DynamicsMinion(store),
        "retrosynthesis_minion": RetrosynthesisMinion(store),
        "tox_minion": ToxMinion(store),
        "cyp_minion": CypMinion(store),
        "herg_minion": HergMinion(store),
        "grounding_minion": GroundingMinion(store),
        "export_minion": ExportMinion(store),
        "formulation_minion": FormulationMinion(store),
        "safety_minion": SafetyMinion(store),
        "selectivity_minion": SelectivityMinion(store),
        "validation_minion": ValidationMinion(store),
        "patent_minion": PatentMinion(store),
        "clinical_minion": ClinicalMinion(store),
        "regulatory_minion": RegulatoryMinion(store),
        "orchestrator_minion": OrchestratorMinion(store),
        "memory_minion": MemoryMinion(store),
        "overview_minion": OverviewMinion(store),
        "report_minion": ReportMinion(store),
        "report_pdf_minion": ReportPdfMinion(store),
    }


def _inject_reasoning(tasks: list[RunTask], reasoning: ReasoningConfig | None) -> list[RunTask]:
    if reasoning is None or reasoning.backend in (None, "", "none"):
        return tasks
    for t in tasks:
        if t.agent in ("researcher_minion", "overview_minion", "designer_minion", "retrosynthesis_minion",
                       "patent_minion", "clinical_minion", "regulatory_minion"):
            t.params.setdefault("reasoning_backend", reasoning.backend)
            if reasoning.command:
                t.params.setdefault("reasoning_command", reasoning.command)
            t.params.setdefault("reasoning_max_turns", reasoning.max_turns)
            t.params.setdefault("reasoning_run_budget", reasoning.run_budget)
    return tasks


def _print_report(store: Store, project_id: str, run_id: str) -> None:
    manifest = store.load_run_manifest(project_id, run_id)
    print("=" * 70)
    print(f"project_id : {project_id}")
    print(f"run_id     : {run_id}")
    print(f"run status : {manifest.status.value}")
    print("-" * 70)
    for task_id in manifest.tasks:
        ts = store.load_task_spec(project_id, run_id, task_id)
        print(f"[{task_id}] {ts.status.value}  ({ts.agent}, retries={ts.retry_count})")
        try:
            res = store.load_latest_agent_result(project_id, run_id, task_id)
        except FileNotFoundError:
            continue
        for a in res.artifacts:
            print(f"    artifact : {a.kind}  v{a.version}")
        for f in res.findings:
            print(f"    finding  : {f}")
        for w in res.warnings:
            print(f"    warning  : {w}")
    adir = store.artifacts_dir(project_id, run_id)
    if adir.exists():
        latest = latest_versioned(adir.glob("report_document_*.report_document.v*.json"))
        if latest:
            doc = ReportDocument.model_validate_json(latest.read_text())
            print(f"PDF report : {doc.pdf_uri}")
    print("=" * 70)


def _copy_pdf_to_reports(store: Store, project_id: str, run_id: str,
                         reports_dir: Path) -> Path | None:
    """Copy the run's exported PDF to a stable, human-facing reports directory
    (e.g. <reports_dir>/<project_id>.pdf), so it is easy to find outside the
    versioned store. Returns the destination path, or None if no PDF exists."""
    adir = store.artifacts_dir(project_id, run_id)
    if not adir.exists():
        return None
    latest = latest_versioned(adir.glob("report_document_*.report_document.v*.json"))
    if latest is None:
        return None
    doc = ReportDocument.model_validate_json(latest.read_text())
    src = Path(doc.pdf_uri)
    if not src.exists():
        return None
    reports_dir.mkdir(parents=True, exist_ok=True)
    dest = reports_dir / f"{project_id}.pdf"
    dest.write_bytes(src.read_bytes())
    return dest


def _copy_exports_to_reports(store: Store, project_id: str, run_id: str, reports_dir: Path) -> list[Path]:
    """Copy the machine-readable export sidecars (CSV/SDF) + candidate-table JSON
    into <reports_dir> with a project_id prefix, so they are easy to find outside
    the versioned store. Returns the list of destinations written."""
    adir = store.artifacts_dir(project_id, run_id)
    if not adir.exists():
        return []
    reports_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for pattern, suffix in [
        ("candidates_*.candidate_table.v*.json", "candidates.json"),
        ("candidates_*.candidate_csv.v*.csv", "candidates.csv"),
        ("candidates_*.candidate_sdf.v*.sdf", "candidates.sdf"),
    ]:
        latest = latest_versioned(adir.glob(pattern))
        if latest is None:
            continue
        dest = reports_dir / f"{project_id}_{suffix}"
        dest.write_bytes(latest.read_bytes())
        out.append(dest)
    return out


def _with_orchestration(tasks: list[RunTask], iterations: int) -> list[RunTask]:
    """Wrap a task list with an orchestrator (first) and a memory_minion (after
    report), wiring entry tasks to the orchestrator and validation to memory.
    Returns a fresh task list; does not mutate the input."""
    base = [t.model_copy(deep=True) for t in tasks]
    task_ids = {t.task_id for t in base}
    for t in base:
        if not t.depends_on:
            t.depends_on = ["orchestrator"]
        if t.task_id == "validation":
            t.depends_on = list(t.depends_on) + ["memory"]
    if "report" in task_ids:
        base.append(RunTask(task_id="memory", agent="memory_minion", depends_on=["report"]))
    orch = RunTask(task_id="orchestrator", agent="orchestrator_minion",
                   params={"iterations": iterations,
                           "tasks": [{"task_id": t.task_id, "agent": t.agent,
                                      "depends_on": list(t.depends_on)} for t in base]})
    return [orch] + base


def _read_generation_memory(store: Store, project_id: str, run_id: str) -> GenerationMemory | None:
    try:
        res = store.load_latest_agent_result(project_id, run_id, "memory")
    except FileNotFoundError:
        return None
    for a in res.artifacts:
        if a.kind == "generation_memory" and a.uri:
            return GenerationMemory.model_validate_json(Path(a.uri).read_text())
    return None


def _inject_grounding(tasks: list[RunTask]) -> list[RunTask]:
    """Insert an opt-in grounding task before the report task and make report
    depend on it. Grounding is network-dependent and non-deterministic, so it is
    NOT part of DEFAULT_PIPELINE; it is added only when --grounding is passed."""
    base = [t.model_copy(deep=True) for t in tasks]
    ids = {t.task_id for t in base}
    gdep = ["screen"] if "screen" in ids else []
    gt = RunTask(task_id="grounding", agent="grounding_minion", depends_on=gdep)
    report_idx = next((i for i, t in enumerate(base) if t.task_id == "report"), None)
    if report_idx is not None:
        base[report_idx].depends_on = list(base[report_idx].depends_on) + ["grounding"]
        base.insert(report_idx, gt)
    else:
        base.append(gt)
    return base


def _inject_resume(tasks: list[RunTask]) -> list[RunTask]:
    """Opt-in content-addressed resume: mark every task to reuse cached
    docking results, or — for dynamics — to extend the receptor MD trajectory
    from its last checkpoint instead of restarting (deterministic; only the
    docking/dynamics/selectivity agents read the flag)."""
    base = [t.model_copy(deep=True) for t in tasks]
    for t in base:
        t.params["resume"] = True
    return base


def cmd_run(args) -> int:
    req = RunRequest.model_validate_json(Path(args.project_file).read_text())
    # Platform-agnostic default: deterministic reasoning (no LLM). Hermes is an
    # explicit opt-in via 'reasoning.backend = "hermes"' in the JSON, or
    # --reasoning hermes on the command line.
    if req.reasoning is None:
        req.reasoning = ReasoningConfig()  # backend default "none"
    if args.reasoning is not None:
        req.reasoning.backend = args.reasoning

    # deep copies: the reasoning/iupac injection below mutates task params, and
    # must never leak into the module-level DEFAULT_PIPELINE (or the request)
    tasks = [t.model_copy(deep=True) for t in (req.tasks or DEFAULT_PIPELINE)]
    tasks = _inject_reasoning(tasks, req.reasoning)
    if args.grounding:
        tasks = _inject_grounding(tasks)
    if not args.no_iupac:
        for t in tasks:
            if t.agent == "report_minion":
                t.params["iupac"] = True
    if args.resume:
        tasks = _inject_resume(tasks)

    iterations = max(1, int(args.iterations))
    if iterations > 1:
        tasks = _with_orchestration(tasks, iterations)

    store = Store(args.store)
    runner = Runner(store, build_agents(store))

    final = None
    last_manifest = None
    prior_best: set[str] = set()
    for gen in range(iterations):
        gen_tasks = [t.model_copy(deep=True) for t in tasks]
        for t in gen_tasks:
            if t.agent in ("orchestrator_minion", "memory_minion"):
                t.params["generation"] = gen
        task_defs = [TaskDefinition(task_id=t.task_id, agent=t.agent,
                                    depends_on=t.depends_on, params=t.params)
                     for t in gen_tasks]
        manifest = runner.create_run(req.project, task_defs)
        final = runner.run(manifest.run_id, project_id=req.project.project_id)
        last_manifest = manifest
        if iterations > 1:
            mem = _read_generation_memory(store, req.project.project_id, manifest.run_id)
            new_best = {c.smiles for c in mem.best_candidates} if mem else set()
            if gen > 0 and not (new_best - prior_best):
                break  # converged: no new candidates vs the prior generation
            prior_best = new_best

    _print_report(store, req.project.project_id, last_manifest.run_id)
    dest = _copy_pdf_to_reports(store, req.project.project_id, last_manifest.run_id, args.reports_dir)
    if dest:
        print(f"PDF saved to : {dest}")
    for exp in _copy_exports_to_reports(store, req.project.project_id, last_manifest.run_id, args.reports_dir):
        print(f"export saved to : {exp}")
    return 0 if final.status.value == "completed" else 1


def cmd_runs(args) -> int:
    store = Store(args.store)
    root = Path(store.root)
    if not root.exists():
        print("no store at", root)
        return 0
    projects = [p for p in root.iterdir() if p.is_dir()] if args.project_id is None \
        else [root / args.project_id]
    for proj in projects:
        runs_dir = proj / "runs"
        if not runs_dir.exists():
            continue
        runs = sorted(runs_dir.iterdir(), key=lambda p: p.name, reverse=True)
        for r in runs:
            manifest = store.load_run_manifest(proj.name, r.name)
            print(f"{proj.name:30s} {r.name:30s} {manifest.status.value}")
    return 0


def cmd_report(args) -> int:
    store = Store(args.store)
    project_id = args.project_id
    if args.run_id is None:
        runs = sorted((Path(store.root) / project_id / "runs").iterdir(), key=lambda p: p.name)
        if not runs:
            print(f"no runs for project {project_id}")
            return 1
        run_id = runs[-1].name
    else:
        run_id = args.run_id
    _print_report(store, project_id, run_id)
    dest = _copy_pdf_to_reports(store, project_id, run_id, args.reports_dir)
    if dest:
        print(f"PDF saved to : {dest}")
    return 0


def cmd_memory(args) -> int:
    store = Store(args.store)
    entries = MemoryStore(store).load(args.project_id).entries
    if not entries:
        print(f"no memory for project {args.project_id}")
        return 0
    for e in entries:
        print(f"[{e.kind}] {e.id}  (run {e.run_id}, {e.created_at})")
        print(f"    {e.content}")
        for k, v in (e.data or {}).items():
            print(f"    data.{k} = {v}")
    return 0


def cmd_init(args) -> int:
    spec = ProjectSpec(
        project_id=args.project_id,
        input_type=args.input_type,
        objective=args.objective,
        targets=args.targets or [],
        starting_materials=args.smiles or [],
        constraints=[],
        exclusions=[],
        success_criteria=[">= 1 candidate passing screening"],
        allowed_tools=["ddswarm contracts + orchestrator"],
        safety_constraints=["standard research use"],
        status="approved",
    )
    req = RunRequest(project=spec, tasks=[t.model_copy(deep=True) for t in DEFAULT_PIPELINE],
                     reasoning=None)
    Path(args.project_file).write_text(req.model_dump_json(indent=2))
    print(f"wrote {args.project_file}")
    print("note: set 'reasoning': {'backend': 'hermes'} to delegate research/design to Hermes subagents")
    return 0


def cmd_receptors(args) -> int:
    """Query the receptor registry + ADR profile library (deterministic, offline)."""
    if args.action == "list":
        fam = (args.family or "").lower()
        rows = [(n, e) for n, e in sorted(RECEPTOR_REGISTRY.items())
                if not fam or fam in str(e.get("family", "")).lower()]
        if not rows:
            print("no receptors match the family filter")
            return 0
        for n, e in rows:
            pdb = e.get("pdb_id") or "-"
            print(f"{n:12s} {str(e.get('gene', '')):14s} {pdb:6s} {e.get('family', '')}")
            if e.get("reference"):
                print(f"{'':12s} {'':14s} {'':6s} ref: {e['reference']}")
        with_pdb = sum(1 for _, e in rows if e.get("pdb_id"))
        print(f"\n{len(rows)} target(s) shown ({with_pdb} with pdb_id, "
              f"{len(rows) - with_pdb} without)")
        return 0

    if args.action == "lookup":
        key = canonical_receptor_name(args.name)
        e = RECEPTOR_REGISTRY.get(key)
        if e is None:
            print(f"not found: {args.name}")
            return 0
        print(f"name      : {e.get('name')}")
        print(f"gene      : {e.get('gene')}")
        print(f"family    : {e.get('family')}")
        print(f"pdb_id    : {e.get('pdb_id') or '(none — supply pdb_id explicitly)'}")
        print(f"reference : {e.get('reference')}")
        return 0

    if args.action == "adr":
        print("ADR profile library (mechanistic, citable; docking reports binding, not function)")
        print("\nSpecial profiles:")
        for pid, prof in SPECIAL_ADR_PROFILES.items():
            print(f"\n  {pid}")
            print(f"    trigger  : {prof['primary']} primary + "
                  f"{', '.join(prof['off_targets'])} off-target")
            print(f"    adr      : {prof['adr']}")
            print(f"    mechanism: {prof['mechanism']}")
            print(f"    citation : {prof['citation']}")
        print("\nOff-target 'antitarget' ADRs (tiered on occupancy at the reference concentration):")
        for pid, prof in ANTITARGET_PROFILES.items():
            print(f"\n  {pid}  ({prof['receptor']}, {prof['sign']})")
            print(f"    adr      : {prof['adr']}")
            print(f"    mechanism: {prof['mechanism']}")
            print(f"    citation : {prof['citation']}")
        return 0

    if args.action == "match":
        off_targets = [o.strip() for o in args.off_targets.split(",") if o.strip()]
        profiles = select_adr_profiles(args.primary, off_targets)
        print(f"primary: {args.primary}; off-targets: {', '.join(off_targets)}")
        if not profiles:
            print("no built-in ADR profile matches this panel")
        else:
            for pid in profiles:
                print(f"  - {pid}")
        return 0

    print(f"unknown action: {args.action}", file=sys.stderr)
    return 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ddswarm", description="drug-discovery swarm runner")
    sub = p.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("run", help="run a pipeline from a project.json file")
    pr.add_argument("project_file")
    pr.add_argument("--reasoning", choices=["none", "hermes"], default=None,
                    help="override reasoning backend for the reasoning agents (default: none — deterministic, platform-agnostic; hermes = opt-in LLM for research/overview/design/retrosynth/patent/clinical/regulatory)")
    pr.add_argument("--no-iupac", action="store_true",
                    help="disable IUPAC name fetching for the ranked candidates (default: fetch from the NIH CACTUS web service as a readability label — non-deterministic, network-dependent, never authoritative)")
    pr.add_argument("--grounding", action="store_true",
                    help="resolve molecules against ChEMBL/PubChem/PubMed and record observed external records (URL + identifier; network-dependent, non-deterministic)")
    pr.add_argument("--iterations", type=int, default=1,
                    help="number of generations to iterate (>=2 injects orchestrator + memory agents and feeds prior survivors forward)")
    pr.add_argument("--resume", action="store_true",
                    help="reuse cached docking results, and extend receptor MD from its last checkpoint (content-addressed, deterministic) instead of recomputing")
    pr.add_argument("--store", type=Path, default=_default_store())
    pr.add_argument("--reports-dir", type=Path, default=_default_reports_dir(),
                    help="directory to copy the exported PDF into (default: <repo>/reports)")
    pr.set_defaults(func=cmd_run)

    ps = sub.add_parser("runs", help="list runs in the store")
    ps.add_argument("project_id", nargs="?")
    ps.add_argument("--store", type=Path, default=_default_store())
    ps.set_defaults(func=cmd_runs)

    pt = sub.add_parser("report", help="print a run report")
    pt.add_argument("project_id")
    pt.add_argument("run_id", nargs="?")
    pt.add_argument("--store", type=Path, default=_default_store())
    pt.add_argument("--reports-dir", type=Path, default=_default_reports_dir(),
                    help="directory to copy the exported PDF into (default: <repo>/reports)")
    pt.set_defaults(func=cmd_report)

    pm = sub.add_parser("memory", help="show project memory (cross-run conclusions)")
    pm.add_argument("project_id")
    pm.add_argument("--store", type=Path, default=_default_store())
    pm.set_defaults(func=cmd_memory)

    pi = sub.add_parser("init", help="scaffold a project.json file")
    pi.add_argument("project_file")
    pi.add_argument("--project-id", required=True)
    pi.add_argument("--objective", required=True)
    pi.add_argument("--targets", nargs="*", default=[])
    pi.add_argument("--smiles", nargs="*", default=[])
    pi.add_argument("--input-type", default="mixed")
    pi.set_defaults(func=cmd_init)

    prc = sub.add_parser("receptors", help="query the receptor registry and ADR profile library")
    rsub = prc.add_subparsers(dest="action", required=True)

    rl = rsub.add_parser("list", help="list all dockable targets")
    rl.add_argument("--family", help="filter by a substring of the family string "
                    "(e.g. GPCR, enzyme, channel, transporter, cyclooxygenase)")
    rl.set_defaults(func=cmd_receptors)

    rk = rsub.add_parser("lookup", help="resolve a receptor name/alias to its registry entry")
    rk.add_argument("name")
    rk.set_defaults(func=cmd_receptors)

    ra = rsub.add_parser("adr", help="list the pluggable ADR profile library")
    ra.set_defaults(func=cmd_receptors)

    rm = rsub.add_parser("match", help="show which ADR profiles match a panel (dry-run, no docking)")
    rm.add_argument("--primary", required=True)
    rm.add_argument("--off-targets", required=True,
                    help="comma-separated off-target receptor names (aliases resolved)")
    rm.set_defaults(func=cmd_receptors)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValidationError, BlockedByGate, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
