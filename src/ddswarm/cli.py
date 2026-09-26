"""ddswarm CLI: run a pipeline from a project.json file and inspect the store.

    ddswarm run project.json [--reasoning none|hermes] [--store DIR]
    ddswarm runs [project_id] [--store DIR]
    ddswarm report project_id [run_id] [--store DIR]
    ddswarm init project.json --project-id ID --objective "..." [--targets ...] [--smiles ...]

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
)
from .contracts import (
    GenerationMemory, ProjectSpec, ReportDocument, RunRequest, ReasoningConfig, RunTask, export_schemas,
)
from pydantic import ValidationError

from .orchestrator import Runner, Store, TaskDefinition
from .orchestrator.gates import BlockedByGate
from .orchestrator.memory import MemoryStore

DEFAULT_PIPELINE = [
    RunTask(task_id="safety", agent="safety_minion"),
    RunTask(task_id="research", agent="researcher_minion", params={"seed_entries": []}),
    RunTask(task_id="design", agent="designer_minion", depends_on=["research", "safety"]),
    RunTask(task_id="screen", agent="screener_minion", depends_on=["design"]),
    RunTask(task_id="predict", agent="predictor_minion", depends_on=["screen"],
            params={"providers": ["rule_admet"]}),
    RunTask(task_id="retrosynth", agent="retrosynthesis_minion", depends_on=["screen"]),
    RunTask(task_id="tox", agent="tox_minion", depends_on=["screen"]),
    RunTask(task_id="formulation", agent="formulation_minion", depends_on=["screen"]),
    RunTask(task_id="report", agent="report_minion", depends_on=["predict", "retrosynth", "tox", "formulation"]),
    RunTask(task_id="validation", agent="validation_minion", depends_on=["report"]),
    RunTask(task_id="report_pdf", agent="report_pdf_minion", depends_on=["validation"]),
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
        "formulation_minion": FormulationMinion(store),
        "safety_minion": SafetyMinion(store),
        "selectivity_minion": SelectivityMinion(store),
        "validation_minion": ValidationMinion(store),
        "patent_minion": PatentMinion(store),
        "clinical_minion": ClinicalMinion(store),
        "regulatory_minion": RegulatoryMinion(store),
        "orchestrator_minion": OrchestratorMinion(store),
        "memory_minion": MemoryMinion(store),
        "report_minion": ReportMinion(store),
        "report_pdf_minion": ReportPdfMinion(store),
    }


def _inject_reasoning(tasks: list[RunTask], reasoning: ReasoningConfig | None) -> list[RunTask]:
    if reasoning is None or reasoning.backend in (None, "", "none"):
        return tasks
    for t in tasks:
        if t.agent in ("researcher_minion", "designer_minion", "retrosynthesis_minion",
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
        docs = sorted(adir.glob("report_document_*.report_document.v*.json"))
        if docs:
            doc = ReportDocument.model_validate_json(docs[-1].read_text())
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
    docs = sorted(adir.glob("report_document_*.report_document.v*.json"))
    if not docs:
        return None
    doc = ReportDocument.model_validate_json(docs[-1].read_text())
    src = Path(doc.pdf_uri)
    if not src.exists():
        return None
    reports_dir.mkdir(parents=True, exist_ok=True)
    dest = reports_dir / f"{project_id}.pdf"
    dest.write_bytes(src.read_bytes())
    return dest


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


def cmd_run(args) -> int:
    req = RunRequest.model_validate_json(Path(args.project_file).read_text())
    # Hermes reasoning is the project standard: default ON unless the JSON sets
    # an explicit backend or the user passes --reasoning none.
    if req.reasoning is None:
        req.reasoning = ReasoningConfig()  # backend default "hermes"
    if args.reasoning is not None:
        req.reasoning.backend = args.reasoning

    tasks = req.tasks or list(DEFAULT_PIPELINE)
    tasks = _inject_reasoning(list(tasks), req.reasoning)
    if args.iupac:
        for t in tasks:
            if t.agent == "report_minion":
                t.params["iupac"] = True

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
    req = RunRequest(project=spec, tasks=list(DEFAULT_PIPELINE), reasoning=None)
    Path(args.project_file).write_text(req.model_dump_json(indent=2))
    print(f"wrote {args.project_file}")
    print("note: set 'reasoning': {'backend': 'hermes'} to delegate research/design to Hermes subagents")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ddswarm", description="drug-discovery swarm runner")
    sub = p.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("run", help="run a pipeline from a project.json file")
    pr.add_argument("project_file")
    pr.add_argument("--reasoning", choices=["none", "hermes"], default=None,
                    help="override reasoning backend for the reasoning agents (research/design/retrosynth/patent/clinical/regulatory)")
    pr.add_argument("--iupac", action="store_true",
                    help="fetch IUPAC names (NIH CACTUS web service) for the ranked candidates as a readability label (non-deterministic)")
    pr.add_argument("--iterations", type=int, default=1,
                    help="number of generations to iterate (>=2 injects orchestrator + memory agents and feeds prior survivors forward)")
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
