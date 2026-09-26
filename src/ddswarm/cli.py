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
    PatentMinion,
    PredictorMinion,
    RegulatoryMinion,
    ReportMinion,
    ResearcherMinion,
    RetrosynthesisMinion,
    SafetyMinion,
    ScreenerMinion,
    ToxMinion,
    ValidationMinion,
)
from .contracts import ProjectSpec, RunRequest, ReasoningConfig, RunTask, export_schemas
from .orchestrator import Runner, Store, TaskDefinition
from .orchestrator.memory import MemoryStore

DEFAULT_PIPELINE = [
    RunTask(task_id="safety", agent="safety_minion"),
    RunTask(task_id="research", agent="researcher_minion", params={"seed_entries": []}),
    RunTask(task_id="design", agent="designer_minion", depends_on=["research", "safety"]),
    RunTask(task_id="screen", agent="screener_minion", depends_on=["design"]),
    RunTask(task_id="predict", agent="predictor_minion", depends_on=["screen"],
            params={"providers": ["null_admet"]}),
    RunTask(task_id="retrosynth", agent="retrosynthesis_minion", depends_on=["screen"]),
    RunTask(task_id="tox", agent="tox_minion", depends_on=["screen"]),
    RunTask(task_id="formulation", agent="formulation_minion", depends_on=["screen"]),
    RunTask(task_id="report", agent="report_minion", depends_on=["predict", "retrosynth", "tox", "formulation"]),
    RunTask(task_id="validation", agent="validation_minion", depends_on=["report"]),
]


def _default_store() -> Path:
    return Path(os.environ.get("DDSWARM_STORE", "./store"))


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
        "validation_minion": ValidationMinion(store),
        "patent_minion": PatentMinion(store),
        "clinical_minion": ClinicalMinion(store),
        "regulatory_minion": RegulatoryMinion(store),
        "report_minion": ReportMinion(store),
    }


def _inject_reasoning(tasks: list[RunTask], reasoning: ReasoningConfig | None) -> list[RunTask]:
    if reasoning is None or reasoning.backend in (None, "", "none"):
        return tasks
    for t in tasks:
        if t.agent in ("researcher_minion", "designer_minion", "retrosynthesis_minion"):
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
    print("=" * 70)


def cmd_run(args) -> int:
    req = RunRequest.model_validate_json(Path(args.project_file).read_text())
    if args.reasoning is not None:
        req.reasoning = req.reasoning or ReasoningConfig()
        req.reasoning.backend = args.reasoning

    tasks = req.tasks or list(DEFAULT_PIPELINE)
    tasks = _inject_reasoning(list(tasks), req.reasoning)

    store = Store(args.store)
    runner = Runner(store, build_agents(store))
    task_defs = [TaskDefinition(task_id=t.task_id, agent=t.agent, depends_on=t.depends_on, params=t.params)
                 for t in tasks]

    manifest = runner.create_run(req.project, task_defs)
    final = runner.run(manifest.run_id, project_id=req.project.project_id)
    _print_report(store, req.project.project_id, manifest.run_id)
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
                    help="override reasoning backend for research/design")
    pr.add_argument("--store", type=Path, default=_default_store())
    pr.set_defaults(func=cmd_run)

    ps = sub.add_parser("runs", help="list runs in the store")
    ps.add_argument("project_id", nargs="?")
    ps.add_argument("--store", type=Path, default=_default_store())
    ps.set_defaults(func=cmd_runs)

    pt = sub.add_parser("report", help="print a run report")
    pt.add_argument("project_id")
    pt.add_argument("run_id", nargs="?")
    pt.add_argument("--store", type=Path, default=_default_store())
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
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
