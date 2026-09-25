"""End-to-end demo: EGFR / gefitinib scaffold.

research -> design -> screen -> predict -> dock -> retrosynth -> tox
         -> formulation -> dynamics -> report

Run from the repo root after `pip install -e .`:
    .venv/bin/python scripts/demo_egfr.py
"""

from pathlib import Path

from ddswarm.agents import (
    DesignerMinion,
    DockingMinion,
    DynamicsMinion,
    FormulationMinion,
    PredictorMinion,
    ReportMinion,
    ResearcherMinion,
    RetrosynthesisMinion,
    ScreenerMinion,
    ToxMinion,
)
from ddswarm.contracts import ProjectSpec, export_schemas
from ddswarm.orchestrator import Runner, Store, TaskDefinition

REPO = Path(__file__).resolve().parent.parent
STORE_ROOT = REPO / "store"
SCHEMAS_DIR = REPO / "schemas"

GEFITINIB_SMILES = "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4"


def main() -> None:
    export_schemas(SCHEMAS_DIR)

    spec = ProjectSpec(
        project_id="egfr_gefitinib_analogs",
        input_type="mixed",
        objective=(
            "Propose a small, verifiable set of EGFR inhibitor analogs scaffolded on "
            "gefitinib (4-anilinoquinazoline), then flag developability liabilities for later increments."
        ),
        targets=["EGFR (epidermal growth factor receptor)"],
        starting_materials=[GEFITINIB_SMILES],
        constraints=[
            "no covalent warheads in this increment",
            "molecular weight < 500 Da (enforced by RDKit in Increment 1)",
        ],
        exclusions=["covalent inhibitors", "PAINS-containing scaffolds"],
        success_criteria=[">= 1 candidate design per scaffold with explicit transformation + rationale"],
        allowed_tools=["ddswarm contracts + orchestrator", "PubChem (provenance only)"],
        safety_constraints=["standard research use", "no controlled substances"],
        status="approved",
    )

    tasks = [
        TaskDefinition(
            task_id="research",
            agent="researcher_minion",
            params={"seed_entries": [{
                "id": "ref_gefitinib",
                "type": "reference",
                "source": "PubChem",
                "url": "https://pubchem.ncbi.nlm.nih.gov/compound/123631",
                "identifier": "CID 123631 (ChEMBL CHEMBL939, DrugBank DB00317)",
                "observed_value": (
                    "gefitinib: approved reversible EGFR tyrosine kinase inhibitor "
                    "(FDA 2003; NSCLC first-line 2015)"
                ),
                "evidence_level": "observed",
                "notes": "reference ligand + scaffold source for analog design",
            }]},
        ),
        TaskDefinition(task_id="design", agent="designer_minion", depends_on=["research"]),
        TaskDefinition(
            task_id="screen", agent="screener_minion", depends_on=["design"],
            params={"filters": {"enable_brenk": False}},
        ),
        TaskDefinition(
            task_id="predict", agent="predictor_minion", depends_on=["screen"],
            params={"providers": ["null_admet"]},
        ),
        TaskDefinition(
            task_id="dock", agent="docking_minion", depends_on=["screen"],
            params={"receptor": {"kind": "pdb_id", "identifier": "2ITY"}},
        ),
        TaskDefinition(task_id="retrosynth", agent="retrosynthesis_minion", depends_on=["screen"]),
        TaskDefinition(task_id="tox", agent="tox_minion", depends_on=["screen"]),
        TaskDefinition(task_id="formulation", agent="formulation_minion", depends_on=["screen"]),
        TaskDefinition(
            task_id="dynamics", agent="dynamics_minion", depends_on=["dock"],
            params={"receptor": {"kind": "pdb_id", "identifier": "2ITY"}, "engine": "openmm",
                    "steps": 200, "minimize_max_iterations": 100},
        ),
        TaskDefinition(
            task_id="report", agent="report_minion",
            depends_on=["predict", "dock", "retrosynth", "tox", "formulation", "dynamics"],
        ),
    ]

    store = Store(STORE_ROOT)
    agents = {
        "researcher_minion": ResearcherMinion(store),
        "designer_minion": DesignerMinion(store),
        "screener_minion": ScreenerMinion(store),
        "predictor_minion": PredictorMinion(store),
        "docking_minion": DockingMinion(store),
        "dynamics_minion": DynamicsMinion(store),
        "retrosynthesis_minion": RetrosynthesisMinion(store),
        "tox_minion": ToxMinion(store),
        "formulation_minion": FormulationMinion(store),
        "report_minion": ReportMinion(store),
    }
    runner = Runner(store, agents)

    manifest = runner.create_run(spec, tasks)
    final = runner.run(manifest.run_id, project_id=spec.project_id)

    print("=" * 70)
    print(f"project_id : {spec.project_id}")
    print(f"run_id     : {manifest.run_id}")
    print(f"run status : {final.status.value}")
    print(f"events log : {store.events_path(spec.project_id, manifest.run_id)}")
    print("-" * 70)
    for tid in ["research", "design", "screen", "predict", "dock", "retrosynth", "tox", "formulation", "dynamics", "report"]:
        ts = store.load_task_spec(spec.project_id, manifest.run_id, tid)
        res = store.load_latest_agent_result(spec.project_id, manifest.run_id, tid)
        print(f"[{tid}] task_status={ts.status.value}  agent={ts.agent}  retries={ts.retry_count}")
        for a in res.artifacts:
            print(f"    artifact : {a.kind}  v{a.version}  {a.uri}")
        for f in res.findings:
            print(f"    finding  : {f}")
        for w in res.warnings:
            print(f"    warning  : {w}")
    print("NOTE: Brenk is disabled in THIS run (screen.filters.enable_brenk=false) so gefitinib")
    print("      passes screening and the full chain reaches predict + dock + retrosynth + tox.")
    print("      With Brenk on (default), gefitinib is flagged 'Aliphatic_long_chain' - see demo_screen.py.")
    print("NOTE: the dock task requires tools/vina + network (RCSB) + pdbfixer/meeko/openbabel.")
    print("      Docking scores and tox alerts are in-silico (calculated/predicted), never experimental evidence.")
    print("NOTE: the dynamics task runs a SHORT (ps-scale) receptor MD relaxation (amber14 + GBSA),")
    print("      reporting backbone RMSD + energy fluctuation only - NOT production MD, NOT a binding free energy.")
    print("=" * 70)


if __name__ == "__main__":
    main()
