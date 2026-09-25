"""Increment 3 showcase: dock gefitinib into EGFR (PDB 2ITY) and validate the
protocol by redocking against the co-crystallized ligand.

Run from the repo root:
    .venv/bin/python scripts/demo_docking.py

Requires the AutoDock Vina binary (tools/vina), network access to RCSB, and
pdbfixer + openbabel-wheel + meeko. The search box is auto-derived from the
co-crystal ligand's centroid."""

from pathlib import Path

from ddswarm.agents import DockingMinion
from ddswarm.contracts import DockingResult, ProjectSpec
from ddswarm.orchestrator import Runner, Store, TaskDefinition

REPO = Path(__file__).resolve().parent.parent
STORE_ROOT = REPO / "store"

GEFITINIB_SMILES = "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4"


def main() -> None:
    spec = ProjectSpec(
        project_id="egfr_docking_showcase",
        input_type="target",
        objective=(
            "Dock gefitinib into the EGFR kinase domain (PDB 2ITY) and validate "
            "the docking protocol by redocking against the co-crystallized ligand."
        ),
        targets=["EGFR (PDB 2ITY)"],
        starting_materials=[GEFITINIB_SMILES],
        constraints=["docking scores are in-silico, not experimental evidence"],
        success_criteria=[">= 1 docked pose with provenance and a redocking RMSD"],
        allowed_tools=["AutoDock Vina", "PDBFixer", "Open Babel", "Meeko"],
        safety_constraints=["standard research use"],
        status="approved",
    )

    tasks = [
        TaskDefinition(
            task_id="dock",
            agent="docking_minion",
            params={
                "receptor": {"kind": "pdb_id", "identifier": "2ITY"},
                "exhaustiveness": 8,
                "num_modes": 9,
                "seed": 42,
            },
        ),
    ]

    store = Store(STORE_ROOT)
    agents = {"docking_minion": DockingMinion(store)}
    runner = Runner(store, agents)

    manifest = runner.create_run(spec, tasks)
    final = runner.run(manifest.run_id, project_id=spec.project_id)

    print("=" * 70)
    print(f"project_id : {spec.project_id}")
    print(f"run_id     : {manifest.run_id}")
    print(f"run status : {final.status.value}")
    print("-" * 70)

    ts = store.load_task_spec(spec.project_id, manifest.run_id, "dock")
    res = store.load_latest_agent_result(spec.project_id, manifest.run_id, "dock")
    print(f"[dock] task_status={ts.status.value}  agent={ts.agent}")
    for f in res.findings:
        print(f"  finding  : {f}")
    for w in res.warnings:
        print(f"  warning  : {w}")

    if not res.artifacts:
        print("no artifact produced")
        return
    batch = store.load_artifact(res.artifacts[0].uri, DockingResult)

    if batch.receptor:
        print("  receptor  :", batch.receptor.identifier, "| prep:", batch.receptor.preparation_tool)
        print("  QC status :", batch.receptor.qc_status)
        for c in batch.receptor.qc_checks:
            print(f"      [{c.status:5s}] {c.check}: {c.detail}")

    if batch.protocol:
        print("  engine    :", batch.protocol.engine, batch.protocol.engine_version)
        if batch.protocol.search_box:
            print("  box center:", [round(v, 2) for v in batch.protocol.search_box.center],
                  "size:", [round(v, 1) for v in batch.protocol.search_box.size])

    print("  poses:")
    for p in batch.poses:
        print(f"      rank {p.rank}: {p.molecule_id}  affinity={p.affinity_kcal_per_mol} kcal/mol  "
              f"rmsd(lb/ub)={p.rmsd_lower_bound}/{p.rmsd_upper_bound}")

    if batch.redocking_validation:
        rv = batch.redocking_validation
        print("  redocking:")
        print(f"      reference : {rv.reference_ligand} ({rv.reference_source})")
        print(f"      RMSD      : {rv.rmsd_angstrom} A (threshold {rv.threshold_angstrom} A) -> {rv.status}")
        print(f"      note      : {rv.note}")

    print("=" * 70)


if __name__ == "__main__":
    main()
