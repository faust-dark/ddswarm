"""Increment 1 screening showcase: one molecule per outcome.

Run from the repo root:
    .venv/bin/python scripts/demo_screen.py
"""

from pathlib import Path

from ddswarm.agents import ScreenerMinion
from ddswarm.agents.base import AgentContext
from ddswarm.contracts import ProjectSpec, ScreenedBatch
from ddswarm.orchestrator import Store

REPO = Path(__file__).resolve().parent.parent
STORE = Store(REPO / "store")

GEFITINIB = "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4"

CASES = [
    ("benzene", "c1ccccc1", "clean: passes all filters"),
    ("gefitinib", GEFITINIB, "approved drug; Brenk flags Aliphatic_long_chain"),
    ("rhodanine", "O=C1CSC(=S)N1", "canonical PAINS scaffold"),
    ("C45_alkane", "C" * 45, "MW + logP exceed Ro5"),
    ("broken", "CC(C", "invalid SMILES"),
]


def main() -> None:
    spec = ProjectSpec(
        project_id="screen_showcase", input_type="mixed",
        objective="deterministic screening demo", status="approved",
    )
    agent = ScreenerMinion(STORE)
    ctx = AgentContext(
        project_id=spec.project_id, run_id="demo", task_id="screen", spec=spec,
        params={"molecules": [{"id": name, "smiles": smiles} for (name, smiles, _) in CASES]},
    )
    result = agent.run(ctx)
    batch = STORE.load_artifact(result.artifacts[0].uri, ScreenedBatch)

    print("=" * 78)
    print(f"project : {spec.project_id}  status: {result.status.value}")
    print(f"summary : total={batch.summary.total} passed={batch.summary.passed} "
          f"rejected={batch.summary.rejected} invalid={batch.summary.invalid}")
    print("-" * 78)
    for m in batch.molecules:
        if m.valid:
            p = m.properties
            verdict = "PASS" if m.passed else "REJECT"
            print(f"[{verdict}] {m.id:12} MW={p.molecular_weight:8.2f} "
                  f"logP={p.logp:5.2f} HBD={p.hbd} HBA={p.hba}")
        else:
            print(f"[INVALID] {m.id:12} {m.sanitization_error}")
        for r in m.elimination_reasons:
            print(f"           reason: {r}")
    print("=" * 78)
    print("Note: Brenk is a conservative liability flag, not a kill criterion;")
    print("approved drugs like gefitinib can trip 'Aliphatic_long_chain'.")
    print("Disable it per-project via params.filters.enable_brenk=false.")


if __name__ == "__main__":
    main()
