# ddswarm — drug discovery multi-agent swarm (Increment 1: RDKit deterministic screening)

A reproducible, traceable, verifiable agentic drug-discovery pipeline.
**Increment 1** adds `screener_minion` + deterministic RDKit screening on top of
Increment 0 (contracts + orchestrator + logging + researcher/designer).
ADMET, docking, dynamics, etc. land in later increments.

## Architecture

- **Harness**: Hermes (this agent's runtime) drives execution.
  `ddswarm` provides the contract + state-machine layer that Hermes invokes.
- **Orchestrator** (`orchestrator_minion`): task graph, gating, retry,
  idempotent versioning, resume. Produces no chemistry.
- **Agents**:
  - `researcher_minion` — KnowledgePack from ProjectSpec + seed (refuses to fabricate).
  - `designer_minion` — DesignBatch from upstream scaffolds (labels only; no de novo yet).
  - `screener_minion` — deterministic structural + physicochemical screening (RDKit).
- **Chemistry layer** (`chemistry/rdkit_ops.py`): sanitization, canonicalization,
  descriptors, PAINS/Brenk alerts. The agent reasons; this module computes.

## Contracts (JSON, versioned, validated)

| Contract | Purpose |
|---|---|
| `ProjectSpec` | normalized input, objectives, constraints, success criteria |
| `AgentResult` | per-agent outcome: status, findings, warnings, errors, provenance |
| `RunManifest` | run-level state + task DAG |
| `TaskSpec` | execution ledger for one task (attempts, retries) |
| `EventLog` | append-only JSONL with input/output hashes |
| `KnowledgePack` | researcher output (targets/ligands/references with evidence levels) |
| `DesignBatch` | designer output (candidates with transformation + rationale) |
| `ScreenedBatch` | screener output (per-molecule validity, properties, alerts, verdicts) |

Every artifact carries `evidence_level` (`observed|calculated|predicted|inferred|hypothetical`)
and provenance (source, URL/identifier, timestamps, tool versions, hashes).

## Screening (Increment 1)

Hard elimination (default, all configurable via `params.filters`):
1. invalid SMILES (sanitization/canonicalization failed)
2. MW > 500
3. logP > 5 (Crippen)
4. HBD > 5
5. HBA > 10
6. any PAINS match (RDKit FilterCatalog)
7. any Brenk match (RDKit FilterCatalog)

Advisory (reported, not eliminating): TPSA > 140, rotatable bonds > 10,
|formal charge| > 1.

Strict Ro5: any single violation eliminates (approved default).

Known consequence: Brenk is conservative and flags the approved drug gefitinib
as `Aliphatic_long_chain` (its OCCCN propyl-morpholine chain). This is a
liability flag, not a kill criterion; disable per-project with
`params.filters.enable_brenk=false`.

## Run

```bash
cd ddswarm
python3 -m venv .venv
.venv/bin/pip install -e .            # ddswarm + pydantic + rdkit
.venv/bin/pip install pytest          # for tests
.venv/bin/python -m pytest -q         # run the test suite
.venv/bin/python scripts/demo_egfr.py # full pipeline: research -> design -> screen
.venv/bin/python scripts/demo_screen.py # one molecule per screening outcome
```

## Layout

```
store/<project_id>/runs/<run_id>/
  run_manifest.json
  tasks/<task_id>.json                 # TaskSpec ledger
  tasks/<task_id>.result.v<N>.json     # versioned AgentResult
  artifacts/<id>.<kind>.v<N>.json      # versioned artifacts (never overwritten)
  events.jsonl                         # append-only event log
schemas/*.schema.json                  # exported JSON Schema for every contract
```

## Gating

- **Gate A — input validity**: objective non-empty; target/ligand inputs match
  `input_type`; ProjectSpec must be `approved`.
- **Gate C — chemical validity**: molecules must sanitize + canonicalize
  (invalid SMILES are eliminated with a reason, not silently dropped).
- **Result gating**: `failed`/`blocked`/`insufficient_evidence` stops downstream
  tasks; `success`/`partial` lets them proceed.

## Verification status

Run `pytest -q` — all tests must pass before this increment is considered done.
