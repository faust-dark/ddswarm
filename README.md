# ddswarm — drug discovery multi-agent swarm (Increment 2: ADMET/PK prediction abstraction)

A reproducible, traceable, verifiable agentic drug-discovery pipeline.
**Increment 2** adds `predictor_minion` + a pluggable ADMET/PK prediction
abstraction (applicability domain + uncertainty) on top of Increment 0-1
(contracts + orchestrator + logging + researcher/designer/screener).
Docking, dynamics, etc. land in later increments.

## Architecture

- **Harness**: Hermes (this agent's runtime) drives execution.
  `ddswarm` provides the contract + state-machine layer that Hermes invokes.
- **Orchestrator** (`orchestrator_minion`): task graph, gating, retry,
  idempotent versioning, resume. Produces no chemistry.
- **Agents**:
  - `researcher_minion` — KnowledgePack from ProjectSpec + seed (refuses to fabricate).
  - `designer_minion` — DesignBatch from upstream scaffolds (labels only; no de novo yet).
  - `screener_minion` — deterministic structural + physicochemical screening (RDKit).
  - `predictor_minion` — ADMET/PK predictions through pluggable providers.
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
| `AdmetPrediction` | predictor output (per-molecule, per-endpoint: model/version/AD/uncertainty) |

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

## Prediction (Increment 2)

`predictor_minion` runs ADMET/PK predictions through a pluggable provider
interface (`predictors/`). Each endpoint prediction carries:

- `model` + `model_version` (documented)
- `predicted_value` + `units`
- `prediction_status` (`predicted|not_available|out_of_domain|error`)
- `evidence_level` = `predicted` (a computational score is never experimental evidence)
- `applicability_domain` (status/method/note)
- `uncertainty` (type/value/note)

Providers are registered in `predictors/registry.py`. This increment ships
`NullAdmetProvider` — a deterministic offline provider with NO model, which
returns `not_available` for a canonical endpoint set (solubility, Caco-2, BBB,
hERG, CYP3A4/2D6, Ames, PPB, half-life, clearance). It exercises the contract
end-to-end without fabricating any value. A real model (e.g. `admet_ai`, which
needs a heavy torch+CUDA install) can be dropped in later without touching the
contract or orchestrator.

No provider configured, or no molecules => `insufficient_evidence`.

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
