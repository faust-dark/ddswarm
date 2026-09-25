# ddswarm — drug discovery multi-agent swarm (Increment 0 PoC)

A reproducible, traceable, verifiable agentic drug-discovery pipeline.
This is **Increment 0**: contracts + orchestrator + logging + two minimal
agents. Chemistry (RDKit), ADMET, docking, etc. land in later increments and
are deliberately *not* implemented yet.

## Architecture

- **Harness**: Hermes (this agent's runtime) drives execution.
  `ddswarm` provides the contract + state-machine layer that Hermes invokes.
- **Orchestrator** (`orchestrator_minion`): task graph, gating, retry,
  idempotent versioning, resume. Produces no chemistry.
- **Agents** (`researcher_minion`, `designer_minion`): minimal, honest stubs.
  They refuse to fabricate (return `insufficient_evidence`) instead of inventing
  results a tool should produce.

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

Every artifact carries `evidence_level` (`observed|calculated|predicted|inferred|hypothetical`)
and provenance (source, URL/identifier, timestamps, tool versions, hashes).

## Run

```bash
cd ddswarm
python3 -m venv .venv
.venv/bin/pip install -e .            # install ddswarm + pydantic
.venv/bin/pip install pytest          # for tests
.venv/bin/python -m pytest -q         # run the test suite
.venv/bin/python scripts/demo_egfr.py # end-to-end demo (writes to store/)
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

## Gating (this increment)

- **Gate A — input validity**: objective non-empty; target/ligand inputs match
  `input_type`; ProjectSpec must be `approved`.
- **Result gating**: `failed`/`blocked`/`insufficient_evidence` stops downstream
  tasks; `success`/`partial` lets them proceed.

## Verification status

Run `pytest -q` — all tests must pass before this increment is considered done.
