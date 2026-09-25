# ddswarm — drug discovery multi-agent swarm (Increment 3: docking + redocking validation)

A reproducible, traceable, verifiable agentic drug-discovery pipeline.
**Increment 3** adds `docking_minion` + a real AutoDock Vina engine with
receptor structural QC and redocking validation (RMSD vs the co-crystal
ligand) on top of Increments 0-2 (contracts + orchestrator + logging +
researcher/designer/screener/predictor). Dynamics and beyond land later.

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
  - `docking_minion` — receptor prep + QC, docking (AutoDock Vina), redocking RMSD.
- **Chemistry layer** (`chemistry/rdkit_ops.py`, `chemistry/pdb_utils.py`):
  sanitization, canonicalization, descriptors, PAINS/Brenk alerts, and pure-Python
  PDB structural analysis (missing residues/atoms, waters, heterogens, hydrogens).
  The agent reasons; these modules compute.

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
| `DockingResult` | docking output (receptor QC, protocol, ranked poses, redocking RMSD) |

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

## Docking (Increment 3)

`docking_minion` runs a real AutoDock Vina engine (`docking/vina_engine.py`):

1. **Receptor**: raw PDB -> `PDBFixer` (clean, add hydrogens) -> Open Babel `-xr`
   (rigid PDBQT). Structural QC runs on the *raw* structure and reports, as
   observed facts: missing residues (SEQRES vs observed), missing backbone
   atoms, waters, heterogens, hydrogens.
2. **Ligand**: SMILES -> `Meeko` -> PDBQT with a `REMARK SMILES IDX` table
   (maps SMILES atom order to PDBQT serials).
3. **Docking**: the Vina binary via subprocess; poses/affinities are parsed
   straight from Vina's output — nothing is invented.
4. **Redocking validation**: when the source PDB carries a co-crystal ligand
   (e.g. gefitinib `IRE` in 2ITY), the rank-1 docked pose is compared to it
   with a symmetry-aware heavy-atom RMSD (`rdMolAlign.CalcRMS`, no
   pre-alignment). Threshold is 2.0 A by default.

Every pose carries rank/affinity/rmsd-lb/rmsd-ub; the result carries receptor
QC, the search box, engine version, and redocking RMSD. Docking scores are
`evidence_level = calculated` — in-silico, never experimental evidence.

Honest result (this increment's demo): gefitinib redocks into 2ITY with RMSD
~3.5 A (above the 2 A success threshold). The protocol reproduces the pocket
(centroid ~1 A) but not the exact pose; that is reported as-is, not polished.

The engine needs the AutoDock Vina binary (not on PyPI). Download it to
`tools/vina` (gitignored):

```bash
curl -sL https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.7/vina_1.2.7_linux_x86_64 -o tools/vina
chmod +x tools/vina
# plus: .venv/bin/pip install -e '.[docking]'   # meeko, pdbfixer, openbabel-wheel, scipy, gemmi
```

Or point the agent at it with `params.vina_binary` / the `VINA_BINARY` env var.

## Run

```bash
cd ddswarm
python3 -m venv .venv
.venv/bin/pip install -e .            # ddswarm + pydantic + rdkit
.venv/bin/pip install pytest          # for tests
.venv/bin/python -m pytest -q         # run the test suite
.venv/bin/python scripts/demo_egfr.py # full pipeline: research -> design -> screen -> predict -> dock
.venv/bin/python scripts/demo_screen.py # one molecule per screening outcome
.venv/bin/python scripts/demo_docking.py # docking + redocking showcase (needs vina + network)
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

Run `pytest -q` — 54 tests pass. The real-Vina smoke test auto-skips when the
binary is absent; the receptor QC, PDBQT parsing, RMSD, and docking_minion
gating tests are offline and always run.
