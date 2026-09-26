# ddswarm — drug discovery multi-agent swarm (Increment 9: rule-based ADMET)

A reproducible, traceable, verifiable agentic drug-discovery pipeline.
**Increment 9** replaces the no-op ADMET provider with a **deterministic
rule-based provider** (`rule_admet`): five published QSAR rules computed from
RDKit descriptors — ESOL aqueous solubility (Delaney 2004), QED drug-likeness
(Bickerton 2012), Lipinski rule-of-5 violations, Veber bioavailability, and
Egan passive absorption — each with an exact formula/threshold and a literature
citation. No ML model, no torch/CUDA. On top of Increments 0-8 (contracts +
orchestrator + logging + research/design/screen/predict/docking/dynamics/
retrosynthesis/tox/formulation + memory + report + feedback loop + safety gate +
validation + knowledge agents + CLI + Hermes reasoning).

## Architecture

- **Harness**: Hermes (this agent's runtime) drives execution.
  `ddswarm` provides the contract + state-machine layer that Hermes invokes.
- **Orchestrator** (`orchestrator_minion`): task graph, gating, retry,
  idempotent versioning, resume. Produces no chemistry.
- **Agents**:
  - `researcher_minion` — KnowledgePack from ProjectSpec + seed, plus optional
    delegation to a reasoning backend (Hermes subagent) for literature research.
  - `designer_minion` — DesignBatch; deterministic labels, or real analog SMILES
    from a reasoning backend when configured.
  - `screener_minion` — deterministic structural + physicochemical screening (RDKit).
  - `predictor_minion` — ADMET/PK predictions through pluggable providers
    (default: deterministic `rule_admet` rules; `null_admet` for an explicit
    no-model run).
  - `docking_minion` — receptor prep + QC, docking (AutoDock Vina), redocking RMSD.
  - `retrosynthesis_minion` — RDKit synthetic-accessibility score per candidate
    (calculated) + easy/moderate/hard classification; optional LLM-proposed
    synthesis routes (hypothetical) via the reasoning backend.
  - `tox_minion` — deterministic, sourced structural toxicophore alerts (SMARTS),
    reported as *predicted* potential liabilities, never experimental evidence.
  - `dynamics_minion` — receptor molecular dynamics (OpenMM, amber14 + OBC2 GBSA
    implicit solvent, Langevin NVT): backbone-RMSD stability + potential-energy
    fluctuation, reported as *calculated*. A short ps-scale relaxation snapshot,
    NOT production MD and never a binding free energy. Ligand MD is deferred
    (openff-toolkit is unavailable on Python 3.14).
  - `formulation_minion` — deterministic developability estimate per molecule:
    aqueous-solubility class, permeability class, BCS classification, ionizable
    groups, and salt/prodrug suggestions. All *predicted* rule-based estimates
    from RDKit logP/TPSA — never measured properties.
  - `safety_minion` — deterministic gating: controlled substances (exact
    canonical-SMILES match) and dual-use/biosafety targets (name match) against
    a curated, non-exhaustive list. A `block` verdict stops every downstream
    task via the orchestrator's result gating.
  - `validation_minion` — audits a finished run from the store: artifact schema
    validity, provenance completeness, hash integrity (anti-tamper), cross-agent
    consistency, evidence-level sanity, numeric sanity, and report-breakdown
    consistency. Reports per-check pass/fail/warn; never a scientific judgment.
  - `patent_minion` — patent / freedom-to-operate landscape for the candidates
    and/or target, delegated to the Hermes reasoning backend; every entry is LLM
    recall forced to `inferred` + "verify against primary source". A research
    aid, NOT a legal freedom-to-operate opinion.
  - `clinical_minion` — clinical-stage knowledge (known drugs, trial phase,
    indication, status) for the target/candidates, delegated to the reasoning
    backend; every entry `inferred`.
  - `regulatory_minion` — regulatory knowledge (agency designations, approval
    status, scheduling), delegated to the reasoning backend; every entry
    `inferred`.
  - `report_minion` — aggregates a finished run into a structured Report
    (executive summary, per-stage findings, evidence-level breakdown, gaps,
    next steps, caveats) and writes `finding` + `feedback` to project memory.
- **Project memory** (`orchestrator/memory.py`): an append-only, cross-run log
  under `store/<project_id>/memory.json`. Conclusions are `inferred` (the
  swarm's reasoning), never observed data.
- **Feedback loop**: report_minion records survivors + gaps to memory; in the
  next run, `designer_minion` adds prior survivors as design scaffolds and
  `researcher_minion` re-seeds prior findings (evidence_level=inferred).
- **Reasoning layer** (`reasoning/`): a pluggable backend that the reasoning
  agents use. `NullReasoningBackend` (offline, deterministic baseline) and
  `HermesSubprocessBackend` (shells out to `hermes chat -q`). LLM output is
  never trusted as observed data — the agents force `evidence_level` to
  `inferred`/`hypothetical` and tag provenance accordingly.
- **CLI** (`ddswarm`): `run`/`runs`/`report`/`init` — see below.
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
| `RetrosynthesisPlan` | SA score + accessibility per molecule, optional synthesis routes |
| `ToxProfile` | per-molecule structural toxicophore alerts (predicted liabilities) |
| `DynamicsResult` | receptor MD run: protocol, backbone-RMSD + energy metrics (calculated) |
| `FormulationAssessment` | per-molecule solubility/permeability/BCS + salt/prodrug suggestions |
| `SafetyAssessment` | safety gate: flags (controlled substance / dual-use) + verdict pass/flag/block |
| `ValidationReport` | per-check reproducibility/consistency audit of one run |
| `PatentLandscape` | patent/FTO entries (LLM-recalled, inferred) + relevance summary |
| `ClinicalAssessment` | clinical findings (LLM-recalled, inferred) + phase counts |
| `RegulatoryAssessment` | regulatory findings (LLM-recalled, inferred) + agency counts |
| `Report` | structured run summary (exec summary, evidence breakdown, gaps, next steps) |
| `ProjectMemory` | cross-run conclusions (finding/decision/lesson/constraint/feedback) |
| `RunRequest` | project.json: ProjectSpec + task DAG + optional reasoning config |

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
- `predicted_value` + `units` (continuous) and/or `predicted_label` (categorical)
- `citation` (literature source for the rule/model)
- `prediction_status` (`predicted|not_available|out_of_domain|error`)
- `evidence_level` = `predicted` (a computational score is never experimental evidence)
- `applicability_domain` (status/method/note)
- `uncertainty` (type/value/note)

Providers are registered in `predictors/registry.py`. The default is
`RuleBasedAdmetProvider` (`rule_admet`), a deterministic RDKit-descriptor
provider with five cited rules — every output `evidence_level = predicted`:

- `LogS_ESOL` (Delaney 2004): LogS = 0.16 - 0.63·clogP - 0.0062·MW + 0.066·RB - 0.74·AP (continuous, std 0.75)
- `QED` (Bickerton 2012): RDKit `QED.default()`, 0-1
- `Lipinski_Violations` (Lipinski 1997): rule-of-5 violation count
- `Veber_Bioavailability` (Veber 2002): good/poor (RB <= 10 and TPSA <= 140 A^2)
- `Egan_Absorption` (Egan 2000): good/poor (box approximation, TPSA < 131.6 and 0.4 < clogP < 5.88)

`NullAdmetProvider` (`null_admet`) stays registered for tests and explicit
no-model runs, returning `not_available` for the admet_ai-style canonical set.
A heavier ML model (e.g. `admet_ai`, torch+CUDA) can still be dropped in later
behind the same interface. BBB / hERG / CYP / Ames / half-life / clearance are
deferred — they need trained models, not precise RDKit rules.

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

## Dynamics (Increment 6)

`dynamics_minion` runs real receptor MD through a pluggable `MdEngine`
(`dynamics/`), mirroring the docking engine pattern:

- **`NullMdEngine`** — deterministic no-op: reports `not_run` with an honest
  note (no MD engine configured). The offline default.
- **`OpenMMMdEngine`** — real: `PDBFixer` prep (the same receptor QC as
  docking) -> `amber14-all.xml` + OBC2 GBSA implicit solvent -> Langevin NVT
  (fixed seed) -> bounded minimization + a short trajectory. Reports, per
  receptor run: backbone heavy-atom (N/CA/C) RMSD (mean/max, Kabsch-aligned)
  and potential-energy mean/std. All `evidence_level = calculated`.

Honesty rules: this is a **short ps-scale relaxation snapshot**, not production
MD; backbone RMSD is never presented as a converged stability verdict, and no
MM/PBSA-style binding free energy is computed or implied. **Ligand MD is
explicitly deferred** — it needs a small-molecule force field (openff-toolkit),
which is not installable on Python 3.14; the `MdEngine` abstraction makes it a
drop-in when available.

The engine needs OpenMM + PDBFixer (already pulled in by the `[docking]` extra).
A receptor is configured via `params.receptor` (`{kind: pdb_id, identifier: 2ITY}`
downloads from RCSB; `{kind: file, path: ...}` uses a local PDB). Note: GBSA on
a ~330-residue receptor is genuinely expensive (~5 min for a bounded run) —
see `scripts/demo_egfr.py`, which uses `steps=200, minimize_max_iterations=100`.

## Formulation (Increment 6)

`formulation_minion` is deterministic and offline (RDKit only). Per molecule it
reports, from `logP`/`TPSA` heuristics (documented in `chemistry/formulation.py`):

- **aqueous-solubility class** (`high|moderate|low`, crude logP heuristic)
- **permeability class** (`high|moderate|low`, TPSA/logP; Veber TPSA<=140 A^2)
- **BCS class** (`I|II|III|IV`, the 2x2 solubility x permeability grid)
- **ionizable groups** (acidic: COOH/SO3H/tetrazole; basic: non-amide amine +
  pyridine-like aromatic N), plus **salt** and **prodrug** suggestions

Every field is `evidence_level = predicted` — a rule-based estimate, explicitly
not a measured solubility/permeability.

## Safety gate (Increment 7)

`safety_minion` is a deterministic gate over two curated lists
(`chemistry/safety.py`): controlled substances (matched by EXACT canonical
SMILES — fentanyl, methamphetamine, MDMA) and dual-use/biosafety targets
(matched by name — ricin, botulinum toxin, anthrax, variola, ebola, sarin,
sulfur mustard). A match is an *observed* fact against that list, and it
returns `verdict=block` -> `AgentResultStatus.blocked`, which the orchestrator
propagates to every downstream task. The list is deliberately small and
non-exhaustive, and the note on every flag says so — this is a policy aid,
NOT a legal determination and NOT a database.

## Validation (Increment 7)

`validation_minion` reads a finished run straight from the store and emits a
`ValidationReport` with one check per concern (each pass/fail/warn):

- `artifact_schema` — every artifact re-validates against its contract model
- `provenance_complete` — every artifact carries >= 1 provenance entry
- `hash_integrity` — `ArtifactRef.hash` equals a fresh hash of the file
- `cross_agent_consistency` — screened survivors appear in retrosynth/tox/formulation
- `evidence_level_sanity` — computed/predicted values are never labelled `observed`
- `numeric_sanity` — finite, plausible docking affinity / SA score / RMSD
- `report_breakdown` — `Report.evidence_breakdown` matches a recount of the artifacts

A failed check means the run's own bookkeeping is inconsistent; the report
status is `success` when no check fails, else `partial` (it never fabricates a
green result, and it never blocks the run — it is an audit, not a gate).

## Knowledge agents (Increment 8)

`patent_minion`, `clinical_minion`, and `regulatory_minion` are *knowledge*
agents with no deterministic offline source, so they are opt-in (not in the
default pipeline) and run only when `reasoning.backend=hermes` is configured.
Each delegates through three new `ReasoningBackend` methods
(`patent_landscape`, `clinical_assessment`, `regulatory_assessment`) and
validates the returned raw dicts against its contract. Without a backend they
return `insufficient_evidence` — they never fabricate patent numbers, trial
data, or designations.

Every entry is forced to `evidence_level='inferred'` and tagged "verify against
primary source": an LLM-recalled patent number is not an observed fact. The
patent agent only counts relevance (how many entries the LLM labelled
`covers_candidate`); it does NOT render a freedom-to-operate or infringement
opinion. To run them, add the three tasks with `params.reasoning_backend =
"hermes"` (plus `reasoning_command`), or set the top-level `reasoning.backend`,
which the CLI propagates to every task.

## CLI & reasoning backend

The `ddswarm` console script runs a whole pipeline from one versioned file:

```bash
ddswarm init project.json --project-id egfr --objective "inhibit EGFR" \
    --targets "EGFR" --smiles "COC1=..."       # scaffold (default pipeline)
ddswarm run project.json [--reasoning none|hermes] [--store ./store]
ddswarm runs [project_id]                       # list runs
ddswarm report project_id [run_id]              # per-task report
ddswarm memory project_id                       # cross-run conclusions
```

`project.json` is a `RunRequest`: `{project: ProjectSpec, tasks: [...], reasoning: {...}}`.
If `tasks` is omitted a default pipeline runs (safety -> research -> design ->
screen -> predict -> retrosynth -> tox -> formulation -> report -> validation);
docking and dynamics are opt-in (need the Vina binary / OpenMM + network).
`report` writes `finding` + `feedback` to project memory, closing the loop: the
next run's designer/researcher read it.

`"reasoning": {"backend": "hermes"}` makes `researcher_minion`, `designer_minion`,
`retrosynthesis_minion` (routes only), and the knowledge agents `patent_minion`/
`clinical_minion`/`regulatory_minion` delegate to one-shot Hermes subagents
(`hermes chat -q`), merging the LLM output with the deterministic baseline.
Without it (or when the backend is unavailable) the deterministic agents fall
back to their baseline — seed assembly, transformation labels, SA score — while
the knowledge agents return `insufficient_evidence` (they have no offline
baseline). LLM-contributed entries are forced to `evidence_level`
`inferred`/`hypothetical` and tagged "verify against primary source", because a
recalled PubChem ID or patent number is not experimental evidence.

## Run

```bash
cd ddswarm
python3 -m venv .venv
.venv/bin/pip install -e .            # ddswarm + pydantic + rdkit + the ddswarm CLI
.venv/bin/pip install pytest          # for tests
.venv/bin/python -m pytest -q         # run the test suite
.venv/bin/ddswarm init p.json --project-id egfr --objective "inhibit EGFR" --smiles "COC1=..."
.venv/bin/ddswarm run p.json          # default pipeline (null reasoning)
.venv/bin/python scripts/demo_egfr.py # full pipeline incl. docking + dynamics + validation
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

Run `pytest -q` — 129 tests pass. The real-Vina smoke test auto-skips when the
binary is absent, and the real OpenMM MD tests auto-skip when OpenMM/PDBFixer
are absent. Receptor QC, PDBQT parsing, RMSD, docking gating, reasoning
backends (with a fake `hermes` script), the CLI, SA-score classification, the
toxicophore matcher, project memory, report aggregation, the feedback loop,
MD-engine metrics (real MD on a small peptide fixture), the formulation rules,
the rule-based ADMET provider (exact ESOL/QED/Ro5/Veber/Egan values on aspirin,
hexadecane and sucrose + out-of-domain + determinism), the safety gate (controlled-substance + dual-use matching), the validation
audit (all checks + tamper detection), and the knowledge agents
(patent/clinical/regulatory delegation via a fake `hermes` script +
insufficient_evidence without a backend) are tested offline. Real
Hermes-subagent delegation was verified once manually (researcher returned 24
sourced entries, designer 8 analog SMILES); real receptor MD was verified on
the 2ITY kinase domain (backbone RMSD ~0.19 A over a 0.4 ps bounded relaxation).
