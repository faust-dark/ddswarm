# ddswarm — drug discovery multi-agent swarm (Increment 19: target-selectivity / sexual-ADR scoring)

A reproducible, traceable, verifiable agentic drug-discovery pipeline.

## Design principles

Modeling is **mechanistically constrained by default**. Wherever domain
knowledge supplies governing equations (compartmental PK mass balance,
receptor-occupancy saturation, turnover kinetics), those equations are the
model. Machine-learning predictions are only admissible when they do **not**
violate conservation laws, mass balance, stoichiometry, thermodynamic
relationships, or monotonicity constraints — and any such relaxation requires
explicit, cited justification, never silent.

When sufficient mechanistic structure exists, ddswarm uses a **hybrid
mechanistic/ML** approach rather than a purely data-driven one: ML components
may estimate unknown parameters, residuals, or latent relationships, while the
mechanistic model enforces the physical constraints.

These principles are enforced, not aspirational: `validation_minion` runs a
`mechanistic_constraints` check that (a) verifies the linear-compartment
mass-balance identity `AUC = Cmax·V/CL`, (b) requires the systemic
concentration curve to be monotonically non-increasing (no spurious
re-absorption), and (c) flags any ADMET endpoint labelled `predicted` without
a citable rule. Today every computational model is deterministic and rule- or
physics-based (published QSAR rules, AutoDock Vina scoring, OpenMM force
fields, exact PK/PD ODEs); no unconstrained ML exists in the pipeline.

**Increment 9** replaces the no-op ADMET provider with a **deterministic
rule-based provider** (`rule_admet`): five published QSAR rules computed from
RDKit descriptors — ESOL aqueous solubility (Delaney 2004), QED drug-likeness
(Bickerton 2012), Lipinski rule-of-5 violations, Veber bioavailability, and
Egan passive absorption — each with an exact formula/threshold and a literature
citation. No ML model, no torch/CUDA. On top of Increments 0-8 (contracts +
orchestrator + logging + research/design/screen/predict/docking/dynamics/
retrosynthesis/tox/formulation + memory + report + feedback loop + safety gate +
validation + knowledge agents + CLI + Hermes reasoning).

**Increment 10** adds a deterministic **PDF report export**: every run ends in a
self-contained, human-readable "scientific essay" PDF (reportlab + matplotlib,
both deterministic and headless). The body is narrative prose — a verbose
interpretation of every result in plain language — with figures kept only as
auxiliary aids to the text. The PDF is a binary artifact alongside the JSON
artifacts (a `ReportDocument` metadata contract references it and is
versioned/hashed like any other artifact), and a copy is exported to
`reports/<project_id>.pdf` for easy access.

**Increment 11** turns ddswarm into a *discovery* pipeline, not just a
characterization one. `designer_minion` now generates genuinely NEW analog
molecules offline through a deterministic RDKit transformation library
(`chemistry/analog_generator.py`: Cl→F/Br/I halogen swaps, methyl→ethyl
homologation, N-demethylation) — real SMILES, not label-only copies of the
input. `report_minion` ranks the screening survivors by a deterministic
composite (QED + BCS bonus − 0.1·SA − 0.2·tox alerts) and names the top one the
`identified_candidate`; the PDF report draws its 2D structure (RDKit) with
SMILES + InChIKey, plus a ranked candidate table. A reasoning backend still
proposes LLM analogs; offline, the generator is the deterministic path.

**Increment 12** makes the report readable and complete. `retrosynthesis_minion`
now computes a deterministic **BRICS retrosynthetic disconnection** — the
theoretical building blocks the molecule breaks into (or an honest "no simple
disconnection" for fused/aromatic scaffolds) — alongside the SA score. The PDF
report labels every stage as "Stage — agent" (e.g. "Synthesis —
retrosynthesis_minion"), and the Methodology section lists, in plain language,
what each agent does.

**Increment 13** adds an opt-in **IUPAC name** as a human-readability label
alongside the canonical identifiers. SMILES + InChIKey remain the machine
backbone (every compute step consumes SMILES); the IUPAC name is fetched only
when asked for, from the NIH CACTUS web service (`--iupac`), and is clearly
marked non-deterministic + non-authoritative — it is excluded from the report's
reproducibility hash.

**Increment 14** adds deterministic **PK/PD** and turns Hermes reasoning on by
default. `clinical_minion` now emits a `ClinicalSimulation` artifact alongside
its clinical knowledge: a deterministic one-compartment PK model (t1/2, Cmax,
AUC, concentration-time curve) computed from clearance + volume of distribution
(or t1/2 + one of them), supplied explicitly or recalled by the reasoning
backend. It runs only when sufficient inputs exist — otherwise it reports
`insufficient_inputs`. The `predictor_minion` gained a `BBB_Permeability` rule
(Pajouhesh & Lenz 2005) and honestly reports the PK endpoints that need a QSPR
model (clearance, Vd, t1/2, protein binding) as `not_available`. The project
input now carries `mechanism_of_action`, `indication`, `existing_drugs`,
`chemical_scaffold`, `adme_issue`, `dose_mg`, `route_of_administration` — the
swarm workflow schema's primary-input fields. **Hermes reasoning is now the
default** (`reasoning.backend` defaults to `hermes`); `--reasoning none` opts
out.

**Increment 15** adds PK/PD model *complexity selection* and mechanism-based PD
models, per the schema's 7C requirement to not assume the simplest model is
correct:

- **PK**: `one_compartment` (existing) + `two_compartment` (new; analytic
  biexponential solution with distribution α / terminal β half-lives, V1/Q/V2).
- **PD**: `emax` and `hill` (sigmoid-Emax) direct-effect models on the PK
  concentration curve.
- **Model selection** is two-level: `mechanism_hint` (explicit
  `params.pk_model`/`params.pd_model`, or a documented keyword match against
  `mechanism_of_action`/`adme_issue`), then `data_driven` (goodness-of-fit
  RSS/AIC/BIC + a parameter-identifiability note against supplied observed
  concentration/effect data), else a labelled `default` (never claimed as
  validated). Parameters (V1/Q/V2, Emax/EC50/Hill-n) come from `params`
  (user) or LLM recall (inferred); the math is always `calculated`.
- `indirect_response`, `effect_compartment`, `receptor_occupancy` PD models and
  the tissue/organ PBPK models (well-stirred, blood-flow-limited,
  permeability-limited) are deferred to later increments.

**Increment 16** adds the mechanism-based PD model families, selected by
mechanism (explicit `params.pd_model` or a keyword match on
`mechanism_of_action`/`adme_issue`):

- `indirect_response` — turnover model (Dayneka Type I, inhibition of kin):
  dR/dt = kin·(1 − Imax·C/(IC50+C)) − kout·R, R(0) = baseline; integrated with a
  fixed-step RK4 ODE solver (deterministic, no scipy).
- `effect_compartment` — link model for delayed response: dCe/dt = ke0·(Cp − Ce),
  E = E0 + Emax·Ce/(EC50+Ce); RK4-integrated.
- `receptor_occupancy` — fractional target engagement RO = C/(C + Kd), algebraic.

`params.pd_model` (and `pk_model`) may be a single string for all molecules or a
`{molecule_id: model}` map for per-molecule overrides. The PD model is still
selected by mechanism — the goodness-of-fit/AIC-BIC data-driven comparison
remains within the direct-effect family (emax vs hill) only, since the other
families have incommensurate output scales.

**Increment 17** completes the PK/PD complexity axis with multi-compartment and
tissue-level (organ) models:

- `multi_compartment` — mammillary N-compartment model (central + N peripheral
  compartments, explicit `peripheral_compartments` list of {V, Q}), integrated
  with a vector fixed-step RK4; AUC via the mass-balance identity D/CL_total,
  terminal t1/2 from a log-linear least-squares tail fit.
- Organ/tissue models (single target tissue, opt-in via `params.tissue_model`
  or a keyword match; layered on the systemic PK curve):
  - `well_stirred` — Ct = Kp·C_plasma (algebraic).
  - `blood_flow_limited` — dCt/dt = (Qt/Vt)·(C − Ct/Kp) (perfusion-limited, RK4).
  - `permeability_limited` — dCt/dt = (PS/Vt)·(C − Ct/Kp) (membrane barrier, RK4).

`PKModelKind` now covers systemic models only (one/two/multi-compartment);
`TissueSimulation`/`TissueParameters` hold the optional tissue-level model. The
data-driven model selection compares ALL feasible systemic models (1 vs 2 vs N)
via a total-order fit key (perfect fit < BIC < RSS); whole-body multi-organ PBPK
is deliberately out of scope (no deterministic parameter source).

**Increment 18** ties the remaining schema items together — orchestration,
iteration, and memory as one linked loop:

- `orchestrator_minion` runs first and emits an `ExecutionPlan` (ordered
  stages + roles + convergence config + a deterministic prose rationale) — the
  auditable record of *why* and *how* the pipeline was assembled.
- `memory_minion` runs after report and distils the generation into a
  `GenerationMemory` (best candidates, avoided scaffolds, insights), also
  written as a durable `generation` memory entry for the next cycle.
- Iteration loop: `ddswarm run --iterations N` runs N generations (each a full
  run); the designer already consumes prior survivors via the feedback loop,
  and the CLI stops early when no new candidate emerges (convergence).
- `dynamics_minion` gains an opt-in four-branch gate (`params.gate_branches=true`):
  receptor MD runs only when predictor + docking + tox + formulation all produced
  results for at least one candidate.
- `regulatory_minion` now also emits a seven-dimension `RegulatoryChecklist`
  (quality/CMC, nonclinical tox, efficacy, safety, PK-PD, IP/FTO, pathway).
- The PDF opens with a "Reasoning and plan" section (why/how we reached the
  candidates, from the plan + generation memory + ranking rationale) and a
  "Workflow schema" section that reflects the exact stage -> agent -> role
  mapping (single source of truth in `agents/roles.py`).

Single-generation runs are unchanged (backward compatible); orchestration +
memory agents are injected only when `--iterations >= 2`.

**Increment 19** adds mechanistic **target-selectivity / sexual-adverse-effect
scoring** (`selectivity_minion`, opt-in). Each candidate is docked against SERT,
5-HT2A and 5-HT2C (AutoDock Vina); the docking free energy is converted to a
dissociation constant via the Gibbs identity (ΔG = RT·ln Ki), a fractional
receptor occupancy is computed at a fixed reference concentration
(C/(C+Ki), saturable and monotonic), and a selectivity index
(Ki_5HT2C/Ki_SERT, Ki_5HT2A/Ki_SERT) plus a deterministic sexual-ADR risk label
are derived. The classification encodes the SSRI mechanism — sexual dysfunction
is an *indirect* effect driven by SERT occupancy (elevated synaptic 5-HT
activating 5-HT2A/2C), so a candidate with negligible direct 5-HT2C binding
keeps the class effect (`class_effect`); a drop in SERT occupancy gives a
reduced ADR only via reduced efficacy (`efficacy_tradeoff`); and a candidate
that gains direct 5-HT2C binding (`direct_5ht2c_engagement`) is flagged as of
unknown sign because docking reports binding, not agonist/antagonist function.

Two honesty constraints are built in: Vina scores are relative, so every
Ki/occupancy is a *ranking* estimate (never an absolute affinity), and the
reference concentration (default 10 nM) is a documented modeling assumption
whose choice affects occupancy magnitude but not the monotonic ordering.

## Architecture

- **Harness**: Hermes (this agent's runtime) drives execution.
  `ddswarm` provides the contract + state-machine layer that Hermes invokes.
- **Runner** (`orchestrator/runner.py`): the state machine — task graph, gating,
  retry, idempotent versioning, resume. Produces no chemistry.
- **Agents**:
  - `orchestrator_minion` — runs first; emits the ExecutionPlan (stages, roles,
    convergence, deterministic rationale).
  - `memory_minion` — runs after report; emits the GenerationMemory and writes a
    durable `generation` memory entry for the next cycle.
  - `researcher_minion` — KnowledgePack from ProjectSpec + seed, plus optional
    delegation to a reasoning backend (Hermes subagent) for literature research.
  - `designer_minion` — DesignBatch. Offline: deterministic analog generation
    via an RDKit transformation library (halogen swaps, homologation,
    N-demethylation) — real new SMILES. With a reasoning backend: LLM-proposed
    analog SMILES (hypothetical).
  - `screener_minion` — deterministic structural + physicochemical screening (RDKit).
  - `predictor_minion` — ADMET/PK predictions through pluggable providers
    (default: deterministic `rule_admet` rules; `null_admet` for an explicit
    no-model run).
  - `docking_minion` — receptor prep + QC, docking (AutoDock Vina), redocking RMSD.
  - `retrosynthesis_minion` — RDKit synthetic-accessibility score per candidate
    (calculated) + easy/moderate/hard classification; a deterministic BRICS
    retrosynthetic disconnection (building blocks, or an honest "no simple
    disconnection" for fused scaffolds); optional LLM-proposed synthesis routes
    (hypothetical) via the reasoning backend.
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
    next steps, caveats), ranks the screening survivors by a deterministic
    composite score, and names the top one the `identified_candidate`; also
    writes `finding` + `feedback` to project memory.
  - `report_pdf_minion` — renders the finished run into a self-contained PDF in
    plain, verbose human language: every stage is a labeled subsection
    ("Synthesis — retrosynthesis_minion"), the Methodology lists what each agent
    does, and figures are only auxiliary aids; runs last and persists the PDF as
    a binary sidecar plus a `ReportDocument` metadata artifact, and the CLI
    copies it to `reports/<project_id>.pdf`.
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

Advisory (reported, not eliminating): TPSA > 140, rotatable bonds > 10,
|formal charge| > 1, and any Brenk match (RDKit FilterCatalog, reported in
`brenk_alerts`).

Strict Ro5: any single violation eliminates (approved default).

Brenk is advisory by default: its "unwanted functionality" alerts are
liability flags, not kill criteria — e.g. it flags the approved drug gefitinib
as `Aliphatic_long_chain` on the OCCCN propyl-morpholine chain. Restore hard
Brenk per-project with `params.filters.brenk_hard=true`.

Screen is a soft gate: downstream agents consume screened_batch survivors
(`passed=true`); when none pass they fall back to the starting material(s), so
ADMET/tox/formulation/knowledge still run on the parent scaffold rather than
the pipeline stalling on zero survivors.

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
ddswarm run project.json [--reasoning none|hermes] [--iupac] [--store ./store] [--reports-dir reports/]
ddswarm runs [project_id]                       # list runs
ddswarm report project_id [run_id]              # per-task report
ddswarm memory project_id                       # cross-run conclusions
```

`project.json` is a `RunRequest`: `{project: ProjectSpec, tasks: [...], reasoning: {...}}`.
If `tasks` is omitted a default pipeline runs (safety -> research -> design ->
screen -> predict -> retrosynth -> tox -> formulation -> report -> validation ->
report_pdf);
docking and dynamics are opt-in (need the Vina binary / OpenMM + network).
`report` writes `finding` + `feedback` to project memory, closing the loop: the
next run's designer/researcher read it. `report_pdf` exports the run as a PDF;
`ddswarm run` copies it to `reports/<project_id>.pdf` (override with
`--reports-dir DIR` or the `DDSWARM_REPORTS_DIR` env var).

`"reasoning": {"backend": "hermes"}` makes `researcher_minion`, `designer_minion`,
`retrosynthesis_minion` (routes only), and the knowledge agents `patent_minion`/
`clinical_minion`/`regulatory_minion` delegate to one-shot Hermes subagents
(`hermes chat -q`), merging the LLM output with the deterministic baseline.
This is now the **default**: `reasoning.backend` defaults to `hermes`, so
`ddswarm run` delegates by default; pass `--reasoning none` (or
`"reasoning": {"backend": "none"}`) to run the deterministic baseline only.
When the backend is unavailable the deterministic agents fall
back to their baseline — seed assembly, transformation labels, SA score — while
the knowledge agents return `insufficient_evidence` (they have no offline
baseline). LLM-contributed entries are forced to `evidence_level`
`inferred`/`hypothetical` and tagged "verify against primary source", because a
recalled PubChem ID or patent number is not experimental evidence.

`--iupac` fetches IUPAC names for the ranked candidates from the NIH CACTUS web
service and prints them in the report as a readability label (alongside the 2D
structure, SMILES and InChIKey). It is opt-in because it is network-dependent
and third-party: the names are non-deterministic across time and not
authoritative, so they are excluded from the report's reproducibility hash.
Canonical SMILES + InChIKey remain the identifiers of record.

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

Run `pytest -q` — 140 tests pass. The real-Vina smoke test auto-skips when the
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

Hardening (point 4) adds: filesystem-ID validation (project/task/artifact IDs
reject path traversal at both the contract and Store layers), CLI error
handling (missing file / bad JSON / bad IDs exit 1 with a clean message), and
an end-to-end reproducibility test (two runs of the default pipeline produce
identical artifact fingerprints, excluding timestamp/run_id metadata).
