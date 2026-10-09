# ddswarm

> A reproducible, traceable, verifiable multi-agent drug-discovery pipeline —
> usable as a CLI and as an MCP server for any agentic stack (Google ADK, Hermes,
> OpenMinion, Claude, …).

**ddswarm** turns a therapeutic target and a starting molecule into a ranked set
of candidate analogs — each with a full characterization dossier (docking,
selectivity, ADMET, PK/PD, toxicology, synthesis) and a plain-language PDF
report — from a single command. Every step is deterministic and auditable by
default, with an optional LLM reasoning layer for literature recall and
structure proposal.

---

## What it does

Give it a target (e.g. EGFR, SERT, a GABA-A subunit) and one or more starting
molecules (SMILES). The swarm:

1. **Designs** new analog molecules — offline via a deterministic RDKit
   transformation library (halogen swaps, homologation, N-demethylation), or via
   an LLM when a reasoning backend is enabled.
2. **Screens** them against medicinal-chemistry filters (PAINS, rule-of-five,
   Brenk).
3. **Characterizes** every survivor: molecular docking and target-selectivity
   scoring, ADMET prediction, PK/PD simulation, retrosynthetic accessibility,
   structural toxicology (CYP, hERG, tox), and developability (formulation/BCS).
4. **Ranks** the survivors with a deterministic composite score and names the
   top molecule the *identified candidate*.
5. **Reports** the whole run as a self-contained PDF essay plus machine-readable
   exports (CSV / JSON / SDF).

Everything is stored as versioned, hashed JSON artifacts, so a run can be
replayed, audited, and reproduced.

---

## Design principles

Modeling is **mechanistically constrained by default**. Wherever domain
knowledge supplies governing equations — compartmental PK mass balance,
receptor-occupancy saturation, turnover kinetics — those equations *are* the
model. Machine-learning predictions are admissible only when they do **not**
violate conservation laws, mass balance, stoichiometry, thermodynamic
relationships, or monotonicity constraints; any such relaxation requires
explicit, cited justification, never a silent override.

When enough mechanistic structure exists, ddswarm uses a **hybrid
mechanistic/ML** approach: ML components may estimate unknown parameters,
residuals, or latent relationships, while the mechanistic model enforces the
physical constraints. This is not aspirational — the `validation_minion`
auditor enforces it with a `mechanistic_constraints` check that verifies the
mass-balance identity, requires systemic concentration curves to be monotonically
non-increasing, and flags any `predicted` endpoint lacking a citable rule.

Every computational model in the pipeline today is deterministic and rule- or
physics-based (published QSAR rules, AutoDock Vina scoring, OpenMM force fields,
exact PK/PD ODEs); there is no unconstrained ML.

---

## The pipeline

The default pipeline is 18 agents, wired as a task graph and executed by a
deterministic state machine (the **Runner**) that handles gating, retries,
versioning, and resume:

```
safety → research → overview → design → screen → docking → selectivity →
predict → retrosynth → tox → cyp → herg → formulation → patent →
report → export → validation → report_pdf
```

Two more agents are injected automatically when you ask for iteration
(`--iterations N`): `orchestrator` (first — writes the execution plan) and
`memory` (after report — distils the generation for the next cycle). Several
more are opt-in (see below).

### Gating

- **Input validity** — objective must be non-empty; target/ligand inputs must
  match the declared `input_type`; the spec must be `approved`.
- **Chemical validity** — every molecule must sanitize and canonicalize; invalid
  SMILES are eliminated with a reason, never silently dropped.
- **Result gating** — only a `blocked` verdict (a safety hit) stops downstream
  tasks. A `failed` agent still marks the run `failed` but does not block
  downstream (the error is surfaced in the report); `insufficient_evidence` is a
  reported gap, not a crash.

The `safety_minion` is the single hard gate: a controlled-substance or
dual-use/biosafety hit halts every downstream task and marks the run `blocked`.

---

## Agents

### Default pipeline

| Agent | Role |
|---|---|
| `safety_minion` | Hard gate: exact-SMILES controlled substances + name-matched dual-use targets. |
| `researcher_minion` | Assembles a KnowledgePack (targets, ligands, references) from the spec; optional LLM literature research. |
| `overview_minion` | Overview report — deterministic receptor-registry facts plus an optional LLM narrative (mechanism, structure, uses). |
| `designer_minion` | Proposes new analog molecules (RDKit transformations offline; LLM proposals with a backend). |
| `screener_minion` | Structural + physicochemical screening (PAINS, Ro5, Brenk). |
| `docking_minion` | Receptor prep + AutoDock Vina docking + redocking RMSD validation. |
| `selectivity_minion` | Docks an off-target panel; derives Ki/occupancy/selectivity index and ADR risk. |
| `predictor_minion` | ADMET/PK prediction through a pluggable provider (default: cited RDKit rules). |
| `retrosynthesis_minion` | Synthetic-accessibility score + deterministic BRICS disconnection (+ optional LLM routes). |
| `tox_minion` | Structural toxicophore alerts (SMARTS). |
| `cyp_minion` | CYP inhibition / induction flags. |
| `herg_minion` | hERG (Kv11.1) cardiotoxicity flag. |
| `formulation_minion` | Solubility / permeability / BCS class + salt and prodrug suggestions. |
| `patent_minion` | Patent / freedom-to-operate landscape (non-gating "UNKNOWN" baseline without a backend). |
| `report_minion` | Aggregates the run into a structured Report; ranks survivors; names the identified candidate. |
| `export_minion` | Machine-readable candidate export (CSV + JSON + SDF) with a go/no-go verdict. |
| `validation_minion` | Terminal audit: schema, provenance, hash integrity, consistency, evidence-level sanity. |
| `report_pdf_minion` | Renders the run into a self-contained plain-language PDF. |

### Opt-in

| Agent | Role |
|---|---|
| `dynamics_minion` | Receptor molecular dynamics (OpenMM, implicit solvent). |
| `grounding_minion` | Resolves molecules against ChEMBL / PubChem / PubMed (live, network). |
| `clinical_minion` | Clinical-stage knowledge + deterministic PK/PD simulation. |
| `regulatory_minion` | Regulatory knowledge + a seven-dimension regulatory checklist. |

---

## Capabilities

### Molecular design & screening

`designer_minion` generates genuinely **new** analogs offline through a
deterministic RDKit transformation library (`Cl→F/Br/I` halogen swap,
methyl→ethyl homologation, N-demethylation) — real new SMILES, not copies of the
input. With a reasoning backend it can also accept LLM-proposed analogs.

Screening eliminates on hard filters — invalid SMILES, MW > 500, logP > 5,
HBD > 5, HBA > 10, any PAINS match — and reports (without eliminating) TPSA >
140 Å², rotatable bonds > 10, formal charge outside ±1, and Brenk "unwanted
functionality" alerts. Brenk is advisory by default: it flags the approved drug
gefitinib as `Aliphatic_long_chain`, so it is a liability flag, not a kill
criterion (`filters.brenk_hard=true` restores hard Brenk per project).

### Structure-based modeling

- **Docking** (`docking_minion`) runs a real AutoDock Vina engine: `PDBFixer`
  receptor prep → Open Babel rigid PDBQT, `Meeko` ligand prep, Vina subprocess
  docking, and — when the source PDB carries a co-crystal ligand — a
  symmetry-aware redocking RMSD against it (2.0 Å success threshold). Receptor
  quality (missing residues/atoms, waters, heterogens) is reported from the raw
  structure as observed facts.
- **Target selectivity** (`selectivity_minion`) docks a configurable panel and
  derives mechanistic metrics: docking free energy → dissociation constant via
  the Gibbs identity, fractional occupancy at a reference concentration, and a
  selectivity index `Ki(off)/Ki(primary)`. A pluggable ADR-profile library
  interprets the panel (SSRI sexual-ADR, NSAID COX-2/COX-1 GI risk, and
  antitarget profiles for M1/H1/D2/α1/5-HT2B/κ), including a GABA-A profile
  (α2/α3 anxiolytic on-target; α1 sedation + α5 amnesia off-target).
- **Vina calibration** corrects docking scores against a fixed, committed
  literature-Ki set (NIMH PDSP) with a per-receptor intercept (slope fixed at
  1, so the Vina→Ki mapping stays monotonic); the selectivity ratio is
  approximately calibration-invariant. A ~61-target receptor registry
  (RCSB-verified PDBs) serves any panel.
- **Molecular dynamics** (`dynamics_minion`, opt-in) runs receptor MD with
  OpenMM (amber14 + OBC2 GBSA implicit solvent, Langevin NVT), reporting
  backbone RMSD stability and potential-energy fluctuation. This is a short
  ps-scale relaxation snapshot, not production MD, and never a binding free
  energy. Both docking and MD support content-addressed caching / checkpoint
  resume (`--resume`).

### ADMET & PK/PD

- **ADMET prediction** (`predictor_minion`) ships five cited, exact RDKit rules:
  ESOL solubility (Delaney 2004), QED drug-likeness (Bickerton 2012), Lipinski
  rule-of-five, Veber bioavailability, Egan absorption — plus a BBB permeability
  rule (Pajouhesh & Lenz 2005). Every prediction carries its model, version,
  citation, applicability domain, and uncertainty, and is labelled `predicted`
  (never experimental evidence). A heavier ML provider can be dropped in behind
  the same interface.
- **PK/PD simulation** (`clinical_minion`, hybrid) integrates exact ODEs with a
  deterministic fixed-step RK4 solver (no SciPy): one-, two-, and
  multi-compartment PK; `emax`, `hill`, indirect-response (turnover),
  effect-compartment (link), and receptor-occupancy PD models; and single-tissue
  organ models (well-stirred, blood-flow-limited, permeability-limited). Model
  complexity is selected by mechanism hint, by fit against observed data
  (RSS/AIC/BIC), or by a labelled default — never claimed as validated.

### Safety & toxicology

- **Structural toxicology** matches candidates against curated, cited SMARTS
  libraries: 22 toxicophores (aromatic amine, azide, acyl halide, catechol,
  glyoxal, …), CYP inhibition/induction chemotypes, and hERG pharmacophores
  (protonatable basic amine + ≥2 aromatic rings, gated by cLogP ≥ 3.0). A SMARTS
  match is a deterministic structural fact; the label is always
  "potential"/"-like" (`predicted`), never a functional-assay verdict.
- **Safety gate** (`safety_minion`) blocks on exact-canonical-SMILES controlled
  substances (fentanyl, methamphetamine, MDMA) and name-matched dual-use
  targets. The list is deliberately small and non-exhaustive — a policy aid, not
  a database or a legal determination.

### Retrosynthesis

`retrosynthesis_minion` scores each candidate for synthetic accessibility and
computes a deterministic **BRICS retrosynthetic disconnection** (building
blocks, or an honest "no simple disconnection" for fused/aromatic scaffolds).
With a reasoning backend it also proposes a human-readable synthesis *method* —
a narrative a bench chemist could start from, with literature references.

### Knowledge & grounding

- **Knowledge agents** (`patent_minion`, `clinical_minion`, `regulatory_minion`)
  have no deterministic offline source, so they delegate to the reasoning
  backend. Patent/FTO sits in the default pipeline with a non-gating baseline:
  without a backend it returns an empty landscape with an explicit "patent
  status UNKNOWN" note — never a fabricated clear, never a legal opinion. Every
  entry is forced to `inferred` and tagged "verify against primary source".
- **Grounding** (`grounding_minion`, opt-in) resolves each molecule by standard
  InChIKey against ChEMBL, PubChem, and PubMed, recording observed external
  records with IDs, URLs, and retrieval timestamps — the first machine-readable
  link to live databases — and optionally a binding-affinity profile.
- **On-demand ChEMBL layer** fetches per-compound literature at query time
  (target resolution, affinity, reference ligands) rather than committing a
  large static dataset.

### Reporting & export

- **PDF report** (`report_pdf_minion`) renders every finished run into a
  self-contained, plain-language "scientific essay": the user's prompt, an
  overview, the reasoning and plan, every stage as a labelled subsection,
  a "Structural modifications rationale" (what was changed on the parent
  scaffold and why), the identified candidate with its 2D structure, and an
  evidence-and-confidence discussion. Figures are auxiliary aids to the prose,
  never a data dump.
- **Candidate export** (`export_minion`) writes one flat row per screened
  molecule with every score plus a deterministic go/no-go verdict (no-go =
  screening reject or safety block; review = advisory liability; go = clean) as
  CSV, JSON, and SDF.
- **IUPAC names** are fetched (NIH CACTUS, then PubChem by InChIKey) for the
  ranked candidates as readability labels alongside SMILES + InChIKey — the
  machine identifiers of record. Names are non-authoritative and excluded from
  the reproducibility hashes; disable with `--no-iupac`.

---

## Reasoning backends

The swarm is **platform-agnostic by default**: `reasoning.backend` defaults to
`none`, so a bare `ddswarm run` executes the deterministic baseline with no LLM
or harness dependency. The reasoning layer is a pluggable backend:

- `NullReasoningBackend` — offline, deterministic fallback.
- `HermesSubprocessBackend` — shells out to `hermes chat -q` for one-shot LLM
  recall.

Opt into the LLM with `--reasoning hermes` (or `"reasoning": {"backend":
"hermes"}` in the project file), which makes the researcher, designer,
retrosynthesis (routes), and knowledge agents (patent/clinical/regulatory)
delegate to one-shot subagents and merge LLM output with the deterministic
baseline. LLM output is never trusted as observed data — those agents force
`evidence_level` to `inferred`/`hypothetical` and tag provenance accordingly.
When the backend is unavailable, deterministic agents fall back to their
baseline, the knowledge agents return `insufficient_evidence`, and patent
returns its non-gating "UNKNOWN" state.

---

## Reproducibility & traceability

Every artifact is a versioned, hashed JSON contract under the store, and every
value carries an **evidence level** — `observed` (a fact), `calculated`
(in-silico math), `predicted` (model/heuristic), `inferred`/`hypothetical`
(LLM recall) — plus provenance (source, identifier, timestamps, tool versions,
hashes). Input/output hashes fingerprint deterministic content (never run IDs or
timestamps), an append-only event log records each step, and two deterministic
runs produce identical fingerprints. Network-sourced labels (IUPAC names,
grounded records) are explicitly excluded from the hashes so they can't break
reproducibility.

---

## Installation

Requires Python ≥ 3.11 (developed on 3.14).

```bash
git clone <this repo> && cd ddswarm
python3 -m venv .venv
.venv/bin/pip install -e .            # core: pydantic + rdkit + reportlab + matplotlib
.venv/bin/pip install pytest          # to run the test suite
```

Optional extras:

- **Docking / MD** — `.venv/bin/pip install -e '.[docking]'` (meeko, pdbfixer,
  openbabel-wheel, scipy, gemmi; OpenMM for MD), plus the AutoDock Vina binary:

  ```bash
  curl -sL https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.7/vina_1.2.7_linux_x86_64 -o tools/vina
  chmod +x tools/vina
  ```

  (or point the agent at it via `params.vina_binary` / the `VINA_BINARY` env var).

## Quick start

```bash
# scaffold a project file
.venv/bin/ddswarm init p.json --project-id egfr --objective "inhibit EGFR" \
    --targets "EGFR" --smiles "COC1=..."

# run the default deterministic pipeline
.venv/bin/ddswarm run p.json

# fully offline (also skips the network IUPAC name lookup)
.venv/bin/ddswarm run p.json --no-iupac

# with the LLM reasoning layer (when running under Hermes)
.venv/bin/ddswarm run p.json --reasoning hermes
```

The run writes its report PDF to `reports/<project_id>.pdf`, the candidate
exports alongside it, and the full artifact ledger under `store/`. Docking and
target-selectivity (always-on) download receptor PDBs from RCSB at runtime, so
fully offline they degrade to a non-blocking "receptor unavailable" gap — the
run still completes and says so honestly.

## CLI

```bash
ddswarm init project.json --project-id P --objective "..." [--targets T] [--smiles S]
ddswarm run project.json [--reasoning none|hermes] [--no-iupac] [--grounding]
                         [--iterations N] [--resume] [--store DIR] [--reports-dir DIR]
ddswarm runs [project_id]                 # list runs
ddswarm report project_id [run_id]        # per-task report
ddswarm memory project_id                 # cross-run conclusions
ddswarm receptors list [--family GPCR]    # the ~61-target registry
ddswarm receptors lookup SERT             # resolve a name/alias
ddswarm receptors adr                     # the ADR-profile library
ddswarm receptors match --primary COX2 --off-targets COX1   # dry-run profile match
```

## Configuration (`project.json`)

`project.json` is a `RunRequest`: `{project: ProjectSpec, tasks: [...],
reasoning: {...}}`.

- `project` — the normalized spec: objective, targets, starting materials,
  `input_type`, success criteria, and optional context fields
  (`mechanism_of_action`, `indication`, `existing_drugs`, `chemical_scaffold`,
  `adme_issue`, `dose_mg`, `route_of_administration`).
- `tasks` — an optional task DAG; omitted (or `[]`) runs the default pipeline.
- `reasoning` — `{"backend": "none"|"hermes", "command", "max_turns",
  "run_budget"}`; optional.

## Store layout

```
store/<project_id>/runs/<run_id>/
  run_manifest.json
  tasks/<task_id>.json                 # TaskSpec ledger
  tasks/<task_id>.result.v<N>.json     # versioned AgentResult
  artifacts/<id>.<kind>.v<N>.json      # versioned artifacts (never overwritten)
  events.jsonl                         # append-only event log
schemas/*.schema.json                  # exported JSON Schema for every contract
```

## Use as an MCP server

The swarm is also an **MCP (Model Context Protocol) server**, so any agentic
stack that speaks MCP — Google ADK/AG2, Hermes, OpenMinion, Claude Desktop,
Cursor — can drive the deterministic drug-discovery tools directly, with the
calling LLM as the orchestrator.

```bash
.venv/bin/ddswarm-mcp          # starts the server on stdio
```

Tools (16):

| Tool | Description |
|---|---|
| `design_analogs` | Generate analog molecules from starting SMILES (deterministic RDKit transformations). |
| `screen_molecules` | Screen molecules (rule-of-five, PAINS, Brenk) with pass/reject verdicts. |
| `predict_admet` | Predict ADMET endpoints (rule-based provider). |
| `retrosynthesis` | Propose retrosynthetic disconnections (SA + BRICS). |
| `toxicity_alerts` / `cyp_alerts` / `herg_flag` | Flag toxicophore / CYP / hERG liabilities (SMARTS). |
| `formulation_assess` | Assess formulation/biopharmaceutics (BCS). |
| `safety_check` | Hard safety gate (controlled substances + dual-use targets). |
| `pkpd_simulate` | Deterministic PK/PD simulation (RK4, mechanistic PD). |
| `dock_molecule` | Dock against a receptor (registry name or explicit PDB). |
| `selectivity_profile` | Selectivity/ADR profile (docking → Ki → occupancy → SI). |
| `receptor_list` / `receptor_lookup` / `adr_profiles` | Registry + ADR-library data. |
| `run_pipeline` | Run the full 18-agent pipeline from a ProjectSpec dict and return the report. |

Docking and selectivity accept an `allow_download` flag. With
`allow_download=false`, a receptor that needs a PDB download returns
`insufficient_evidence` with a message telling the caller to pass a local PDB
file instead — unless the PDB was already downloaded once, in which case the
local cache is used (no network). Downloaded PDBs are cached under
`~/.cache/ddswarm/pdbs` (override with `$DDWARM_PDB_CACHE`), so a receptor is
only fetched from RCSB the first time it is used.

Connect from your agent stack — the server speaks stdio, so any MCP client
launches it as a subprocess. Use the absolute path to the venv binary (or just
`ddswarm-mcp` once the venv is activated / on `PATH`):

**Claude Desktop / Cursor / Gemini CLI / VS Code** (standard `mcpServers` JSON):

```json
{
  "mcpServers": {
    "ddswarm": {
      "command": "/home/faust/ddswarm/.venv/bin/ddswarm-mcp",
      "args": []
    }
  }
}
```

**Hermes Agent** (`~/.hermes/config.yaml`, tools register as `mcp_ddswarm_*`):

```yaml
mcp_servers:
  ddswarm:
    command: "/home/faust/ddswarm/.venv/bin/ddswarm-mcp"
    args: []
```

**Google ADK** (Python `MCPToolset`):

```python
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters

tools, exit_stack = await MCPToolset.from_server(
    connection_params=StdioServerParameters(
        command="/home/faust/ddswarm/.venv/bin/ddswarm-mcp",
        args=[],
    )
)
```

**Any other MCP client** (OpenMinion, OpenAI Agents SDK, …) connects the same
way: point its stdio transport at `ddswarm-mcp` and it discovers the 16 tools
via `tools/list`.

### HTTP transport (streamable HTTP)

The same 16 tools are also served over a remote HTTP endpoint — useful for a
hosted or shared instance:

```bash
.venv/bin/ddswarm-mcp --transport streamable-http --host 0.0.0.0 --port 8000
# serves at http://0.0.0.0:8000/mcp (default path)
```

Options: `--host`, `--port` (default 8000), `--path` (default `/mcp`), and
`--stateless-http` (no session tracking — each request is self-contained, which
suits ddswarm's stateless tools). The legacy SSE transport is available with
`--transport sse`. A URL-based client (e.g. Claude Desktop over HTTP, or the
`mcpServers` `url` form) just points at the endpoint:

```json
{
  "mcpServers": {
    "ddswarm": {
      "url": "http://127.0.0.1:8000/mcp"
    }
  }
}
```

Every tool returns a JSON dict `{status, findings, warnings, errors,
artifacts}`; `artifacts` embeds the full contract payload (e.g. `design_batch`,
`screened_batch`), so the client never has to read the on-disk store. Docking
and target-selectivity resolve their receptor from the project targets and
degrade non-blocking (`insufficient_evidence`) when a PDB is unavailable.

---

## Verification

Run `pytest -q` — **351 tests** pass, covering every agent offline with a fake
`hermes` script: screening, the rule-based ADMET provider (exact ESOL/QED/Ro5/
Veber/Egan values plus out-of-domain and determinism), docking gating and RMSD,
selectivity/ADR classification, PK/PD ODEs, retrosynthesis, the toxicophore /
CYP / hERG matchers, the safety gate, the validation audit (including tamper
detection), the knowledge agents, project memory, report aggregation, and the
CLI. Real-Vina and real-OpenMM tests auto-skip when the binaries/engines are
absent; an end-to-end reproducibility test locks in identical artifact
fingerprints across two deterministic runs.

---

## License

[MIT](LICENSE) — © 2026 faust-dark.
