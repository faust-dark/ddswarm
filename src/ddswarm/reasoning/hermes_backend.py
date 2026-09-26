"""Hermes-subprocess reasoning backend: delegate research/design to a one-shot
Hermes subagent (`hermes chat --query-file ... -Q`) and parse its JSON.

LLM output is reasoning, not verified data — the agent overrides evidence_level
(inferred/hypothetical) and tags provenance accordingly, so a recalled PubChem
ID is never reported as 'observed'."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from ..contracts.project_spec import ProjectSpec
from .base import ReasoningBackend, ReasoningResult

_RESEARCH_PROMPT = """You are a drug-discovery research agent inside a reproducible pipeline.

Project objective: {objective}
Target(s): {targets}
Starting material SMILES: {smiles}

Research the target(s) and starting material(s): mechanism of action, known
references/ligands (with source + URL + identifier), and relevant ADMET or
clinical data where available. Do NOT fabricate: if you are unsure of a value,
use null and say so in notes.

Respond with ONLY a single JSON array of objects — no markdown, no prose, no
code fences. Each object must have exactly these keys:
  "type": one of "target" | "ligand" | "reference" | "admet" | "mechanism",
  "source": source name (e.g. "PubChem", "ChEMBL", "DrugBank"),
  "url": source URL or null,
  "identifier": source identifier (e.g. "CID 123631") or null,
  "observed_value": the factual value or statement,
  "notes": any caveats or uncertainty.
"""

_DESIGN_PROMPT = """You are a medicinal chemist proposing drug-candidate analogs.

Project objective: {objective}
Target(s): {targets}
Parent scaffold(s) (id: SMILES):
{scaffolds}

Propose 3-6 analog molecules derived from the parent scaffold(s) via
bioisosteric replacement, scaffold hopping, or analog design. Each analog must
be a valid, neutral SMILES string.

Respond with ONLY a single JSON array of objects — no markdown, no prose, no
code fences, no comments. Each object must have exactly these keys:
  "smiles": the SMILES string (one token, no spaces, no quotes),
  "parent_scaffold": the parent id it derives from,
  "transformation": the transformation name,
  "rationale": one-sentence medicinal-chemistry rationale.

The array must be complete and closed with a "]" — never truncate mid-array.
"""

_SYNTHESIS_PROMPT = """You are a synthetic chemist proposing synthesis routes.

Project objective: {objective}
Molecule(s) to plan (id: SMILES):
{molecules}

For each molecule propose 1-2 viable synthesis routes. Each route is an ordered
list of synthetic steps plus the key reagents. Be honest about feasibility; if
a step is speculative, say so in the feasibility note.

Respond with ONLY a single JSON array of objects — no markdown, no prose, no
code fences. Each object must have exactly these keys:
  "molecule_id": the id from the list above,
  "steps": [ordered synthetic/retrosynthetic steps as strings],
  "key_reagents": [reagent names],
  "feasibility_note": one-sentence feasibility assessment.
"""

_PATENT_PROMPT = """You are a patent analyst inside a reproducible drug-discovery pipeline.

Project objective: {objective}
Target(s): {targets}
Candidate molecule(s) (id: SMILES):
{molecules}

Identify patents and published applications relevant to the candidate
molecule(s) and/or the target: composition-of-matter, method-of-use, and
scaffold patents. Classify each entry's relevance. Do NOT fabricate patent
numbers: if unsure of an identifier, use null and say so in notes. This is a
research aid, NOT a legal freedom-to-operate opinion.

Respond with ONLY a single JSON array of objects — no markdown, no prose, no
code fences. Each object must have exactly these keys:
  "molecule_id": the molecule id it concerns, or null,
  "patent_id": the publication/patent number (e.g. "US1234567B2"),
  "title": short title or null,
  "assignee": assignee name or null,
  "status": "granted" | "pending" | "expired" | "unknown",
  "jurisdiction": e.g. "US" | "EP" | "WO" or null,
  "relevance": "covers_candidate" | "related_scaffold" | "background",
  "source": source name (e.g. "Google Patents", "USPTO", "Espacenet"),
  "url": source URL or null,
  "notes": any caveats or uncertainty.
"""

_CLINICAL_PROMPT = """You are a clinical-development analyst inside a reproducible drug-discovery pipeline.

Project objective: {objective}
Target(s): {targets}
Candidate molecule(s) (id: SMILES):
{molecules}

Summarize clinical-stage knowledge for the target(s) and candidate(s): known
drugs in development or approved for the target, their trial phase, indication,
and status. Do NOT fabricate trial details: if unsure, use null and say so.

Respond with ONLY a single JSON array of objects — no markdown, no prose, no
code fences. Each object must have exactly these keys:
  "molecule_id": molecule id it concerns, or null,
  "entity": the drug/candidate/target name,
  "phase": "preclinical" | "phase_1" | "phase_2" | "phase_3" | "approved" | "unknown",
  "indication": the indication or null,
  "status": e.g. "ongoing" | "completed" | "terminated" or null,
  "source": source name (e.g. "ClinicalTrials.gov", "DrugBank", "FDA"),
  "url": source URL or null,
  "notes": caveats or uncertainty.
"""

_REGULATORY_PROMPT = """You are a regulatory-affairs analyst inside a reproducible drug-discovery pipeline.

Project objective: {objective}
Target(s): {targets}
Candidate molecule(s) (id: SMILES):
{molecules}

Summarize regulatory knowledge: agency designations and approval status
relevant to the target(s) and candidate(s) (e.g. FDA/EMA approval, orphan drug,
fast track, breakthrough therapy, scheduling/controlled-substance status). Do
NOT fabricate: if unsure, use null and say so.

Respond with ONLY a single JSON array of objects — no markdown, no prose, no
code fences. Each object must have exactly these keys:
  "molecule_id": molecule id it concerns, or null,
  "agency": the agency (e.g. "FDA", "EMA", "WHO", "DEA"),
  "designation": e.g. "approved" | "orphan drug" | "fast track" | "schedule II" or null,
  "status": current status or null,
  "source": source name (e.g. "FDA Orange Book", "EMA", "DEA"),
  "url": source URL or null,
  "notes": caveats or uncertainty.
"""


def extract_json_list(text: str) -> list | None:
    """Parse a JSON array from a subagent's text output, tolerating surrounding
    prose, markdown fences, a trailing truncated tail, and multiple arrays.

    Returns the first complete, non-empty JSON array (or a lone object wrapped
    as a one-element list). None when nothing parses."""
    text = (text or "").strip()
    if not text:
        return None
    # strip a surrounding markdown code fence (```json ... ```)
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    # fast path: the whole text is one JSON value
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, list) else [obj]
    except json.JSONDecodeError:
        pass
    # robust path: scan for the first complete JSON value at a '[' or '{', so
    # leading prose, a trailing truncated tail, and multiple arrays are all
    # tolerated. Prefer a non-empty array.
    decoder = json.JSONDecoder()
    empty_list = None
    for idx in range(len(text)):
        if text[idx] not in "[{":
            continue
        try:
            val, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(val, list):
            if val:
                return val
            if empty_list is None:
                empty_list = val
        elif isinstance(val, dict):
            return [val]
    return empty_list


class HermesSubprocessBackend(ReasoningBackend):
    name = "hermes"

    def __init__(self, command: str = "hermes", max_turns: int = 30, run_budget: int = 300):
        self.command = command
        self.max_turns = max_turns
        self.run_budget = run_budget

    def available(self) -> tuple[bool, str]:
        if "/" in self.command or Path(self.command).is_absolute():
            if not Path(self.command).exists():
                return False, f"reasoning binary not found: {self.command}"
        elif not shutil.which(self.command):
            return False, f"reasoning binary not found on PATH: {self.command}"
        try:
            r = subprocess.run([self.command, "--version"], capture_output=True, text=True, timeout=30)
            line = (r.stdout or "").strip().splitlines()[0] if (r.stdout or "").strip() else ""
            return True, line or "unknown"
        except (OSError, subprocess.SubprocessError):
            return True, "unknown"

    # ---- research ----
    def research(self, spec: ProjectSpec, params: dict) -> ReasoningResult:
        prompt = _RESEARCH_PROMPT.format(
            objective=spec.objective,
            targets="; ".join(spec.targets) or "(none)",
            smiles="; ".join(spec.starting_materials) or "(none)",
        )
        return self._run(prompt)

    # ---- design ----
    def design(self, spec: ProjectSpec, scaffolds: list[tuple[str, str]], params: dict) -> ReasoningResult:
        scaffold_lines = "\n".join(f"- {pid}: {smi}" for pid, smi in scaffolds) or "(none)"
        prompt = _DESIGN_PROMPT.format(
            objective=spec.objective,
            targets="; ".join(spec.targets) or "(none)",
            scaffolds=scaffold_lines,
        )
        return self._run(prompt)

    # ---- synthesis routes ----
    def synthesis_routes(self, spec: ProjectSpec, molecules: list[tuple[str, str]], params: dict) -> ReasoningResult:
        mol_lines = "\n".join(f"- {mid}: {smi}" for mid, smi in molecules) or "(none)"
        prompt = _SYNTHESIS_PROMPT.format(objective=spec.objective, molecules=mol_lines)
        return self._run(prompt)

    # ---- patent landscape ----
    def patent_landscape(self, spec: ProjectSpec, molecules: list[tuple[str, str]], params: dict) -> ReasoningResult:
        mol_lines = "\n".join(f"- {mid}: {smi}" for mid, smi in molecules) or "(none)"
        prompt = _PATENT_PROMPT.format(
            objective=spec.objective,
            targets="; ".join(spec.targets) or "(none)",
            molecules=mol_lines,
        )
        return self._run(prompt)

    # ---- clinical assessment ----
    def clinical_assessment(self, spec: ProjectSpec, molecules: list[tuple[str, str]], params: dict) -> ReasoningResult:
        mol_lines = "\n".join(f"- {mid}: {smi}" for mid, smi in molecules) or "(none)"
        prompt = _CLINICAL_PROMPT.format(
            objective=spec.objective,
            targets="; ".join(spec.targets) or "(none)",
            molecules=mol_lines,
        )
        return self._run(prompt)

    # ---- regulatory assessment ----
    def regulatory_assessment(self, spec: ProjectSpec, molecules: list[tuple[str, str]], params: dict) -> ReasoningResult:
        mol_lines = "\n".join(f"- {mid}: {smi}" for mid, smi in molecules) or "(none)"
        prompt = _REGULATORY_PROMPT.format(
            objective=spec.objective,
            targets="; ".join(spec.targets) or "(none)",
            molecules=mol_lines,
        )
        return self._run(prompt)

    # ---- shared ----
    def _run(self, prompt: str) -> ReasoningResult:
        # LLM output is stochastic: retry once if the first response is not
        # parseable JSON before surfacing the failure to the agent.
        for attempt in range(2):
            r, stderr = self._invoke(prompt)
            if r is None:
                return ReasoningResult(ok=False, note=stderr)
            if r.returncode != 0:
                return ReasoningResult(ok=False, note=f"reasoning subprocess exited {r.returncode}: {stderr.strip()[:300]}")
            parsed = extract_json_list(r.stdout)
            if parsed is not None:
                return ReasoningResult(entries=parsed, ok=True, note=f"hermes subagent ({self.command})")
        return ReasoningResult(ok=False, note="could not parse JSON from subagent response (after 1 retry)")

    def _invoke(self, prompt: str) -> tuple[subprocess.CompletedProcess | None, str]:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
            fh.write(prompt)
            prompt_file = fh.name
        try:
            cmd = [self.command, "chat", "--query-file", prompt_file, "-Q",
                   "--max-turns", str(self.max_turns), "--run-budget", str(self.run_budget)]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=self.run_budget + 60)
            return r, r.stderr or ""
        except (OSError, subprocess.SubprocessError, subprocess.TimeoutExpired) as exc:
            return None, f"reasoning subprocess failed: {exc}"
        finally:
            Path(prompt_file).unlink(missing_ok=True)
