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

Propose 4-8 analog molecules derived from the parent scaffold(s) via
bioisosteric replacement, scaffold hopping, or analog design. Each analog must
be a valid, neutral SMILES string.

Respond with ONLY a single JSON array of objects — no markdown, no prose, no
code fences. Each object must have exactly these keys:
  "smiles": the SMILES string,
  "parent_scaffold": the parent id it derives from,
  "transformation": the transformation name,
  "rationale": one-sentence medicinal-chemistry rationale.
"""


def extract_json_list(text: str) -> list | None:
    """Parse a JSON array from a subagent's text output, tolerating surrounding
    prose and markdown fences."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, list) else [obj]
    except json.JSONDecodeError:
        pass
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            return obj if isinstance(obj, list) else [obj]
        except json.JSONDecodeError:
            return None
    return None


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

    # ---- shared ----
    def _run(self, prompt: str) -> ReasoningResult:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
            fh.write(prompt)
            prompt_file = fh.name
        try:
            cmd = [self.command, "chat", "--query-file", prompt_file, "-Q",
                   "--max-turns", str(self.max_turns), "--run-budget", str(self.run_budget)]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=self.run_budget + 60)
        except (OSError, subprocess.SubprocessError, subprocess.TimeoutExpired) as exc:
            return ReasoningResult(ok=False, note=f"reasoning subprocess failed: {exc}")
        finally:
            Path(prompt_file).unlink(missing_ok=True)

        if r.returncode != 0:
            return ReasoningResult(ok=False, note=f"reasoning subprocess exited {r.returncode}: {r.stderr.strip()[:300]}")
        parsed = extract_json_list(r.stdout)
        if parsed is None:
            return ReasoningResult(ok=False, note="could not parse JSON from subagent response")
        return ReasoningResult(entries=parsed, ok=True, note=f"hermes subagent ({self.command})")
