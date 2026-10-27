"""safety_minion: deterministic gating for controlled substances, dual-use,
biosafety, and compliance. Matches a curated (small, non-exhaustive) list of
controlled substances by exact canonical SMILES and dual-use targets by name.

A `block` verdict returns AgentResultStatus.blocked, which the orchestrator
propagates to every downstream task. A flag is an *observed* match against a
list — never a legal determination, and the list is not a database."""

from __future__ import annotations

from rdkit import rdBase

from ddswarm import __version__

from ..chemistry.rdkit_ops import sanitize_and_canonicalize
from ..chemistry.safety import match_controlled_substance, match_dual_use_target
from ..contracts.agent_result import AgentResult
from ..contracts.enums import AgentResultStatus
from ..contracts.provenance import ProvenanceEntry, ToolVersion
from ..contracts.safety_assessment import SafetyAssessment, SafetyFlag, SafetySummary
from ..logging.hashing import hash_jsonable
from .base import Agent, AgentContext

TOOL = ToolVersion(name="ddswarm", version=__version__)
RDKIT_TOOL = ToolVersion(name="rdkit", version=rdBase.rdkitVersion)


class SafetyMinion(Agent):
    name = "safety_minion"

    def run(self, ctx: AgentContext) -> AgentResult:
        flags: list[SafetyFlag] = []
        n = 0

        # dual-use / biosafety targets
        for target in ctx.spec.targets:
            for e in match_dual_use_target(target):
                n += 1
                flags.append(SafetyFlag(
                    id=f"du_{n}", kind="dual_use", level="block", subject=target,
                    reason=f"target matches '{e['name']}' ({e['category']})", source=e["source"],
                ))

        # controlled substances across input + designed molecules
        for mid, smi in self._collect_molecules(ctx):
            s = sanitize_and_canonicalize(smi)
            if not s.ok or not s.canonical_smiles:
                continue
            for e in match_controlled_substance(s.canonical_smiles):
                n += 1
                flags.append(SafetyFlag(
                    id=f"cs_{n}", kind="controlled_substance", level="block", subject=mid,
                    reason=f"'{e['name']}' ({e['schedule']})", source=e["source"],
                ))

        n_block = sum(1 for f in flags if f.level == "block")
        n_flag = sum(1 for f in flags if f.level == "flag")
        verdict = "block" if n_block else ("flag" if n_flag else "pass")

        input_hash = hash_jsonable({"targets": ctx.spec.targets,
                                    "molecules": [(m, s) for m, s in self._collect_molecules(ctx)]})
        output_hash = hash_jsonable([f.model_dump() for f in flags])

        note = ("curated non-exhaustive list; exact canonical-SMILES / name match only — "
                "not a legal determination")
        assessment = SafetyAssessment(
            project_id=ctx.project_id,
            status=AgentResultStatus.blocked if verdict == "block" else AgentResultStatus.success,
            verdict=verdict,
            flags=flags,
            summary=SafetySummary(total_flags=len(flags), block=n_block, flag=n_flag),
            note=note,
            provenance=[ProvenanceEntry(
                source="curated safety gate (controlled substances + dual-use targets)",
                tool_versions=[TOOL, RDKIT_TOOL],
                params={"targets": ctx.spec.targets},
                input_hash=input_hash, output_hash=output_hash,
            )],
        )
        art = self.persist_artifact(ctx, "safety_assessment", f"safety_{ctx.project_id}", assessment)

        warnings: list[str] = [note] if flags else []
        findings = [f"safety gate: {verdict} ({len(flags)} flag(s))"]
        if verdict == "block":
            findings.append("; ".join(f.reason for f in flags))

        return AgentResult(
            project_id=ctx.project_id, agent=self.name, task_id=ctx.task_id,
            status=AgentResultStatus.blocked if verdict == "block" else AgentResultStatus.success,
            artifacts=[art], input_refs=ctx.input_refs, findings=findings, warnings=warnings,
            provenance=[ProvenanceEntry(source="safety gate", output_hash=art.hash,
                                        tool_versions=[TOOL, RDKIT_TOOL])],
            tool_versions=[TOOL, RDKIT_TOOL],
            reproducibility={"input_hash": input_hash, "verdict": verdict},
        )

    def _collect_molecules(self, ctx: AgentContext) -> list[tuple[str, str]]:
        mols: list[tuple[str, str]] = []
        explicit = ctx.params.get("molecules", []) or []
        for i, m in enumerate(explicit):
            if isinstance(m, dict) and "smiles" in m:
                mols.append((m.get("id", f"param_{i}"), m["smiles"]))
        if mols:
            return mols
        for sb in ctx.inputs.get("screened_batch", []):
            for m in getattr(sb, "molecules", []):
                if m.passed and m.canonical_smiles:
                    mols.append((m.id, m.canonical_smiles))
        if mols:
            return mols
        for i, smi in enumerate(ctx.spec.starting_materials):
            mols.append((f"starting_material_{i}", smi))
        return mols
