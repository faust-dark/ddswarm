"""patent_minion: patent-landscape / freedom-to-operate knowledge for the
candidate molecules and/or the project target, delegated to a reasoning backend
(Hermes subagent).

There is NO deterministic offline source for this knowledge — without
`reasoning.backend=hermes` the agent returns insufficient_evidence (it never
fabricates patent data). Every entry is LLM recall: evidence_level is forced to
'inferred' and tagged 'verify against primary source'. This is a research aid,
NOT a legal freedom-to-operate opinion.
"""

from __future__ import annotations

from ..contracts.agent_result import AgentResult
from ..contracts.enums import AgentResultStatus
from ..contracts.patent_landscape import PatentEntry, PatentLandscape, PatentSummary
from ..contracts.provenance import ProvenanceEntry, ToolVersion
from ..logging.hashing import hash_jsonable
from ..reasoning import make_reasoning_backend
from .base import Agent, AgentContext
from ddswarm import __version__

TOOL = ToolVersion(name="ddswarm", version=__version__)


class PatentMinion(Agent):
    name = "patent_minion"

    def run(self, ctx: AgentContext) -> AgentResult:
        molecules = self._collect_molecules(ctx)
        warnings: list[str] = []
        input_hash = hash_jsonable({"targets": ctx.spec.targets, "molecules": list(molecules)})
        backend_name = ctx.params.get("reasoning_backend", "none")

        if backend_name in (None, "", "none"):
            return self._not_assessed(
                ctx, input_hash, backend_name,
                "patent/FTO landscape not performed: requires reasoning.backend=hermes")

        backend, note = self._resolve_backend(ctx, warnings)
        if backend is None:
            return self._not_assessed(ctx, input_hash, backend_name, note)

        res = backend.patent_landscape(ctx.spec, molecules, ctx.params)
        if not res.ok:
            return self._not_assessed(
                ctx, input_hash, backend_name, f"patent/FTO reasoning failed: {res.note}")

        entries: list[PatentEntry] = []
        for i, raw in enumerate(res.entries):
            try:
                entries.append(self._from_reasoning(raw, i))
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"patent entry {i} invalid, skipped: {exc}")

        if not entries:
            return self._not_assessed(
                ctx, input_hash, backend_name,
                "patent/FTO landscape empty: reasoning returned no valid patent entries")

        relevance_counts: dict[str, int] = {}
        for e in entries:
            relevance_counts[e.relevance] = relevance_counts.get(e.relevance, 0) + 1
        summary = PatentSummary(total_entries=len(entries),
                                relevance_counts=relevance_counts,
                                flagged=relevance_counts.get("covers_candidate", 0))

        payload = PatentLandscape(
            project_id=ctx.project_id, status=AgentResultStatus.success,
            entries=entries, summary=summary,
            provenance=[ProvenanceEntry(
                source="Hermes subagent (LLM reasoning)", tool_versions=[TOOL],
                input_hash=input_hash,
                output_hash=hash_jsonable([e.model_dump() for e in entries]),
                params={"reasoning_backend": backend_name, "molecules": len(molecules)},
            )],
        )
        art = self.persist_artifact(ctx, "patent_landscape", f"pl_{ctx.project_id}", payload)
        finding = (f"patent landscape: {summary.total_entries} entries; "
                   f"{summary.flagged} covering candidate(s) (all inferred)")
        return AgentResult(
            project_id=ctx.project_id, agent=self.name, task_id=ctx.task_id,
            status=AgentResultStatus.success, artifacts=[art], input_refs=ctx.input_refs,
            findings=[finding], warnings=warnings,
            provenance=[ProvenanceEntry(source="patent landscape", output_hash=art.hash, tool_versions=[TOOL])],
            tool_versions=[TOOL], reproducibility={"input_hash": input_hash},
        )

    def _not_assessed(self, ctx: AgentContext, input_hash: str, backend_name: str | None,
                      reason: str) -> AgentResult:
        """Non-gating baseline so patent_minion can sit in DEFAULT_PIPELINE.

        Without a reasoning backend (or when it fails) the agent returns SUCCESS
        with an empty PatentLandscape and an explicit 'patent status UNKNOWN'
        note — never a fabricated clear, and never insufficient_evidence (which
        would gate report/validation in the default graph)."""
        note = (f"{reason}; patent status UNKNOWN — this run does NOT establish "
                "novelty or freedom-to-operate")
        payload = PatentLandscape(
            project_id=ctx.project_id, status=AgentResultStatus.success,
            entries=[], summary=PatentSummary(total_entries=0, relevance_counts={}, flagged=0),
            note=note,
            provenance=[ProvenanceEntry(
                source="patent_minion (no reasoning backend or backend failure)",
                tool_versions=[TOOL], input_hash=input_hash,
                params={"reasoning_backend": backend_name},
            )],
        )
        art = self.persist_artifact(ctx, "patent_landscape", f"pl_{ctx.project_id}", payload)
        return AgentResult(
            project_id=ctx.project_id, agent=self.name, task_id=ctx.task_id,
            status=AgentResultStatus.success, artifacts=[art], input_refs=ctx.input_refs,
            findings=[f"patent/FTO landscape: not assessed ({reason})"],
            warnings=[note],
            provenance=[ProvenanceEntry(source="patent landscape", output_hash=art.hash,
                                        tool_versions=[TOOL])],
            tool_versions=[TOOL], reproducibility={"input_hash": input_hash},
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

    @staticmethod
    def _resolve_backend(ctx: AgentContext, warnings: list[str]):
        name = ctx.params.get("reasoning_backend", "none")
        try:
            backend = make_reasoning_backend(
                name,
                command=ctx.params.get("reasoning_command"),
                max_turns=ctx.params.get("reasoning_max_turns", 30),
                run_budget=ctx.params.get("reasoning_run_budget", 600),
            )
        except KeyError as exc:
            return None, str(exc)
        avail, msg = backend.available()
        if not avail:
            return None, f"reasoning backend '{name}' unavailable: {msg}"
        return backend, ""

    @staticmethod
    def _from_reasoning(raw: dict, i: int) -> PatentEntry:
        data = dict(raw)
        data.setdefault("id", f"patent_{i}")
        data["evidence_level"] = "inferred"  # LLM recall is not observed data
        data["source"] = data.get("source") or "Hermes subagent (LLM reasoning)"
        notes = data.get("notes") or ""
        if "verify against primary source" not in notes:
            data["notes"] = (notes + " | LLM-recalled via reasoning backend; verify against primary source").strip(" |")
        return PatentEntry.model_validate(data)
