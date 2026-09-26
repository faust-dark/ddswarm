"""memory_minion: distils one generation into a GenerationMemory — the swarm's
structured recollection of what worked (best candidates), what to avoid, and
what it learned — and writes it as both an artifact and a durable 'generation'
entry in project memory for the next generation's orchestrator/designer.

Deterministic: it reads the report (ranked candidates) and screened batch
(survivors + rejected) and aggregates them. No LLM required.
"""

from __future__ import annotations

from ..contracts.agent_result import AgentResult
from ..contracts.enums import AgentResultStatus, EvidenceLevel
from ..contracts.generation_memory import GenerationCandidate, GenerationMemory
from ..contracts.memory import MemoryEntry
from ..contracts.provenance import ProvenanceEntry, ToolVersion
from ..contracts.report import Report
from ..contracts.screened_batch import ScreenedBatch
from ..logging.hashing import hash_jsonable
from ..orchestrator.memory import MemoryStore
from .base import Agent, AgentContext

TOOL = ToolVersion(name="ddswarm", version="0.1.0")


class MemoryMinion(Agent):
    name = "memory_minion"

    def run(self, ctx: AgentContext) -> AgentResult:
        generation = int(ctx.params.get("generation", 0))
        report = (ctx.inputs.get("report") or [None])[0]
        sbs = ctx.inputs.get("screened_batch", [])

        best: list[GenerationCandidate] = []
        avoided: list[str] = []
        insights: list[str] = []

        if isinstance(report, Report):
            for c in report.ranked_candidates:
                best.append(GenerationCandidate(
                    smiles=c.smiles, score=c.score, reasons=list(c.criteria),
                ))
            insights.extend(report.gaps)
        for sb in sbs:
            if isinstance(sb, ScreenedBatch):
                for m in sb.molecules:
                    if not m.passed and m.canonical_smiles and m.canonical_smiles not in avoided:
                        avoided.append(m.canonical_smiles)

        summary = (f"generation {generation}: {len(best)} ranked candidate(s), "
                   f"{len(avoided)} rejected scaffold(s) recorded as avoid")

        input_hash = hash_jsonable({
            "report": report.model_dump(mode="json", exclude={"provenance", "run_id"})
            if isinstance(report, Report) else None,
            "generation": generation,
        })
        output_hash = hash_jsonable({
            "best": [b.model_dump() for b in best], "avoided": avoided, "insights": insights,
        })

        mem = GenerationMemory(
            project_id=ctx.project_id, run_id=ctx.run_id, generation=generation,
            summary=summary, best_candidates=best, avoided=avoided, insights=insights,
            converged=not best,
            provenance=[ProvenanceEntry(
                source="generation aggregation (deterministic)",
                tool_versions=[TOOL], input_hash=input_hash, output_hash=output_hash,
                params={"generation": generation},
            )],
        )
        art = self.persist_artifact(ctx, "generation_memory", f"genmem_{ctx.project_id}", mem)

        # Durable cross-run recollection for the next generation.
        MemoryStore(self.store).append(MemoryEntry(
            id=f"generation_{ctx.run_id}", project_id=ctx.project_id, kind="generation",
            content=summary,
            data={"best_candidates": [b.model_dump() for b in best], "avoided": avoided,
                  "generation": generation, "run_id": ctx.run_id},
            evidence_level=EvidenceLevel.inferred,
            source_agent=self.name, run_id=ctx.run_id,
            provenance=[ProvenanceEntry(source="generation aggregation", tool_versions=[TOOL],
                                        input_hash=input_hash, output_hash=output_hash)],
        ))

        return AgentResult(
            project_id=ctx.project_id, agent=self.name, task_id=ctx.task_id,
            status=AgentResultStatus.success, artifacts=[art], input_refs=ctx.input_refs,
            findings=[summary],
            provenance=[ProvenanceEntry(source="generation aggregation", output_hash=art.hash,
                                        tool_versions=[TOOL])],
            tool_versions=[TOOL],
            reproducibility={"generation": generation, "best": len(best), "avoided": len(avoided)},
        )
