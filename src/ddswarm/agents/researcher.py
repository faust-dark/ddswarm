"""researcher_minion: assembles a KnowledgePack from ProjectSpec + seed data.

Minimal in Increment 0: no literature/DB client yet. It canonicalizes whatever
inputs are present and REFUSES to fabricate entries when there is nothing to
ingest (returns insufficient_evidence)."""

from __future__ import annotations

from ..contracts.agent_result import AgentResult
from ..contracts.enums import AgentResultStatus, EvidenceLevel
from ..contracts.knowledge_pack import KnowledgePack, KnowledgePackEntry
from ..contracts.provenance import ProvenanceEntry, ToolVersion
from ..logging.hashing import hash_jsonable
from .base import Agent, AgentContext

TOOL = ToolVersion(name="ddswarm", version="0.1.0")


class ResearcherMinion(Agent):
    name = "researcher_minion"

    def run(self, ctx: AgentContext) -> AgentResult:
        entries: list[KnowledgePackEntry] = []
        warnings: list[str] = []

        seed = ctx.params.get("seed_entries", [])
        if not isinstance(seed, list):
            warnings.append("seed_entries must be a list; ignored")
            seed = []

        input_digest = {
            "targets": ctx.spec.targets,
            "starting_materials": ctx.spec.starting_materials,
            "seed_entries": seed,
        }
        input_hash = hash_jsonable(input_digest)

        for i, t in enumerate(ctx.spec.targets):
            entries.append(KnowledgePackEntry(
                id=f"target_{i}",
                type="target",
                source="ProjectSpec",
                observed_value=t,
                evidence_level=EvidenceLevel.inferred,
                notes="target premise from ProjectSpec input (user-provided)",
            ))

        for i, smi in enumerate(ctx.spec.starting_materials):
            entries.append(KnowledgePackEntry(
                id=f"ligand_{i}",
                type="ligand",
                source="ProjectSpec",
                observed_value=smi,
                evidence_level=EvidenceLevel.inferred,
                notes="starting-material SMILES from ProjectSpec input; not yet canonicalized (RDKit in Increment 1)",
            ))

        for i, se in enumerate(seed):
            try:
                entries.append(KnowledgePackEntry.model_validate(se))
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"seed entry {i} invalid, skipped: {exc}")

        if not entries:
            return AgentResult(
                project_id=ctx.project_id,
                agent=self.name,
                task_id=ctx.task_id,
                status=AgentResultStatus.insufficient_evidence,
                warnings=["no targets, starting_materials, or valid seed_entries; refusing to fabricate a KnowledgePack"],
                provenance=[ProvenanceEntry(source="ProjectSpec", input_hash=input_hash, params={"seed_entries_count": len(seed)})],
                tool_versions=[TOOL],
                reproducibility={"reason": "no_input_data"},
            )

        kp = KnowledgePack(
            project_id=ctx.project_id,
            status=AgentResultStatus.success,
            entries=entries,
            provenance=[ProvenanceEntry(
                source="ProjectSpec + seed",
                input_hash=input_hash,
                output_hash=hash_jsonable([e.model_dump() for e in entries]),
                params={"seed_entries_count": len(seed)},
                tool_versions=[TOOL],
            )],
        )
        art = self.persist_artifact(ctx, "knowledge_pack", f"kp_{ctx.project_id}", kp)
        return AgentResult(
            project_id=ctx.project_id,
            agent=self.name,
            task_id=ctx.task_id,
            status=AgentResultStatus.success,
            artifacts=[art],
            findings=[f"KnowledgePack assembled with {len(entries)} entries"],
            warnings=warnings,
            provenance=[ProvenanceEntry(source="ProjectSpec + seed", input_hash=input_hash, output_hash=art.hash, tool_versions=[TOOL])],
            tool_versions=[TOOL],
            reproducibility={"input_hash": input_hash},
        )
