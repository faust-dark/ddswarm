"""researcher_minion: assembles a KnowledgePack from ProjectSpec + seed data,
optionally delegating the actual literature research to a reasoning backend
(e.g. a Hermes subagent).

Deterministic baseline (targets/ligands/seed) is always built. A reasoning
backend *adds* research entries; its output is LLM reasoning, so every entry it
contributes is forced to evidence_level='inferred' and tagged 'verify against
primary source'. No backend, or no input at all => insufficient_evidence
(never fabricate)."""

from __future__ import annotations

from ..contracts.agent_result import AgentResult
from ..contracts.enums import AgentResultStatus, EvidenceLevel
from ..contracts.knowledge_pack import KnowledgePack, KnowledgePackEntry
from ..contracts.provenance import ProvenanceEntry, ToolVersion
from ..logging.hashing import hash_jsonable
from ..orchestrator.memory import MemoryStore
from ..reasoning import make_reasoning_backend
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

        # ---- feedback loop: remember prior-run conclusions ----
        memory_findings = MemoryStore(self.store).by_kind(ctx.project_id, "finding")

        input_digest = {
            "targets": ctx.spec.targets,
            "starting_materials": ctx.spec.starting_materials,
            "seed_entries": seed,
            "memory_findings": [e.content for e in memory_findings],
        }
        input_hash = hash_jsonable(input_digest)

        # ---- deterministic baseline ----
        for i, t in enumerate(ctx.spec.targets):
            entries.append(KnowledgePackEntry(
                id=f"target_{i}", type="target", source="ProjectSpec",
                observed_value=t, evidence_level=EvidenceLevel.inferred,
                notes="target premise from ProjectSpec input (user-provided)",
            ))
        for i, smi in enumerate(ctx.spec.starting_materials):
            entries.append(KnowledgePackEntry(
                id=f"ligand_{i}", type="ligand", source="ProjectSpec",
                observed_value=smi, evidence_level=EvidenceLevel.inferred,
                notes="starting-material SMILES from ProjectSpec input",
            ))
        for i, se in enumerate(seed):
            try:
                entries.append(KnowledgePackEntry.model_validate(se))
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"seed entry {i} invalid, skipped: {exc}")

        for mf in memory_findings:
            entries.append(KnowledgePackEntry(
                id=f"memory_{mf.id}", type="reference", source="project memory",
                observed_value=mf.content, evidence_level=EvidenceLevel.inferred,
                notes=f"prior-run conclusion (run {mf.run_id}); swarm inference, verify against primary sources",
            ))

        # ---- reasoning backend (adds research) ----
        backend_name = ctx.params.get("reasoning_backend", "none")
        if backend_name not in (None, "", "none"):
            backend, note = self._resolve_backend(ctx, warnings)
            if backend is not None:
                res = backend.research(ctx.spec, ctx.params)
                if res.ok:
                    for i, raw in enumerate(res.entries):
                        try:
                            entries.append(self._from_reasoning(raw, i))
                        except Exception as exc:  # noqa: BLE001
                            warnings.append(f"reasoning entry {i} invalid, skipped: {exc}")
                else:
                    warnings.append(f"reasoning failed: {res.note}; kept deterministic baseline only")
            else:
                warnings.append(note)

        if not entries:
            return AgentResult(
                project_id=ctx.project_id, agent=self.name, task_id=ctx.task_id,
                status=AgentResultStatus.insufficient_evidence,
                warnings=["no targets, starting_materials, valid seed_entries, or reasoning output; refusing to fabricate a KnowledgePack"],
                provenance=[ProvenanceEntry(source="ProjectSpec", input_hash=input_hash, params={"seed_entries_count": len(seed)})],
                tool_versions=[TOOL], reproducibility={"reason": "no_input_data"},
            )

        kp = KnowledgePack(
            project_id=ctx.project_id,
            status=AgentResultStatus.success,
            entries=entries,
            provenance=[ProvenanceEntry(
                source="ProjectSpec + seed" + (" + reasoning backend" if backend_name not in (None, "", "none") else ""),
                input_hash=input_hash,
                output_hash=hash_jsonable([e.model_dump() for e in entries]),
                params={"seed_entries_count": len(seed), "reasoning_backend": backend_name},
                tool_versions=[TOOL],
            )],
        )
        art = self.persist_artifact(ctx, "knowledge_pack", f"kp_{ctx.project_id}", kp)
        return AgentResult(
            project_id=ctx.project_id, agent=self.name, task_id=ctx.task_id,
            status=AgentResultStatus.success, artifacts=[art],
            findings=[f"KnowledgePack assembled with {len(entries)} entries"],
            warnings=warnings,
            provenance=[ProvenanceEntry(source="ProjectSpec + seed", input_hash=input_hash, output_hash=art.hash, tool_versions=[TOOL])],
            tool_versions=[TOOL], reproducibility={"input_hash": input_hash},
        )

    @staticmethod
    def _from_reasoning(raw: dict, i: int) -> KnowledgePackEntry:
        data = dict(raw)
        data.setdefault("id", f"reasoning_{i}")
        data.setdefault("source", "Hermes subagent (LLM reasoning)")
        data["evidence_level"] = "inferred"  # LLM recall is not observed data
        notes = data.get("notes") or ""
        if "verify against primary source" not in notes:
            data["notes"] = (notes + " | LLM-recalled via reasoning backend; verify against primary source").strip(" |")
        return KnowledgePackEntry.model_validate(data)

    @staticmethod
    def _resolve_backend(ctx: AgentContext, warnings: list[str]):
        name = ctx.params.get("reasoning_backend", "none")
        try:
            backend = make_reasoning_backend(
                name,
                command=ctx.params.get("reasoning_command"),
                max_turns=ctx.params.get("reasoning_max_turns", 30),
                run_budget=ctx.params.get("reasoning_run_budget", 300),
            )
        except KeyError as exc:
            return None, str(exc)
        avail, msg = backend.available()
        if not avail:
            return None, f"reasoning backend '{name}' unavailable: {msg}; kept deterministic baseline only"
        return backend, ""
