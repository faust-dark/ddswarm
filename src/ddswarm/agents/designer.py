"""designer_minion: proposes a DesignBatch from upstream scaffolds, optionally
delegating actual analog design to a reasoning backend (e.g. a Hermes subagent).

When no backend is configured it falls back to deterministic transformation
labels over the parent SMILES (no new structures). A reasoning backend proposes
real analog SMILES; its output is LLM reasoning, so every molecule it
contributes is forced to evidence_level='hypothetical'. No scaffold input, and
no backend => insufficient_evidence (never invent molecules)."""

from __future__ import annotations

from ..contracts.agent_result import AgentResult
from ..contracts.design_batch import DesignBatch, MoleculeDesign
from ..contracts.enums import AgentResultStatus, EvidenceLevel
from ..contracts.provenance import ProvenanceEntry, ToolVersion
from ..logging.hashing import hash_jsonable
from ..orchestrator.memory import MemoryStore
from ..reasoning import make_reasoning_backend
from .base import Agent, AgentContext

TOOL = ToolVersion(name="ddswarm", version="0.1.0")


class DesignerMinion(Agent):
    name = "designer_minion"

    TRANSFORMATIONS = [
        "bioisostere substitution",
        "scaffold hopping",
        "homologation",
        "analog design",
    ]

    def run(self, ctx: AgentContext) -> AgentResult:
        scaffolds: list[tuple[str, str]] = self._collect_scaffolds(ctx)
        warnings: list[str] = []

        # ---- feedback loop: prior-cycle survivors become extra scaffolds ----
        feedback_entries = MemoryStore(self.store).by_kind(ctx.project_id, "feedback")
        consumed = 0
        saw_survivors = False
        for fe in feedback_entries:
            survivors = (fe.data or {}).get("survivors", [])
            if survivors:
                saw_survivors = True
            for i, smi in enumerate(survivors):
                if smi and not any(s == smi for _, s in scaffolds):
                    scaffolds.append((f"prior_{fe.run_id}_{i}", smi))
                    consumed += 1
        if feedback_entries and not saw_survivors:
            warnings.append("prior cycle recorded no survivors; reconsider filters/scaffold")
        if consumed:
            warnings.append(f"consumed {consumed} prior-cycle survivor(s) as design scaffolds (feedback loop)")

        if not scaffolds:
            return AgentResult(
                project_id=ctx.project_id, agent=self.name, task_id=ctx.task_id,
                status=AgentResultStatus.insufficient_evidence,
                warnings=warnings + ["no ligand scaffolds available from KnowledgePack, starting_materials, or prior-cycle memory; refusing to invent molecules"],
                tool_versions=[TOOL], reproducibility={"reason": "no_scaffolds"},
            )

        molecules: list[MoleculeDesign] = []
        backend_name = ctx.params.get("reasoning_backend", "none")
        used_reasoning = False

        if backend_name in (None, "", "none"):
            molecules = self._labels(scaffolds)
        else:
            backend, note = self._resolve_backend(ctx)
            if backend is None:
                warnings.append(note)
                molecules = self._labels(scaffolds)
            else:
                res = backend.design(ctx.spec, scaffolds, ctx.params)
                if res.ok and res.entries:
                    for i, raw in enumerate(res.entries):
                        try:
                            molecules.append(self._from_reasoning(raw, i))
                        except Exception as exc:  # noqa: BLE001
                            warnings.append(f"reasoning molecule {i} invalid, skipped: {exc}")
                    if molecules:
                        used_reasoning = True
                    else:
                        warnings.append("reasoning produced no valid molecules; using labels")
                        molecules = self._labels(scaffolds)
                else:
                    warnings.append(f"reasoning failed: {res.note}; using labels")
                    molecules = self._labels(scaffolds)

        if used_reasoning:
            warnings.append("molecules proposed by reasoning backend are LLM-generated (evidence_level=hypothetical); not RDKit-validated here")

        input_hash = hash_jsonable({"scaffolds": scaffolds, "reasoning_backend": backend_name})
        output_hash = hash_jsonable([m.model_dump() for m in molecules])
        db = DesignBatch(
            project_id=ctx.project_id,
            status=AgentResultStatus.success,
            molecules=molecules,
            provenance=[ProvenanceEntry(
                source="KnowledgePack + ProjectSpec" + (" + reasoning backend" if used_reasoning else ""),
                tool_versions=[TOOL],
                params={"scaffold_count": len(scaffolds), "transformations": self.TRANSFORMATIONS,
                        "reasoning_backend": backend_name},
                input_hash=input_hash,
                output_hash=output_hash,
            )],
        )
        art = self.persist_artifact(ctx, "design_batch", f"db_{ctx.project_id}", db)
        return AgentResult(
            project_id=ctx.project_id, agent=self.name, task_id=ctx.task_id,
            status=AgentResultStatus.success, artifacts=[art], input_refs=ctx.input_refs,
            findings=[f"DesignBatch with {len(molecules)} candidate designs"
                      f"{' (reasoning backend)' if used_reasoning else ' (labels only)'}"],
            warnings=warnings,
            provenance=[ProvenanceEntry(source="KnowledgePack + ProjectSpec", output_hash=art.hash, tool_versions=[TOOL])],
            tool_versions=[TOOL], reproducibility={"scaffold_count": len(scaffolds)},
        )

    def _labels(self, scaffolds: list[tuple[str, str]]) -> list[MoleculeDesign]:
        molecules: list[MoleculeDesign] = []
        for i, (parent, smi) in enumerate(scaffolds):
            for j, t in enumerate(self.TRANSFORMATIONS):
                molecules.append(MoleculeDesign(
                    id=f"mol_{i}_{j}", smiles=smi, parent_scaffold=parent, transformation=t,
                    rationale=f"minimal analog derived from {parent} via '{t}' (label only; no new structure)",
                    evidence_level=EvidenceLevel.hypothetical,
                ))
        return molecules

    @staticmethod
    def _from_reasoning(raw: dict, i: int) -> MoleculeDesign:
        data = dict(raw)
        data.setdefault("id", f"reasoning_mol_{i}")
        data["evidence_level"] = "hypothetical"
        return MoleculeDesign.model_validate(data)

    @staticmethod
    def _collect_scaffolds(ctx: AgentContext) -> list[tuple[str, str]]:
        scaffolds: list[tuple[str, str]] = []
        for kp in ctx.inputs.get("knowledge_pack", []):
            for e in getattr(kp, "entries", []):
                if e.type == "ligand" and e.observed_value:
                    scaffolds.append((e.id, e.observed_value))
        for smi in ctx.spec.starting_materials:
            if not any(s == smi for _, s in scaffolds):
                scaffolds.append(("starting_material", smi))
        return scaffolds

    @staticmethod
    def _resolve_backend(ctx: AgentContext):
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
            return None, f"reasoning backend '{name}' unavailable: {msg}; using labels"
        return backend, ""
