"""designer_minion: proposes a DesignBatch from upstream scaffolds.

Minimal in Increment 0: no RDKit yet, so it does NOT generate new structures.
It carries each parent scaffold forward under a set of named transformations,
marked evidence_level=hypothetical, and refuses to invent molecules when there
is no scaffold input (returns insufficient_evidence)."""

from __future__ import annotations

from ..contracts.agent_result import AgentResult
from ..contracts.design_batch import DesignBatch, MoleculeDesign
from ..contracts.enums import AgentResultStatus, EvidenceLevel
from ..contracts.provenance import ProvenanceEntry, ToolVersion
from ..logging.hashing import hash_jsonable
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
        scaffolds: list[tuple[str, str]] = []  # (parent_id, smiles)
        for kp in ctx.inputs.get("knowledge_pack", []):
            for e in getattr(kp, "entries", []):
                if e.type == "ligand" and e.observed_value:
                    scaffolds.append((e.id, e.observed_value))
        for smi in ctx.spec.starting_materials:
            if not any(s == smi for _, s in scaffolds):
                scaffolds.append(("starting_material", smi))

        if not scaffolds:
            return AgentResult(
                project_id=ctx.project_id,
                agent=self.name,
                task_id=ctx.task_id,
                status=AgentResultStatus.insufficient_evidence,
                warnings=["no ligand scaffolds available from KnowledgePack or starting_materials; refusing to invent molecules"],
                tool_versions=[TOOL],
                reproducibility={"reason": "no_scaffolds"},
            )

        molecules: list[MoleculeDesign] = []
        for i, (parent, smi) in enumerate(scaffolds):
            for j, t in enumerate(self.TRANSFORMATIONS):
                molecules.append(MoleculeDesign(
                    id=f"mol_{i}_{j}",
                    smiles=smi,
                    parent_scaffold=parent,
                    transformation=t,
                    rationale=(
                        f"minimal analog derived from {parent} via '{t}' "
                        "(label only; structural generation + RDKit validation land in Increment 1)"
                    ),
                    evidence_level=EvidenceLevel.hypothetical,
                ))

        input_hash = hash_jsonable({"scaffolds": scaffolds})
        output_hash = hash_jsonable([m.model_dump() for m in molecules])
        db = DesignBatch(
            project_id=ctx.project_id,
            status=AgentResultStatus.success,
            molecules=molecules,
            provenance=[ProvenanceEntry(
                source="KnowledgePack + ProjectSpec",
                tool_versions=[TOOL],
                params={"scaffold_count": len(scaffolds), "transformations": self.TRANSFORMATIONS},
                input_hash=input_hash,
                output_hash=output_hash,
            )],
        )
        art = self.persist_artifact(ctx, "design_batch", f"db_{ctx.project_id}", db)
        warnings = [
            "designs are transformation labels over parent SMILES only; "
            "no new structures generated and no RDKit validation in Increment 0"
        ]
        return AgentResult(
            project_id=ctx.project_id,
            agent=self.name,
            task_id=ctx.task_id,
            status=AgentResultStatus.success,
            artifacts=[art],
            findings=[f"DesignBatch with {len(molecules)} candidate designs (labels only)"],
            warnings=warnings,
            provenance=[ProvenanceEntry(source="KnowledgePack + ProjectSpec", output_hash=art.hash, tool_versions=[TOOL])],
            tool_versions=[TOOL],
            reproducibility={"scaffold_count": len(scaffolds)},
        )
