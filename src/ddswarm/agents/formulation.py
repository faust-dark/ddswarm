"""formulation_minion: deterministic developability assessment (aqueous
solubility class, permeability class, BCS classification, ionizable groups,
salt/prodrug suggestions) from RDKit descriptors. Every field is a *predicted*
rule-based estimate — never a measured property.

Consumes molecules from params['molecules'], then screened_batch survivors
(passed=True), then starting_materials. No molecules => insufficient_evidence."""

from __future__ import annotations

from rdkit import Chem
from rdkit import rdBase

from ..chemistry.formulation import assess_formulation
from ..chemistry.rdkit_ops import sanitize_and_canonicalize
from ..contracts.agent_result import AgentResult
from ..contracts.enums import AgentResultStatus
from ..contracts.formulation_assessment import (
    FormulationAssessment,
    FormulationSummary,
    MoleculeFormulation,
)
from ..contracts.provenance import ProvenanceEntry, ToolVersion
from ..logging.hashing import hash_jsonable
from .base import Agent, AgentContext
from ddswarm import __version__

TOOL = ToolVersion(name="ddswarm", version=__version__)
RDKIT_TOOL = ToolVersion(name="rdkit", version=rdBase.rdkitVersion)


class FormulationMinion(Agent):
    name = "formulation_minion"

    def run(self, ctx: AgentContext) -> AgentResult:
        molecules = self._collect_molecules(ctx)
        if not molecules:
            return AgentResult(
                project_id=ctx.project_id, agent=self.name, task_id=ctx.task_id,
                status=AgentResultStatus.insufficient_evidence,
                warnings=["no molecules to assess (params['molecules'], screened_batch survivors, or starting_materials empty)"],
                tool_versions=[TOOL, RDKIT_TOOL], reproducibility={"reason": "no_molecules"},
            )

        assessments: list[MoleculeFormulation] = []
        skipped = 0
        for mid, smi in molecules:
            s = sanitize_and_canonicalize(smi)
            if not s.ok:
                skipped += 1
                continue
            mol = Chem.MolFromSmiles(s.canonical_smiles)
            a = assess_formulation(mol)
            assessments.append(MoleculeFormulation(
                id=mid, canonical_smiles=s.canonical_smiles,
                logp=a["logp"], tpsa=a["tpsa"],
                solubility_class=a["solubility_class"],
                permeability_class=a["permeability_class"],
                bcs_class=a["bcs_class"],
                ionizable_groups=a["ionizable_groups"],
                salt_suggestion=a["salt_suggestion"],
                prodrug_suggestion=a["prodrug_suggestion"],
                note=a["note"],
            ))

        warnings: list[str] = []
        if skipped:
            warnings.append(f"skipped {skipped} molecule(s) that failed sanitization")

        bcs_counts: dict[str, int] = {}
        for m in assessments:
            bcs_counts[m.bcs_class] = bcs_counts.get(m.bcs_class, 0) + 1
        summary = FormulationSummary(total_molecules=len(assessments), bcs_counts=bcs_counts)

        input_hash = hash_jsonable({"molecules": [(m, s) for m, s in molecules]})
        output_hash = hash_jsonable([m.model_dump() for m in assessments])

        batch = FormulationAssessment(
            project_id=ctx.project_id,
            status=AgentResultStatus.success,
            molecules=assessments,
            summary=summary,
            provenance=[ProvenanceEntry(
                source="RDKit descriptor rules (formulation)",
                tool_versions=[TOOL, RDKIT_TOOL],
                params={},
                input_hash=input_hash,
                output_hash=output_hash,
            )],
        )
        art = self.persist_artifact(ctx, "formulation_assessment", f"fa_{ctx.project_id}", batch)
        finding = (f"formulation: {summary.total_molecules} molecules; "
                   f"BCS {bcs_counts or 'n/a'}")
        return AgentResult(
            project_id=ctx.project_id, agent=self.name, task_id=ctx.task_id,
            status=AgentResultStatus.success, artifacts=[art], input_refs=ctx.input_refs,
            findings=[finding], warnings=warnings,
            provenance=[ProvenanceEntry(source="formulation assessment", output_hash=art.hash,
                                        tool_versions=[TOOL, RDKIT_TOOL])],
            tool_versions=[TOOL, RDKIT_TOOL],
            reproducibility={"input_hash": input_hash},
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
