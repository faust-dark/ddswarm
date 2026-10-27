"""herg_minion: deterministic hERG (Kv11.1) cardiotoxicity-liability flag.

Matches molecules against the hERG pharmacophore (basic amine + >=2 aromatic
rings) gated by lipophilicity (cLogP), and reports each match as a *predicted*
potential hERG liability — not a patch-clamp assay. Consumes molecules from
params['molecules'], then screened_batch survivors (passed=True), then
starting_materials.
"""

from __future__ import annotations

from rdkit import rdBase

from ..chemistry.herg_alerts import assess_herg
from ..chemistry.rdkit_ops import sanitize_and_canonicalize
from ..contracts.agent_result import AgentResult
from ..contracts.enums import AgentResultStatus
from ..contracts.herg_assessment import HergAssessment, HergSummary, MoleculeHergProfile
from ..contracts.provenance import ProvenanceEntry, ToolVersion
from ..logging.hashing import hash_jsonable
from .base import Agent, AgentContext
from ddswarm import __version__

TOOL = ToolVersion(name="ddswarm", version=__version__)
RDKIT_TOOL = ToolVersion(name="rdkit", version=rdBase.rdkitVersion)


class HergMinion(Agent):
    name = "herg_minion"

    def run(self, ctx: AgentContext) -> AgentResult:
        molecules = self._collect_molecules(ctx)
        if not molecules:
            return AgentResult(
                project_id=ctx.project_id, agent=self.name, task_id=ctx.task_id,
                status=AgentResultStatus.insufficient_evidence,
                warnings=["no molecules to assess (params['molecules'], screened_batch "
                          "survivors, or starting_materials empty)"],
                tool_versions=[TOOL, RDKIT_TOOL], reproducibility={"reason": "no_molecules"},
            )

        profiles: list[MoleculeHergProfile] = []
        warnings: list[str] = []
        skipped = 0
        for mid, smi in molecules:
            s = sanitize_and_canonicalize(smi)
            if not s.ok:
                skipped += 1
                continue
            call = assess_herg(s.canonical_smiles)
            profiles.append(MoleculeHergProfile(
                molecule_id=mid, canonical_smiles=s.canonical_smiles,
                flagged=call["flagged"],
                basic_amine_motifs=call["basic_amine_motifs"],
                aromatic_rings=call["aromatic_rings"],
                clogp=call["clogp"],
                note=call["reason"],
            ))

        if skipped:
            warnings.append(f"skipped {skipped} molecule(s) that failed sanitization")

        summary = HergSummary(
            total_molecules=len(profiles),
            flagged=sum(1 for p in profiles if p.flagged),
        )

        input_hash = hash_jsonable({"molecules": [(m, s) for m, s in molecules]})
        output_hash = hash_jsonable([p.model_dump() for p in profiles])

        assessment = HergAssessment(
            project_id=ctx.project_id,
            status=AgentResultStatus.success,
            molecules=profiles,
            summary=summary,
            note=("hERG flags are structural/physicochemical predictions, not a "
                  "patch-clamp assay; IC50 potency is not estimated from a 2D pharmacophore."),
            provenance=[ProvenanceEntry(
                source="deterministic hERG pharmacophore + lipophilicity rule",
                tool_versions=[TOOL, RDKIT_TOOL],
                params={"aromatic_ring_min": 2, "clogp_min": 3.0},
                input_hash=input_hash,
                output_hash=output_hash,
            )],
        )
        art = self.persist_artifact(ctx, "herg_assessment", f"herg_{ctx.project_id}", assessment)
        finding = (f"hERG: {summary.total_molecules} molecules; "
                   f"{summary.flagged} flagged as potential hERG liability (predicted)")
        return AgentResult(
            project_id=ctx.project_id, agent=self.name, task_id=ctx.task_id,
            status=AgentResultStatus.success, artifacts=[art], input_refs=ctx.input_refs,
            findings=[finding], warnings=warnings,
            provenance=[ProvenanceEntry(source="deterministic hERG pharmacophore + lipophilicity rule",
                                        output_hash=art.hash, tool_versions=[TOOL, RDKIT_TOOL])],
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
