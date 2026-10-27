"""cyp_minion: deterministic CYP450 DDI (inhibition + induction) flags.

Matches a curated, sourced set of CYP inhibitor/inducer chemotype SMARTS and
reports each match as a *predicted* potential DDI liability (structural alert
from a literature rule) — not a functional enzyme assay. Consumes molecules
from params['molecules'], then screened_batch survivors (passed=True), then
starting_materials.
"""

from __future__ import annotations

from rdkit import rdBase

from ..chemistry.cyp_alerts import assess_cyp
from ..chemistry.rdkit_ops import sanitize_and_canonicalize
from ..contracts.agent_result import AgentResult
from ..contracts.cyp_assessment import (
    CypAssessment,
    CypLiability,
    CypSummary,
    MoleculeCypProfile,
)
from ..contracts.enums import AgentResultStatus
from ..contracts.provenance import ProvenanceEntry, ToolVersion
from ..logging.hashing import hash_jsonable
from .base import Agent, AgentContext
from ddswarm import __version__

TOOL = ToolVersion(name="ddswarm", version=__version__)
RDKIT_TOOL = ToolVersion(name="rdkit", version=rdBase.rdkitVersion)


class CypMinion(Agent):
    name = "cyp_minion"

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

        profiles: list[MoleculeCypProfile] = []
        warnings: list[str] = []
        skipped = 0
        for mid, smi in molecules:
            s = sanitize_and_canonicalize(smi)
            if not s.ok:
                skipped += 1
                continue
            hits = assess_cyp(s.canonical_smiles)
            liabilities = [
                CypLiability(enzyme=e, effect=ef, alert_slug=slug, citation=cit)  # type: ignore[arg-type]
                for e, ef, slug, cit in hits
            ]
            if liabilities:
                note = (f"{len(liabilities)} DDI alert(s): "
                        + ", ".join(f"{l.enzyme} {l.effect} ({l.alert_slug})" for l in liabilities)
                        + " — predicted potential DDI from literature rules, NOT a functional assay")
            else:
                note = "no matches in the curated CYP chemotype set (not a clean-DDI declaration)"
            profiles.append(MoleculeCypProfile(
                molecule_id=mid, canonical_smiles=s.canonical_smiles,
                liabilities=liabilities, note=note,
            ))

        if skipped:
            warnings.append(f"skipped {skipped} molecule(s) that failed sanitization")

        summary = CypSummary(
            total_molecules=len(profiles),
            flagged_inhibition=sum(1 for p in profiles
                                   if any(l.effect == "inhibition" for l in p.liabilities)),
            flagged_induction=sum(1 for p in profiles
                                  if any(l.effect == "induction" for l in p.liabilities)),
            total_alerts=sum(len(p.liabilities) for p in profiles),
        )

        input_hash = hash_jsonable({"molecules": [(m, s) for m, s in molecules]})
        output_hash = hash_jsonable([p.model_dump() for p in profiles])

        assessment = CypAssessment(
            project_id=ctx.project_id,
            status=AgentResultStatus.success,
            molecules=profiles,
            summary=summary,
            note=("CYP inhibition/induction flags are structural alerts (predicted), not a "
                  "functional enzyme assay; induction is a weak 2D-structure signal (PXR/CAR/AhR)."),
            provenance=[ProvenanceEntry(
                source="deterministic CYP DDI structural alerts",
                tool_versions=[TOOL, RDKIT_TOOL],
                params={"inhibition_alert_count": 6, "induction_alert_count": 3},
                input_hash=input_hash,
                output_hash=output_hash,
            )],
        )
        art = self.persist_artifact(ctx, "cyp_assessment", f"cyp_{ctx.project_id}", assessment)
        finding = (f"CYP DDI: {summary.total_molecules} molecules; "
                   f"{summary.flagged_inhibition} inhibitor-flagged, "
                   f"{summary.flagged_induction} inducer-flagged "
                   f"({summary.total_alerts} total alerts, all predicted)")
        return AgentResult(
            project_id=ctx.project_id, agent=self.name, task_id=ctx.task_id,
            status=AgentResultStatus.success, artifacts=[art], input_refs=ctx.input_refs,
            findings=[finding], warnings=warnings,
            provenance=[ProvenanceEntry(source="deterministic CYP DDI structural alerts",
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
