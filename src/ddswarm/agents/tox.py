"""tox_minion: deterministic structural toxicophore screening.

Matches a curated, sourced set of toxicophore SMARTS and reports each match as a
*predicted* potential liability (structural alert from a literature rule) — not
experimental evidence of toxicity. Consumes molecules from params['molecules'],
then screened_batch survivors (passed=True), then starting_materials."""

from __future__ import annotations

from rdkit import Chem
from rdkit import rdBase

from ..chemistry.rdkit_ops import sanitize_and_canonicalize
from ..chemistry.tox_alerts import match_tox_alerts
from ..contracts.agent_result import AgentResult
from ..contracts.enums import AgentResultStatus
from ..contracts.provenance import ProvenanceEntry, ToolVersion
from ..contracts.tox_profile import MoleculeToxAssessment, ToxAlert, ToxProfile, ToxSummary
from ..logging.hashing import hash_jsonable
from .base import Agent, AgentContext
from ddswarm import __version__

TOOL = ToolVersion(name="ddswarm", version=__version__)
RDKIT_TOOL = ToolVersion(name="rdkit", version=rdBase.rdkitVersion)


class ToxMinion(Agent):
    name = "tox_minion"

    def run(self, ctx: AgentContext) -> AgentResult:
        molecules = self._collect_molecules(ctx)
        if not molecules:
            return AgentResult(
                project_id=ctx.project_id, agent=self.name, task_id=ctx.task_id,
                status=AgentResultStatus.insufficient_evidence,
                warnings=["no molecules to assess (params['molecules'], screened_batch survivors, or starting_materials empty)"],
                tool_versions=[TOOL, RDKIT_TOOL], reproducibility={"reason": "no_molecules"},
            )

        assessments: list[MoleculeToxAssessment] = []
        warnings: list[str] = []
        skipped = 0
        for mid, smi in molecules:
            s = sanitize_and_canonicalize(smi)
            if not s.ok:
                skipped += 1
                continue
            mol = Chem.MolFromSmiles(s.canonical_smiles)
            alerts = [ToxAlert(**a) for a in match_tox_alerts(mol)]
            if alerts:
                names = ", ".join(a.name for a in alerts)
                note = (f"{len(alerts)} structural alert(s): {names} — predicted potential liability "
                        f"from literature rules, NOT experimental toxicity evidence")
            else:
                note = "no matches in the curated toxicophore set (not a clean-toxicity declaration)"
            assessments.append(MoleculeToxAssessment(
                id=mid, canonical_smiles=s.canonical_smiles, alerts=alerts,
                liability_count=len(alerts), risk_note=note,
            ))

        if skipped:
            warnings.append(f"skipped {skipped} molecule(s) that failed sanitization")

        summary = ToxSummary(
            total_molecules=len(assessments),
            molecules_with_alerts=sum(1 for a in assessments if a.alerts),
            total_alerts=sum(a.liability_count for a in assessments),
        )

        input_hash = hash_jsonable({"molecules": [(m, s) for m, s in molecules]})
        output_hash = hash_jsonable([a.model_dump() for a in assessments])

        profile = ToxProfile(
            project_id=ctx.project_id,
            status=AgentResultStatus.success,
            molecules=assessments,
            summary=summary,
            provenance=[ProvenanceEntry(
                source="deterministic toxicophore structural alerts",
                tool_versions=[TOOL, RDKIT_TOOL],
                params={"alert_count": 7},
                input_hash=input_hash,
                output_hash=output_hash,
            )],
        )
        art = self.persist_artifact(ctx, "tox_profile", f"tx_{ctx.project_id}", profile)
        finding = (f"assessed {summary.total_molecules} molecules; "
                   f"{summary.molecules_with_alerts} with >=1 structural alert "
                   f"({summary.total_alerts} total alerts, all predicted)")
        return AgentResult(
            project_id=ctx.project_id, agent=self.name, task_id=ctx.task_id,
            status=AgentResultStatus.success, artifacts=[art], input_refs=ctx.input_refs,
            findings=[finding], warnings=warnings,
            provenance=[ProvenanceEntry(source="deterministic toxicophore structural alerts",
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
