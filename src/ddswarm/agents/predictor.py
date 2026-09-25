"""predictor_minion: ADMET/PK prediction through a pluggable provider abstraction.

Consumes molecules from params['molecules'], then screened_batch survivors
(passed=True), then starting_materials. Each provider returns per-endpoint
predictions with applicability domain and uncertainty. No provider configured,
or no molecules, => insufficient_evidence (never fabricate a prediction)."""

from __future__ import annotations

from rdkit import rdBase

from ..chemistry.rdkit_ops import sanitize_and_canonicalize
from ..contracts.admet_prediction import (
    AdmetPrediction,
    AdmetSummary,
    MoleculeAdmetResult,
)
from ..contracts.agent_result import AgentResult
from ..contracts.enums import AgentResultStatus
from ..contracts.provenance import ProvenanceEntry, ToolVersion
from ..logging.hashing import hash_jsonable
from ..predictors.registry import resolve_providers
from .base import Agent, AgentContext

TOOL = ToolVersion(name="ddswarm", version="0.1.0")
RDKIT_TOOL = ToolVersion(name="rdkit", version=rdBase.rdkitVersion)


class PredictorMinion(Agent):
    name = "predictor_minion"

    def run(self, ctx: AgentContext) -> AgentResult:
        provider_names = ctx.params.get("providers", []) or []
        if not provider_names:
            return self._result(
                ctx, AgentResultStatus.insufficient_evidence,
                warnings=["no ADMET provider configured (params['providers'] empty); refusing to fabricate predictions"],
                reproducibility={"reason": "no_provider"},
            )
        try:
            providers = resolve_providers(provider_names)
        except KeyError as exc:
            return self._result(
                ctx, AgentResultStatus.failed,
                errors=[str(exc)], reproducibility={"reason": "unknown_provider"},
            )

        molecules = self._collect_molecules(ctx)
        if not molecules:
            return self._result(
                ctx, AgentResultStatus.insufficient_evidence,
                warnings=["no molecules to predict (params['molecules'], screened_batch survivors, or starting_materials empty)"],
                reproducibility={"reason": "no_molecules"},
            )

        results: list[MoleculeAdmetResult] = []
        skipped = 0
        for mid, smi in molecules:
            s = sanitize_and_canonicalize(smi)
            if not s.ok:
                skipped += 1
                continue
            endpoints = []
            for p in providers:
                endpoints.extend(p.predict(s.canonical_smiles))
            results.append(MoleculeAdmetResult(id=mid, canonical_smiles=s.canonical_smiles, endpoints=endpoints))

        summary = self._summarize(results)
        warnings = []
        if skipped:
            warnings.append(f"skipped {skipped} molecule(s) that failed sanitization")

        input_hash = hash_jsonable({"molecules": [(m, s) for m, s in molecules], "providers": provider_names})
        output_hash = hash_jsonable([r.model_dump() for r in results])

        batch = AdmetPrediction(
            project_id=ctx.project_id,
            status=AgentResultStatus.success,
            molecules=results,
            summary=summary,
            provenance=[ProvenanceEntry(
                source="ADMET/PK prediction (provider abstraction)",
                tool_versions=[TOOL, RDKIT_TOOL],
                params={"providers": provider_names},
                input_hash=input_hash,
                output_hash=output_hash,
            )],
        )
        art = self.persist_artifact(ctx, "admet_prediction", f"ap_{ctx.project_id}", batch)
        return self._result(
            ctx, AgentResultStatus.success,
            input_refs=ctx.input_refs, artifacts=[art],
            findings=[f"predicted {summary.total_endpoints} endpoints across {summary.total_molecules} molecules "
                      f"(predicted={summary.predicted}, not_available={summary.not_available})"],
            warnings=warnings,
            provenance=[ProvenanceEntry(source="ADMET/PK prediction", output_hash=art.hash, tool_versions=[TOOL, RDKIT_TOOL])],
            reproducibility={"input_hash": input_hash, "providers": provider_names},
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

    def _summarize(self, results: list[MoleculeAdmetResult]) -> AdmetSummary:
        total_molecules = len(results)
        total_endpoints = sum(len(r.endpoints) for r in results)
        predicted = sum(1 for r in results for e in r.endpoints if e.prediction_status == "predicted")
        not_available = sum(1 for r in results for e in r.endpoints if e.prediction_status == "not_available")
        out_of_domain = sum(1 for r in results for e in r.endpoints if e.prediction_status == "out_of_domain")
        error = sum(1 for r in results for e in r.endpoints if e.prediction_status == "error")
        return AdmetSummary(
            total_molecules=total_molecules, total_endpoints=total_endpoints,
            predicted=predicted, not_available=not_available,
            out_of_domain=out_of_domain, error=error,
        )

    def _result(self, ctx: AgentContext, status: AgentResultStatus, **kw) -> AgentResult:
        return AgentResult(
            project_id=ctx.project_id, agent=self.name, task_id=ctx.task_id, status=status,
            tool_versions=[TOOL, RDKIT_TOOL], **kw,
        )
