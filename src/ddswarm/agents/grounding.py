"""grounding_minion: resolve molecules against open external databases (opt-in).

Fetches real, cited external records (ChEMBL, PubChem, PubMed; DrugBank
identifier via PubChem xref) for each molecule and records them as *observed*
entries with an external ID + URL + retrieval timestamp. This is the project's
first machine-readable link to live external data; LLM-recalled knowledge stays
*inferred*, grounded entries are *observed*.

Opt-in because live APIs are non-deterministic across time — a grounded run is
NOT byte-reproducible, so grounding is excluded from the reproducibility hash.

Grounds (deduplicated): params['molecules'], then starting_materials, then
screened_batch survivors (the survivors double as a novelty check: a designed
analog that IS in ChEMBL/PubChem is not novel).
"""

from __future__ import annotations

from datetime import datetime, timezone

from rdkit import rdBase

from ..chemistry.chembl_client import compound_affinities
from ..chemistry.rdkit_ops import sanitize_and_canonicalize
from ..contracts.agent_result import AgentResult
from ..contracts.enums import AgentResultStatus
from ..contracts.grounding_report import (
    BindingAffinity,
    GroundedEntry,
    GroundingReport,
    GroundingSummary,
    MoleculeGrounding,
)
from ..contracts.provenance import ProvenanceEntry, ToolVersion
from ..grounding.client import ground_molecule
from ..logging.hashing import hash_jsonable
from .base import Agent, AgentContext

TOOL = ToolVersion(name="ddswarm", version="0.1.0")
RDKIT_TOOL = ToolVersion(name="rdkit", version=rdBase.rdkitVersion)


class GroundingMinion(Agent):
    name = "grounding_minion"

    def run(self, ctx: AgentContext) -> AgentResult:
        molecules = self._collect_molecules(ctx)
        if not molecules:
            return AgentResult(
                project_id=ctx.project_id, agent=self.name, task_id=ctx.task_id,
                status=AgentResultStatus.insufficient_evidence,
                warnings=["no molecules to ground (params['molecules'], starting_materials, "
                          "or screened_batch survivors empty)"],
                tool_versions=[TOOL, RDKIT_TOOL], reproducibility={"reason": "no_molecules"},
            )

        names = ctx.params.get("names", {}) or {}
        retrieved_at = datetime.now(timezone.utc).isoformat()
        results: list[MoleculeGrounding] = []
        warnings: list[str] = []
        skipped = 0
        for mid, smi in molecules:
            s = sanitize_and_canonicalize(smi)
            if not s.ok:
                skipped += 1
                continue
            g = ground_molecule(s.canonical_smiles, name=names.get(mid))
            entries = [GroundedEntry(
                source=e["source"], external_id=e["external_id"], url=e["url"],
                title=e["title"], detail=e.get("detail"), retrieved_at=retrieved_at,
            ) for e in g["entries"]]
            affinities: list[BindingAffinity] = []
            if ctx.params.get("affinities") and g.get("inchi_key"):
                try:
                    for a in compound_affinities(g["inchi_key"], limit=10):
                        affinities.append(BindingAffinity(
                            target_chembl_id=a["target_chembl_id"],
                            target_pref_name=a["target_pref_name"],
                            standard_type=a["standard_type"],
                            value_nm=a["value_nm"],
                            document_year=a["document_year"],
                        ))
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"affinity fetch failed for {mid}: {exc}")
            results.append(MoleculeGrounding(
                molecule_id=mid, canonical_smiles=s.canonical_smiles,
                inchi_key=g["inchi_key"], status=g["status"],
                entries=entries, binding_affinities=affinities, errors=g["errors"],
            ))
        if skipped:
            warnings.append(f"skipped {skipped} molecule(s) that failed sanitization")

        summary = GroundingSummary(
            total_molecules=len(results),
            grounded=sum(1 for r in results if r.status == "grounded"),
            not_found=sum(1 for r in results if r.status == "not_found"),
            failed=sum(1 for r in results if r.status == "failed"),
            total_entries=sum(len(r.entries) for r in results),
            total_affinities=sum(len(r.binding_affinities) for r in results),
        )

        input_hash = hash_jsonable({"molecules": [(m, s) for m, s in molecules]})
        # NOTE: output_hash is intentionally OMITTED from provenance — live API
        # responses are non-deterministic across time, so the grounded artifact is
        # excluded from the reproducibility fingerprint.
        report = GroundingReport(
            project_id=ctx.project_id,
            status=AgentResultStatus.success,
            molecules=results,
            summary=summary,
            note=("Grounded entries are observed external records (URL + identifier + "
                  "retrieval timestamp); LLM-recalled knowledge remains inferred. "
                  "DrugBank record content requires a license (403); its identifier/link "
                  "is resolved via PubChem cross-reference. Network failures are reported "
                  "as failed/not_found — never fabricated."),
            provenance=[ProvenanceEntry(
                source="external-database grounding (ChEMBL + PubChem + PubMed)",
                tool_versions=[TOOL, RDKIT_TOOL],
                params={"sources": ["chembl", "pubchem", "pubmed", "drugbank-via-pubchem-xref"]},
                input_hash=input_hash,
            )],
        )
        art = self.persist_artifact(ctx, "grounding_report", f"grounding_{ctx.project_id}", report)
        finding = (f"grounding: {summary.total_molecules} molecules; "
                   f"{summary.grounded} grounded, {summary.not_found} not_found, "
                   f"{summary.failed} failed ({summary.total_entries} observed entries)")
        if ctx.params.get("affinities"):
            finding += f", {summary.total_affinities} binding affinities"
        return AgentResult(
            project_id=ctx.project_id, agent=self.name, task_id=ctx.task_id,
            status=AgentResultStatus.success, artifacts=[art], input_refs=ctx.input_refs,
            findings=[finding], warnings=warnings,
            provenance=[ProvenanceEntry(source="external-database grounding",
                                        output_hash=art.hash, tool_versions=[TOOL, RDKIT_TOOL])],
            tool_versions=[TOOL, RDKIT_TOOL],
            reproducibility={"input_hash": input_hash, "non_deterministic": ["external_network"]},
        )

    def _collect_molecules(self, ctx: AgentContext) -> list[tuple[str, str]]:
        mols: list[tuple[str, str]] = []
        seen: set[str] = set()

        def add(mid: str, smi: str) -> None:
            if smi and smi not in seen:
                mols.append((mid, smi))
                seen.add(smi)

        explicit = ctx.params.get("molecules", []) or []
        for i, m in enumerate(explicit):
            if isinstance(m, dict) and "smiles" in m:
                add(m.get("id", f"param_{i}"), m["smiles"])
        if not mols:
            for i, smi in enumerate(ctx.spec.starting_materials):
                add(f"starting_material_{i}", smi)
            for sb in ctx.inputs.get("screened_batch", []):
                for m in getattr(sb, "molecules", []):
                    if m.passed and m.canonical_smiles:
                        add(m.id, m.canonical_smiles)
        return mols
