"""screener_minion: deterministic structural + physicochemical screening (RDKit).

Consumes molecules from params['molecules'] (explicit), then design_batch
upstream, then ProjectSpec.starting_materials. Applies hard filters (validity,
Ro5, PAINS, Brenk) and advisory checks. Never fabricates: with no molecules it
returns insufficient_evidence."""

from __future__ import annotations

from rdkit import Chem
from rdkit import rdBase

from ..chemistry.rdkit_ops import compute_properties, match_alerts, sanitize_and_canonicalize
from ..contracts.agent_result import AgentResult
from ..contracts.enums import AgentResultStatus
from ..contracts.provenance import ProvenanceEntry, ToolVersion
from ..contracts.screened_batch import (
    MoleculeScreenResult,
    ScreenSummary,
    ScreenedBatch,
)
from ..logging.hashing import hash_jsonable
from .base import Agent, AgentContext

TOOL = ToolVersion(name="ddswarm", version="0.1.0")
RDKIT_TOOL = ToolVersion(name="rdkit", version=rdBase.rdkitVersion)

DEFAULT_FILTERS = {
    "max_mw": 500.0,
    "max_logp": 5.0,
    "max_hbd": 5,
    "max_hba": 10,
    "enable_pains": True,
    "enable_brenk": True,
    "brenk_hard": False,
    "advisory_tpsa": 140.0,
    "advisory_rotatable_bonds": 10,
    "advisory_max_abs_charge": 1,
}


class ScreenerMinion(Agent):
    name = "screener_minion"

    def run(self, ctx: AgentContext) -> AgentResult:
        filters = dict(DEFAULT_FILTERS)
        filters.update(ctx.params.get("filters", {}) or {})

        molecules = self._collect_molecules(ctx)
        if not molecules:
            return AgentResult(
                project_id=ctx.project_id,
                agent=self.name,
                task_id=ctx.task_id,
                status=AgentResultStatus.insufficient_evidence,
                warnings=["no molecules to screen (params['molecules'], design_batch, or starting_materials empty)"],
                tool_versions=[TOOL, RDKIT_TOOL],
                reproducibility={"reason": "no_molecules"},
            )

        results = [self._screen_one(mid, smi, filters) for mid, smi in molecules]
        summary = ScreenSummary(
            total=len(results),
            passed=sum(1 for r in results if r.passed),
            rejected=sum(1 for r in results if r.valid and not r.passed),
            invalid=sum(1 for r in results if not r.valid),
            brenk_flagged=sum(1 for r in results if r.brenk_alerts),
        )

        warnings = self._exclusion_warnings(ctx)
        input_hash = hash_jsonable({"molecules": [(m, s) for m, s in molecules], "filters": filters})
        output_hash = hash_jsonable([r.model_dump() for r in results])

        batch = ScreenedBatch(
            project_id=ctx.project_id,
            status=AgentResultStatus.success,
            molecules=results,
            summary=summary,
            provenance=[ProvenanceEntry(
                source="RDKit deterministic screening",
                tool_versions=[TOOL, RDKIT_TOOL],
                params={"filters": filters},
                input_hash=input_hash,
                output_hash=output_hash,
            )],
        )
        art = self.persist_artifact(ctx, "screened_batch", f"sb_{ctx.project_id}", batch)
        return AgentResult(
            project_id=ctx.project_id,
            agent=self.name,
            task_id=ctx.task_id,
            status=AgentResultStatus.success,
            input_refs=ctx.input_refs,
            artifacts=[art],
            findings=[f"screened {summary.total} molecules: {summary.passed} passed, {summary.rejected} rejected, {summary.invalid} invalid"],
            warnings=warnings,
            provenance=[ProvenanceEntry(source="RDKit deterministic screening", output_hash=art.hash, tool_versions=[TOOL, RDKIT_TOOL])],
            tool_versions=[TOOL, RDKIT_TOOL],
            reproducibility={"input_hash": input_hash, "filters": filters},
        )

    def _collect_molecules(self, ctx: AgentContext) -> list[tuple[str, str]]:
        mols: list[tuple[str, str]] = []
        explicit = ctx.params.get("molecules", []) or []
        for i, m in enumerate(explicit):
            if isinstance(m, dict) and "smiles" in m:
                mols.append((m.get("id", f"param_{i}"), m["smiles"]))
        if mols:
            return mols
        for db in ctx.inputs.get("design_batch", []):
            for m in getattr(db, "molecules", []):
                mols.append((m.id, m.smiles))
        if mols:
            return mols
        for i, smi in enumerate(ctx.spec.starting_materials):
            mols.append((f"starting_material_{i}", smi))
        return mols

    def _screen_one(self, mid: str, smi: str, filters: dict) -> MoleculeScreenResult:
        s = sanitize_and_canonicalize(smi)
        if not s.ok:
            return MoleculeScreenResult(
                id=mid,
                input_smiles=smi,
                valid=False,
                sanitization_error=s.error,
                passed=False,
                elimination_reasons=[f"invalid SMILES: {s.error}"],
            )
        mol = Chem.MolFromSmiles(s.canonical_smiles)
        props = compute_properties(mol)

        ro5: list[str] = []
        reasons: list[str] = []
        if props["molecular_weight"] > filters["max_mw"]:
            ro5.append("MW")
            reasons.append(f"molecular_weight {props['molecular_weight']} > {filters['max_mw']}")
        if props["logp"] > filters["max_logp"]:
            ro5.append("logP")
            reasons.append(f"logp {props['logp']} > {filters['max_logp']}")
        if props["hbd"] > filters["max_hbd"]:
            ro5.append("HBD")
            reasons.append(f"hbd {props['hbd']} > {filters['max_hbd']}")
        if props["hba"] > filters["max_hba"]:
            ro5.append("HBA")
            reasons.append(f"hba {props['hba']} > {filters['max_hba']}")

        pains: list[str] = []
        brenk: list[str] = []
        if filters.get("enable_pains", True):
            pains += match_alerts(mol, "pains")
        if filters.get("enable_brenk", True):
            brenk += match_alerts(mol, "brenk")
        if pains:
            reasons.append("PAINS alert(s): " + "; ".join(pains))
        if brenk and filters.get("brenk_hard", False):
            reasons.append("Brenk alert(s): " + "; ".join(brenk))

        advisories: list[str] = []
        if props["tpsa"] > filters.get("advisory_tpsa", 140.0):
            advisories.append(f"tpsa {props['tpsa']} > {filters.get('advisory_tpsa', 140.0)}")
        if props["rotatable_bonds"] > filters.get("advisory_rotatable_bonds", 10):
            advisories.append(f"rotatable_bonds {props['rotatable_bonds']} > {filters.get('advisory_rotatable_bonds', 10)}")
        if abs(props["formal_charge"]) > filters.get("advisory_max_abs_charge", 1):
            advisories.append(f"formal_charge {props['formal_charge']} outside +/- {filters.get('advisory_max_abs_charge', 1)}")

        hard_alerts = pains + (brenk if filters.get("brenk_hard", False) else [])
        passed = not ro5 and not hard_alerts
        return MoleculeScreenResult(
            id=mid,
            input_smiles=smi,
            canonical_smiles=s.canonical_smiles,
            valid=True,
            properties=props,
            rule_of_five_violations=ro5,
            structural_alerts=pains,
            brenk_alerts=brenk,
            advisories=advisories,
            passed=passed,
            elimination_reasons=reasons,
        )

    def _exclusion_warnings(self, ctx: AgentContext) -> list[str]:
        warnings: list[str] = []
        recognized = {"pains", "brenk"}
        for exc in ctx.spec.exclusions:
            low = exc.strip().lower()
            if not low:
                continue
            key = next((k for k in recognized if k in low), None)
            if key is None:
                warnings.append(f"exclusion '{exc}' not mapped to a deterministic filter (not enforced)")
        return warnings
