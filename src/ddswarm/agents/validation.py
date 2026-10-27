"""validation_minion: deterministic verification of a finished run.

Reads the run's artifacts directly from the store and checks, per run:
  - artifact schema validity (re-validates every artifact against its contract)
  - provenance completeness (every artifact carries >= 1 provenance entry)
  - hash integrity (ArtifactRef.hash == fresh hash of the artifact file)
  - cross-agent consistency (screened survivors appear in downstream agents)
  - evidence-level sanity (computed/predicted values are not labelled 'observed')
  - numeric sanity (finite, plausible docking/SA/RMSD values)
  - report-breakdown consistency (Report.evidence_breakdown matches a recount)
  - mechanistic-constraint sanity (mass balance + monotonicity hold; every
    predicted endpoint carries a citable rule/justification)

A failed check is a reproducibility/bookkeeping inconsistency, not a scientific
judgment. The report status is 'success' when no check fails, else 'partial'."""

from __future__ import annotations

import math

from rdkit import rdBase

from ddswarm import __version__

from ..contracts.agent_result import AgentResult
from ..contracts.enums import AgentResultStatus
from ..contracts.provenance import ProvenanceEntry, ToolVersion
from ..contracts.validation_report import ValidationCheck, ValidationReport, ValidationSummary
from ..logging.hashing import hash_content, hash_jsonable
from ..orchestrator.runner import ARTIFACT_MODELS
from .base import Agent, AgentContext
from .report import ReportMinion

TOOL = ToolVersion(name="ddswarm", version=__version__)
RDKIT_TOOL = ToolVersion(name="rdkit", version=rdBase.rdkitVersion)


class ValidationMinion(Agent):
    name = "validation_minion"
    reads_whole_run = True  # scans every task's result in the run manifest

    def run(self, ctx: AgentContext) -> AgentResult:
        manifest = self.store.load_run_manifest(ctx.project_id, ctx.run_id)

        all_refs: list = []
        # artifacts of THIS task's earlier attempt (a re-run / retry of
        # validation): not re-audited for content, but they are hashed files on
        # disk like any other, so tampering with them must still be detected
        own_prior_refs: list = []
        for task_id in manifest.tasks:
            try:
                res = self.store.load_latest_agent_result(ctx.project_id, ctx.run_id, task_id)
            except FileNotFoundError:
                continue  # not run yet (incl. our own first attempt)
            if task_id == ctx.task_id:
                own_prior_refs.extend(res.artifacts)
                continue
            all_refs.extend(res.artifacts)

        if not all_refs:
            check = ValidationCheck(check_id="no_artifacts", name="artifacts present",
                                    status="warn", detail="no artifacts found to validate")
            return self._finish(ctx, [check], {})

        checks: list[ValidationCheck] = []
        schema_fail: list[str] = []
        missing_prov: list[str] = []
        hash_mismatch: list[str] = []
        by_kind: dict[str, list] = {}
        n_checked = 0

        for a in all_refs:
            if not a.uri or a.kind not in ARTIFACT_MODELS:
                continue
            n_checked += 1
            raw = self.store.artifact_path(a.uri).read_text()
            if a.hash and hash_content(raw) != a.hash:
                hash_mismatch.append(f"{a.kind}:{a.artifact_id}")
            try:
                payload = ARTIFACT_MODELS[a.kind].model_validate_json(raw)
            except Exception as exc:  # noqa: BLE001
                schema_fail.append(f"{a.kind}:{a.artifact_id} ({type(exc).__name__})")
                continue
            if not getattr(payload, "provenance", None):
                missing_prov.append(f"{a.kind}:{a.artifact_id}")
            by_kind.setdefault(a.kind, []).append(payload)

        for a in own_prior_refs:
            if not a.uri or not a.hash:
                continue
            n_checked += 1
            try:
                raw = self.store.artifact_path(a.uri).read_text()
            except OSError:
                hash_mismatch.append(f"{a.kind}:{a.artifact_id} (missing)")
                continue
            if hash_content(raw) != a.hash:
                hash_mismatch.append(f"{a.kind}:{a.artifact_id}")

        checks.append(self._check("artifact_schema", "artifact schema validity",
                                  "fail" if schema_fail else "pass", _join(schema_fail)))
        checks.append(self._check("provenance_complete", "provenance completeness",
                                  "fail" if missing_prov else "pass",
                                  f"{len(missing_prov)} artifact(s) without provenance" if missing_prov else None))
        checks.append(self._check("hash_integrity", "artifact hash integrity",
                                  "fail" if hash_mismatch else "pass", _join(hash_mismatch)))
        checks.append(self._check("cross_agent_consistency", "cross-agent consistency",
                                  "fail" if (ci := self._cross_agent(by_kind)) else "pass", _join(ci)))
        checks.append(self._check("evidence_level_sanity", "evidence-level sanity",
                                  "fail" if (el := self._evidence_levels(by_kind)) else "pass", _join(el)))
        checks.append(self._check("numeric_sanity", "numeric sanity",
                                  "fail" if (ns := self._numeric(by_kind)) else "pass", _join(ns)))
        checks.append(self._check("report_breakdown", "report breakdown consistency",
                                  "fail" if (rb := self._report_breakdown(by_kind)) else "pass", _join(rb)))
        checks.append(self._check("mechanistic_constraints", "mechanistic-constraint sanity",
                                  "fail" if (mc := self._mechanistic_constraints(by_kind)) else "pass", _join(mc)))

        return self._finish(ctx, checks, {
            "n_artifacts_checked": n_checked,
            "n_missing_provenance": len(missing_prov),
            "n_hash_mismatch": len(hash_mismatch),
        }, by_kind)

    # ---- checks ----
    @staticmethod
    def _cross_agent(by_kind: dict[str, list]) -> list[str]:
        survivors = set()
        for sb in by_kind.get("screened_batch", []):
            for m in sb.molecules:
                if m.passed and m.canonical_smiles:
                    survivors.add(m.canonical_smiles)
        if not survivors:
            return []
        issues: list[str] = []
        for kind in ("retrosynthesis_plan", "tox_profile", "formulation_assessment"):
            if kind not in by_kind:
                continue
            got = set()
            for payload in by_kind[kind]:
                for m in payload.molecules:
                    got.add(m.canonical_smiles)
            missing = survivors - got
            if missing:
                issues.append(f"{kind} missing {len(missing)} screened survivor(s)")
        return issues

    @staticmethod
    def _evidence_levels(by_kind: dict[str, list]) -> list[str]:
        issues: list[str] = []
        for rp in by_kind.get("retrosynthesis_plan", []):
            for m in rp.molecules:
                if m.evidence_level.value != "calculated":
                    issues.append(f"retrosynth {m.id}: {m.evidence_level.value} (expected calculated)")
                for r in m.routes:
                    if r.evidence_level.value != "hypothetical":
                        issues.append(f"retrosynth route: {r.evidence_level.value} (expected hypothetical)")
        for tp in by_kind.get("tox_profile", []):
            for m in tp.molecules:
                for a in m.alerts:
                    if a.evidence_level.value != "predicted":
                        issues.append(f"tox alert {a.name}: {a.evidence_level.value} (expected predicted)")
        for fa in by_kind.get("formulation_assessment", []):
            for m in fa.molecules:
                if m.evidence_level.value != "predicted":
                    issues.append(f"formulation {m.id}: {m.evidence_level.value} (expected predicted)")
        for md in by_kind.get("dynamics_result", []):
            for r in md.runs:
                if r.evidence_level.value != "calculated":
                    issues.append(f"dynamics run: {r.evidence_level.value} (expected calculated)")
        return issues

    @staticmethod
    def _numeric(by_kind: dict[str, list]) -> list[str]:
        issues: list[str] = []
        for dr in by_kind.get("docking_result", []):
            for p in dr.poses:
                v = p.affinity_kcal_per_mol
                if v is not None and (not math.isfinite(v) or v > 0.0):
                    issues.append(f"docking affinity implausible: {v}")
            rv = dr.redocking_validation
            if rv and rv.rmsd_angstrom is not None and (not math.isfinite(rv.rmsd_angstrom) or rv.rmsd_angstrom < 0):
                issues.append(f"redocking RMSD implausible: {rv.rmsd_angstrom}")
        for rp in by_kind.get("retrosynthesis_plan", []):
            for m in rp.molecules:
                if not math.isfinite(m.sa_score) or not (1.0 <= m.sa_score <= 10.0):
                    issues.append(f"SA score out of range: {m.sa_score}")
        for md in by_kind.get("dynamics_result", []):
            for r in md.runs:
                if r.status == "success" and r.metrics.backbone_rmsd_mean_angstrom is not None:
                    v = r.metrics.backbone_rmsd_mean_angstrom
                    if not math.isfinite(v) or v < 0:
                        issues.append(f"backbone RMSD implausible: {v}")
        return issues

    @staticmethod
    def _report_breakdown(by_kind: dict[str, list]) -> list[str]:
        reports = by_kind.get("report", [])
        if not reports:
            return []
        recount = ReportMinion._evidence_counts(by_kind)
        actual = reports[0].evidence_breakdown.model_dump()
        return [f"{lvl}: report={actual.get(lvl, 0)} recount={count}"
                for lvl, count in recount.items() if actual.get(lvl, 0) != count]

    @staticmethod
    def _mechanistic_constraints(by_kind: dict[str, list]) -> list[str]:
        """Enforce the mechanistic-modeling principle: no prediction may violate
        mass balance, monotonicity, or go uncited without explicit justification.

        - PK/PD: the linear compartmental mass-balance identity
          AUC = Cmax * V_central / (CL * 0.06) must hold (weight-independent,
          follows from Cmax = F*D/(V*W) and AUC = F*D/(CL*W*0.06)); the systemic
          concentration curve must be monotonically non-increasing (no spurious
          re-absorption); Cmax and half-life must be positive.
        - ADMET: every endpoint labelled 'predicted' must carry a citable rule
          (a prediction without a mechanistic/rule citation is flagged).
        """
        issues: list[str] = []
        for cs in by_kind.get("clinical_simulation", []):
            for s in cs.simulations:
                p = s.parameters
                v_central = (p.volume_of_distribution_l_kg
                             if s.model == "one_compartment" else p.central_volume_l_kg)
                if (v_central and p.clearance_ml_min_kg
                        and s.cmax_mg_l is not None and s.auc_mg_h_l is not None):
                    expected = s.cmax_mg_l * v_central / (p.clearance_ml_min_kg * 0.06)
                    if expected > 0 and not math.isclose(s.auc_mg_h_l, expected, rel_tol=0.02, abs_tol=1e-6):
                        issues.append(
                            f"PK {s.molecule_id}: AUC {s.auc_mg_h_l:.4g} violates mass balance "
                            f"(expected {expected:.4g} from Cmax*V/CL)")
                if s.cmax_mg_l is not None and s.cmax_mg_l <= 0:
                    issues.append(f"PK {s.molecule_id}: Cmax <= 0")
                if s.half_life_hours is not None and s.half_life_hours <= 0:
                    issues.append(f"PK {s.molecule_id}: terminal half-life <= 0")
                curve = [c.concentration_mg_l for c in s.concentration_curve
                         if c.concentration_mg_l is not None]
                for i in range(1, len(curve)):
                    if curve[i] > curve[i - 1] + 1e-4:  # 1e-4 > 4-dp rounding noise
                        issues.append(
                            f"PK {s.molecule_id}: systemic concentration curve non-monotonic "
                            f"(re-absorption without justification)")
                        break
        for ap in by_kind.get("admet_prediction", []):
            for m in ap.molecules:
                for ep in m.endpoints:
                    if ep.prediction_status == "predicted" and not ep.citation:
                        issues.append(
                            f"ADMET {m.canonical_smiles}: {ep.endpoint} predicted "
                            f"without a citable rule (no mechanistic justification)")
        return issues

    # ---- plumbing ----
    @staticmethod
    def _check(check_id: str, name: str, status: str, detail: str | None) -> ValidationCheck:
        return ValidationCheck(check_id=check_id, name=name, status=status, detail=detail)

    def _finish(self, ctx: AgentContext, checks: list[ValidationCheck],
                reproducibility: dict, by_kind: dict | None = None) -> AgentResult:
        summary = ValidationSummary(
            total_checks=len(checks),
            passed=sum(1 for c in checks if c.status == "pass"),
            failed=sum(1 for c in checks if c.status == "fail"),
            warned=sum(1 for c in checks if c.status == "warn"),
        )
        status = AgentResultStatus.success if summary.failed == 0 else AgentResultStatus.partial
        failed = [c.name for c in checks if c.status == "fail"]

        input_hash = hash_jsonable({
            kind: [a.model_dump(mode="json", exclude={"provenance", "run_id"}) for a in arts]
            for kind, arts in sorted((by_kind or {}).items())
        })
        output_hash = hash_jsonable([c.model_dump() for c in checks])
        report = ValidationReport(
            project_id=ctx.project_id, run_id=ctx.run_id, status=status,
            checks=checks, summary=summary, reproducibility=reproducibility,
            provenance=[ProvenanceEntry(
                source="run validation (reproducibility + consistency audit)",
                tool_versions=[TOOL, RDKIT_TOOL],
                input_hash=input_hash, output_hash=output_hash,
            )],
        )
        art = self.persist_artifact(ctx, "validation_report", f"validation_{ctx.project_id}", report)

        findings = [f"validation: {summary.passed}/{summary.total_checks} checks passed"]
        if failed:
            findings.append("failed checks: " + "; ".join(failed))
        return AgentResult(
            project_id=ctx.project_id, agent=self.name, task_id=ctx.task_id,
            status=status, artifacts=[art], input_refs=ctx.input_refs, findings=findings,
            provenance=[ProvenanceEntry(source="run validation", output_hash=art.hash,
                                        tool_versions=[TOOL, RDKIT_TOOL])],
            tool_versions=[TOOL, RDKIT_TOOL],
            reproducibility={"run_id": ctx.run_id, **reproducibility},
        )


def _join(items: list[str]) -> str | None:
    if not items:
        return None
    shown = "; ".join(items[:5])
    return shown + ("; ..." if len(items) > 5 else "")
