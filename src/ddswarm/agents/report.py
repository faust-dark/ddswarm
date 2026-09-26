"""report_minion: aggregates a finished run into a structured, evidence-aware
Report and closes the feedback loop by writing `finding` + `feedback` entries
to project memory (read by researcher/designer in later runs).

Reads the run's task specs and artifacts directly from the store (not via
ctx.inputs), so it works for any task graph. Runs last; if an upstream leaf
failed, the orchestrator blocks it (a failed run produces no survivors, so
there is no feedback to record anyway)."""

from __future__ import annotations

from pathlib import Path

from ..contracts.admet_prediction import AdmetPrediction
from ..contracts.agent_result import AgentResult
from ..contracts.clinical_assessment import ClinicalAssessment
from ..contracts.clinical_simulation import ClinicalSimulation
from ..contracts.design_batch import DesignBatch
from ..contracts.docking_result import DockingResult
from ..contracts.dynamics_result import DynamicsResult
from ..contracts.enums import AgentResultStatus, EvidenceLevel
from ..contracts.formulation_assessment import FormulationAssessment
from ..contracts.knowledge_pack import KnowledgePack
from ..contracts.memory import MemoryEntry
from ..contracts.patent_landscape import PatentLandscape
from ..contracts.provenance import ProvenanceEntry, ToolVersion
from ..contracts.report import CandidateSelection, EvidenceBreakdown, Report, StageFinding
from ..contracts.regulatory_assessment import RegulatoryAssessment
from ..contracts.retrosynthesis_plan import RetrosynthesisPlan
from ..contracts.safety_assessment import SafetyAssessment
from ..contracts.screened_batch import ScreenedBatch
from ..contracts.tox_profile import ToxProfile
from ..logging.hashing import hash_jsonable
from ..chemistry.iupac import fetch_iupac_name
from ..orchestrator.memory import MemoryStore
from .base import Agent, AgentContext

TOOL = ToolVersion(name="ddswarm", version="0.1.0")

_KIND_MODELS: dict[str, type] = {
    "knowledge_pack": KnowledgePack,
    "design_batch": DesignBatch,
    "screened_batch": ScreenedBatch,
    "admet_prediction": AdmetPrediction,
    "docking_result": DockingResult,
    "dynamics_result": DynamicsResult,
    "formulation_assessment": FormulationAssessment,
    "retrosynthesis_plan": RetrosynthesisPlan,
    "tox_profile": ToxProfile,
    "safety_assessment": SafetyAssessment,
    "patent_landscape": PatentLandscape,
    "clinical_assessment": ClinicalAssessment,
    "clinical_simulation": ClinicalSimulation,
    "regulatory_assessment": RegulatoryAssessment,
}


class ReportMinion(Agent):
    name = "report_minion"

    def run(self, ctx: AgentContext) -> AgentResult:
        manifest = self.store.load_run_manifest(ctx.project_id, ctx.run_id)
        stage_findings: list[StageFinding] = []
        by_kind: dict[str, list] = {}
        for task_id in manifest.tasks:
            ts = self.store.load_task_spec(ctx.project_id, ctx.run_id, task_id)
            try:
                res = self.store.load_latest_agent_result(ctx.project_id, ctx.run_id, task_id)
            except FileNotFoundError:
                res = None
            kinds: list[str] = []
            if res:
                for a in res.artifacts:
                    if a.uri and a.kind in _KIND_MODELS:
                        kinds.append(a.kind)
                        by_kind.setdefault(a.kind, []).append(
                            _KIND_MODELS[a.kind].model_validate_json(Path(a.uri).read_text())
                        )
            stage_findings.append(StageFinding(
                task_id=task_id, agent=ts.agent, status=ts.status.value,
                findings=list(res.findings) if res else [],
                warnings=list(res.warnings) if res else [],
                artifact_kinds=kinds,
            ))

        # The run's final status is set *after* report (the last task) finishes,
        # so derive it from the other tasks' statuses, mirroring Runner._final_status.
        statuses = [sf.status for sf in stage_findings if sf.task_id != ctx.task_id]
        if any(s == "failed" for s in statuses):
            run_status = "failed"
        elif any(s in ("blocked", "insufficient_evidence") for s in statuses):
            run_status = "blocked"
        else:
            run_status = "completed"

        counts = self._evidence_counts(by_kind)
        exec_summary = self._executive_summary(by_kind)
        gaps = self._gaps(by_kind)
        next_steps = self._next_steps(gaps)
        caveats = self._caveats()
        ranked = self._rank_survivors(by_kind)
        # Opt-in IUPAC readability labels (network service; non-deterministic).
        iupac_enabled = bool(ctx.params.get("iupac"))
        if iupac_enabled and ranked:
            for c in ranked:
                c.iupac_name = fetch_iupac_name(c.smiles)
            caveats.append("IUPAC names were fetched from the NIH CACTUS web service — "
                           "readability-only, non-deterministic (network + third-party), "
                           "not authoritative.")
        identified = ranked[0] if ranked else None

        input_hash = hash_jsonable({
            kind: [a.model_dump(mode="json", exclude={"provenance", "run_id"}) for a in arts]
            for kind, arts in sorted(by_kind.items())
        })
        output_hash = hash_jsonable({
            "stage_findings": [s.model_dump() for s in stage_findings],
            "executive_summary": exec_summary, "gaps": gaps, "next_steps": next_steps,
            # iupac_name is a network-fetched readability label -> excluded so the
            # report hash stays deterministic across runs.
            "ranked_candidates": [c.model_dump(exclude={"iupac_name"}) for c in ranked],
        })

        report = Report(
            project_id=ctx.project_id, run_id=ctx.run_id,
            status=AgentResultStatus.success, run_status=run_status,
            executive_summary=exec_summary, stage_findings=stage_findings,
            evidence_breakdown=EvidenceBreakdown(**counts),
            identified_candidate=identified, ranked_candidates=ranked,
            gaps=gaps, next_steps=next_steps, caveats=caveats,
            provenance=[ProvenanceEntry(
                source="run aggregation", tool_versions=[TOOL],
                input_hash=input_hash, output_hash=output_hash,
            )],
        )
        art = self.persist_artifact(ctx, "report", f"report_{ctx.project_id}", report)

        # ---- close the loop: write finding + feedback to project memory ----
        memory = MemoryStore(self.store)
        finding_id = f"finding_{ctx.run_id}"
        feedback_id = f"feedback_{ctx.run_id}"
        memory.append(MemoryEntry(
            id=finding_id, project_id=ctx.project_id, kind="finding",
            content=exec_summary,
            data={"run_id": ctx.run_id, "run_status": run_status,
                  "evidence_breakdown": counts, "gaps": gaps},
            evidence_level=EvidenceLevel.inferred,
            source_agent=self.name, run_id=ctx.run_id,
            provenance=[ProvenanceEntry(source="run aggregation", tool_versions=[TOOL],
                                        input_hash=input_hash, output_hash=output_hash)],
        ))
        memory.append(MemoryEntry(
            id=feedback_id, project_id=ctx.project_id, kind="feedback",
            content="prior-cycle feedback for the next design cycle",
            data={"survivors": self._survivors(by_kind),
                  "prior_designs": self._prior_designs(by_kind), "gaps": gaps},
            evidence_level=EvidenceLevel.inferred,
            source_agent=self.name, run_id=ctx.run_id,
            provenance=[ProvenanceEntry(source="run aggregation", tool_versions=[TOOL],
                                        input_hash=input_hash, output_hash=output_hash)],
        ))

        return AgentResult(
            project_id=ctx.project_id, agent=self.name, task_id=ctx.task_id,
            status=AgentResultStatus.success, artifacts=[art], input_refs=ctx.input_refs,
            findings=[exec_summary],
            provenance=[ProvenanceEntry(source="run aggregation", output_hash=art.hash, tool_versions=[TOOL])],
            tool_versions=[TOOL],
            reproducibility={"run_id": ctx.run_id, "memory_finding_id": finding_id,
                            "memory_feedback_id": feedback_id},
        )

    # ---- aggregation helpers ----
    @staticmethod
    def _evidence_counts(by_kind: dict[str, list]) -> dict[str, int]:
        c = {lvl.value: 0 for lvl in EvidenceLevel}
        for kp in by_kind.get("knowledge_pack", []):
            for e in kp.entries:
                c[e.evidence_level.value] += 1
        for db in by_kind.get("design_batch", []):
            for m in db.molecules:
                c[m.evidence_level.value] += 1
        for sb in by_kind.get("screened_batch", []):
            c["calculated"] += sum(1 for m in sb.molecules if m.valid)
        for ap in by_kind.get("admet_prediction", []):
            for m in ap.molecules:
                c["predicted"] += sum(1 for ep in m.endpoints if ep.prediction_status == "predicted")
        for dr in by_kind.get("docking_result", []):
            c["calculated"] += len(dr.poses)
        for md in by_kind.get("dynamics_result", []):
            for r in md.runs:
                c[r.evidence_level.value] += 1
        for fa in by_kind.get("formulation_assessment", []):
            c["predicted"] += len(fa.molecules)
        for rp in by_kind.get("retrosynthesis_plan", []):
            for m in rp.molecules:
                c[m.evidence_level.value] += 1
                for d in m.disconnections:
                    c[d.evidence_level.value] += 1
                for route in m.routes:
                    c[route.evidence_level.value] += 1
        for tp in by_kind.get("tox_profile", []):
            for m in tp.molecules:
                for a in m.alerts:
                    c[a.evidence_level.value] += 1
        for sa in by_kind.get("safety_assessment", []):
            for f in sa.flags:
                c[f.evidence_level.value] += 1
        for pl in by_kind.get("patent_landscape", []):
            for e in pl.entries:
                c[e.evidence_level.value] += 1
        for ca in by_kind.get("clinical_assessment", []):
            for e in ca.entries:
                c[e.evidence_level.value] += 1
        for cs in by_kind.get("clinical_simulation", []):
            c["calculated"] += len(cs.simulations) + len(cs.pd_simulations)  # deterministic math
        for ra in by_kind.get("regulatory_assessment", []):
            for e in ra.entries:
                c[e.evidence_level.value] += 1
        return c

    @staticmethod
    def _executive_summary(by_kind: dict[str, list]) -> str:
        parts: list[str] = []
        n_entries = sum(len(k.entries) for k in by_kind.get("knowledge_pack", []))
        if n_entries:
            parts.append(f"{n_entries} knowledge entries")
        n_designs = sum(len(d.molecules) for d in by_kind.get("design_batch", []))
        if n_designs:
            parts.append(f"{n_designs} designs")
        sbs = by_kind.get("screened_batch", [])
        if sbs:
            parts.append(f"{sum(s.summary.passed for s in sbs)}/{sum(s.summary.total for s in sbs)} passed screening")
            brenk = sum(s.summary.brenk_flagged for s in sbs)
            if brenk:
                parts.append(f"{brenk} Brenk advisories")
        aps = by_kind.get("admet_prediction", [])
        if aps:
            parts.append(f"ADMET {sum(a.summary.predicted for a in aps)} predicted / "
                         f"{sum(a.summary.not_available for a in aps)} not_available")
        best = None
        for dr in by_kind.get("docking_result", []):
            for p in dr.poses:
                if p.affinity_kcal_per_mol is not None:
                    best = p.affinity_kcal_per_mol if best is None else min(best, p.affinity_kcal_per_mol)
        if best is not None:
            parts.append(f"best docking {best:.2f} kcal/mol")
        rps = by_kind.get("retrosynthesis_plan", [])
        if rps:
            parts.append(f"SA easy={sum(r.summary.easy for r in rps)} "
                         f"moderate={sum(r.summary.moderate for r in rps)} "
                         f"hard={sum(r.summary.hard for r in rps)}")
        tps = by_kind.get("tox_profile", [])
        if tps:
            parts.append(f"{sum(t.summary.total_alerts for t in tps)} tox alerts")
        fas = by_kind.get("formulation_assessment", [])
        if fas:
            bcs: dict[str, int] = {}
            for fa in fas:
                for m in fa.molecules:
                    bcs[m.bcs_class] = bcs.get(m.bcs_class, 0) + 1
            parts.append("formulation BCS " + ", ".join(f"{k}:{v}" for k, v in sorted(bcs.items())))
        for md in by_kind.get("dynamics_result", []):
            for r in md.runs:
                if r.status == "success" and r.metrics.backbone_rmsd_mean_angstrom is not None:
                    parts.append(f"receptor MD backbone RMSD {r.metrics.backbone_rmsd_mean_angstrom:.2f} A")
        sas = by_kind.get("safety_assessment", [])
        if sas:
            parts.append(f"safety {sas[0].verdict} ({sum(s.summary.total_flags for s in sas)} flags)")
        pls = by_kind.get("patent_landscape", [])
        if pls:
            parts.append(f"patent {sum(p.summary.total_entries for p in pls)} entries "
                         f"({sum(p.summary.flagged for p in pls)} covering)")
        cas = by_kind.get("clinical_assessment", [])
        if cas:
            phases: dict[str, int] = {}
            for ca in cas:
                for e in ca.entries:
                    phases[e.phase] = phases.get(e.phase, 0) + 1
            parts.append("clinical " + ", ".join(f"{k}:{v}" for k, v in sorted(phases.items())))
        css = by_kind.get("clinical_simulation", [])
        if css:
            sim = sum(c.summary.simulated for c in css)
            tot = sum(c.summary.total_molecules for c in css)
            pds = sum(c.summary.pd_simulated for c in css)
            t12s = [f"{s.molecule_id}: {s.model} t1/2 {s.half_life_hours}h"
                    for c in css for s in c.simulations[:2]]
            parts.append(f"PK/PD {sim}/{tot} PK simulated"
                         + (f" ({', '.join(t12s)})" if t12s else "")
                         + (f", {pds} PD" if pds else ""))
        ras = by_kind.get("regulatory_assessment", [])
        if ras:
            parts.append(f"regulatory {sum(r.summary.total_entries for r in ras)} findings")
        if not parts:
            return "no artifacts to summarize"
        return "Run summary: " + "; ".join(parts) + "."

    @staticmethod
    def _gaps(by_kind: dict[str, list]) -> list[str]:
        gaps: list[str] = []
        sbs = by_kind.get("screened_batch", [])
        if sbs and sum(s.summary.passed for s in sbs) == 0:
            gaps.append("0 molecules passed screening; reconsider filters or starting scaffold")
        aps = by_kind.get("admet_prediction", [])
        if aps and sum(a.summary.predicted for a in aps) == 0:
            gaps.append("no ADMET endpoints predicted (null provider configured)")
        if "docking_result" not in by_kind:
            gaps.append("no docking performed this run")
        if "dynamics_result" not in by_kind:
            gaps.append("no receptor MD performed this run")
        if "formulation_assessment" not in by_kind:
            gaps.append("no formulation assessment performed this run")
        if "patent_landscape" not in by_kind:
            gaps.append("no patent/FTO landscape (opt-in; needs reasoning.backend=hermes)")
        if "clinical_assessment" not in by_kind:
            gaps.append("no clinical assessment (opt-in; needs reasoning.backend=hermes)")
        if "clinical_simulation" not in by_kind:
            gaps.append("no PK/PD simulation (insufficient PK inputs: clearance+Vd via params or reasoning recall)")
        if "regulatory_assessment" not in by_kind:
            gaps.append("no regulatory assessment (opt-in; needs reasoning.backend=hermes)")
        rps = by_kind.get("retrosynthesis_plan", [])
        if rps and sum(r.summary.molecules_with_routes for r in rps) == 0:
            gaps.append("no synthesis routes proposed (no reasoning backend)")
        tps = by_kind.get("tox_profile", [])
        if tps and sum(t.summary.total_alerts for t in tps) == 0:
            gaps.append("no structural tox alerts (curated set is small; not a clean-toxicity declaration)")
        if "knowledge_pack" not in by_kind:
            gaps.append("no research performed")
        return gaps

    @staticmethod
    def _next_steps(gaps: list[str]) -> list[str]:
        steps: list[str] = []
        if any("ADMET" in g for g in gaps):
            steps.append("configure a real ADMET provider (params.providers) for actual predictions")
        if any("docking" in g for g in gaps):
            steps.append("add a docking task with a receptor (pdb_id) to score poses")
        if any("receptor MD" in g for g in gaps):
            steps.append("add a dynamics task with params.receptor + engine=openmm to assess receptor stability")
        if any("0 molecules passed" in g for g in gaps):
            steps.append("reconsider screening filters or the starting scaffold")
        if any("routes" in g for g in gaps):
            steps.append("enable reasoning.backend=hermes to propose synthesis routes and analogs")
        if any("patent" in g or "clinical" in g or "regulatory" in g for g in gaps):
            steps.append("enable reasoning.backend=hermes and add patent/clinical/regulatory tasks for a knowledge landscape")
        if any("clean-toxicity" in g for g in gaps):
            steps.append("extend the toxicophore library with more sourced alerts")
        if not steps:
            steps.append("validate top candidates experimentally (in-vitro/in-vivo) before progressing")
        return steps

    @staticmethod
    def _caveats() -> list[str]:
        return [
            "All scores are in-silico (calculated/predicted); none are experimental evidence.",
            "Evidence breakdown is a coarse rollup; individual entries carry finer-grained evidence_level.",
            "Structural tox alerts are predicted liabilities from literature rules, not observed toxicity.",
            "Brenk 'unwanted functionality' alerts are advisory, not eliminating (PAINS stays hard); set filters.brenk_hard=true to make Brenk eliminate.",
            "LLM-contributed entries are inferred/hypothetical; verify against primary sources.",
            "Docking affinity and redocking RMSD are calculated; a failed redocking (RMSD > 2 A) is reported as failed.",
            "Receptor MD is a short (ps-scale) relaxation snapshot, not production MD; backbone RMSD is not a converged stability verdict.",
            "Formulation classes (solubility/permeability/BCS) are rule-based estimates from logP/TPSA, not measured properties.",
            "Patent/clinical/regulatory entries are LLM-recalled and inferred — NOT a legal freedom-to-operate opinion or regulatory determination.",
            "PK/PD simulations are compartmental models (one- or two-compartment) on predicted/inferred parameters — not measured human pharmacokinetics.",
        ]

    @staticmethod
    def _survivors(by_kind: dict[str, list]) -> list[str]:
        out: list[str] = []
        for sb in by_kind.get("screened_batch", []):
            for m in sb.molecules:
                if m.passed and m.canonical_smiles and m.canonical_smiles not in out:
                    out.append(m.canonical_smiles)
        return out

    @staticmethod
    def _prior_designs(by_kind: dict[str, list]) -> list[str]:
        out: list[str] = []
        for db in by_kind.get("design_batch", []):
            for m in db.molecules:
                if m.smiles and m.smiles not in out:
                    out.append(m.smiles)
        return out

    @staticmethod
    def _rank_survivors(by_kind: dict[str, list]) -> list[CandidateSelection]:
        """Rank screening survivors by a deterministic composite score over
        their computed/predicted properties:

            score = QED + BCS_bonus - 0.1*SA - 0.2*tox_alerts

        BCS bonus: I=1.0, II/III=0.5, else 0.0. Missing properties use neutral
        defaults (QED 0.0, SA 5.0, tox 0, BCS unknown). Ties broken by canonical
        SMILES (lexicographic) so the ordering is always deterministic. The score
        is a ranking aid, not experimental merit."""
        survivors = ReportMinion._survivors(by_kind)
        if not survivors:
            return []

        qed: dict[str, float] = {}
        for ap in by_kind.get("admet_prediction", []):
            for m in ap.molecules:
                for ep in m.endpoints:
                    if (ep.endpoint == "QED" and ep.prediction_status == "predicted"
                            and ep.predicted_value is not None):
                        qed[m.canonical_smiles] = ep.predicted_value
        sa: dict[str, float] = {}
        for rp in by_kind.get("retrosynthesis_plan", []):
            for m in rp.molecules:
                sa[m.canonical_smiles] = m.sa_score
        tox: dict[str, int] = {}
        for tp in by_kind.get("tox_profile", []):
            for m in tp.molecules:
                tox[m.canonical_smiles] = len(m.alerts)
        bcs: dict[str, str] = {}
        for fa in by_kind.get("formulation_assessment", []):
            for m in fa.molecules:
                bcs[m.canonical_smiles] = m.bcs_class

        def bonus(b: str) -> float:
            return 1.0 if b == "I" else (0.5 if b in ("II", "III") else 0.0)

        scored: list[tuple[str, float, dict[str, str]]] = []
        for smi in survivors:
            q = qed.get(smi, 0.0)
            s = sa.get(smi, 5.0)
            t = tox.get(smi, 0)
            b = bcs.get(smi, "unknown")
            score = q + bonus(b) - 0.1 * s - 0.2 * t
            crit = {"QED": f"{q:.2f}", "SA": f"{s:.2f}", "tox alerts": str(t), "BCS": b}
            scored.append((smi, score, crit))

        scored.sort(key=lambda x: (-x[1], x[0]))
        out: list[CandidateSelection] = []
        for rank, (smi, score, crit) in enumerate(scored, 1):
            out.append(CandidateSelection(
                rank=rank, smiles=smi, score=round(score, 4),
                criteria=[f"{k}: {v}" for k, v in crit.items()],
                rationale="composite = QED + BCS bonus - 0.1*SA - 0.2*tox alerts",
            ))
        return out
