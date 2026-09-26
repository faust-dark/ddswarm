"""clinical_minion: clinical-development knowledge AND PK/PD simulation.

Two independent outputs, per the swarm workflow schema (7C):

1. ClinicalAssessment — clinical-development knowledge (known drugs, trial
   phase, indication, status) for the target(s)/candidates, delegated to a
   reasoning backend (Hermes subagent). No deterministic offline source: without
   reasoning.backend=hermes this part yields no entries. LLM recall, forced
   evidence_level='inferred'.

2. ClinicalSimulation — a deterministic one-compartment PK/PD simulation for the
   candidate molecules (evidence_level='calculated' math). It runs ONLY when
   sufficient inputs exist (clearance + Vd, or t1/2 + one of them), supplied
   explicitly via params or recalled by the reasoning backend (`pk_parameters`).
   The simulation is a model, NOT measured human pharmacokinetics.

The agent succeeds if EITHER output is produced; both are absent -> insufficient.
"""

from __future__ import annotations

from ..chemistry.pkpd import resolve_cl_vd, simulate_one_compartment
from ..contracts.agent_result import AgentResult
from ..contracts.clinical_assessment import ClinicalAssessment, ClinicalFinding, ClinicalSummary
from ..contracts.clinical_simulation import (
    ClinicalSimulation,
    ClinicalSimulationSummary,
    ConcentrationPoint,
    PKParameters,
    PKSimulation,
)
from ..contracts.enums import AgentResultStatus
from ..contracts.provenance import ProvenanceEntry, ToolVersion
from ..logging.hashing import hash_jsonable
from ..reasoning import make_reasoning_backend
from .base import Agent, AgentContext

TOOL = ToolVersion(name="ddswarm", version="0.1.0")

_DEFAULT_DOSE_MG = 10.0


class ClinicalMinion(Agent):
    name = "clinical_minion"

    def run(self, ctx: AgentContext) -> AgentResult:
        molecules = self._collect_molecules(ctx)
        warnings: list[str] = []
        input_hash = hash_jsonable({"targets": ctx.spec.targets, "molecules": list(molecules)})
        backend_name = ctx.params.get("reasoning_backend", "none")

        artifacts = []
        findings: list[str] = []
        provenance: list[ProvenanceEntry] = []

        # ---- 1. clinical knowledge (ClinicalAssessment) ----
        if backend_name in (None, "", "none"):
            warnings.append("clinical knowledge requires reasoning.backend=hermes")
        else:
            backend, note = self._resolve_backend(ctx, warnings)
            if backend is None:
                warnings.append(note)
            else:
                ca = self._clinical_assessment(ctx, backend, molecules, backend_name, warnings)
                if ca is not None:
                    artifacts.append(ca[0])
                    findings.append(ca[1])
                    provenance.append(ca[2])

        # ---- 2. PK/PD simulation (ClinicalSimulation) ----
        sim = self._pk_simulation(ctx, molecules, backend_name, warnings)
        if sim is not None:
            artifacts.append(sim[0])
            findings.append(sim[1])
            provenance.append(sim[2])

        if not artifacts:
            reason = "; ".join(warnings) or "no clinical knowledge and no PK/PD inputs"
            return self._insufficient(ctx, input_hash, reason)

        return AgentResult(
            project_id=ctx.project_id, agent=self.name, task_id=ctx.task_id,
            status=AgentResultStatus.success, artifacts=artifacts, input_refs=ctx.input_refs,
            findings=findings, warnings=warnings,
            provenance=provenance,
            tool_versions=[TOOL], reproducibility={"input_hash": input_hash},
        )

    # ---- clinical knowledge ----
    def _clinical_assessment(self, ctx, backend, molecules, backend_name, warnings):
        res = backend.clinical_assessment(ctx.spec, molecules, ctx.params)
        if not res.ok:
            warnings.append(f"clinical assessment reasoning failed: {res.note}")
            return None
        entries: list[ClinicalFinding] = []
        for i, raw in enumerate(res.entries):
            try:
                entries.append(self._from_reasoning(raw, i))
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"clinical entry {i} invalid, skipped: {exc}")
        if not entries:
            warnings.append("reasoning returned no valid clinical entries")
            return None
        phase_counts: dict[str, int] = {}
        for e in entries:
            phase_counts[e.phase] = phase_counts.get(e.phase, 0) + 1
        summary = ClinicalSummary(total_entries=len(entries), phase_counts=phase_counts)
        payload = ClinicalAssessment(
            project_id=ctx.project_id, status=AgentResultStatus.success,
            entries=entries, summary=summary,
            provenance=[ProvenanceEntry(
                source="Hermes subagent (LLM reasoning)", tool_versions=[TOOL],
                input_hash=hash_jsonable({"targets": ctx.spec.targets, "molecules": len(molecules)}),
                output_hash=hash_jsonable([e.model_dump() for e in entries]),
                params={"reasoning_backend": backend_name, "molecules": len(molecules)},
            )],
        )
        art = self.persist_artifact(ctx, "clinical_assessment", f"ca_{ctx.project_id}", payload)
        finding = (f"clinical: {summary.total_entries} findings; "
                   f"phases {phase_counts or 'n/a'} (all inferred)")
        prov = ProvenanceEntry(source="clinical assessment", output_hash=art.hash, tool_versions=[TOOL])
        return art, finding, prov

    # ---- PK/PD simulation ----
    def _pk_simulation(self, ctx, molecules, backend_name, warnings):
        dose_mg = ctx.params.get("dose_mg") or ctx.spec.dose_mg or _DEFAULT_DOSE_MG
        route = ctx.spec.route_of_administration or ctx.params.get("route", "oral")

        # explicit per-molecule params win; reasoning recall fills the rest
        pk_map: dict[str, dict] = {}
        explicit_ids: set[str] = set()
        for p in ctx.params.get("pk_parameters", []) or []:
            if isinstance(p, dict) and p.get("molecule_id"):
                pk_map[p["molecule_id"]] = p
                explicit_ids.add(p["molecule_id"])

        if backend_name not in (None, "", "none"):
            try:
                backend, _ = self._resolve_backend(ctx, [])
            except Exception:  # noqa: BLE001
                backend = None
            if backend is not None:
                try:
                    res = backend.pk_parameters(ctx.spec, molecules, ctx.params)
                    if res.ok:
                        for raw in res.entries:
                            if isinstance(raw, dict) and raw.get("molecule_id"):
                                pk_map.setdefault(raw["molecule_id"], raw)
                    else:
                        warnings.append(f"pk_parameters recall failed: {res.note}")
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"pk_parameters recall failed: {exc}")

        simulations: list[PKSimulation] = []
        insufficient = 0
        for mid, smi in molecules:
            p = pk_map.get(mid) or {}
            resolved = resolve_cl_vd(
                clearance_ml_min_kg=p.get("clearance_ml_min_kg"),
                volume_of_distribution_l_kg=p.get("volume_of_distribution_l_kg"),
                half_life_hours=p.get("half_life_hours"),
            )
            if resolved is None:
                insufficient += 1
                continue
            cl, vd = resolved
            f = p.get("bioavailability")
            f = float(f) if f is not None else 1.0
            try:
                r = simulate_one_compartment(
                    dose_mg=dose_mg, clearance_ml_min_kg=cl,
                    volume_of_distribution_l_kg=vd, bioavailability=f, route=route)
            except ValueError as exc:
                warnings.append(f"PK simulation skipped for {mid}: {exc}")
                insufficient += 1
                continue
            source = "user" if mid in explicit_ids else "inferred"
            simulations.append(PKSimulation(
                molecule_id=mid, canonical_smiles=smi, dose_mg=dose_mg, route=route,
                parameters=PKParameters(
                    clearance_ml_min_kg=cl, volume_of_distribution_l_kg=vd,
                    bioavailability=f, half_life_hours=round(r.half_life_hours, 3),
                    source=source, source_ref=p.get("source")),
                elimination_rate_per_hour=round(r.elimination_rate_per_hour, 5),
                half_life_hours=round(r.half_life_hours, 3),
                cmax_mg_l=round(r.cmax_mg_l, 4),
                auc_mg_h_l=round(r.auc_mg_h_l, 4),
                concentration_curve=[ConcentrationPoint(time_hours=t, concentration_mg_l=round(c, 4))
                                     for t, c in r.curve],
                note=("one-compartment model, instant absorption; parameters "
                      + ("user-supplied" if source == "user" else "LLM-recalled (inferred)")),
            ))

        if not simulations:
            warnings.append(
                f"PK/PD simulation: insufficient inputs for all {len(molecules)} molecule(s) "
                "(need clearance + Vd, or t1/2 + one of them)")
            return None

        summary = ClinicalSimulationSummary(
            total_molecules=len(molecules), simulated=len(simulations),
            insufficient_inputs=insufficient)
        payload = ClinicalSimulation(
            project_id=ctx.project_id, status=AgentResultStatus.success,
            simulations=simulations, summary=summary,
            note="deterministic one-compartment PK; parameters predicted/inferred, NOT measured human data",
            provenance=[ProvenanceEntry(
                source="clinical PK/PD simulation", tool_versions=[TOOL],
                input_hash=hash_jsonable({"dose_mg": dose_mg, "route": route, "pk_map": pk_map}),
                output_hash=hash_jsonable([s.model_dump() for s in simulations]),
                params={"dose_mg": dose_mg, "route": route, "molecules": len(molecules)},
            )],
        )
        art = self.persist_artifact(ctx, "clinical_simulation", f"cs_{ctx.project_id}", payload)
        t12s = ", ".join(f"{s.molecule_id}: t1/2 {s.half_life_hours}h" for s in simulations[:3])
        finding = (f"pkpd: {summary.simulated}/{summary.total_molecules} simulated "
                   f"(one-compartment; {t12s}); parameters predicted/inferred")
        prov = ProvenanceEntry(source="clinical PK/PD simulation", output_hash=art.hash, tool_versions=[TOOL])
        return art, finding, prov

    def _insufficient(self, ctx: AgentContext, input_hash: str, message: str) -> AgentResult:
        return AgentResult(
            project_id=ctx.project_id, agent=self.name, task_id=ctx.task_id,
            status=AgentResultStatus.insufficient_evidence, warnings=[message],
            provenance=[ProvenanceEntry(source="clinical assessment", input_hash=input_hash)],
            tool_versions=[TOOL], reproducibility={"reason": "requires_reasoning_backend_or_pk_inputs"},
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

    @staticmethod
    def _resolve_backend(ctx: AgentContext, warnings: list[str]):
        name = ctx.params.get("reasoning_backend", "none")
        try:
            backend = make_reasoning_backend(
                name,
                command=ctx.params.get("reasoning_command"),
                max_turns=ctx.params.get("reasoning_max_turns", 30),
                run_budget=ctx.params.get("reasoning_run_budget", 300),
            )
        except KeyError as exc:
            return None, str(exc)
        avail, msg = backend.available()
        if not avail:
            return None, f"reasoning backend '{name}' unavailable: {msg}"
        return backend, ""

    @staticmethod
    def _from_reasoning(raw: dict, i: int) -> ClinicalFinding:
        data = dict(raw)
        data.setdefault("id", f"clinical_{i}")
        data["evidence_level"] = "inferred"  # LLM recall is not observed data
        data["source"] = data.get("source") or "Hermes subagent (LLM reasoning)"
        notes = data.get("notes") or ""
        if "verify against primary source" not in notes:
            data["notes"] = (notes + " | LLM-recalled via reasoning backend; verify against primary source").strip(" |")
        return ClinicalFinding.model_validate(data)
