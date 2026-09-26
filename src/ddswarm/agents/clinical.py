"""clinical_minion: clinical-development knowledge AND PK/PD simulation.

Two independent outputs, per the swarm workflow schema (7C):

1. ClinicalAssessment — clinical-development knowledge (known drugs, trial
   phase, indication, status), delegated to a reasoning backend (Hermes
   subagent). No deterministic offline source: without reasoning.backend=hermes
   this part yields no entries. LLM recall, forced evidence_level='inferred'.

2. ClinicalSimulation — deterministic PK/PD models for the candidate molecules
   (evidence_level='calculated' math). PK complexity (one- vs two-compartment)
   and PD model (Emax/Hill) are selected on a two-level basis:
     mechanism_hint — an explicit `params.pk_model`/`params.pd_model`, or a
       documented keyword match against mechanism_of_action/adme_issue;
     data_driven — goodness-of-fit/identifiability/predictive-performance
       (AIC/BIC/RSS) against supplied observed data;
   otherwise the model is a labelled `default` (never claimed as validated).

   The simulation is a model, NOT measured human pharmacokinetics; it runs only
   when sufficient inputs exist.

The agent succeeds if EITHER output is produced; both absent -> insufficient.
"""

from __future__ import annotations

import math

from ..chemistry.pkpd import (
    emax_effect,
    fit_metrics,
    hill_effect,
    resolve_cl_vd,
    simulate_one_compartment,
    simulate_two_compartment,
)
from ..contracts.agent_result import AgentResult
from ..contracts.clinical_assessment import ClinicalAssessment, ClinicalFinding, ClinicalSummary
from ..contracts.clinical_simulation import (
    ClinicalSimulation,
    ClinicalSimulationSummary,
    ConcentrationPoint,
    EffectPoint,
    ModelSelectionMetrics,
    PDParameters,
    PDSimulation,
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

# documented keyword -> model-kind hints (deterministic, overridable by params)
_PK_HINTS = {
    "two_compartment": ("distribution", "tissue", "peripheral", "biphasic",
                        "multi-compartment", "multicompartment", "large volume"),
}
_PD_HINTS = {
    "emax": ("agonist", "direct effect", "direct", "inhibitor", "antagonist"),
    "hill": ("sigmoid", "cooperative", "hill"),
    "indirect_response": ("turnover", "indirect"),
    "effect_compartment": ("delay", "delayed", "hysteresis", "effect compartment"),
    "receptor_occupancy": ("occupancy", "target engagement", "receptor occupancy"),
}
# implemented in this increment; the rest are deferred to later increments
_IMPLEMENTED_PD = {"emax", "hill"}


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

        pk_map, explicit_ids = self._gather_pk_map(ctx, molecules, backend_name, warnings)
        pd_map, pd_explicit_ids = self._gather_pd_params(ctx, warnings)
        observed_conc = self._gather_observed(ctx, "observed_concentrations", "concentration_mg_l")
        observed_eff = self._gather_observed(ctx, "observed_effects", "effect")

        simulations: list[PKSimulation] = []
        pd_simulations: list[PDSimulation] = []
        insufficient = 0

        for mid, smi in molecules:
            p = pk_map.get(mid) or {}
            kind, reason, metrics, sim_info = self._select_pk(
                ctx, p, mid, observed_conc.get(mid), dose_mg, route, warnings)
            if kind is None:
                insufficient += 1
                continue
            sim = self._build_pk_simulation(mid, smi, kind, reason, metrics, sim_info, p,
                                            dose_mg, route, explicit_ids)
            simulations.append(sim)

            pd_sim = self._build_pd_simulation(
                ctx, mid, smi, sim.concentration_curve, sim_info.get("fn"),
                pd_map.get(mid), observed_eff.get(mid), pd_explicit_ids, warnings)
            if pd_sim is not None:
                pd_simulations.append(pd_sim)

        if not simulations:
            warnings.append(
                f"PK/PD simulation: insufficient inputs for all {len(molecules)} molecule(s) "
                "(need clearance + Vd, or t1/2 + one of them)")
            return None

        summary = ClinicalSimulationSummary(
            total_molecules=len(molecules), simulated=len(simulations),
            insufficient_inputs=insufficient, pd_simulated=len(pd_simulations))
        payload = ClinicalSimulation(
            project_id=ctx.project_id, status=AgentResultStatus.success,
            simulations=simulations, pd_simulations=pd_simulations, summary=summary,
            note="deterministic PK/PD models; parameters predicted/inferred, NOT measured human data",
            provenance=[ProvenanceEntry(
                source="clinical PK/PD simulation", tool_versions=[TOOL],
                input_hash=hash_jsonable({"dose_mg": dose_mg, "route": route,
                                          "pk_map": pk_map, "pd_map": pd_map}),
                output_hash=hash_jsonable(
                    [s.model_dump() for s in simulations]
                    + [s.model_dump() for s in pd_simulations]),
                params={"dose_mg": dose_mg, "route": route, "molecules": len(molecules)},
            )],
        )
        art = self.persist_artifact(ctx, "clinical_simulation", f"cs_{ctx.project_id}", payload)
        t12s = ", ".join(f"{s.molecule_id}: {s.model} t1/2 {s.half_life_hours}h"
                         for s in simulations[:3])
        finding = (f"pkpd: {summary.simulated}/{summary.total_molecules} PK simulated "
                   f"({t12s}), {summary.pd_simulated} PD; parameters predicted/inferred")
        prov = ProvenanceEntry(source="clinical PK/PD simulation", output_hash=art.hash,
                               tool_versions=[TOOL])
        return art, finding, prov

    def _gather_pk_map(self, ctx, molecules, backend_name, warnings):
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
        return pk_map, explicit_ids

    def _gather_pd_params(self, ctx, warnings):
        pd_map: dict[str, dict] = {}
        explicit_ids: set[str] = set()
        for p in ctx.params.get("pd_parameters", []) or []:
            if isinstance(p, dict) and p.get("molecule_id"):
                pd_map[p["molecule_id"]] = p
                explicit_ids.add(p["molecule_id"])
        return pd_map, explicit_ids

    @staticmethod
    def _gather_observed(ctx, key: str, value_key: str) -> dict[str, list[tuple[float, float]]]:
        out: dict[str, list[tuple[float, float]]] = {}
        for row in ctx.params.get(key, []) or []:
            if not isinstance(row, dict):
                continue
            mid = row.get("molecule_id")
            t = row.get("time_hours")
            v = row.get(value_key)
            if mid is None or t is None or v is None:
                continue
            out.setdefault(str(mid), []).append((float(t), float(v)))
        return out

    def _select_pk(self, ctx, p, mid, observed, dose_mg, route, warnings):
        """Return (kind, reason, metrics, sim_info) or (None, None, None, None)."""
        explicit = ctx.params.get("pk_model")
        hint = self._keyword_hint(ctx.spec, _PK_HINTS)
        requested = explicit or hint
        one = self._one_compartment_result(p, dose_mg, route)
        two = self._two_compartment_result(p, dose_mg, route)

        if requested == "two_compartment":
            if two is None:
                warnings.append(f"{mid}: pk_model=two_compartment but V1/Q/V2 unavailable; "
                                "using available model")
            else:
                return "two_compartment", "mechanism_hint", None, two
        elif requested == "one_compartment":
            if one is not None:
                return "one_compartment", "mechanism_hint", None, one

        # data-driven: goodness-of-fit / identifiability / predictive-performance
        if observed and one is not None and two is not None:
            m1 = fit_metrics(observed, one["fn"], one["n_params"])
            m2 = fit_metrics(observed, two["fn"], two["n_params"])
            if self._better_fit(m2, m1):
                return "two_compartment", "data_driven", m2, two
            return "one_compartment", "data_driven", m1, one

        if one is not None:
            return "one_compartment", "default", None, one
        if two is not None:
            return "two_compartment", "default", None, two
        return None, None, None, None

    @staticmethod
    def _one_compartment_result(p, dose_mg, route):
        resolved = resolve_cl_vd(
            clearance_ml_min_kg=p.get("clearance_ml_min_kg"),
            volume_of_distribution_l_kg=p.get("volume_of_distribution_l_kg"),
            half_life_hours=p.get("half_life_hours"),
        )
        if resolved is None:
            return None
        cl, vd = resolved
        f = float(p.get("bioavailability")) if p.get("bioavailability") is not None else 1.0
        try:
            r = simulate_one_compartment(dose_mg=dose_mg, clearance_ml_min_kg=cl,
                                         volume_of_distribution_l_kg=vd,
                                         bioavailability=f, route=route)
        except ValueError:
            return None
        ke = r.elimination_rate_per_hour
        cmax = r.cmax_mg_l
        fn = lambda t: cmax * math.exp(-ke * t)  # noqa: E731
        return {"result": r, "cl": cl, "vd": vd, "f": f, "fn": fn, "n_params": 2}

    @staticmethod
    def _two_compartment_result(p, dose_mg, route):
        v1 = p.get("central_volume_l_kg")
        q = p.get("intercompartmental_clearance_ml_min_kg")
        v2 = p.get("peripheral_volume_l_kg")
        cl = p.get("clearance_ml_min_kg")
        if v1 is None or q is None or v2 is None or cl is None:
            return None
        f = float(p.get("bioavailability")) if p.get("bioavailability") is not None else 1.0
        try:
            r = simulate_two_compartment(dose_mg=dose_mg, v1_l_kg=v1,
                                         clearance_ml_min_kg=cl,
                                         intercompartmental_clearance_ml_min_kg=q,
                                         v2_l_kg=v2, bioavailability=f, route=route)
        except ValueError:
            return None
        a, b = r.a_coefficient, r.b_coefficient
        alpha, beta = r.alpha_per_hour, r.beta_per_hour
        fn = lambda t: a * math.exp(-alpha * t) + b * math.exp(-beta * t)  # noqa: E731
        return {"result": r, "cl": cl, "v1": v1, "q": q, "v2": v2, "f": f,
                "fn": fn, "n_params": 4}

    @staticmethod
    def _keyword_hint(spec, hints) -> str | None:
        text = " ".join(filter(None, [spec.mechanism_of_action, spec.adme_issue])).lower()
        for kind, keywords in hints.items():
            if any(kw in text for kw in keywords):
                return kind
        return None

    @staticmethod
    def _better_fit(m1: dict, m2: dict) -> bool:
        """True if model 1 fits strictly better than model 2.

        Ordering: perfect fit (RSS=0) wins; then lower BIC; then lower RSS.
        Ties on perfect fit prefer fewer parameters (parsimony).
        """
        r1 = m1.get("residual_sum_of_squares")
        r2 = m2.get("residual_sum_of_squares")
        if r1 is None or r2 is None:
            return False
        if r1 == 0.0 and r2 == 0.0:
            return m1["n_parameters"] < m2["n_parameters"]
        if r1 == 0.0:
            return True
        if r2 == 0.0:
            return False
        b1, b2 = m1.get("bic"), m2.get("bic")
        if b1 is not None and b2 is not None and b1 != b2:
            return b1 < b2
        return r1 < r2

    def _build_pk_simulation(self, mid, smi, kind, reason, metrics, sim_info, p,
                             dose_mg, route, explicit_ids):
        source = "user" if mid in explicit_ids else "inferred"
        sel_metrics = self._metrics(metrics)
        r = sim_info["result"]
        if kind == "two_compartment":
            return PKSimulation(
                molecule_id=mid, canonical_smiles=smi, dose_mg=dose_mg, route=route,
                model="two_compartment",
                parameters=PKParameters(
                    clearance_ml_min_kg=sim_info["cl"], central_volume_l_kg=sim_info["v1"],
                    intercompartmental_clearance_ml_min_kg=sim_info["q"],
                    peripheral_volume_l_kg=sim_info["v2"], bioavailability=sim_info["f"],
                    half_life_hours=round(r.terminal_half_life_hours, 3),
                    source=source, source_ref=p.get("source")),
                elimination_rate_per_hour=round(r.beta_per_hour, 5),
                half_life_hours=round(r.terminal_half_life_hours, 3),
                alpha_per_hour=round(r.alpha_per_hour, 5),
                distribution_half_life_hours=round(r.distribution_half_life_hours, 3),
                cmax_mg_l=round(r.cmax_mg_l, 4), auc_mg_h_l=round(r.auc_mg_h_l, 4),
                concentration_curve=[ConcentrationPoint(time_hours=t, concentration_mg_l=round(c, 4))
                                     for t, c in r.curve],
                selection_reason=reason, selection_metrics=sel_metrics,
                note=("two-compartment (biexponential) model; parameters "
                      + ("user-supplied" if source == "user" else "LLM-recalled (inferred)")),
            )
        return PKSimulation(
            molecule_id=mid, canonical_smiles=smi, dose_mg=dose_mg, route=route,
            model="one_compartment",
            parameters=PKParameters(
                clearance_ml_min_kg=sim_info["cl"], volume_of_distribution_l_kg=sim_info["vd"],
                bioavailability=sim_info["f"], half_life_hours=round(r.half_life_hours, 3),
                source=source, source_ref=p.get("source")),
            elimination_rate_per_hour=round(r.elimination_rate_per_hour, 5),
            half_life_hours=round(r.half_life_hours, 3),
            cmax_mg_l=round(r.cmax_mg_l, 4), auc_mg_h_l=round(r.auc_mg_h_l, 4),
            concentration_curve=[ConcentrationPoint(time_hours=t, concentration_mg_l=round(c, 4))
                                 for t, c in r.curve],
            selection_reason=reason, selection_metrics=sel_metrics,
            note=("one-compartment model; parameters "
                  + ("user-supplied" if source == "user" else "LLM-recalled (inferred)")),
        )

    def _build_pd_simulation(self, ctx, mid, smi, pk_curve, pk_fn, p, observed_eff,
                             pd_explicit_ids, warnings):
        if not p:
            return None
        e0 = p.get("baseline_effect")
        emax = p.get("max_effect")
        ec50 = p.get("ec50_mg_l")
        hill = p.get("hill_coefficient")
        if emax is None or ec50 is None:
            return None  # insufficient PD inputs; PD is additive, so skip silently
        emax = float(emax)
        ec50 = float(ec50)
        e0 = float(e0) if e0 is not None else 0.0

        explicit = ctx.params.get("pd_model")
        hint = self._keyword_hint(ctx.spec, _PD_HINTS)
        requested = explicit or hint
        reason = "mechanism_hint" if requested else "default"
        if requested not in _IMPLEMENTED_PD:
            if requested in _PD_HINTS:
                warnings.append(f"{mid}: pd_model={requested} deferred to a later increment")
            requested = None
        if requested == "hill":
            if hill is None or hill <= 0:
                warnings.append(f"{mid}: pd_model=hill requires hill_coefficient; using emax")
                requested = "emax"
            else:
                hill = float(hill)
        if requested is None:
            requested = "emax"

        metrics = None
        # data-driven: emax vs hill (complexity: Hill adds one parameter)
        if (observed_eff and pk_fn is not None and hill is not None and hill > 0
                and emax is not None and ec50 is not None):
            m_emax = fit_metrics(observed_eff, lambda t: emax_effect(pk_fn(t), e0, emax, ec50), 3)
            m_hill = fit_metrics(observed_eff, lambda t: hill_effect(pk_fn(t), e0, emax, ec50, hill), 4)
            if self._better_fit(m_hill, m_emax):
                requested, reason, metrics = "hill", "data_driven", m_hill
            else:
                requested, reason, metrics = "emax", "data_driven", m_emax

        curve: list[EffectPoint] = []
        try:
            for cp in pk_curve:
                c = cp.concentration_mg_l
                if requested == "hill":
                    e = hill_effect(c, e0, emax, ec50, hill)
                else:
                    e = emax_effect(c, e0, emax, ec50)
                curve.append(EffectPoint(time_hours=cp.time_hours, effect=round(e, 4)))
        except ValueError as exc:
            warnings.append(f"PD simulation skipped for {mid}: {exc}")
            return None

        source = "user" if mid in pd_explicit_ids else "inferred"
        return PDSimulation(
            molecule_id=mid, canonical_smiles=smi, model=requested,
            parameters=PDParameters(baseline_effect=e0, max_effect=emax, ec50_mg_l=ec50,
                                    hill_coefficient=hill, source=source,
                                    source_ref=p.get("source")),
            effect_curve=curve, selection_reason=reason,
            selection_metrics=self._metrics(metrics),
            note="direct-effect PD model on the PK concentration curve",
        )

    @staticmethod
    def _metrics(metrics: dict | None) -> ModelSelectionMetrics | None:
        if not metrics:
            return None
        return ModelSelectionMetrics(
            n_observations=metrics.get("n_observations"),
            n_parameters=metrics.get("n_parameters"),
            residual_sum_of_squares=metrics.get("residual_sum_of_squares"),
            aic=metrics.get("aic"),
            bic=metrics.get("bic"),
            identifiability_note=metrics.get("identifiability_note"),
        )

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
