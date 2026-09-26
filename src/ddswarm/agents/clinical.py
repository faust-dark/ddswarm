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
    _sample,
    emax_effect,
    fit_metrics,
    hill_effect,
    receptor_occupancy,
    resolve_cl_vd,
    simulate_blood_flow_limited,
    simulate_effect_compartment,
    simulate_indirect_response,
    simulate_multi_compartment,
    simulate_one_compartment,
    simulate_permeability_limited,
    simulate_two_compartment,
    simulate_well_stirred,
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
    PeripheralCompartment,
    PKParameters,
    PKSimulation,
    TissueParameters,
    TissueSimulation,
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
# implemented in this increment; anything else is deferred to later increments
_IMPLEMENTED_PD = {"emax", "hill", "indirect_response", "effect_compartment",
                   "receptor_occupancy"}

_PD_NOTE = {
    "emax": "direct-effect Emax model on the PK concentration curve",
    "hill": "direct-effect sigmoid-Emax (Hill) model on the PK concentration curve",
    "indirect_response": "indirect-response (turnover) model — drug inhibits response synthesis",
    "effect_compartment": "effect-compartment (link) model — delayed effect vs plasma",
    "receptor_occupancy": "receptor-occupancy (target-engagement) model",
}

# documented keyword -> tissue model-kind hints (tissue-level representation)
_TISSUE_HINTS = {
    "well_stirred": ("well-stirred", "well stirred", "instant equilibrium", "rapid equilibrium"),
    "blood_flow_limited": ("blood-flow-limited", "blood flow limited", "perfusion-limited",
                           "perfusion limited"),
    "permeability_limited": ("permeability-limited", "permeability limited",
                             "membrane barrier", "permeability barrier"),
}

_TISSUE_NOTE = {
    "well_stirred": "well-stirred tissue model (Ct = Kp*C_plasma)",
    "blood_flow_limited": "blood-flow-limited tissue model (perfusion-limited uptake)",
    "permeability_limited": "permeability-limited tissue model (membrane PS barrier)",
}


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
            sim = self._build_pk_simulation(ctx, mid, smi, kind, reason, metrics, sim_info, p,
                                            dose_mg, route, explicit_ids, warnings)
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
        explicit = self._resolve_override(ctx, "pk_model", mid)
        hint = self._keyword_hint(ctx.spec, _PK_HINTS)
        requested = explicit or hint
        one = self._one_compartment_result(p, dose_mg, route)
        two = self._two_compartment_result(p, dose_mg, route)
        multi = self._multi_compartment_result(p, dose_mg, route)
        candidates: dict[str, dict] = {}
        if one is not None:
            candidates["one_compartment"] = one
        if two is not None:
            candidates["two_compartment"] = two
        if multi is not None:
            candidates["multi_compartment"] = multi

        if requested is not None:
            if requested in candidates:
                return requested, "mechanism_hint", None, candidates[requested]
            warnings.append(f"{mid}: pk_model={requested} but parameters unavailable; "
                            "using available model")

        # data-driven: goodness-of-fit / identifiability / predictive-performance
        if observed and len(candidates) >= 2:
            fitted = {k: fit_metrics(observed, c["fn"], c["n_params"])
                      for k, c in candidates.items()}
            best = min(fitted, key=lambda k: self._fit_key(fitted[k]))
            return best, "data_driven", fitted[best], candidates[best]

        # default: simplest feasible model, labelled (not claimed validated)
        for kind in ("one_compartment", "two_compartment", "multi_compartment"):
            if kind in candidates:
                return kind, "default", None, candidates[kind]
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
    def _multi_compartment_result(p, dose_mg, route):
        v1 = p.get("central_volume_l_kg")
        cl = p.get("clearance_ml_min_kg")
        raw = p.get("peripheral_compartments", []) or []
        if v1 is None or cl is None or not raw:
            return None
        peripherals: list[tuple[float, float]] = []
        for pc in raw:
            if isinstance(pc, dict):
                vi, qi = pc.get("volume_l_kg"), pc.get("intercompartmental_clearance_ml_min_kg")
            else:
                vi, qi = pc[0], pc[1]
            if vi is None or qi is None:
                return None
            peripherals.append((float(vi), float(qi)))
        if not peripherals:
            return None
        f = float(p.get("bioavailability")) if p.get("bioavailability") is not None else 1.0
        try:
            r = simulate_multi_compartment(dose_mg=dose_mg, v1_l_kg=v1,
                                           clearance_ml_min_kg=cl,
                                           peripherals=peripherals,
                                           bioavailability=f, route=route)
        except ValueError:
            return None
        fn = lambda t: _sample(r.ts, r.central, [t])[0]  # noqa: E731
        return {"result": r, "cl": cl, "v1": v1, "peripherals": peripherals, "f": f,
                "fn": fn, "n_params": 2 + 2 * len(peripherals)}

    @staticmethod
    def _keyword_hint(spec, hints) -> str | None:
        text = " ".join(filter(None, [spec.mechanism_of_action, spec.adme_issue])).lower()
        for kind, keywords in hints.items():
            if any(kw in text for kw in keywords):
                return kind
        return None

    @staticmethod
    def _resolve_override(ctx, key: str, mid: str):
        """Resolve a per-molecule override param that may be a str (all) or {mid: value}."""
        v = ctx.params.get(key)
        if isinstance(v, dict):
            return v.get(mid)
        return v

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

    @staticmethod
    def _fit_key(m: dict) -> tuple:
        """Total-order sort key for model fit: perfect fit < (BIC, RSS) < unassessable."""
        rss = m.get("residual_sum_of_squares")
        if rss is None:
            return (2, 0.0, float("inf"))
        if rss == 0.0:
            return (0, float(m["n_parameters"]), 0.0)
        bic = m.get("bic")
        return (1, bic if bic is not None else float("inf"), float(rss))

    def _build_pk_simulation(self, ctx, mid, smi, kind, reason, metrics, sim_info, p,
                             dose_mg, route, explicit_ids, warnings):
        source = "user" if mid in explicit_ids else "inferred"
        sel_metrics = self._metrics(metrics)
        r = sim_info["result"]
        if kind == "two_compartment":
            sim = PKSimulation(
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
        elif kind == "multi_compartment":
            sim = PKSimulation(
                molecule_id=mid, canonical_smiles=smi, dose_mg=dose_mg, route=route,
                model="multi_compartment",
                parameters=PKParameters(
                    clearance_ml_min_kg=sim_info["cl"], central_volume_l_kg=sim_info["v1"],
                    peripheral_compartments=[
                        PeripheralCompartment(volume_l_kg=v,
                                              intercompartmental_clearance_ml_min_kg=q)
                        for v, q in sim_info["peripherals"]],
                    bioavailability=sim_info["f"],
                    half_life_hours=round(r.terminal_half_life_hours, 3),
                    source=source, source_ref=p.get("source")),
                half_life_hours=round(r.terminal_half_life_hours, 3),
                cmax_mg_l=round(r.cmax_mg_l, 4), auc_mg_h_l=round(r.auc_mg_h_l, 4),
                concentration_curve=[ConcentrationPoint(time_hours=t, concentration_mg_l=round(c, 4))
                                     for t, c in r.curve],
                selection_reason=reason, selection_metrics=sel_metrics,
                note=(f"multi-compartment model ({len(sim_info['peripherals'])} peripheral "
                      "compartment(s)); parameters "
                      + ("user-supplied" if source == "user" else "LLM-recalled (inferred)")),
            )
        else:
            sim = PKSimulation(
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
        tissue = self._build_tissue(ctx, mid, p, sim_info["fn"], [t for t, _ in r.curve],
                                    explicit_ids, warnings)
        if tissue is not None:
            sim.tissue = tissue
        return sim

    def _build_tissue(self, ctx, mid, p, pk_fn, times, explicit_ids, warnings):
        """Optional tissue-level (organ) model, layered on the systemic PK curve."""
        explicit = self._resolve_override(ctx, "tissue_model", mid)
        hint = self._keyword_hint(ctx.spec, _TISSUE_HINTS)
        requested = explicit or hint
        if requested is None:
            return None
        reason = "mechanism_hint" if requested else "default"
        kp = p.get("tissue_partition_coefficient")
        vt = p.get("tissue_volume_l_kg")
        qt = p.get("tissue_blood_flow_ml_min_kg")
        ps = p.get("permeability_surface_product_ml_min_kg")
        source = "user" if mid in explicit_ids else "inferred"
        try:
            if requested == "well_stirred":
                if kp is None or kp <= 0:
                    warnings.append(f"{mid}: tissue_model=well_stirred requires Kp")
                    return None
                vals = simulate_well_stirred(pk_fn, times, kp=float(kp))
                params = TissueParameters(tissue_partition_coefficient=float(kp),
                                          source=source, source_ref=p.get("source"))
            elif requested == "blood_flow_limited":
                if kp is None or vt is None or qt is None or kp <= 0 or vt <= 0 or qt <= 0:
                    warnings.append(f"{mid}: tissue_model=blood_flow_limited requires Kp, Vt, Qt")
                    return None
                vals = simulate_blood_flow_limited(pk_fn, times, qt_ml_min_kg=float(qt),
                                                   vt_l_kg=float(vt), kp=float(kp))
                params = TissueParameters(tissue_partition_coefficient=float(kp),
                                          tissue_blood_flow_ml_min_kg=float(qt),
                                          tissue_volume_l_kg=float(vt),
                                          source=source, source_ref=p.get("source"))
            elif requested == "permeability_limited":
                if kp is None or vt is None or ps is None or kp <= 0 or vt <= 0 or ps <= 0:
                    warnings.append(f"{mid}: tissue_model=permeability_limited requires Kp, Vt, PS")
                    return None
                vals = simulate_permeability_limited(pk_fn, times, ps_ml_min_kg=float(ps),
                                                     vt_l_kg=float(vt), kp=float(kp))
                params = TissueParameters(tissue_partition_coefficient=float(kp),
                                          tissue_volume_l_kg=float(vt),
                                          permeability_surface_product_ml_min_kg=float(ps),
                                          source=source, source_ref=p.get("source"))
            else:
                warnings.append(f"{mid}: tissue_model={requested} not implemented")
                return None
        except ValueError as exc:
            warnings.append(f"tissue simulation skipped for {mid}: {exc}")
            return None
        curve = [ConcentrationPoint(time_hours=t, concentration_mg_l=round(v, 4))
                 for t, v in zip(times, vals)]
        return TissueSimulation(model=requested, parameters=params, tissue_curve=curve,
                                selection_reason=reason,
                                note=_TISSUE_NOTE.get(requested, "tissue model"))

    def _build_pd_simulation(self, ctx, mid, smi, pk_curve, pk_fn, p, observed_eff,
                             pd_explicit_ids, warnings):
        if not p:
            return None

        explicit = self._resolve_override(ctx, "pd_model", mid)
        hint = self._keyword_hint(ctx.spec, _PD_HINTS)
        requested = explicit or hint
        reason = "mechanism_hint" if requested else "default"
        if requested is not None and requested not in _IMPLEMENTED_PD:
            warnings.append(f"{mid}: pd_model={requested} not implemented")
            requested = None
        if requested is None:
            requested = "emax"  # default direct-effect model (requires Emax+EC50)

        times = [cp.time_hours for cp in pk_curve]
        source = "user" if mid in pd_explicit_ids else "inferred"
        metrics = None
        curve: list[EffectPoint] = []
        params: PDParameters

        try:
            if requested in ("emax", "hill"):
                e0 = float(p["baseline_effect"]) if p.get("baseline_effect") is not None else 0.0
                emax = p.get("max_effect")
                ec50 = p.get("ec50_mg_l")
                hill = p.get("hill_coefficient")
                if emax is None or ec50 is None:
                    return None  # insufficient PD inputs (silent: PD is additive)
                emax, ec50 = float(emax), float(ec50)
                if requested == "hill":
                    if hill is None or hill <= 0:
                        warnings.append(f"{mid}: pd_model=hill requires hill_coefficient; using emax")
                        requested = "emax"
                    else:
                        hill = float(hill)
                # data-driven: emax vs hill (Hill adds one parameter)
                if (observed_eff and pk_fn is not None and hill is not None and hill > 0
                        and emax is not None and ec50 is not None):
                    m_emax = fit_metrics(observed_eff,
                                         lambda t: emax_effect(pk_fn(t), e0, emax, ec50), 3)
                    m_hill = fit_metrics(observed_eff,
                                         lambda t: hill_effect(pk_fn(t), e0, emax, ec50, hill), 4)
                    if self._better_fit(m_hill, m_emax):
                        requested, reason, metrics = "hill", "data_driven", m_hill
                    else:
                        requested, reason, metrics = "emax", "data_driven", m_emax
                for cp in pk_curve:
                    c = cp.concentration_mg_l
                    e = (hill_effect(c, e0, emax, ec50, hill) if requested == "hill"
                         else emax_effect(c, e0, emax, ec50))
                    curve.append(EffectPoint(time_hours=cp.time_hours, effect=round(e, 4)))
                params = PDParameters(
                    baseline_effect=e0, max_effect=emax, ec50_mg_l=ec50,
                    hill_coefficient=(hill if requested == "hill" else None),
                    source=source, source_ref=p.get("source"))

            elif requested == "indirect_response":
                baseline = p.get("baseline_effect")
                kout = p.get("kout_per_hour")
                imax = p.get("imax")
                ic50 = p.get("ic50_mg_l")
                if baseline is None or kout is None or imax is None or ic50 is None:
                    warnings.append(f"{mid}: pd_model=indirect_response requires "
                                    "baseline_effect, kout_per_hour, imax, ic50_mg_l")
                    return None
                vals = simulate_indirect_response(pk_fn, times, baseline=float(baseline),
                                                  kout_per_hour=float(kout), imax=float(imax),
                                                  ic50_mg_l=float(ic50))
                curve = [EffectPoint(time_hours=t, effect=round(v, 4)) for t, v in zip(times, vals)]
                params = PDParameters(baseline_effect=float(baseline), kout_per_hour=float(kout),
                                      imax=float(imax), ic50_mg_l=float(ic50),
                                      source=source, source_ref=p.get("source"))

            elif requested == "effect_compartment":
                ke0 = p.get("ke0_per_hour")
                e0 = float(p["baseline_effect"]) if p.get("baseline_effect") is not None else 0.0
                emax = p.get("max_effect")
                ec50 = p.get("ec50_mg_l")
                if ke0 is None or emax is None or ec50 is None:
                    warnings.append(f"{mid}: pd_model=effect_compartment requires "
                                    "ke0_per_hour, max_effect, ec50_mg_l")
                    return None
                vals = simulate_effect_compartment(pk_fn, times, ke0_per_hour=float(ke0),
                                                   e0=e0, emax=float(emax), ec50=float(ec50))
                curve = [EffectPoint(time_hours=t, effect=round(v, 4)) for t, v in zip(times, vals)]
                params = PDParameters(baseline_effect=e0, max_effect=float(emax),
                                      ec50_mg_l=float(ec50), ke0_per_hour=float(ke0),
                                      source=source, source_ref=p.get("source"))

            elif requested == "receptor_occupancy":
                kd = p.get("kd_mg_l")
                if kd is None or kd <= 0:
                    warnings.append(f"{mid}: pd_model=receptor_occupancy requires kd_mg_l")
                    return None
                for cp in pk_curve:
                    curve.append(EffectPoint(time_hours=cp.time_hours,
                                             effect=round(receptor_occupancy(cp.concentration_mg_l, kd), 4)))
                params = PDParameters(kd_mg_l=float(kd), source=source,
                                      source_ref=p.get("source"))

            else:
                return None
        except ValueError as exc:
            warnings.append(f"PD simulation skipped for {mid}: {exc}")
            return None

        return PDSimulation(
            molecule_id=mid, canonical_smiles=smi, model=requested,
            parameters=params, effect_curve=curve, selection_reason=reason,
            selection_metrics=self._metrics(metrics),
            note=_PD_NOTE.get(requested, "PD model"),
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
