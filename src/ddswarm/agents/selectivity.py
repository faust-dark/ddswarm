"""selectivity_minion: mechanistic target-selectivity scoring over a configurable
receptor panel.

Docks each candidate against every configured receptor (AutoDock Vina), converts
the docking free energy to a dissociation constant via the Gibbs identity
(ΔG = RT·ln Ki), computes a fractional occupancy at a fixed reference
concentration (C/(C+Ki)), and derives a general selectivity index
(Ki_off / Ki_primary) for each off-target relative to the designated primary
(on-target) receptor.

An optional adverse-effect interpretation is attached only when the panel
matches a profile from the pluggable ADR library — the SSRI profile (SERT
primary + 5-HT2C off-target -> sexual-ADR risk), the NSAID profile (COX-2
primary + COX-1 off-target -> GI risk), and off-target "antitarget" ADRs (H1
sedation, M1 anticholinergic, D2 extrapyramidal/prolactin, alpha-1 orthostatic
hypotension, 5-HT2B valvulopathy, kappa dysphoria). Any other panel reports
pure affinity + selectivity indices without inventing an ADR label. Docking
reports binding, not functional sign, so the SSRI profile's direct 5-HT2C
engagement is refined by documented-chemotype matching (antagonist-like/
agonist-like via chemistry/ht2_function); unmatched chemotypes remain of
unknown sign.

Opt-in (like docking/dynamics): needs the Vina binary + network for PDB download.
"""

from __future__ import annotations

from pathlib import Path

from rdkit import rdBase

from ..chemistry.ht2_function import classify_ht2_function
from ..chemistry.receptors import canonical_receptor_name, expand_receptor
from ..chemistry.selectivity import (
    ANTITARGET_PROFILES,
    classify_cox_gi,
    classify_occupancy_adr,
    classify_sexual_adr,
    docking_ki,
    occupancy_at,
    select_adr_profiles,
    selectivity_index,
)
from ..chemistry.vina_calibration import (
    CALIBRATION_SET,
    calibrated_ki,
    delta_g_from_ki,
    fit_offset,
)
from ..contracts.agent_result import AgentResult
from ..contracts.enums import AgentResultStatus
from ..contracts.provenance import ProvenanceEntry, ToolVersion
from ..contracts.selectivity_assessment import (
    AdrInterpretation,
    MoleculeSelectivity,
    ReceptorAffinity,
    SelectivityAssessment,
    SelectivityIndex,
    SelectivitySummary,
    VinaCalibration,
)
from ..docking import make_engine
from ..docking.base import DockProtocol, ReceptorSource, box_from_reference_mol
from ..logging.hashing import hash_jsonable
from ..orchestrator.checkpoint import ComputeCache, cached_dock
from .base import Agent, AgentContext

TOOL = ToolVersion(name="ddswarm", version="0.1.0")
RDKIT_TOOL = ToolVersion(name="rdkit", version=rdBase.rdkitVersion)

# Default panel is the SSRI triad (backward-compatible). Any receptor list can be
# passed via params["receptors"]; names without a pdb_id are expanded from the
# registry, and explicit pdb_id/identifier/url always wins.
_DEFAULT_RECEPTORS = [
    {"name": "SERT", "pdb_id": "5I73"},
    {"name": "HTR2A", "pdb_id": "6A93"},
    {"name": "HTR2C", "pdb_id": "6BQH"},
]


class SelectivityMinion(Agent):
    name = "selectivity_minion"

    def run(self, ctx: AgentContext) -> AgentResult:
        engine_name = ctx.params.get("engine", "vina")
        try:
            engine = make_engine(engine_name, vina_binary=ctx.params.get("vina_binary"))
        except KeyError as exc:
            return self._result(ctx, AgentResultStatus.failed, errors=[str(exc)])

        ok, msg = engine.available()
        if not ok:
            return self._result(
                ctx, AgentResultStatus.failed,
                errors=[f"docking engine '{engine_name}' unavailable: {msg}"],
                reproducibility={"reason": "engine_unavailable"},
            )

        raw_receptors = ctx.params.get("receptors") or _DEFAULT_RECEPTORS
        receptors = []
        for rc in raw_receptors:
            d = expand_receptor(dict(rc))
            # canonicalise names (PTGS2->COX2, M1->CHRM1, 5-HT2C->HTR2C, ...) so the
            # affinity/occupancy dicts and the ADR interpreters key on one spelling.
            d["name"] = canonical_receptor_name(str(d.get("name", "")))
            receptors.append(d)
        if not receptors:
            return self._result(
                ctx, AgentResultStatus.insufficient_evidence,
                warnings=["no receptors configured (params['receptors'] empty)"],
                reproducibility={"reason": "no_receptors"},
            )

        receptor_names = [str(rc.get("name", "")) for rc in receptors]
        primary = canonical_receptor_name(
            str(ctx.params.get("primary") or (receptor_names[0] if receptor_names else "")))
        if primary not in receptor_names:
            return self._result(
                ctx, AgentResultStatus.failed,
                errors=[f"primary receptor {primary!r} not in the configured panel {receptor_names}"],
                reproducibility={"reason": "primary_not_in_panel"},
            )
        off_targets = [r for r in receptor_names if r != primary]

        molecules = self._collect_molecules(ctx)
        if not molecules:
            return self._result(
                ctx, AgentResultStatus.insufficient_evidence,
                warnings=["no molecules (screened_batch survivors or starting_materials empty)"],
                reproducibility={"reason": "no_molecules"},
            )

        # parent (starting material) docks alongside the survivors as the
        # efficacy reference for the primary-receptor occupancy comparison.
        parent = None
        if ctx.spec.starting_materials:
            parent = ctx.spec.starting_materials[0]
            if all(smi != parent for _mid, smi in molecules):
                molecules.append(("parent", parent))

        c_ref = float(ctx.params.get("reference_concentration_molar", 1e-8))
        exhaustiveness = int(ctx.params.get("exhaustiveness", 8))
        num_modes = int(ctx.params.get("num_modes", 9))
        seed = int(ctx.params.get("seed", 42))
        calibrate = bool(ctx.params.get("calibrate", True))
        work_root = Path(self.store.run_dir(ctx.project_id, ctx.run_id)) / "selectivity" / ctx.task_id
        cache = ComputeCache(self.store, bool(ctx.params.get("resume")))

        # best affinity per (receptor_name, molecule_id)
        best: dict[tuple[str, str], float] = {}
        pdb_by_receptor: dict[str, str] = {}
        calib_pairs: dict[str, list[tuple[float, float]]] = {}  # receptor -> (vina, ΔG_lit)
        notes: list[str] = []

        for rc in receptors:
            rname = str(rc.get("name", ""))
            source = ReceptorSource(
                kind=rc.get("kind", "pdb_id"),
                identifier=str(rc.get("pdb_id") or rc.get("identifier", "")),
                url=rc.get("url"),
            )
            if not source.identifier:
                notes.append(f"receptor {rname} has no pdb_id/identifier; skipped")
                continue
            receptor = engine.prepare_receptor(source, work_root / rname)
            box_center, box_size = self._resolve_box(rc, receptor, ctx)
            if box_center is None:
                notes.append(f"receptor {rname}: {receptor.note or 'no search box resolvable'}")
                continue
            protocol = DockProtocol(
                box_center=box_center, box_size=box_size,
                exhaustiveness=exhaustiveness, num_modes=num_modes, seed=seed,
            )
            pdb_by_receptor[rname] = source.identifier
            for mid, smi in molecules:
                ligand = engine.prepare_ligand(smi, work_root / rname / f"lig_{mid}")
                if not ligand.pdbqt_path:
                    continue
                poses = cached_dock(engine, cache, source.identifier, receptor, ligand,
                                    protocol, work_root / rname / f"dock_{mid}")
                if poses:
                    best[(rname, mid)] = min(p.affinity_kcal_per_mol for p in poses
                                             if p.affinity_kcal_per_mol is not None)
            if calibrate:
                for ci, (cal_smi, ki_exp, _cit) in enumerate(CALIBRATION_SET.get(rname, [])):
                    ligand = engine.prepare_ligand(cal_smi, work_root / rname / f"cal_{ci}")
                    if not ligand.pdbqt_path:
                        continue
                    poses = cached_dock(engine, cache, source.identifier, receptor, ligand,
                                        protocol, work_root / rname / f"caldock_{ci}")
                    if poses:
                        vina = min(p.affinity_kcal_per_mol for p in poses
                                   if p.affinity_kcal_per_mol is not None)
                        calib_pairs.setdefault(rname, []).append((vina, delta_g_from_ki(ki_exp)))

        # fit the hybrid Vina->Ki offset calibration (per receptor)
        calib: dict[str, tuple[float | None, int, float | None]] = {}
        if calibrate:
            for rname in receptor_names:
                calib[rname] = fit_offset(calib_pairs.get(rname, []))

        def ki_of(rname: str, dg: float | None) -> float | None:
            if calibrate:
                return calibrated_ki(dg, calib.get(rname, (None, 0, None))[0])
            return docking_ki(dg)

        # assemble per-molecule selectivity
        assessed: list[MoleculeSelectivity] = []
        # parent primary-receptor occupancy = efficacy reference
        parent_primary_occ = None
        if parent is not None:
            parent_primary_occ = occupancy_at(c_ref, ki_of(primary, best.get((primary, "parent"))))

        adr_profiles = select_adr_profiles(primary, off_targets)

        for mid, smi in molecules:
            if mid == "parent":
                continue  # reference only, not a candidate
            affs: list[ReceptorAffinity] = []
            ki: dict[str, float | None] = {}
            occ: dict[str, float | None] = {}
            for rname in receptor_names:
                dg = best.get((rname, mid))
                k = ki_of(rname, dg)
                o = occupancy_at(c_ref, k)
                ki[rname] = k
                occ[rname] = o
                affs.append(ReceptorAffinity(
                    receptor=rname,
                    pdb_id=pdb_by_receptor.get(rname),
                    best_affinity_kcal_per_mol=dg,
                    ki_molar=k,
                    occupancy=o,
                    note=("hybrid-calibrated Ki (offset from literature Ki, slope a=1)"
                          if calibrate
                          else "relative estimate (Vina scoring is not calibrated absolute free energy)")
                    if dg is not None else None,
                ))
            indices = [
                SelectivityIndex(
                    off_target=r, primary=primary,
                    value=selectivity_index(ki.get(r), ki.get(primary)),
                )
                for r in off_targets
            ]
            adrs = self._interpret_adr(adr_profiles, occ, ki, parent_primary_occ, smi)
            assessed.append(MoleculeSelectivity(
                molecule_id=mid, canonical_smiles=smi,
                primary_receptor=primary, affinities=affs,
                selectivity_indices=indices, adr_interpretations=adrs,
            ))

        insufficient = sum(1 for m in assessed if all(a.ki_molar is None for a in m.affinities))
        summary = SelectivitySummary(
            total_molecules=len(assessed),
            assessed=len(assessed) - insufficient,
            insufficient_data=insufficient,
            primary_receptor=primary,
        )
        note = self._note(calibrate, adr_profiles, primary, off_targets, notes)

        calibration = None
        if calibrate:
            calibration = VinaCalibration(
                intercepts={r: calib[r][0] for r in receptor_names},
                n_points={r: calib[r][1] for r in receptor_names},
                rmsd_kcal_per_mol={r: calib[r][2] for r in receptor_names},
                note="per-receptor offset fit on a fixed, committed literature-Ki set; slope "
                     "fixed at 1 (no rescaling) so the Vina->Ki mapping is monotonic. The "
                     "calibrated Ki is a calculated estimate, not an experimental affinity.",
            )

        input_hash = hash_jsonable({"receptors": receptors, "molecules": molecules,
                                    "primary": primary, "c_ref": c_ref, "seed": seed,
                                    "calibrate": calibrate})
        output_hash = hash_jsonable([m.model_dump() for m in assessed])
        assessment = SelectivityAssessment(
            project_id=ctx.project_id,
            status=AgentResultStatus.success,
            molecules=assessed,
            summary=summary,
            reference_concentration_molar=c_ref,
            primary_receptor=primary,
            reference_molecule=parent,
            reference_primary_occupancy=parent_primary_occ,
            calibration=calibration,
            note=note,
            provenance=[ProvenanceEntry(
                source="target-selectivity scoring (AutoDock Vina + Gibbs/occupancy)",
                tool_versions=[TOOL, RDKIT_TOOL, ToolVersion(name=engine_name, version=engine.version)],
                params={"receptors": receptors, "primary": primary, "c_ref": c_ref,
                        "exhaustiveness": exhaustiveness, "seed": seed},
                input_hash=input_hash,
                output_hash=output_hash,
            )],
        )
        art = self.persist_artifact(ctx, "selectivity_assessment", f"sel_{ctx.project_id}", assessment)
        findings = [f"selectivity: {summary.assessed}/{summary.total_molecules} assessed "
                    f"({summary.insufficient_data} insufficient data), primary={primary}"]
        if adr_profiles:
            label_counts: dict[str, int] = {}
            for m in assessed:
                for a in m.adr_interpretations:
                    key = f"{a.profile}:{a.label}"
                    label_counts[key] = label_counts.get(key, 0) + 1
            if label_counts:
                findings.append("ADR " + ", ".join(
                    f"{k}:{v}" for k, v in sorted(label_counts.items())))
        return self._result(
            ctx, AgentResultStatus.success,
            input_refs=ctx.input_refs, artifacts=[art], findings=findings,
            provenance=[ProvenanceEntry(source="target-selectivity scoring", output_hash=art.hash,
                                        tool_versions=[TOOL, RDKIT_TOOL])],
            reproducibility={"input_hash": input_hash, "engine": engine_name},
        )

    # ---- helpers ----
    @staticmethod
    def _interpret_adr(profiles: list[str], occ: dict[str, float | None],
                       ki: dict[str, float | None], parent_primary_occ: float | None,
                       smi: str) -> list[AdrInterpretation]:
        """Interpret every matching ADR profile into AdrInterpretation entries."""
        out: list[AdrInterpretation] = []
        for profile in profiles:
            if profile == "sert_5ht2c_sexual_adr":
                out.append(SelectivityMinion._interpret_ssri_adr(occ, parent_primary_occ, smi))
            elif profile == "cox_gi_risk":
                label, rationale = classify_cox_gi(ki.get("COX1"), ki.get("COX2"))
                out.append(AdrInterpretation(profile=profile, label=label, rationale=rationale))
            else:
                prof = ANTITARGET_PROFILES.get(profile)
                if prof is None:
                    continue
                label, rationale = classify_occupancy_adr(profile, occ.get(prof["receptor"]))
                out.append(AdrInterpretation(profile=profile, label=label, rationale=rationale))
        return out

    @staticmethod
    def _interpret_ssri_adr(occ: dict[str, float | None], parent_primary_occ: float | None,
                            smi: str) -> AdrInterpretation:
        label, rationale = classify_sexual_adr(
            sert_occupancy=occ.get("SERT"),
            htr2c_occupancy=occ.get("HTR2C"),
            reference_sert_occupancy=parent_primary_occ,
        )
        ht2_sign, ht2_motifs, ht2_citations = classify_ht2_function(smi)
        note = None
        if ht2_motifs:
            note = "; ".join(f"{m} ({c})" for m, c in zip(ht2_motifs, ht2_citations))
        if label == "direct_5ht2c_engagement" and ht2_sign != "unknown":
            joined = ", ".join(ht2_motifs)
            cites = "; ".join(ht2_citations)
            if ht2_sign == "antagonist_like":
                label = "direct_5ht2c_antagonism"
                rationale = (f"direct 5-HT2C binding + documented antagonist chemotype ({joined}); "
                             f"5-HT2C antagonism may REDUCE sexual ADR — confirm functionally. {cites}")
            else:
                label = "direct_5ht2c_agonism"
                rationale = (f"direct 5-HT2C binding + documented agonist chemotype ({joined}); "
                             f"5-HT2C agonism may WORSEN sexual ADR — confirm functionally. {cites}")
        return AdrInterpretation(
            profile="sert_5ht2c_sexual_adr", label=label, rationale=rationale, note=note)

    @staticmethod
    def _note(calibrate: bool, adr_profiles: list[str], primary: str,
              off_targets: list[str], notes: list[str]) -> str:
        base = (
            "Ki/occupancy are "
            + ("hybrid-calibrated estimates (per-receptor offset from a fixed literature-Ki set, "
               "slope a = 1 so monotonicity is exact)" if calibrate
               else "relative estimates (Vina scoring is not calibrated absolute free energy)")
            + f"; primary (on-target) = {primary}, off-targets = {', '.join(off_targets) or 'none'}. "
            "Docking reports binding, not function"
        )
        if adr_profiles:
            base += (f"; the panel matches ADR profile(s) {', '.join(adr_profiles)} — blockade-driven "
                     "antitarget ADRs are tiered on occupancy at the reference concentration, while "
                     "sign-dependent ones (agonism) stay 'engagement' until a functional assay or "
                     "chemotype match resolves the sign")
        else:
            base += "; the panel matches no built-in ADR profile, so no adverse-effect label is "
            "asserted (pure affinity + selectivity indices)"
        base += "."
        if notes:
            base += " Receptor notes: " + "; ".join(notes)
        return base

    def _collect_molecules(self, ctx: AgentContext) -> list[tuple[str, str]]:
        mols: list[tuple[str, str]] = []
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
    def _resolve_box(rc, receptor, ctx):
        box = rc.get("box") or ctx.params.get("box") or {}
        if box.get("center") and box.get("size"):
            return (tuple(box["center"]), tuple(box["size"]))
        if receptor.reference_ligand_mol is not None:
            margin = float(ctx.params.get("box_margin", 11.25))
            return box_from_reference_mol(receptor.reference_ligand_mol, margin=margin)
        return (None, None)

    def _result(self, ctx: AgentContext, status: AgentResultStatus, **kw) -> AgentResult:
        return AgentResult(
            project_id=ctx.project_id, agent=self.name, task_id=ctx.task_id, status=status,
            tool_versions=[TOOL, RDKIT_TOOL], **kw,
        )
