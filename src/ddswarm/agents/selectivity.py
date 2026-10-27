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

from ..chemistry.chembl_client import receptor_reference_ligands, search_target
from ..chemistry.ht2_function import classify_ht2_function
from ..chemistry.receptors import RECEPTOR_REGISTRY, canonical_receptor_name, expand_receptor
from ..chemistry.selectivity import (
    ANTITARGET_PROFILES,
    SPECIAL_ADR_PROFILES,
    classify_cox_gi,
    classify_gabaa_adr,
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
from ..docking.parallel import DockJob, dock_jobs
from ..logging.hashing import hash_jsonable
from ..orchestrator.checkpoint import ComputeCache, dock_cache_enabled
from .base import Agent, AgentContext
from ddswarm import __version__

TOOL = ToolVersion(name="ddswarm", version=__version__)
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
            # An optional engine that is not installed (e.g. no Vina binary on a
            # core install) is a reported gap, not a crash: degrade to
            # insufficient_evidence so the run still completes (README contract).
            return self._result(
                ctx, AgentResultStatus.insufficient_evidence,
                warnings=[f"docking engine '{engine_name}' unavailable: {msg}; selectivity not performed"],
                reproducibility={"reason": "engine_unavailable"},
            )

        raw_receptors = ctx.params.get("receptors") or self._default_panel(ctx)
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
        cache = ComputeCache(self.store, dock_cache_enabled(ctx.params))
        if cache.enabled and hasattr(engine, "receptor_cache_dir"):
            engine.receptor_cache_dir = cache.root / "receptor"

        # best affinity per (receptor_name, molecule_id)
        best: dict[tuple[str, str], float] = {}
        pdb_by_receptor: dict[str, str] = {}
        calib_pairs: dict[str, list[tuple[float, float]]] = {}  # receptor -> (vina, ΔG_lit)
        cal_sources: dict[str, str] = {}  # receptor -> "static" | "chembl" | "none"
        chembl_ligands: dict[str, list[tuple[str, float, str]]] = {}  # fetched (for hash)
        notes: list[str] = []

        # each distinct SMILES is prepared ONCE and reused for every receptor
        # (ligand prep does not depend on the receptor)
        ligand_cache: dict[str, object] = {}

        def ligand_for(smi: str):
            if smi not in ligand_cache:
                ligand_cache[smi] = engine.prepare_ligand(
                    smi, work_root / "_ligands" / f"lig_{len(ligand_cache)}")
            return ligand_cache[smi]

        # 1) prepare receptors + collect every (receptor x ligand) docking job
        jobs: list[DockJob] = []
        cal_meta: dict[str, list[tuple[int, float]]] = {}  # receptor -> [(cal index, Ki)]
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
            if not receptor.identifier:
                receptor.identifier = source.identifier
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
                ligand = ligand_for(smi)
                if ligand.pdbqt_path:
                    jobs.append(DockJob(("mol", rname, mid), source.identifier, receptor, ligand,
                                        protocol, work_root / rname / f"dock_{mid}"))
            if calibrate:
                cal_ligands, cal_source = self._calibration_ligands(rname, ctx)
                cal_sources[rname] = cal_source
                if cal_source == "chembl":
                    chembl_ligands[rname] = cal_ligands
                for ci, (cal_smi, ki_exp, _cit) in enumerate(cal_ligands):
                    ligand = ligand_for(cal_smi)
                    if not ligand.pdbqt_path:
                        continue
                    cal_meta.setdefault(rname, []).append((ci, ki_exp))
                    jobs.append(DockJob(("cal", rname, ci), source.identifier, receptor, ligand,
                                        protocol, work_root / rname / f"caldock_{ci}"))

        # 2) dock everything concurrently (cache hits — e.g. the fixed calibration
        #    ligands on a later run — skip Vina entirely)
        results = dock_jobs(engine, cache, jobs, ctx.params.get("max_workers"))

        # 3) read results back in declaration order (completion order never leaks)
        for job in jobs:
            kind, rname, mid = job.key
            if kind != "mol":
                continue
            affs = [p.affinity_kcal_per_mol for p in results.get(job.key) or []
                    if p.affinity_kcal_per_mol is not None]
            if affs:
                best[(rname, mid)] = min(affs)
        for rname, entries in cal_meta.items():
            for ci, ki_exp in entries:
                affs = [p.affinity_kcal_per_mol for p in results.get(("cal", rname, ci)) or []
                        if p.affinity_kcal_per_mol is not None]
                if affs:
                    calib_pairs.setdefault(rname, []).append((min(affs), delta_g_from_ki(ki_exp)))

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
            if chembl_ligands:
                cal_note = (
                    "per-receptor offset fit on the fixed committed set plus ON-DEMAND ChEMBL "
                    "reference ligands for " + ", ".join(sorted(chembl_ligands)) + " "
                    "(network-dependent, non-reproducible); slope fixed at 1 so the Vina->Ki "
                    "mapping is monotonic. The calibrated Ki is a calculated estimate, not an "
                    "experimental affinity."
                )
            else:
                cal_note = (
                    "per-receptor offset fit on a fixed, committed literature-Ki set; slope "
                    "fixed at 1 (no rescaling) so the Vina->Ki mapping is monotonic. The "
                    "calibrated Ki is a calculated estimate, not an experimental affinity."
                )
            calibration = VinaCalibration(
                intercepts={r: calib[r][0] for r in receptor_names},
                n_points={r: calib[r][1] for r in receptor_names},
                rmsd_kcal_per_mol={r: calib[r][2] for r in receptor_names},
                note=cal_note,
            )

        hash_input = {"receptors": receptors, "molecules": molecules,
                      "primary": primary, "c_ref": c_ref, "seed": seed,
                      "calibrate": calibrate}
        if chembl_ligands:
            hash_input["chembl_calibration"] = chembl_ligands
        input_hash = hash_jsonable(hash_input)
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
    def _calibration_ligands(rname: str, ctx: AgentContext) -> tuple[list[tuple[str, float, str]], str]:
        """(smiles, ki_molar, citation) for a receptor + the source ("static"|"chembl"|"none").

        The fixed, committed CALIBRATION_SET wins; when a receptor is absent
        from it and params['chembl_calibration'] is set, reference ligands are
        fetched from ChEMBL on-demand (network-dependent, opt-in — the fetched
        values are re-hashed by the caller, not silently folded into the output).
        """
        static = CALIBRATION_SET.get(rname, [])
        if static:
            return static, "static"
        if ctx.params.get("chembl_calibration"):
            gene = (RECEPTOR_REGISTRY.get(rname) or {}).get("gene")
            if gene:
                try:
                    tid = search_target(gene)
                    if tid:
                        refs = receptor_reference_ligands(tid)
                        if refs:
                            return ([(r["smiles"], r["ki_molar"], r["citation"]) for r in refs],
                                    "chembl")
                except Exception:
                    return [], "none"
        return [], "none"

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
            elif profile == "gabaa_alpha2_anxiolytic":
                label, rationale = classify_gabaa_adr(
                    alpha1_ki=ki.get("GABRA1"),
                    alpha2_ki=ki.get("GABRA2"),
                    alpha5_occupancy=occ.get("GABRA5"),
                )
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
            + ("hybrid-calibrated estimates (per-receptor offset from a literature-Ki set, "
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

    @staticmethod
    def _default_panel(ctx: AgentContext) -> list[dict]:
        """Default receptor panel derived from the project targets when no
        explicit ``params['receptors']`` is given (Increment 39 — selectivity is
        always-on, so it must resolve a sensible panel generically).

        primary = ``spec.targets[0]`` (canonicalised); off-targets = the
        off-target list of whichever built-in ADR profile has that primary
        (GABRA2 -> GABRA1/GABRA5 sedation+amnesia; COX2 -> COX1; SERT -> HTR2C).
        No matching profile -> a primary-only panel (pure affinity, no SI). No
        targets at all -> the legacy SSRI triad (backward-compatible).
        """
        targets = [canonical_receptor_name(str(t)) for t in (ctx.spec.targets or [])]
        if not targets:
            return list(_DEFAULT_RECEPTORS)
        primary = targets[0]
        off_targets: list[str] = []
        for prof in SPECIAL_ADR_PROFILES.values():
            if primary == canonical_receptor_name(str(prof.get("primary", ""))):
                off_targets = [canonical_receptor_name(str(o))
                               for o in prof.get("off_targets", [])]
                break
        return [{"name": primary}] + [{"name": o} for o in off_targets]

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
