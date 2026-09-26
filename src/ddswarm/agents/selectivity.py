"""selectivity_minion: mechanistic target-selectivity / sexual-ADR scoring.

Docks each candidate against SERT, 5-HT2A and 5-HT2C (AutoDock Vina), converts
the docking free energy to a dissociation constant via the Gibbs identity
(ΔG = RT·ln Ki), computes a fractional occupancy at a fixed reference
concentration (C/(C+Ki)), and derives a selectivity index plus a deterministic
sexual-ADR risk label.

The classification encodes the SSRI mechanism: sexual dysfunction is mediated
by elevated synaptic 5-HT activating 5-HT2A/2C — an indirect effect driven by
SERT occupancy (class-effect coupling). Docking reports binding, not functional
sign, so a direct 5-HT2C engagement is flagged as of unknown sign.

Opt-in (like docking/dynamics): needs the Vina binary + network for PDB download.
"""

from __future__ import annotations

from pathlib import Path

from rdkit import rdBase

from ..chemistry.selectivity import (
    classify_sexual_adr,
    docking_ki,
    occupancy_at,
    selectivity_index,
)
from ..contracts.agent_result import AgentResult
from ..contracts.enums import AgentResultStatus
from ..contracts.provenance import ProvenanceEntry, ToolVersion
from ..contracts.selectivity_assessment import (
    MoleculeSelectivity,
    ReceptorAffinity,
    SelectivityAssessment,
    SelectivitySummary,
)
from ..docking import make_engine
from ..docking.base import DockProtocol, ReceptorSource, box_from_reference_mol
from ..logging.hashing import hash_jsonable
from .base import Agent, AgentContext

TOOL = ToolVersion(name="ddswarm", version="0.1.0")
RDKIT_TOOL = ToolVersion(name="rdkit", version=rdBase.rdkitVersion)

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

        receptors = ctx.params.get("receptors") or _DEFAULT_RECEPTORS
        if not receptors:
            return self._result(
                ctx, AgentResultStatus.insufficient_evidence,
                warnings=["no receptors configured (params['receptors'] empty)"],
                reproducibility={"reason": "no_receptors"},
            )

        molecules = self._collect_molecules(ctx)
        if not molecules:
            return self._result(
                ctx, AgentResultStatus.insufficient_evidence,
                warnings=["no molecules (screened_batch survivors or starting_materials empty)"],
                reproducibility={"reason": "no_molecules"},
            )

        # parent (starting material) docks alongside the survivors as the
        # efficacy reference for the SERT-occupancy comparison.
        parent = None
        if ctx.spec.starting_materials:
            parent = ctx.spec.starting_materials[0]
            if all(smi != parent for _mid, smi in molecules):
                molecules.append(("parent", parent))

        c_ref = float(ctx.params.get("reference_concentration_molar", 1e-8))
        exhaustiveness = int(ctx.params.get("exhaustiveness", 8))
        num_modes = int(ctx.params.get("num_modes", 9))
        seed = int(ctx.params.get("seed", 42))
        work_root = Path(self.store.run_dir(ctx.project_id, ctx.run_id)) / "selectivity" / ctx.task_id

        # best affinity per (receptor_name, molecule_id)
        best: dict[tuple[str, str], float] = {}
        pdb_by_receptor: dict[str, str] = {}
        notes: list[str] = []

        for rc in receptors:
            rname = str(rc.get("name", ""))
            if rname not in ("SERT", "HTR2A", "HTR2C"):
                notes.append(f"skipping unknown receptor {rname!r}")
                continue
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
                poses = engine.dock(receptor, ligand, protocol, work_root / rname / f"dock_{mid}")
                if poses:
                    best[(rname, mid)] = min(p.affinity_kcal_per_mol for p in poses
                                             if p.affinity_kcal_per_mol is not None)

        # assemble per-molecule selectivity
        assessed: list[MoleculeSelectivity] = []
        # parent SERT occupancy = efficacy reference (from docking the parent)
        parent_sert_occ = None
        if parent is not None:
            parent_sert_occ = occupancy_at(c_ref, docking_ki(best.get(("SERT", "parent"))))

        for mid, smi in molecules:
            if mid == "parent":
                continue  # reference only, not a candidate
            affs: list[ReceptorAffinity] = []
            ki: dict[str, float | None] = {}
            occ: dict[str, float | None] = {}
            for rname in ("SERT", "HTR2A", "HTR2C"):
                dg = best.get((rname, mid))
                k = docking_ki(dg)
                o = occupancy_at(c_ref, k)
                ki[rname] = k
                occ[rname] = o
                affs.append(ReceptorAffinity(
                    receptor=rname,  # type: ignore[arg-type]
                    pdb_id=pdb_by_receptor.get(rname),
                    best_affinity_kcal_per_mol=dg,
                    ki_molar=k,
                    occupancy=o,
                    note="relative estimate (Vina scoring is not calibrated absolute free energy)"
                    if dg is not None else None,
                ))
            si_2c = selectivity_index(ki.get("HTR2C"), ki.get("SERT"))
            si_2a = selectivity_index(ki.get("HTR2A"), ki.get("SERT"))
            label, rationale = classify_sexual_adr(
                sert_occupancy=occ.get("SERT"),
                htr2c_occupancy=occ.get("HTR2C"),
                reference_sert_occupancy=parent_sert_occ,
            )
            assessed.append(MoleculeSelectivity(
                molecule_id=mid, canonical_smiles=smi, affinities=affs,
                selectivity_index_5ht2c_sert=si_2c,
                selectivity_index_5ht2a_sert=si_2a,
                sert_occupancy=occ.get("SERT"),
                htr2c_occupancy=occ.get("HTR2C"),
                htr2a_occupancy=occ.get("HTR2A"),
                sexual_adr_risk=label,  # type: ignore[arg-type]
                sexual_adr_rationale=rationale,
            ))

        insufficient = sum(1 for m in assessed if m.sexual_adr_risk == "insufficient_data")
        summary = SelectivitySummary(
            total_molecules=len(assessed),
            assessed=len(assessed) - insufficient,
            insufficient_data=insufficient,
        )
        note = ("Ki/occupancy are relative estimates (Vina scoring is not calibrated absolute "
                "free energy); docking reports binding, not functional sign (agonist vs "
                "antagonist) — a direct 5-HT2C engagement is flagged as of unknown sign.")
        if notes:
            note += " Receptor notes: " + "; ".join(notes)

        input_hash = hash_jsonable({"receptors": receptors, "molecules": molecules,
                                    "c_ref": c_ref, "seed": seed})
        output_hash = hash_jsonable([m.model_dump() for m in assessed])
        assessment = SelectivityAssessment(
            project_id=ctx.project_id,
            status=AgentResultStatus.success,
            molecules=assessed,
            summary=summary,
            reference_concentration_molar=c_ref,
            reference_molecule=parent,
            reference_sert_occupancy=parent_sert_occ,
            note=note,
            provenance=[ProvenanceEntry(
                source="target-selectivity scoring (AutoDock Vina + Gibbs/occupancy)",
                tool_versions=[TOOL, RDKIT_TOOL, ToolVersion(name=engine_name, version=engine.version)],
                params={"receptors": receptors, "c_ref": c_ref, "exhaustiveness": exhaustiveness,
                        "seed": seed},
                input_hash=input_hash,
                output_hash=output_hash,
            )],
        )
        art = self.persist_artifact(ctx, "selectivity_assessment", f"sel_{ctx.project_id}", assessment)
        findings = [f"selectivity: {summary.assessed}/{summary.total_molecules} assessed "
                    f"({summary.insufficient_data} insufficient data)"]
        labels = {m.sexual_adr_risk for m in assessed}
        if labels:
            findings.append("sexual-ADR labels: " + ", ".join(sorted(labels)))
        return self._result(
            ctx, AgentResultStatus.success,
            input_refs=ctx.input_refs, artifacts=[art], findings=findings,
            provenance=[ProvenanceEntry(source="target-selectivity scoring", output_hash=art.hash,
                                        tool_versions=[TOOL, RDKIT_TOOL])],
            reproducibility={"input_hash": input_hash, "engine": engine_name},
        )

    # ---- helpers ----
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
