"""docking_minion: prepare the receptor, dock candidates, and validate the
protocol via redocking (Increment 3).

Consumes molecules from screened_batch survivors (passed=True), then
starting_materials. Runs a real docking engine (Vina). Receptor structural QC
and, when a co-crystal ligand is available, redocking RMSD are first-class —
a pose is never reported without its receptor-quality and protocol context.
Docking scores are in-silico calculations, never experimental evidence."""

from __future__ import annotations

from pathlib import Path

from rdkit import rdBase

from ..chemistry.receptors import canonical_receptor_name, registry_entry
from ..contracts.agent_result import AgentResult
from ..contracts.docking_result import (
    DockedPose,
    DockingProtocol,
    DockingResult,
    DockingSummary,
    ReceptorInfo,
    RedockingValidation,
    SearchBox,
)
from ..contracts.enums import AgentResultStatus
from ..contracts.provenance import ProvenanceEntry, ToolVersion
from ..docking import make_engine
from ..docking.base import DockProtocol, box_from_reference_mol
from ..docking.parallel import DockJob, dock_jobs
from ..docking.rmsd import heavy_atom_rmsd, pose_mol_from_coords
from ..logging.hashing import hash_jsonable
from ..orchestrator.checkpoint import ComputeCache, dock_cache_enabled
from .base import Agent, AgentContext

TOOL = ToolVersion(name="ddswarm", version="0.1.0")
RDKIT_TOOL = ToolVersion(name="rdkit", version=rdBase.rdkitVersion)


class DockingMinion(Agent):
    name = "docking_minion"

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
                warnings=[f"docking engine '{engine_name}' unavailable: {msg}; docking not performed"],
                reproducibility={"reason": "engine_unavailable"},
            )

        # receptor source: explicit params['receptor'] wins; otherwise resolve the
        # primary target (spec.targets[0]) through the registry (Increment 39 —
        # docking is always-on, so it resolves a generic default receptor).
        source = self._receptor_source(ctx)
        if source is None:
            return self._result(
                ctx, AgentResultStatus.insufficient_evidence,
                warnings=["no receptor resolved (no params['receptor'] and the primary target has no registry PDB)"],
                reproducibility={"reason": "no_receptor"},
            )

        molecules = self._collect_molecules(ctx)
        if not molecules:
            return self._result(
                ctx, AgentResultStatus.insufficient_evidence,
                warnings=["no molecules to dock (screened_batch survivors or starting_materials empty)"],
                reproducibility={"reason": "no_molecules"},
            )

        work_dir = Path(self.store.run_dir(ctx.project_id, ctx.run_id)) / "docking" / ctx.task_id
        cache = ComputeCache(self.store, dock_cache_enabled(ctx.params))
        if cache.enabled and hasattr(engine, "receptor_cache_dir"):
            engine.receptor_cache_dir = cache.root / "receptor"
        receptor = engine.prepare_receptor(source, work_dir)
        if not receptor.identifier:
            receptor.identifier = source.identifier

        # protocol / search box
        box_center, box_size = self._resolve_box(ctx, receptor)
        if box_center is None:
            return self._result(
                ctx, AgentResultStatus.insufficient_evidence,
                warnings=[(receptor.note or "receptor not prepared") or "no search box resolvable (no box in params, no reference ligand)"],
                reproducibility={"reason": "no_box"},
            )
        protocol = DockProtocol(
            box_center=box_center,
            box_size=box_size,
            exhaustiveness=int(ctx.params.get("exhaustiveness", 8)),
            num_modes=int(ctx.params.get("num_modes", 9)),
            seed=int(ctx.params.get("seed", 42)),
        )

        # dock each molecule
        poses: list[DockedPose] = []
        docked = 0
        failed = 0
        redocking: RedockingValidation | None = None
        redocking_threshold = float(ctx.params.get("redocking_threshold", 2.0))

        # prepare ligands (sequential, cheap), then dock all of them concurrently;
        # results are consumed in molecule order so the artifact is order-stable
        ligands: dict[str, object] = {}
        jobs: list[DockJob] = []
        for mid, smi in molecules:
            ligand = engine.prepare_ligand(smi, work_dir / f"lig_{mid}")
            ligands[mid] = ligand
            if ligand.pdbqt_path:
                jobs.append(DockJob(mid, source.identifier, receptor, ligand, protocol,
                                    work_dir / f"dock_{mid}"))
        docked_by_mid = dock_jobs(engine, cache, jobs, ctx.params.get("max_workers"))

        for mid, _smi in molecules:
            ligand = ligands[mid]
            if not ligand.pdbqt_path:
                failed += 1
                continue
            engine_poses = docked_by_mid.get(mid) or []
            if not engine_poses:
                failed += 1
                continue
            docked += 1
            for ep in engine_poses:
                poses.append(DockedPose(
                    molecule_id=mid,
                    rank=ep.rank,
                    affinity_kcal_per_mol=ep.affinity_kcal_per_mol,
                    rmsd_lower_bound=ep.rmsd_lower_bound,
                    rmsd_upper_bound=ep.rmsd_upper_bound,
                    pose_file=str(ep.pose_file) if ep.pose_file else None,
                    interactions=[],  # reserved: interaction extraction lands in a later increment
                ))
            # redocking validation against the co-crystal ligand (if it matches)
            if redocking is None and receptor.reference_ligand_mol is not None and engine_poses:
                pose_mol = pose_mol_from_coords(
                    ligand.canonical_smiles, ligand.smiles_idx_map, engine_poses[0].pose_atoms
                )
                if pose_mol is not None:
                    rmsd = heavy_atom_rmsd(receptor.reference_ligand_mol, pose_mol)
                    if rmsd is not None:
                        redocking = RedockingValidation(
                            status="passed" if rmsd <= redocking_threshold else "failed",
                            reference_ligand=receptor.reference_ligand_name,
                            reference_source=f"{source.identifier} co-crystal",
                            rmsd_angstrom=round(rmsd, 3),
                            threshold_angstrom=redocking_threshold,
                            note="heavy-atom RMSD, docked rank-1 pose vs co-crystallized ligand (symmetry-aware)",
                        )

        summary = DockingSummary(total_molecules=len(molecules), docked=docked, failed=failed)
        warnings = []
        if failed:
            warnings.append(f"{failed} molecule(s) failed docking/preparation")

        input_hash = hash_jsonable({"receptor": source.identifier, "molecules": molecules,
                                    "box": [box_center, box_size], "seed": protocol.seed})
        output_hash = hash_jsonable([p.model_dump() for p in poses])

        receptor_info = self._receptor_info(receptor, source)
        batch = DockingResult(
            project_id=ctx.project_id,
            status=AgentResultStatus.success,
            receptor=receptor_info,
            protocol=DockingProtocol(
                engine=engine_name,
                engine_version=engine.version,
                search_box=SearchBox(center=list(box_center), size=list(box_size)),
                exhaustiveness=protocol.exhaustiveness,
                num_modes=protocol.num_modes,
                seed=protocol.seed,
                note="search box centred on co-crystal ligand centroid" if receptor.reference_ligand_mol else None,
            ),
            poses=poses,
            redocking_validation=redocking,
            summary=summary,
            provenance=[ProvenanceEntry(
                source="molecular docking (AutoDock Vina)",
                url=source.url,
                identifier=source.identifier,
                tool_versions=[TOOL, RDKIT_TOOL, ToolVersion(name=engine_name, version=engine.version)],
                params={"engine": engine_name, "box": [list(box_center), list(box_size)],
                        "exhaustiveness": protocol.exhaustiveness, "seed": protocol.seed},
                input_hash=input_hash,
                output_hash=output_hash,
            )],
        )
        art = self.persist_artifact(ctx, "docking_result", f"dr_{ctx.project_id}", batch)
        findings = [f"docked {docked}/{len(molecules)} molecules; "
                    f"best affinity {min((p.affinity_kcal_per_mol for p in poses if p.affinity_kcal_per_mol is not None), default=None)} kcal/mol"]
        if redocking and redocking.rmsd_angstrom is not None:
            findings.append(f"redocking RMSD {redocking.rmsd_angstrom} A ({redocking.status})")
        return self._result(
            ctx, AgentResultStatus.success,
            input_refs=ctx.input_refs, artifacts=[art], findings=findings, warnings=warnings,
            provenance=[ProvenanceEntry(source="molecular docking", output_hash=art.hash,
                                        tool_versions=[TOOL, RDKIT_TOOL])],
            reproducibility={"input_hash": input_hash, "engine": engine_name},
        )

    def _resolve_box(self, ctx, receptor):
        box = ctx.params.get("box") or {}
        if box.get("center") and box.get("size"):
            return (tuple(box["center"]), tuple(box["size"]))
        if receptor.reference_ligand_mol is not None:
            margin = float(ctx.params.get("box_margin", 11.25))
            return box_from_reference_mol(receptor.reference_ligand_mol, margin=margin)
        return (None, None)

    @staticmethod
    def _default_receptor(ctx: AgentContext) -> dict | None:
        """Resolve spec.targets[0] to a receptor config through the registry.

        Returns {name, pdb_id} when the primary target has a registry entry with
        a PDB id, else None (the caller degrades to a non-blocking gap).
        """
        targets = [canonical_receptor_name(str(t)) for t in (ctx.spec.targets or [])]
        if not targets:
            return None
        entry = registry_entry(targets[0])
        if not entry or not entry.get("pdb_id"):
            return None
        return {"name": entry["name"], "pdb_id": entry["pdb_id"]}

    def _receptor_source(self, ctx: AgentContext):
        """Build a ReceptorSource from explicit params['receptor'] or the target.

        Accepts both "identifier" (engine shape) and "pdb_id" (registry shape) —
        ``_default_receptor`` returns the registry shape, so reading only
        "identifier" produced an empty id and a malformed RCSB URL (".pdb").
        Returns None when no receptor can be resolved (caller reports a gap).
        """
        receptor_cfg = ctx.params.get("receptor") or {}
        if not receptor_cfg.get("identifier") and not receptor_cfg.get("pdb_id") and not receptor_cfg.get("kind"):
            receptor_cfg = self._default_receptor(ctx) or {}
        if not receptor_cfg:
            return None
        from ..docking.base import ReceptorSource
        return ReceptorSource(
            kind=receptor_cfg.get("kind", "pdb_id"),
            identifier=str(receptor_cfg.get("identifier") or receptor_cfg.get("pdb_id") or ""),
            url=receptor_cfg.get("url"),
        )

    @staticmethod
    def _receptor_info(receptor, source) -> ReceptorInfo:
        return ReceptorInfo(
            source="RCSB PDB" if source.kind == "pdb_id" else "file",
            identifier=source.identifier,
            url=source.url,
            preparation_tool=receptor.preparation_tool or None,
            preparation_status="prepared" if receptor.pdbqt_path else "failed",
            qc_status=("issues" if any(c.status == "issue" for c in receptor.qc_checks) else "ok")
                      if receptor.qc_checks else "not_run",
            qc_checks=[{"check": c.check, "status": c.status, "detail": c.detail} for c in receptor.qc_checks],
            atom_count=receptor.atom_count,
            residue_count=receptor.residue_count,
            chain_count=receptor.chain_count,
            note=receptor.note,
        )

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

    def _result(self, ctx: AgentContext, status: AgentResultStatus, **kw) -> AgentResult:
        return AgentResult(
            project_id=ctx.project_id, agent=self.name, task_id=ctx.task_id, status=status,
            tool_versions=[TOOL, RDKIT_TOOL], **kw,
        )
