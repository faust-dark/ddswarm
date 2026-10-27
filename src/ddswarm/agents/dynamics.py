"""dynamics_minion: receptor molecular dynamics (OpenMM + amber14 + OBC2 GBSA
implicit solvent) to assess receptor stability (backbone RMSD) and energy
fluctuation. A short relaxation snapshot, NOT production MD, and never a
binding free energy. Ligand MD is explicitly deferred (openff-toolkit is
unavailable on Python 3.14); the MdEngine abstraction makes it a drop-in later.

Consumes the receptor from params['receptor'] (kind='pdb_id' downloads from
RCSB; kind='file' uses a local PDB path)."""

from __future__ import annotations

from pathlib import Path

from ..contracts.agent_result import AgentResult
from ..contracts.dynamics_result import DynamicsResult, DynamicsSummary, MdProtocol, MdRun
from ..contracts.enums import AgentResultStatus
from ..contracts.provenance import ProvenanceEntry, ToolVersion
from ..docking.base import download_pdb, is_safe_name, is_valid_pdb_id
from ..dynamics import make_engine
from ..logging.hashing import hash_jsonable
from ..orchestrator.checkpoint import load_md_state, md_state_key, md_state_path, write_md_meta
from .base import Agent, AgentContext
from ddswarm import __version__

TOOL = ToolVersion(name="ddswarm", version=__version__)
_RCSB_URL = "https://files.rcsb.org/download/{pid}.pdb"

# The four upstream branches that must all clear a candidate before receptor MD
# is justified (opt-in via params['gate_branches']=true, paired with depends_on
# predictor + docking + tox + formulation).
_BRANCH_KINDS = {
    "predictor": "admet_prediction",
    "docking": "docking_result",
    "tox": "tox_profile",
    "formulation": "formulation_assessment",
}


class DynamicsMinion(Agent):
    name = "dynamics_minion"

    def run(self, ctx: AgentContext) -> AgentResult:
        ok, gate_msg, eligible = self._branch_gate(ctx)
        if not ok:
            return self._result(
                ctx, AgentResultStatus.insufficient_evidence, warnings=[gate_msg],
                reproducibility={"reason": "branch_gate", "eligible_molecules": len(eligible)},
            )

        engine_name = ctx.params.get("engine", "openmm")
        try:
            engine = make_engine(engine_name)
        except KeyError as exc:
            return self._result(ctx, AgentResultStatus.failed, errors=[str(exc)])

        receptor_cfg = ctx.params.get("receptor") or {}
        kind = receptor_cfg.get("kind", "pdb_id")
        identifier = str(receptor_cfg.get("identifier", ""))
        if not identifier:
            return self._result(
                ctx, AgentResultStatus.insufficient_evidence,
                warnings=["no receptor configured (params['receptor'].identifier missing)"],
                reproducibility={"reason": "no_receptor"},
            )

        pdb_path, err = self._resolve_pdb(receptor_cfg, kind, identifier, ctx)
        if pdb_path is None:
            return self._result(ctx, AgentResultStatus.failed, errors=[err])

        protocol = MdProtocol(
            engine=engine_name,
            engine_version=getattr(engine, "version", None),
            temperature_kelvin=float(ctx.params.get("temperature_kelvin", 300.0)),
            timestep_femtoseconds=float(ctx.params.get("timestep_fs", 2.0)),
            steps=int(ctx.params.get("steps", 1000)),
            seed=int(ctx.params.get("seed", 42)),
            minimize_max_iterations=int(ctx.params.get("minimize_max_iterations", 200)),
        )
        run = self._run_with_checkpoint(engine, pdb_path, identifier, protocol, ctx)

        summary = DynamicsSummary(
            total_runs=1,
            succeeded=1 if run.status == "success" else 0,
            failed=1 if run.status == "failed" else 0,
        )
        input_hash = hash_jsonable({"receptor": identifier, "protocol": protocol.model_dump()})
        output_hash = hash_jsonable([run.model_dump()])

        result = DynamicsResult(
            project_id=ctx.project_id,
            status=AgentResultStatus.success if run.status != "failed" else AgentResultStatus.failed,
            protocol=protocol,
            runs=[run],
            summary=summary,
            provenance=[ProvenanceEntry(
                source="receptor molecular dynamics (OpenMM)",
                identifier=identifier,
                tool_versions=[TOOL, ToolVersion(name=engine_name, version=getattr(engine, "version", None))],
                params={"engine": engine_name, "steps": protocol.steps, "seed": protocol.seed,
                        "resumed": run.prior_steps > 0, "prior_steps": run.prior_steps,
                        "total_steps": run.total_steps},
                input_hash=input_hash,
                output_hash=output_hash,
            )],
        )
        art = self.persist_artifact(ctx, "dynamics_result", f"md_{ctx.project_id}", result)

        warnings: list[str] = []
        findings: list[str] = []
        if run.status == "success" and run.metrics.backbone_rmsd_mean_angstrom is not None:
            findings.append(
                f"receptor MD: backbone RMSD {run.metrics.backbone_rmsd_mean_angstrom:.2f} A "
                f"(max {run.metrics.backbone_rmsd_max_angstrom:.2f} A), "
                f"E {run.metrics.potential_energy_mean_kj_per_mol:.0f} +/- "
                f"{run.metrics.potential_energy_std_kj_per_mol:.0f} kJ/mol"
            )
            if run.prior_steps > 0:
                findings.append(
                    f"receptor MD resumed from a {run.prior_steps}-step checkpoint "
                    f"(cumulative {run.total_steps} steps); metrics over the continued segment"
                )
        elif run.status == "not_run":
            warnings.append("receptor MD not run (no MD engine configured)")
        elif run.status == "failed":
            warnings.append(f"receptor MD failed: {run.error}")
        warnings.append(protocol.note)

        return self._result(
            ctx,
            AgentResultStatus.success if run.status != "failed" else AgentResultStatus.failed,
            input_refs=ctx.input_refs, artifacts=[art], findings=findings, warnings=warnings,
            provenance=[ProvenanceEntry(source="receptor MD", output_hash=art.hash, tool_versions=[TOOL])],
            reproducibility={"input_hash": input_hash, "engine": engine_name, "receptor": identifier,
                             "resumed": run.prior_steps > 0, "prior_steps": run.prior_steps,
                             "total_steps": run.total_steps},
        )

    def _run_with_checkpoint(self, engine, pdb_path: Path, identifier: str,
                             protocol: MdProtocol, ctx: AgentContext) -> MdRun:
        """Run (or resume) receptor MD, persisting the OpenMM State (written by
        the engine directly into the store checkpoint dir) for a future
        continuation. With params['resume']=true and a prior state for the same
        (receptor, protocol-minus-steps, engine), the trajectory is CONTINUED by
        `protocol.steps` more steps instead of restarting; metrics are reported
        over the continued segment, with prior_steps/total_steps tracking the
        cumulative simulation length."""
        state_key = md_state_key(identifier, protocol, engine.name, getattr(engine, "version", None))
        state_path = md_state_path(self.store, state_key)
        resume_state: Path | None = None
        prior_steps = 0
        if ctx.params.get("resume"):
            prior = load_md_state(self.store, state_key)
            if prior is not None:
                resume_state = prior["state_path"]
                prior_steps = prior["total_steps"]
        state_path.parent.mkdir(parents=True, exist_ok=True)
        run = engine.run_md(pdb_path, identifier, protocol,
                            resume_state=resume_state, prior_steps=prior_steps,
                            checkpoint_path=state_path)
        if run.status == "success" and state_path.exists():
            meta = write_md_meta(self.store, state_key, state_path.read_bytes(), run.total_steps)
            run.checkpoint_uri = str(state_path)
            run.checkpoint_hash = meta["state_hash"]
        return run

    def _branch_gate(self, ctx: AgentContext) -> tuple[bool, str, set[str]]:
        """Opt-in gate: receptor MD runs only when predictor + docking + tox +
        formulation all produced results for at least one candidate. Returns
        (ok, message, eligible_smiles)."""
        if not ctx.params.get("gate_branches", False):
            return True, "", set()
        missing = [b for b, k in _BRANCH_KINDS.items()
                   if k not in ctx.inputs or not ctx.inputs[k]]
        if missing:
            return (False,
                    "receptor MD gated: requires results from predictor + docking + tox + "
                    f"formulation; missing {', '.join(missing)}", set())
        smi_sets: list[set[str]] = []
        for k in ("admet_prediction", "tox_profile", "formulation_assessment"):
            s: set[str] = set()
            for art in ctx.inputs.get(k, []):
                for m in getattr(art, "molecules", []):
                    smi = getattr(m, "canonical_smiles", None)
                    if smi:
                        s.add(smi)
            smi_sets.append(s)
        eligible = set.intersection(*smi_sets) if smi_sets else set()
        docked = any(getattr(art, "poses", []) for art in ctx.inputs.get("docking_result", []))
        if not docked:
            return False, "receptor MD gated: docking produced no docked poses", set()
        if not eligible:
            return False, "receptor MD gated: no molecule cleared predictor + tox + formulation", set()
        return True, "", eligible

    def _resolve_pdb(self, receptor_cfg: dict, kind: str, identifier: str, ctx: AgentContext):
        work_dir = Path(self.store.run_dir(ctx.project_id, ctx.run_id)) / "dynamics" / ctx.task_id
        work_dir.mkdir(parents=True, exist_ok=True)
        if kind == "file":
            p = Path(receptor_cfg.get("path") or identifier)
            if not p.exists():
                return None, f"receptor file not found: {p}"
            return p, None
        url = receptor_cfg.get("url") or (_RCSB_URL.format(pid=identifier) if kind == "pdb_id" else None)
        if url is None:
            return None, f"no receptor URL for source kind '{kind}'"
        if kind == "pdb_id" and not is_valid_pdb_id(identifier):
            return None, f"invalid PDB id {identifier!r} (expected 4 alphanumeric characters)"
        if not is_safe_name(identifier):
            return None, f"invalid receptor identifier {identifier!r}"
        raw_path = work_dir / f"{identifier}_raw.pdb"
        err = download_pdb(url, raw_path)
        if err:
            return None, f"could not download receptor ({err})"
        return raw_path, None

    def _result(self, ctx: AgentContext, status: AgentResultStatus, **kw) -> AgentResult:
        return AgentResult(
            project_id=ctx.project_id, agent=self.name, task_id=ctx.task_id, status=status,
            tool_versions=[TOOL], **kw,
        )
