"""retrosynthesis_minion: synthetic-accessibility scoring (deterministic RDKit
SA Score), a deterministic BRICS retrosynthetic disconnection (building blocks),
and, when a reasoning backend is configured, proposed synthesis routes (LLM,
evidence_level='hypothetical').

Consumes molecules from params['molecules'], then screened_batch survivors
(passed=True), then starting_materials. No molecules => insufficient_evidence.
SA score and its easy/moderate/hard classification are always *calculated*;
the BRICS disconnection is *calculated* (a deterministic structural split);
routes are optional reasoning output and never reported as demonstrated
chemistry."""

from __future__ import annotations

from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import BRICS

from ..chemistry.rdkit_ops import sanitize_and_canonicalize
from ..chemistry.sa_score import classify_accessibility, compute_sa_score
from ..contracts.agent_result import AgentResult
from ..contracts.enums import AgentResultStatus
from ..contracts.provenance import ProvenanceEntry, ToolVersion
from ..contracts.retrosynthesis_plan import (
    RetrosynthesisEntry,
    RetrosynthesisPlan,
    RetrosynthesisSummary,
    RetrosyntheticDisconnection,
    SynthesisRoute,
)
from ..logging.hashing import hash_jsonable
from ..reasoning import make_reasoning_backend
from .base import Agent, AgentContext
from ddswarm import __version__

TOOL = ToolVersion(name="ddswarm", version=__version__)
RDKIT_TOOL = ToolVersion(name="rdkit", version=rdBase.rdkitVersion)


class RetrosynthesisMinion(Agent):
    name = "retrosynthesis_minion"

    def run(self, ctx: AgentContext) -> AgentResult:
        molecules = self._collect_molecules(ctx)
        if not molecules:
            return AgentResult(
                project_id=ctx.project_id, agent=self.name, task_id=ctx.task_id,
                status=AgentResultStatus.insufficient_evidence,
                warnings=["no molecules to assess (params['molecules'], screened_batch survivors, or starting_materials empty)"],
                tool_versions=[TOOL, RDKIT_TOOL], reproducibility={"reason": "no_molecules"},
            )

        entries: list[RetrosynthesisEntry] = []
        warnings: list[str] = []
        skipped = 0
        for mid, smi in molecules:
            s = sanitize_and_canonicalize(smi)
            if not s.ok:
                skipped += 1
                continue
            mol = Chem.MolFromSmiles(s.canonical_smiles)
            sa = compute_sa_score(mol)
            acc = classify_accessibility(sa)
            disconnections = self._disconnect(mol)
            entries.append(RetrosynthesisEntry(
                id=mid, canonical_smiles=s.canonical_smiles, sa_score=sa,
                synthetic_accessibility=acc,
                disconnections=disconnections,
                feasibility_note=f"RDKit SA score {sa} => '{acc}' synthetic accessibility (calculated)",
            ))

        # ---- optional synthesis routes via reasoning backend ----
        backend_name = ctx.params.get("reasoning_backend", "none")
        used_reasoning = False
        if backend_name not in (None, "", "none"):
            backend, note = self._resolve_backend(ctx)
            if backend is None:
                warnings.append(note)
            else:
                res = backend.synthesis_routes(ctx.spec, [(e.id, e.canonical_smiles) for e in entries], ctx.params)
                if res.ok:
                    by_id: dict[str, list[SynthesisRoute]] = {}
                    for raw in res.entries:
                        mid = (raw or {}).get("molecule_id")
                        if not mid:
                            continue
                        route = SynthesisRoute(
                            steps=[str(x) for x in (raw.get("steps") or [])],
                            key_reagents=[str(x) for x in (raw.get("key_reagents") or [])],
                            procedure=(raw.get("procedure") or None),
                            references=[str(x) for x in (raw.get("references") or []) if x],
                            feasibility_note=raw.get("feasibility_note"),
                        )
                        by_id.setdefault(str(mid), []).append(route)
                    for e in entries:
                        if e.id in by_id:
                            e.routes = by_id[e.id]
                            used_reasoning = True
                else:
                    warnings.append(f"reasoning failed: {res.note}; no routes proposed")
            if used_reasoning:
                warnings.append("synthesis routes are LLM-proposed (evidence_level=hypothetical); SA score remains calculated")

        if skipped:
            warnings.append(f"skipped {skipped} molecule(s) that failed sanitization")

        summary = RetrosynthesisSummary(
            total_molecules=len(entries),
            easy=sum(1 for e in entries if e.synthetic_accessibility == "easy"),
            moderate=sum(1 for e in entries if e.synthetic_accessibility == "moderate"),
            hard=sum(1 for e in entries if e.synthetic_accessibility == "hard"),
            molecules_with_routes=sum(1 for e in entries if e.routes),
        )

        input_hash = hash_jsonable({"molecules": [(m, s) for m, s in molecules], "reasoning_backend": backend_name})
        output_hash = hash_jsonable([e.model_dump() for e in entries])

        plan = RetrosynthesisPlan(
            project_id=ctx.project_id,
            status=AgentResultStatus.success,
            molecules=entries,
            summary=summary,
            provenance=[ProvenanceEntry(
                source="RDKit SA Score" + (" + reasoning backend" if used_reasoning else ""),
                tool_versions=[TOOL, RDKIT_TOOL],
                params={"reasoning_backend": backend_name},
                input_hash=input_hash,
                output_hash=output_hash,
            )],
        )
        art = self.persist_artifact(ctx, "retrosynthesis_plan", f"rs_{ctx.project_id}", plan)
        finding = (f"SA-scored {summary.total_molecules} molecules "
                   f"(easy={summary.easy}, moderate={summary.moderate}, hard={summary.hard}, "
                   f"with_routes={summary.molecules_with_routes})")
        return AgentResult(
            project_id=ctx.project_id, agent=self.name, task_id=ctx.task_id,
            status=AgentResultStatus.success, artifacts=[art], input_refs=ctx.input_refs,
            findings=[finding], warnings=warnings,
            provenance=[ProvenanceEntry(source="RDKit SA Score", output_hash=art.hash, tool_versions=[TOOL, RDKIT_TOOL])],
            tool_versions=[TOOL, RDKIT_TOOL],
            reproducibility={"input_hash": input_hash, "reasoning_backend": backend_name},
        )

    @staticmethod
    def _disconnect(mol) -> list[RetrosyntheticDisconnection]:
        """Deterministic retrosynthetic disconnection via BRICS rules. Returns a
        single-entry list: building blocks when BRICS finds disconnection bonds,
        or an empty-fragments entry with an honest note for fused/aromatic
        scaffolds that have no BRICS-labile bond."""
        try:
            frags = list(BRICS.BRICSDecompose(mol))
        except Exception:  # noqa: BLE001
            frags = []
        if len(frags) > 1:
            return [RetrosyntheticDisconnection(
                fragments=frags,
                note=(f"BRICS rules disconnect the molecule into {len(frags)} building blocks; "
                      "[n*] markers show the attachment points. A theoretical route, "
                      "not a demonstrated synthesis."),
            )]
        return [RetrosyntheticDisconnection(
            fragments=[],
            note="no BRICS retrosynthetic disconnection point — a fused/aromatic scaffold "
                 "typically built via ring-forming chemistry rather than fragment coupling.",
        )]

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

    def _resolve_backend(self, ctx: AgentContext):
        name = ctx.params.get("reasoning_backend", "none")
        try:
            backend = make_reasoning_backend(
                name,
                command=ctx.params.get("reasoning_command"),
                max_turns=ctx.params.get("reasoning_max_turns", 30),
                run_budget=ctx.params.get("reasoning_run_budget", 600),
                audit_dir=self.store.run_dir(ctx.project_id, ctx.run_id) / "reasoning",
            )
        except KeyError as exc:
            return None, str(exc)
        avail, msg = backend.available()
        if not avail:
            return None, f"reasoning backend '{name}' unavailable: {msg}; no routes proposed"
        return backend, ""
