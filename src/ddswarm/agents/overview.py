"""overview_minion: the opening summary of the final report.

Placed at the TOP of the PDF. Emits an `OverviewReport` with (1) the verbatim
user prompt that started the research and (2) a plain-language overview of
mechanism of action, chemical structure, and therapeutic uses.

Two sources, kept distinct:
- Deterministic own-database facts (receptor registry for the named targets,
  the curated starting materials) — observed, always present.
- Reasoning-backend recall (LLM narrative for MoA / structure / uses) —
  inferred, present only when a backend is configured.

The agent ALWAYS succeeds: without a backend it still echoes the prompt and the
deterministic facts and leaves the narrative fields empty — never fabricating a
mechanism or indication it cannot recall. It is in DEFAULT_PIPELINE (unlike the
opt-in knowledge agents) because it has a deterministic baseline and therefore
never gates the downstream report on a missing backend.
"""

from __future__ import annotations

from ..chemistry.receptors import RECEPTOR_REGISTRY, canonical_receptor_name
from ..contracts.agent_result import AgentResult
from ..contracts.enums import AgentResultStatus
from ..contracts.overview_report import OverviewReport
from ..contracts.provenance import ProvenanceEntry, ToolVersion
from ..logging.hashing import hash_jsonable
from ..reasoning import make_reasoning_backend
from .base import Agent, AgentContext
from ddswarm import __version__

TOOL = ToolVersion(name="ddswarm", version=__version__)


class OverviewMinion(Agent):
    name = "overview_minion"

    def run(self, ctx: AgentContext) -> AgentResult:
        prompt = ctx.spec.user_prompt or ctx.spec.objective
        own_facts = self._own_database_facts(ctx.spec)

        mechanism = chemical = uses = None
        citations: list[str] = []
        note = "deterministic own-database baseline only"
        backend_note = "none"

        backend_name = ctx.params.get("reasoning_backend", "none")
        if backend_name not in (None, "", "none"):
            backend, backend_note = self._resolve_backend(ctx)
            if backend is not None:
                molecules = [(f"starting_material_{i}", smi)
                             for i, smi in enumerate(ctx.spec.starting_materials)]
                res = backend.overview(ctx.spec, molecules, ctx.params)
                if res.ok and res.entries:
                    mechanism, chemical, uses, citations, note = self._from_reasoning(res.entries[0])
                else:
                    note = f"reasoning failed ({res.note}); kept deterministic baseline only"
            else:
                note = f"reasoning backend unavailable ({backend_note}); kept deterministic baseline only"

        input_hash = hash_jsonable({
            "prompt": prompt,
            "targets": ctx.spec.targets,
            "starting_materials": ctx.spec.starting_materials,
        })
        report = OverviewReport(
            project_id=ctx.project_id,
            status=AgentResultStatus.success,
            user_prompt=prompt,
            mechanism_of_action=mechanism,
            chemical_structure=chemical,
            therapeutic_uses=uses,
            citations=citations,
            own_database_facts=own_facts,
            note=note,
            provenance=[ProvenanceEntry(
                source="ProjectSpec + own databases"
                       + (" + reasoning backend" if backend_name not in (None, "", "none") else ""),
                input_hash=input_hash,
                params={"reasoning_backend": backend_name},
                tool_versions=[TOOL],
            )],
        )
        art = self.persist_artifact(ctx, "overview_report", f"overview_{ctx.project_id}", report)
        n_narr = sum(1 for x in (mechanism, chemical, uses) if x)
        return AgentResult(
            project_id=ctx.project_id, agent=self.name, task_id=ctx.task_id,
            status=AgentResultStatus.success, artifacts=[art], input_refs=ctx.input_refs,
            findings=[f"overview assembled: prompt + {n_narr} narrative section(s) "
                      f"+ {len(own_facts)} own-database fact(s)"],
            provenance=[ProvenanceEntry(source="report overview", output_hash=art.hash,
                                        tool_versions=[TOOL])],
            tool_versions=[TOOL],
            reproducibility={"input_hash": input_hash},
        )

    @staticmethod
    def _own_database_facts(spec) -> list[str]:
        facts: list[str] = []
        seen: set[str] = set()
        for t in spec.targets:
            key = canonical_receptor_name(t)
            if key in seen:
                continue
            seen.add(key)
            e = RECEPTOR_REGISTRY.get(key)
            if e is None:
                continue
            gene = e.get("gene") or ""
            family = e.get("family") or ""
            pdb = e.get("pdb_id") or ""
            ref = e.get("reference") or ""
            parts = [f"target {t}", f"gene {gene}"] if gene else [f"target {t}"]
            if family:
                parts.append(f"family {family}")
            if pdb:
                parts.append(f"PDB {pdb}")
            fact = ", ".join(parts) + ("." if not ref else f" — {ref}.")
            facts.append(fact)
        if spec.starting_materials:
            facts.append(
                f"{len(spec.starting_materials)} starting material(s) provided as SMILES: "
                + "; ".join(spec.starting_materials) + ".")
        return facts

    @staticmethod
    def _from_reasoning(raw: dict):
        data = dict(raw or {})
        mechanism = data.get("mechanism_of_action") or None
        chemical = data.get("chemical_structure") or None
        uses = data.get("therapeutic_uses") or None
        citations = [str(c) for c in (data.get("citations") or []) if c]
        note = "LLM-recalled overview (inferred); verify against primary sources"
        return mechanism, chemical, uses, citations, note

    @staticmethod
    def _resolve_backend(ctx: AgentContext):
        name = ctx.params.get("reasoning_backend", "none")
        try:
            backend = make_reasoning_backend(
                name,
                command=ctx.params.get("reasoning_command"),
                max_turns=ctx.params.get("reasoning_max_turns", 30),
                run_budget=ctx.params.get("reasoning_run_budget", 600),
            )
        except KeyError as exc:
            return None, str(exc)
        avail, msg = backend.available()
        if not avail:
            return None, f"backend '{name}' unavailable: {msg}"
        return backend, ""
