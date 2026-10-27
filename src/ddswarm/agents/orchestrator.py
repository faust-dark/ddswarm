"""orchestrator_minion: interprets the project spec + task graph and emits an
ExecutionPlan — the auditable record of *why* and *how* this generation was
assembled (which stages run, in what order, how many generations, and the
convergence rule).

Deterministic by construction: it reads the task list injected via params,
the generation/iteration knobs, and prior generation memory, then writes a
structured plan + a deterministic prose rationale. No LLM required (the
reasoning lives in the derivation, which is fully reproducible).
"""

from __future__ import annotations

from ddswarm import __version__

from ..contracts.agent_result import AgentResult
from ..contracts.enums import AgentResultStatus
from ..contracts.execution_plan import ConvergenceConfig, ExecutionPlan, PlanStage
from ..contracts.provenance import ProvenanceEntry, ToolVersion
from ..logging.hashing import hash_jsonable
from ..orchestrator.memory import MemoryStore
from .base import Agent, AgentContext
from .roles import AGENT_ROLES

TOOL = ToolVersion(name="ddswarm", version=__version__)


class OrchestratorMinion(Agent):
    name = "orchestrator_minion"

    def run(self, ctx: AgentContext) -> AgentResult:
        generation = int(ctx.params.get("generation", 0))
        iterations = int(ctx.params.get("iterations", 1))
        raw_tasks = ctx.params.get("tasks", []) or []

        stages: list[PlanStage] = []
        for t in raw_tasks:
            if not isinstance(t, dict):
                continue
            agent = str(t.get("agent", ""))
            if not agent or agent == self.name:
                continue
            stage, role = AGENT_ROLES.get(agent, (agent, "contributes one stage of the pipeline"))
            stages.append(PlanStage(
                stage_id=str(t.get("task_id", agent)),
                agent=agent,
                role=role,
                depends_on=[d for d in (t.get("depends_on") or []) if d],
            ))

        # Prior generation memory feeds the rationale + the convergence record.
        prior = MemoryStore(self.store).by_kind(ctx.project_id, "generation")
        prior_best: list[str] = []
        prior_avoid: list[str] = []
        if prior:
            data = prior[-1].data or {}
            prior_best = [c.get("smiles", "") for c in data.get("best_candidates", []) if c.get("smiles")]
            prior_avoid = [s for s in data.get("avoided", []) if s]

        rationale = self._rationale(ctx, generation, iterations, len(stages),
                                    prior_best, prior_avoid)

        convergence = ConvergenceConfig(
            max_generations=max(1, iterations),
            stop_on_no_new_candidates=True,
        )
        input_hash = hash_jsonable({"tasks": raw_tasks, "generation": generation,
                                    "iterations": iterations})
        output_hash = hash_jsonable({"stages": [s.model_dump() for s in stages],
                                     "rationale": rationale,
                                     "convergence": convergence.model_dump()})

        plan = ExecutionPlan(
            project_id=ctx.project_id, run_id=ctx.run_id,
            generation=generation, stages=stages, convergence=convergence,
            rationale=rationale,
            provenance=[ProvenanceEntry(
                source="orchestration (deterministic plan derivation)",
                tool_versions=[TOOL], input_hash=input_hash, output_hash=output_hash,
                params={"generation": generation, "iterations": iterations,
                        "prior_best": len(prior_best), "prior_avoid": len(prior_avoid)},
            )],
        )
        art = self.persist_artifact(ctx, "execution_plan", f"plan_{ctx.project_id}", plan)
        return AgentResult(
            project_id=ctx.project_id, agent=self.name, task_id=ctx.task_id,
            status=AgentResultStatus.success, artifacts=[art], input_refs=ctx.input_refs,
            findings=[f"execution plan: generation {generation}, {len(stages)} stages, "
                      f"max {convergence.max_generations} generation(s)"],
            provenance=[ProvenanceEntry(source="orchestration", output_hash=art.hash,
                                        tool_versions=[TOOL])],
            tool_versions=[TOOL],
            reproducibility={"generation": generation, "stages": len(stages)},
        )

    @staticmethod
    def _rationale(ctx: AgentContext, generation: int, iterations: int, n_stages: int,
                   prior_best: list[str], prior_avoid: list[str]) -> str:
        target = ", ".join(ctx.spec.targets) or "the specified target"
        parts: list[str] = []
        parts.append(f"This is generation {generation} of at most {iterations} generation(s) "
                     f"against {target}.")
        if ctx.spec.mechanism_of_action:
            parts.append(f"The stated mechanism of action is '{ctx.spec.mechanism_of_action}'.")
        if ctx.spec.indication:
            parts.append(f"The intended indication is '{ctx.spec.indication}'.")
        if ctx.spec.adme_issue:
            parts.append(f"The ADME/PK concern driving model choice is '{ctx.spec.adme_issue}'.")
        parts.append(f"The pipeline is assembled as {n_stages} stages. Each stage writes a "
                     "versioned, hashed artifact with provenance, so the chain of evidence "
                     "from scaffold to candidate is fully auditable.")
        if prior_best:
            parts.append(f"The prior generation's best candidates "
                         f"({'; '.join(prior_best[:3])}{' ...' if len(prior_best) > 3 else ''}) "
                         "are fed back as design scaffolds.")
        if prior_avoid:
            parts.append(f"{len(prior_avoid)} prior scaffold(s) are carried forward as 'avoid' "
                         "constraints so the next design cycle does not repeat them.")
        parts.append("Candidates are ranked by a deterministic composite score "
                     "(QED + BCS bonus - 0.1 x synthetic-accessibility - 0.2 x tox alerts); "
                     "the top-ranked survivor becomes the identified lead candidate.")
        return " ".join(parts)
