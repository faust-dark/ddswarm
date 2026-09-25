"""Agent base class + execution context."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ..contracts.agent_result import AgentResult
from ..contracts.project_spec import ProjectSpec
from ..contracts.provenance import ArtifactRef, InputRef

if TYPE_CHECKING:
    from ..orchestrator.store import Store


class AgentContext(BaseModel):
    project_id: str
    run_id: str
    task_id: str
    spec: ProjectSpec
    params: dict[str, Any] = Field(default_factory=dict)
    inputs: dict[str, list[Any]] = Field(default_factory=dict)  # kind -> loaded upstream artifacts
    input_refs: list[InputRef] = Field(default_factory=list)


class Agent(ABC):
    """A swarm minion. Reasons over its context, computes nothing by hand that
    a tool should produce, and returns a fully-provenanced AgentResult."""

    name: str

    def __init__(self, store: Store):
        self.store = store

    @abstractmethod
    def run(self, ctx: AgentContext) -> AgentResult:
        ...

    def persist_artifact(
        self, ctx: AgentContext, kind: str, artifact_id: str, payload: BaseModel
    ) -> ArtifactRef:
        path, version, h = self.store.save_artifact(ctx.project_id, ctx.run_id, payload, kind, artifact_id)
        return ArtifactRef(artifact_id=artifact_id, kind=kind, uri=str(path), version=version, hash=h)
