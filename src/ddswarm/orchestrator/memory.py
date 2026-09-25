"""Project memory store: a persistent, append-only, cross-run log under
store/<project_id>/memory.json. Survives across runs (unlike run artifacts)."""

from __future__ import annotations

from pathlib import Path

from ..contracts.memory import MemoryEntry, ProjectMemory
from .store import Store


class MemoryStore:
    def __init__(self, store: Store):
        self.store = store

    def path(self, project_id: str) -> Path:
        return self.store.project_dir(project_id) / "memory.json"

    def load(self, project_id: str) -> ProjectMemory:
        p = self.path(project_id)
        if not p.exists():
            return ProjectMemory(project_id=project_id)
        return ProjectMemory.model_validate_json(p.read_text())

    def append(self, entry: MemoryEntry) -> None:
        """Append unless an entry with the same id already exists (idempotent)."""
        mem = self.load(entry.project_id)
        if any(e.id == entry.id for e in mem.entries):
            return
        mem.entries.append(entry)
        p = self.path(entry.project_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(mem.model_dump_json(indent=2))

    def by_kind(self, project_id: str, kind: str) -> list[MemoryEntry]:
        return [e for e in self.load(project_id).entries if e.kind == kind]
