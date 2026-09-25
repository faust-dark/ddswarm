"""Append-only JSONL event log."""

from __future__ import annotations

import json
from pathlib import Path

from ..contracts.event_log import EventLog


class EventWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: EventLog) -> None:
        line = event.model_dump_json()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        out: list[dict] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out
