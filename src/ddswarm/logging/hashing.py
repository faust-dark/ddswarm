"""Deterministic hashing for input/output provenance."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def hash_content(content: str | bytes) -> str:
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def hash_jsonable(obj: Any) -> str:
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hash_content(canonical)
