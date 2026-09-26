"""Provider registry. Register a real model (e.g. admet_ai) here later."""

from __future__ import annotations

from .base import AdmetProvider
from .null_provider import NullAdmetProvider
from .rule_provider import RuleBasedAdmetProvider

REGISTRY: dict[str, AdmetProvider] = {
    "null_admet": NullAdmetProvider(),
    "rule_admet": RuleBasedAdmetProvider(),
}


def resolve_providers(names: list[str]) -> list[AdmetProvider]:
    out: list[AdmetProvider] = []
    for n in names:
        if n not in REGISTRY:
            raise KeyError(f"unknown ADMET provider '{n}'; registered: {sorted(REGISTRY)}")
        out.append(REGISTRY[n])
    return out
