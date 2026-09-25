"""Reasoning backends for the swarm's LLM-dependent agents (research, design)."""

from .base import ReasoningBackend, ReasoningResult
from .hermes_backend import HermesSubprocessBackend, extract_json_list
from .null_backend import NullReasoningBackend
from .registry import make_reasoning_backend

__all__ = [
    "ReasoningBackend",
    "ReasoningResult",
    "HermesSubprocessBackend",
    "NullReasoningBackend",
    "extract_json_list",
    "make_reasoning_backend",
]
