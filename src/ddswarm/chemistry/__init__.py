"""RDKit-backed deterministic chemistry operations (Increment 1)."""

from .rdkit_ops import compute_properties, match_alerts, sanitize_and_canonicalize

__all__ = ["compute_properties", "match_alerts", "sanitize_and_canonicalize"]
