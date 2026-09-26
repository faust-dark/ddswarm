"""Opt-in IUPAC naming via the NIH CACTUS web service.

IUPAC names are a HUMAN-READABILITY convenience only — they are never the
canonical identifier. Canonical SMILES + InChIKey remain the machine backbone
for every compute step (screening, docking, BRICS, analog generation, ADMET
descriptors). The service is network-dependent and third-party, so its output
is non-deterministic across time and MUST be marked as such in provenance.
"""

from __future__ import annotations

import urllib.parse
import urllib.request

CACTUS_BASE = "https://cactus.nci.nih.gov/chemical/structure"


def fetch_iupac_name(smiles: str, timeout: float = 15.0) -> str | None:
    """Return the IUPAC name for a SMILES from NIH CACTUS, or None on any error
    (network down, unknown molecule, malformed/empty response)."""
    try:
        encoded = urllib.parse.quote(smiles, safe="")
        url = f"{CACTUS_BASE}/{encoded}/iupac_name"
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            name = resp.read().decode("utf-8", "replace").strip()
        return name or None
    except Exception:
        return None
