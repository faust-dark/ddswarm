"""Shared HTTP client — one timeout/retry/backoff/User-Agent implementation.

Every network call in ddswarm goes through here (IUPAC naming, grounding,
ChEMBL, receptor download). Failures return ``None`` (or a short error message)
— never a fabricated value; the caller surfaces the outcome honestly.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from ddswarm import __version__

DEFAULT_TIMEOUT = 20.0
_BACKOFF_BASE = 1.5


def user_agent(purpose: str) -> str:
    """A versioned, purpose-tagged User-Agent string."""
    return f"ddswarm/{__version__} ({purpose})"


def request(
    url: str,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = 0,
    purpose: str = "",
    max_bytes: int | None = None,
) -> tuple[bytes | None, str | None]:
    """GET ``url`` -> ``(body, error)`` — exactly one of the two is set.

    Retries ``retries`` times with backoff (1.5s, 3.0s, ...). ``max_bytes``
    bounds the read; an over-limit body is an error. HTTP errors (4xx/5xx) and
    network errors both surface as a short error string.
    """
    headers = {"User-Agent": user_agent(purpose) if purpose else f"ddswarm/{__version__}"}
    error = "request failed"
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read(max_bytes + 1) if max_bytes is not None else resp.read()
        except urllib.error.HTTPError as exc:
            error = f"HTTP {exc.code}"
        except Exception as exc:  # noqa: BLE001 - network / timeout / parse errors
            error = f"download failed: {exc}"
        else:
            if max_bytes is not None and len(data) > max_bytes:
                return None, f"response exceeds {max_bytes} bytes"
            return data, None
        if attempt < retries:
            time.sleep(_BACKOFF_BASE * (attempt + 1))
    return None, error


def get_text(
    url: str,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = 0,
    purpose: str = "",
) -> str | None:
    """GET ``url`` as decoded text, or None on any failure."""
    data, _ = request(url, timeout=timeout, retries=retries, purpose=purpose)
    return data.decode("utf-8", "replace") if data is not None else None


def get_json(
    url: str,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = 0,
    purpose: str = "",
) -> dict | None:
    """GET ``url`` as a JSON object, or None on any failure / non-dict body."""
    text = get_text(url, timeout=timeout, retries=retries, purpose=purpose)
    if text is None:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None
