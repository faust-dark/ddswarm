"""Small filesystem helpers: atomic writes and exclusive (no-overwrite) creates.

Atomic: content is written to a temp file in the destination directory and
moved into place with ``os.replace``, so a crash mid-write never leaves a
truncated JSON/PDB behind. Exclusive: the final name is claimed with
``os.link`` (fails if it exists), so a versioned artifact is never overwritten.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def _write_temp(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return Path(tmp)


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    """Replace ``path`` with ``data`` atomically (readers see old or new, never partial)."""
    path = Path(path)
    tmp = _write_temp(path, data)
    try:
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def atomic_write_text(path: str | Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def exclusive_write_bytes(path: str | Path, data: bytes) -> None:
    """Create ``path`` with ``data`` atomically; raise FileExistsError if it exists.

    The fully-written temp file is hard-linked to the final name (``os.link``
    refuses to overwrite), so the file is never visible half-written and an
    existing version is never clobbered."""
    path = Path(path)
    tmp = _write_temp(path, data)
    try:
        os.link(tmp, path)
    except FileExistsError:
        raise
    except OSError:
        # filesystem without hard links: fall back to an exclusive create
        with open(path, "xb") as fh:
            fh.write(data)
    finally:
        tmp.unlink(missing_ok=True)


def exclusive_write_text(path: str | Path, text: str) -> None:
    exclusive_write_bytes(path, text.encode("utf-8"))
