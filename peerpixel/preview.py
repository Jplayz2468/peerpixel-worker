"""The last finished picture, so the window has something to show.

One file, replaced atomically, under PEERPIXEL_HOME. Not a gallery and not a
record: a draft is never a file anywhere else in this system and it is not
going to become one here either. This is only the most recent frame, and losing
it costs nothing at all.
"""
from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path

from . import config

_lock = threading.RLock()


def path() -> Path:
    return config.HOME / "preview.jpg"


def save(jpeg: bytes) -> Path:
    """Replace the preview without ever leaving a half-written one readable."""
    with _lock:
        target = path()
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=".preview.", suffix=".tmp",
                                            dir=target.parent)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(jpeg)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(target)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        return target


def read() -> bytes | None:
    try:
        return path().read_bytes()
    except OSError:
        return None
