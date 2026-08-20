"""Optional, atomic runtime state consumed by the localhost dashboard."""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path

from . import config

_lock = threading.RLock()


def state_path() -> Path:
    return config.HOME / config.DASHBOARD_STATE_FILE


def preview_path() -> Path:
    return config.HOME / config.DASHBOARD_PREVIEW_FILE


def _replace_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def read() -> dict:
    with _lock:
        try:
            value = json.loads(state_path().read_text())
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}


def publish(patch: dict) -> dict:
    with _lock:
        merged = {**read(), **patch, "updatedAt": int(time.time() * 1000)}
        _replace_bytes(
            state_path(),
            json.dumps(merged, separators=(",", ":"), sort_keys=True).encode(),
        )
        return merged


def save_preview(jpeg: bytes) -> Path:
    with _lock:
        path = preview_path()
        _replace_bytes(path, jpeg)
        return path
