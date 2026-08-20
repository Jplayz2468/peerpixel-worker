"""Where this install keeps its identity.

The device token is the only secret here and the file is written 0600. It is not
a password: it authorises one machine to render, nothing else. Losing it costs a
re-pair; revoking it is one row in the devices table.
"""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path

API = os.environ.get("PEERPIXEL_API", "https://peerpixel.cc").rstrip("/")
SESSION = os.environ.get("PEERPIXEL_SESSION", "")
HOME = Path(os.environ.get("PEERPIXEL_HOME", Path.home() / ".peerpixel"))
FILE = HOME / "config.json"


def session() -> str:
    """The website's own sign-in cookie, if the owner exported one.

    Everything a worker does day to day runs on the device token. Only the
    free-work switch is different: it belongs to the account, and the API
    authenticates accounts with the browser session and nothing else.
    """
    return SESSION or read().get("session", "")


def read() -> dict:
    try:
        return json.loads(FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def write(**patch) -> dict:
    HOME.mkdir(parents=True, exist_ok=True)
    merged = {**read(), **patch}
    FILE.write_text(json.dumps(merged, indent=2))
    try:
        FILE.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass  # windows
    return merged
