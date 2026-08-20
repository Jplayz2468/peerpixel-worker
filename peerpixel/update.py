"""Is there a newer worker than this one?

A notice, never an install. Someone running a box in a cupboard should hear
that a fix exists; nobody should have their renderer replaced underneath a job.

Every failure here is silent. Being offline, or behind a proxy, or rate
limited by GitHub is not a reason to stop rendering.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from importlib import metadata
from pathlib import Path

from .api import USER_AGENT

RELEASES = "https://api.github.com/repos/Jplayz2468/peerpixel-worker/releases/latest"


def installed() -> str:
    try:
        return metadata.version("peerpixel")
    except metadata.PackageNotFoundError:
        pass
    # Running straight from a clone that was never installed.
    try:
        pyproject = (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text()
    except OSError:
        return "0"
    found = re.search(r'(?m)^version\s*=\s*"([^"]+)"', pyproject)
    return found.group(1) if found else "0"


def _parts(version: str) -> tuple[int, int, int]:
    """0.1.0 and v0.1 and 0.1.0rc2 all have to sort against each other."""
    numbers = [int(n) for n in re.findall(r"\d+", version)[:3]]
    return tuple(numbers + [0] * (3 - len(numbers)))  # type: ignore[return-value]


def check(timeout: float = 2.0) -> None:
    request = urllib.request.Request(
        RELEASES, headers={"user-agent": USER_AGENT, "accept": "application/vnd.github+json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            latest = json.load(response).get("tag_name") or ""
    except urllib.error.HTTPError:
        return  # 404 until the first release is tagged, 403 when rate limited
    except Exception:  # noqa: BLE001 - offline is normal and never fatal
        return

    here = installed()
    if latest and _parts(latest) > _parts(here):
        print(f"Update available: {latest} (you have {here}) - git pull && uv sync")
