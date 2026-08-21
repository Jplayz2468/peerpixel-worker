"""Where this install keeps its Python, and how it starts a working child.

The app is deliberately split in two. The part serving the window is standard
library only and runs on a bare interpreter, so it comes up in a second and
stays answering while a card is pinned. Everything heavy -- torch, diffusers,
the Hub -- runs in a child out of the project environment, which may not even
exist yet the first time somebody opens the app.

That split is what lets the dependency install have a progress bar. An app that
needed torch in order to start could only ever install torch in silence.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = "3.12"

#: Where uv puts itself, on each platform, when its own installer runs. Looking
#: here rather than trusting PATH is not paranoia: uv lands in ~/.local/bin,
#: which is not on the PATH of a shell that was already open, and a bare
#: "uv: command not found" reads to a person like the app is broken.
CANDIDATES = (
    "UV_INSTALL_DIR",
    "XDG_BIN_HOME",
)
PLACES = (
    Path.home() / ".local" / "bin",
    Path.home() / ".cargo" / "bin",
    Path("/opt/homebrew/bin"),
    Path("/usr/local/bin"),
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "uv",
)


def uv() -> str | None:
    """The uv this install should use, wherever the launcher left it."""
    told = os.environ.get("PEERPIXEL_UV")
    if told and Path(told).exists():
        return told
    found = shutil.which("uv")
    if found:
        return found
    name = "uv.exe" if os.name == "nt" else "uv"
    for variable in CANDIDATES:
        directory = os.environ.get(variable)
        if directory and (Path(directory) / name).exists():
            return str(Path(directory) / name)
    for place in PLACES:
        if (place / name).exists():
            return str(place / name)
    return None


def venv_python() -> Path:
    """The interpreter inside the project environment, once it exists."""
    if os.name == "nt":
        return ROOT / ".venv" / "Scripts" / "python.exe"
    return ROOT / ".venv" / "bin" / "python"


def dependencies_ready() -> bool:
    """Is there an environment with the rendering stack in it?

    Presence of the interpreter is not enough -- a half-finished sync leaves
    one behind -- so look for the package that takes all the time and all the
    disk. If torch is there, the sync got to the end.
    """
    python = venv_python()
    if not python.exists():
        return False
    site = list((ROOT / ".venv").rglob("torch/version.py"))
    return bool(site)


def child(command: list[str], *, heavy: bool = True) -> list[str] | None:
    """The argv that runs one of this package's commands in a subprocess.

    Heavy work goes through the project environment. Light work -- a pairing
    call, a status check -- can run on whatever interpreter is already here,
    which means the app can do those before anything is installed.
    """
    if not heavy:
        return [sys.executable, "-m", "peerpixel", *command]
    python = venv_python()
    if python.exists():
        return [str(python), "-m", "peerpixel", *command]
    found = uv()
    if not found:
        return None
    return [found, "run", "--project", str(ROOT), "--python", PYTHON,
            "python", "-m", "peerpixel", *command]


def environment() -> dict:
    """What a child inherits. Unbuffered, so its progress arrives as it happens.

    Without PYTHONUNBUFFERED a child's stdout is a 8 KB block buffer the moment
    it is a pipe rather than a terminal, and every event this app draws would
    arrive in one burst at the end. That is the silent-installer failure with
    extra steps.
    """
    return {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        "PEERPIXEL_EVENTS": "1",
    }
