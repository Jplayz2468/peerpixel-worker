"""Where this install keeps its Python and its libraries.

Two things live here. Finding uv, which is what installs everything and which
is famously not on the PATH of the shell that needs it. And answering whether
the rendering libraries are actually present, which is the first question the
onboarding asks and the one thing it cannot work out from a config file.
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


def python_version() -> str:
    return ".".join(str(n) for n in sys.version_info[:3])


def torch_here() -> bool:
    """Can *this* interpreter import the rendering stack?

    The only question that actually matters, and the one worth asking directly
    rather than inferring from paths.
    """
    import importlib.util

    try:
        return importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):
        return False


def running_in_venv() -> bool:
    """Is this interpreter running out of the project environment?

    `sys.prefix`, not `sys.executable`. A uv venv's `bin/python` is a symlink
    to one shared interpreter, so resolving it gives the same path whether you
    are inside the environment or on the bare interpreter beside it -- and this
    function answered yes to both. The launcher then never switched, and the
    first thing to want torch died with ModuleNotFoundError.

    sys.prefix is the environment's own directory and is not shared.
    """
    try:
        return Path(sys.prefix).resolve() == (ROOT / ".venv").resolve()
    except OSError:
        return False


def use_venv() -> None:
    """Carry on inside the project environment, in place.

    The launcher starts this program on a bare interpreter, because that is the
    only interpreter that exists before anything is installed -- and being able
    to start there is what lets the install have a progress bar rather than
    being the silence before one. The moment the libraries are actually present
    the same command is re-run on the interpreter that can see them, keeping
    its arguments, its environment and its terminal.

    Nothing happens if there is no environment yet, or if this is already it.
    """
    if torch_here():
        return  # already somewhere that can render
    python = venv_python()
    if not python.exists() or not dependencies_ready() or running_in_venv():
        # Nowhere better to go. Being already in the environment and still
        # unable to import torch is a broken install, not a wrong interpreter,
        # and re-running would loop.
        return
    argv = [str(python), "-m", "peerpixel", *sys.argv[1:]]
    if os.name == "nt":
        # execv on Windows detaches the console from the parent in ways that
        # lose Ctrl-C, so wait for the child and inherit its answer instead.
        import subprocess

        raise SystemExit(subprocess.run(argv, cwd=str(ROOT)).returncode)
    os.execv(str(python), argv)
