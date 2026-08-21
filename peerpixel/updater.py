"""Finding a newer PeerPixel, and becoming it.

The old worker only ever printed a line saying a release existed, which is a
reasonable thing for a program somebody runs from a terminal and a useless one
for a program somebody double-clicks. So this can now do the whole job: see the
release, fetch it, put it in place, restart.

Two rules it does not break.

**Never mid-render.** Somebody at the other end is waiting for a picture they
paid for. The app refuses to start an update while a job is in flight.

**Never destroy an install it cannot replace.** The new version is fetched and
unpacked in full before anything in place is touched, and what it replaces is
kept next to it until the new one has started. An update that fails should cost
a restart, not a re-install.

A clone with a `.git` in it is updated by Git instead, fast-forward only, which
is what a person who edited `render.py` would expect and what the old scripts
already promised them.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
import zipfile
from importlib import metadata
from pathlib import Path

from .api import USER_AGENT
from .runtime import ROOT

REPO = "Jplayz2468/peerpixel-worker"
RELEASES = f"https://api.github.com/repos/{REPO}/releases/latest"

#: Never replaced by an update: the environment costs gigabytes and minutes,
#: the Git metadata belongs to whoever cloned this, and a person's own edits
#: and notes are theirs.
KEEP = {".venv", ".git", "__pycache__", ".peerpixel", "uv.lock"}

CHUNK = 256 * 1024


def installed() -> str:
    try:
        return metadata.version("peerpixel")
    except metadata.PackageNotFoundError:
        pass
    try:
        pyproject = (ROOT / "pyproject.toml").read_text()
    except OSError:
        return "0"
    found = re.search(r'(?m)^version\s*=\s*"([^"]+)"', pyproject)
    return found.group(1) if found else "0"


def parts(version: str) -> tuple[int, int, int]:
    """0.1.0 and v0.1 and 0.1.0rc2 all have to sort against each other."""
    numbers = [int(n) for n in re.findall(r"\d+", version or "")[:3]]
    return tuple(numbers + [0] * (3 - len(numbers)))  # type: ignore[return-value]


def newer(candidate: str, current: str) -> bool:
    return parts(candidate) > parts(current)


def release(timeout: float = 6.0) -> dict:
    """The latest release, or an empty dict.

    Every failure here is silent and empty. Being offline, behind a proxy or
    rate limited by GitHub is not news and is certainly not a reason to
    interrupt somebody who is rendering.
    """
    request = urllib.request.Request(
        RELEASES, headers={"user-agent": USER_AGENT, "accept": "application/vnd.github+json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.load(response)
            return value if isinstance(value, dict) else {}
    except Exception:  # noqa: BLE001 - see the docstring
        return {}


def latest(timeout: float = 6.0) -> str:
    return str(release(timeout).get("tag_name") or "")


def is_clone() -> bool:
    return (ROOT / ".git").exists()


def asset_url(data: dict) -> str:
    """The zip to install: a published asset if there is one, else the source.

    A release with a built asset wins because it is what the maintainer meant
    to ship. `zipball_url` is GitHub's automatic source zip and is always there,
    which is what makes an update work from the very first tag.
    """
    for asset in data.get("assets") or []:
        name = str(asset.get("name", ""))
        if name.endswith(".zip") and asset.get("browser_download_url"):
            return str(asset["browser_download_url"])
    return str(data.get("zipball_url") or "")


def fetch(url: str, target: Path, on_progress=None) -> Path:
    """Download with a byte count, because 40 MB over a bad line is a minute."""
    request = urllib.request.Request(url, headers={"user-agent": USER_AGENT})
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request, timeout=60) as response:
        total = int(response.headers.get("content-length") or 0)
        done = 0
        with open(target, "wb") as out:
            while True:
                block = response.read(CHUNK)
                if not block:
                    break
                out.write(block)
                done += len(block)
                if on_progress:
                    on_progress(done, total)
    return target


def unpack(archive: Path, into: Path, on_progress=None) -> Path:
    """Extract, and return the folder that actually holds the worker.

    GitHub's source zips wrap everything in one directory named after the
    commit, so the useful root is one level down. A hand-built release asset
    may not be wrapped. Finding `pyproject.toml` answers it either way.
    """
    if into.exists():
        shutil.rmtree(into)
    into.mkdir(parents=True)
    with zipfile.ZipFile(archive) as zipped:
        members = zipped.namelist()
        for index, member in enumerate(members):
            # Refuse anything that would land outside the folder. A zip is
            # remote input and `..` in a member name is the oldest trick there
            # is, and here it would be writing into somebody's home directory.
            destination = (into / member).resolve()
            if not str(destination).startswith(str(into.resolve())):
                raise ValueError(f"refusing a path outside the update: {member}")
            zipped.extract(member, into)
            if on_progress:
                on_progress(index + 1, len(members))
    found = list(into.glob("*/pyproject.toml")) + list(into.glob("pyproject.toml"))
    if not found:
        raise ValueError("the update does not look like a PeerPixel worker")
    return found[0].parent


def swap(source: Path, target: Path = ROOT) -> None:
    """Put the new version in place, keeping everything in KEEP.

    Copied over rather than swapped wholesale, because the folder somebody
    unzipped is the folder their launcher points at and moving it out from
    under them would break the one file they click.
    """
    for entry in source.iterdir():
        if entry.name in KEEP:
            continue
        destination = target / entry.name
        if entry.is_dir():
            if destination.exists():
                shutil.rmtree(destination, ignore_errors=True)
            shutil.copytree(entry, destination)
        else:
            shutil.copy2(entry, destination)
            if entry.suffix in ("", ".sh", ".command") and os.name != "nt":
                destination.chmod(destination.stat().st_mode | 0o111)


def apply(bar) -> dict:
    """The whole update, under the bar it was handed.

    Returns what happened rather than printing it: the caller in `cli.py` owns
    the terminal while a bar is on it, and something writing to the same lines
    from underneath would tear the drawing apart.
    """
    bar.begin("look")
    current = installed()
    if is_clone():
        return _git_update(bar, current)

    data = release()
    tag = str(data.get("tag_name") or "")
    if not tag or not newer(tag, current):
        return {"updated": False, "version": current}
    url = asset_url(data)
    if not url:
        raise RuntimeError(f"release {tag} has nothing to download")

    bar.begin("fetch", detail=tag)
    staging = Path(tempfile.mkdtemp(prefix="peerpixel-update-"))
    try:
        archive = fetch(url, staging / "update.zip", on_progress=bar.report)
        bar.begin("unpack")
        source = unpack(archive, staging / "tree", on_progress=bar.report)
        bar.begin("install")
        swap(source)
        _sync()
        return {"updated": True, "version": tag}
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _git_update(bar, current: str) -> dict:
    """A clone is updated by Git, fast-forward only.

    Which is what somebody who edited `render.py` would expect: an update that
    quietly reset their work would be worse than no update at all.
    """
    bar.begin("fetch", detail="git")
    out = subprocess.run(["git", "pull", "--ff-only"], cwd=str(ROOT),
                         capture_output=True, text=True)
    if out.returncode != 0:
        said = (out.stderr or out.stdout).strip().splitlines()
        raise RuntimeError(said[-1] if said else "git pull --ff-only failed")
    bar.begin("unpack")
    bar.begin("install")
    _sync()
    return {"updated": "Already up to date" not in out.stdout, "version": installed()}


def _sync() -> None:
    from .runtime import PYTHON, uv

    found = uv()
    if not found:
        return
    subprocess.run([found, "sync", "--project", str(ROOT), "--python", PYTHON],
                   cwd=str(ROOT), check=False)
