"""The long things this worker does, and the shape of each one's bar.

Every task declares its phases before it starts, which is the rule from
`progress.py` made concrete: the last percent of a bar has to be a phase like
any other, so it has to have been written down here first. Nothing may discover
halfway through that there is more work -- if a task can end in two different
amounts of work, both are phases and one of them gets skipped.

The estimates are shipped guesses for a cold machine, deliberately generous.
From the first successful run they are replaced by what this machine actually
did, so the constant-speed part of every bar is calibrated to the disk, the wire
and the card in front of it.
"""
from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from . import config, progress, runtime
from .console import human
from .progress import Phase, Tracker


@dataclass(frozen=True)
class Plan:
    name: str
    title: str
    phases: tuple[Phase, ...]


def plan(name: str, title: str, *phases: Phase) -> Plan:
    return Plan(name, title, tuple(phases))


PLANS: dict[str, Plan] = {
    "install": plan("install", "Installing the rendering libraries",
                    Phase("resolve", "Working out what is needed", 1, 20),
                    Phase("download", "Downloading libraries", 12, 300),
                    Phase("install", "Unpacking and linking", 4, 60)),
    "model": plan("model", "Downloading the model",
                  Phase("plan", "Asking which files to fetch", 1, 6),
                  Phase("fetch", "Downloading weights", 40, 900),
                  Phase("check", "Checking what arrived", 1, 8)),
    "bench": plan("bench", "Checking this machine is fast enough",
                  Phase("load", "Loading the model", 6, 90),
                  Phase("warm", "Warming up the card", 5, 60),
                  Phase("measure", "Timing a render", 5, 45),
                  Phase("submit", "Sending the result", 1, 4)),
    "start": plan("start", "Starting up",
                  Phase("load", "Loading the model", 8, 90),
                  Phase("connect", "Connecting to peerpixel.cc", 1, 8)),
    "job": plan("job", "Rendering",
                Phase("load", "Loading the model", 3, 60),
                Phase("wait", "Waiting for the chosen preview", 1, 4),
                Phase("render", "Rendering", 20, 90),
                Phase("deliver", "Delivering", 1, 6)),
    "update": plan("update", "Updating PeerPixel",
                   Phase("look", "Looking for a newer version", 1, 5),
                   Phase("fetch", "Downloading the update", 6, 40),
                   Phase("unpack", "Unpacking", 2, 10),
                   Phase("install", "Syncing libraries", 6, 90)),
}


def tracker(name: str, estimates: dict | None = None) -> Tracker:
    """A bar for this plan, calibrated to what this machine has done before."""
    shape = PLANS[name]
    history = {**(config.read().get("timings") or {}), **(estimates or {})}
    # Phase names repeat across plans -- "load" means the same thing to the
    # benchmark and to a job -- so the history is scoped by plan.
    scoped = {p.name: history.get(f"{name}.{p.name}") for p in shape.phases}
    made = Tracker(progress.learned(list(shape.phases), scoped))
    made.plan = name       # type: ignore[attr-defined]
    made.title = shape.title  # type: ignore[attr-defined]
    return made


def remember(made: Tracker) -> None:
    """Fold what a finished run actually took back into the history."""
    name = getattr(made, "plan", "")
    if not name or not made.timings:
        return
    config.write(timings=progress.remember(
        config.read().get("timings") or {},
        {f"{name}.{phase}": seconds for phase, seconds in made.timings.items()}))


# -- the dependency install ---------------------------------------------------
#
# The only step still done by a subprocess, because it is the one that installs
# the environment this process would otherwise have to already be inside.

#: uv narrates itself in three lines, and they are exactly the phase boundaries.
UV_MARKERS = (
    (re.compile(r"^\s*Resolved \d+ package"), "download"),
    (re.compile(r"^\s*Prepared \d+ package"), "install"),
)


def _tree_bytes(root: Path, budget: float = 1.0) -> tuple[int, bool]:
    """How much is under here, and whether the walk got to the end of it.

    A synced environment plus a warm uv cache is tens of thousands of files and
    this runs every couple of seconds while somebody watches a bar, so it gives
    up after `budget`. The flag is the point: a half-finished walk returns a
    number that is arbitrarily short, and feeding that to a progress bar makes
    it lurch. An incomplete measurement is discarded and the clock keeps the bar
    moving instead, which is what the clock is for.
    """
    if not root.exists():
        return 0, True
    deadline = time.monotonic() + budget
    total = 0
    stack = [root]
    while stack:
        if time.monotonic() > deadline:
            return total, False
        try:
            with os.scandir(stack.pop()) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat().st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return total, True


def _uv_cache() -> Path | None:
    found = runtime.uv()
    if not found:
        return None
    try:
        out = subprocess.run([found, "cache", "dir"], capture_output=True,
                             text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    path = Path(out.stdout.strip())
    return path if out.returncode == 0 and path.exists() else None


class _Disk:
    """`uv sync` reports nothing useful about itself, so watch the disk.

    It prints three lines and then goes quiet for several minutes while it
    pulls a couple of gigabytes. What it does do is put those gigabytes
    somewhere, and that is the same trick the model download uses -- and the
    only one that survives a resume, because bytes already there are never
    fetched again.
    """

    def __init__(self):
        self.roots = [runtime.ROOT / ".venv"]
        cache = _uv_cache()
        if cache:
            self.roots.append(cache)
        self.baseline, self.usable = self._measure()
        self.total = float(config.read().get("installBytes") or 0)

    def _measure(self) -> tuple[int, bool]:
        seen = [_tree_bytes(root) for root in self.roots]
        return sum(size for size, _ in seen), all(ok for _, ok in seen)

    def grown(self) -> int | None:
        if not self.usable:
            return None
        size, complete = self._measure()
        return max(0, size - self.baseline) if complete else None

    def sample(self, made: Tracker) -> None:
        grown = self.grown()
        if grown is None:
            return
        if self.total > 0:
            made.report(min(grown, self.total), self.total,
                        detail=f"{human(grown)} of {human(self.total)}")
        elif grown:
            # First time on this machine: no total to divide by, so the bar is
            # on the clock. Say the honest number underneath it anyway.
            made.note(f"{human(grown)} downloaded")

    def learn(self) -> None:
        grown = self.grown()
        if grown and grown > 100e6:
            config.write(installBytes=grown)


def install_dependencies(made: Tracker) -> None:
    """Sync the project environment, with a bar on it. Raises on failure."""
    found = runtime.uv()
    if not found:
        raise RuntimeError(
            "cannot find uv, which is what installs everything. Close this and "
            "run the launcher again, or install it from https://docs.astral.sh/uv/")

    made.begin("resolve")
    disk = _Disk()
    stop = threading.Event()

    def watch():
        while not stop.is_set():
            try:
                disk.sample(made)
            except OSError:
                pass
            stop.wait(2.0)

    threading.Thread(target=watch, daemon=True).start()
    process = subprocess.Popen(
        [found, "sync", "--project", str(runtime.ROOT), "--python", runtime.PYTHON],
        cwd=str(runtime.ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, errors="replace",
        env={**os.environ, "PYTHONUNBUFFERED": "1"})
    tail: list[str] = []
    for line in process.stdout:
        text = line.rstrip()
        if text:
            tail.append(text)
            del tail[:-40]
        for pattern, phase in UV_MARKERS:
            if pattern.search(text):
                made.begin(phase)
    process.stdout.close()
    code = process.wait()
    stop.set()
    disk.learn()
    if code != 0:
        raise RuntimeError(_why(tail, code))


def _why(tail: list[str], code: int) -> str:
    """A sentence from the last thing uv said, rather than "exit status 1"."""
    for line in reversed(tail):
        text = line.strip()
        if text and not text.startswith(("Resolved", "Prepared", "Installed", "+", "-")):
            return text[:300]
    return f"installing the libraries failed with status {code}"
