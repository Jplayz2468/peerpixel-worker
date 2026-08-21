"""The long things this app does, and the shape of each one's bar.

Every task declares its phases before it starts. That is the rule from
`progress.py` made concrete: the last percent of a bar has to be a phase like
any other, so it has to have been written down here first. Nothing is allowed
to discover halfway through that there is more work -- if a task can end in two
different amounts of work, both are phases and one of them gets skipped.

Estimates are the shipped guesses for a cold machine, generous on purpose. From
the first successful run they are replaced by what this machine actually did,
so the constant-speed part of every bar is calibrated to the disk, the wire and
the card in front of it.
"""
from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from . import config, events, progress, runtime
from .progress import Phase, Tracker

#: How often the app redraws from a running task, and how often a task that
#: measures the disk goes and looks.
TICK = 0.25
SAMPLE = 2.0
LOG_LINES = 400


@dataclass
class Plan:
    name: str
    title: str
    phases: list[Phase]


PLANS: dict[str, Plan] = {
    "install": Plan("install", "Installing the rendering libraries", [
        Phase("resolve", "Working out what is needed", 1, 20),
        Phase("download", "Downloading libraries", 12, 300),
        Phase("install", "Unpacking and linking", 4, 60),
    ]),
    "model": Plan("model", "Downloading the model", [
        Phase("plan", "Asking which files to fetch", 1, 6),
        Phase("fetch", "Downloading weights", 40, 900),
        Phase("check", "Checking what arrived", 1, 8),
    ]),
    "bench": Plan("bench", "Checking this machine is fast enough", [
        Phase("model", "Making sure the model is here", 1, 10),
        Phase("load", "Loading the model", 6, 90),
        Phase("warm", "Warming up the card", 5, 60),
        Phase("measure", "Timing a render", 5, 45),
        Phase("submit", "Sending the result", 1, 4),
    ]),
    "update": Plan("update", "Updating PeerPixel", [
        Phase("look", "Looking for a newer version", 1, 5),
        Phase("fetch", "Downloading the update", 6, 40),
        Phase("unpack", "Unpacking", 2, 10),
        Phase("install", "Syncing libraries", 6, 90),
        Phase("restart", "Restarting", 1, 5),
    ]),
    # The worker is a service rather than a task, so its bar is per job and the
    # child switches to this plan every time one arrives.
    "job": Plan("job", "Rendering", [
        Phase("load", "Loading the model", 3, 60),
        Phase("wait", "Waiting for the chosen draft", 1, 4),
        Phase("render", "Rendering", 20, 90),
        Phase("deliver", "Delivering", 1, 5),
    ]),
    # Starting the worker is itself a wait worth drawing: loading a 4B model
    # off a cold disk is the longest unexplained pause in the whole app.
    "startup": Plan("startup", "Starting the worker", [
        Phase("load", "Loading the model", 8, 90),
        Phase("connect", "Connecting to peerpixel.cc", 1, 8),
    ]),
}


# -- interpreting output that is not ours -------------------------------------

#: uv narrates itself in three lines and they are exactly the phase boundaries.
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
    it lurch. An incomplete measurement is discarded and the clock keeps the
    bar moving instead, which is what the clock is for.
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


class InstallWatcher:
    """The dependency sync, which reports nothing useful about itself.

    `uv sync` prints three lines and then goes quiet for several minutes while
    it pulls a couple of gigabytes. What it does do is put those gigabytes on
    the disk, so the disk is what gets measured -- the same trick the model
    download uses, and the only one that survives a resume, because bytes
    already there are never fetched again.
    """

    def __init__(self):
        self.roots = [runtime.ROOT / ".venv"]
        cache = _uv_cache_dir()
        if cache:
            self.roots.append(cache)
        self.baseline, self.usable = self._measure()
        self.total = float(config.read().get("installBytes") or 0)

    def _measure(self) -> tuple[int, bool]:
        measured = [_tree_bytes(root) for root in self.roots]
        return sum(size for size, _ in measured), all(ok for _, ok in measured)

    def _grown(self) -> int | None:
        if not self.usable:
            return None
        size, complete = self._measure()
        return max(0, size - self.baseline) if complete else None

    def sample(self, tracker: Tracker) -> None:
        grown = self._grown()
        if grown is None:
            return
        if self.total > 0:
            tracker.report(min(grown, self.total), self.total,
                           detail=f"{_human(grown)} of {_human(self.total)}")
        elif grown:
            # First run on this machine: no total to divide by, so the bar is
            # still on the clock. Say the honest number underneath it anyway.
            tracker.note(f"{_human(grown)} downloaded")

    def finished(self, tracker: Tracker) -> None:
        grown = self._grown()
        if grown and grown > 100e6:
            config.write(installBytes=grown)

    def line(self, text: str, tracker: Tracker) -> None:
        for pattern, phase in UV_MARKERS:
            if pattern.search(text):
                tracker.begin(phase)
                return


def _uv_cache_dir() -> Path | None:
    found = runtime.uv()
    if not found:
        return None
    try:
        out = subprocess.run([found, "cache", "dir"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    path = Path(out.stdout.strip())
    return path if out.returncode == 0 and path.exists() else None


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


# -- running one -------------------------------------------------------------

@dataclass
class Task:
    plan: str
    argv: list[str]
    watcher: object | None = None
    #: True for the worker, which is a service: leaving is normal, not failure.
    service: bool = False


def install_task() -> Task | None:
    found = runtime.uv()
    if not found:
        return None
    return Task("install",
                [found, "sync", "--project", str(runtime.ROOT), "--python", runtime.PYTHON],
                watcher=InstallWatcher())


def command_task(plan: str, command: list[str], *, service: bool = False) -> Task | None:
    argv = runtime.child(command)
    return None if argv is None else Task(plan, argv, service=service)


class Runner:
    """One child process, one bar, and the last few hundred lines it printed.

    The bar is rebuilt whenever the child says it has moved on to a different
    plan, which is what lets the worker -- a service with no end -- still show
    a properly phased bar for each job that comes through it.
    """

    def __init__(self):
        self.lock = threading.RLock()
        self.process: subprocess.Popen | None = None
        self.task: Task | None = None
        self.tracker: Tracker | None = None
        self.log: list[str] = []
        self.result: dict = {}
        self.stopping = False
        self.finished_at = 0.0

    # -- lifecycle
    def start(self, task: Task) -> None:
        with self.lock:
            if self.busy():
                raise ValueError("something is already running")
            self.task = task
            self.log = []
            self.result = {}
            self.stopping = False
            self.finished_at = 0.0
            self.tracker = self._tracker(task.plan)
            self.process = subprocess.Popen(
                task.argv, cwd=str(runtime.ROOT), env=runtime.environment(),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, errors="replace",
            )
            threading.Thread(target=self._pump, args=(self.process,), daemon=True).start()
            if task.watcher is not None:
                threading.Thread(target=self._sample, args=(self.process,), daemon=True).start()

    def stop(self) -> None:
        with self.lock:
            self.stopping = True
            process = self.process
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()

    def busy(self) -> bool:
        return bool(self.process and self.process.poll() is None)

    # -- the two threads
    def _tracker(self, plan_name: str, estimates: dict | None = None) -> Tracker:
        plan = PLANS.get(plan_name) or PLANS["startup"]
        history = {**(config.read().get("timings") or {}), **(estimates or {})}
        # Phase names are shared across plans ("load" means the same thing to
        # the benchmark and to a job), so scope the history by plan.
        scoped = {p.name: history.get(f"{plan.name}.{p.name}", history.get(p.name))
                  for p in plan.phases}
        tracker = Tracker(progress.learned(plan.phases, scoped))
        tracker.title = plan.title  # type: ignore[attr-defined]
        tracker.plan = plan.name    # type: ignore[attr-defined]
        return tracker

    def _pump(self, process: subprocess.Popen) -> None:
        for raw in process.stdout:
            line = raw.rstrip("\n")
            event = events.parse(line)
            with self.lock:
                if event is None:
                    if line.strip():
                        self.log.append(line)
                        del self.log[:-LOG_LINES]
                        if self.task and self.task.watcher is not None and self.tracker:
                            self.task.watcher.line(line, self.tracker)
                else:
                    self._apply(event)
        try:
            process.stdout.close()
        except OSError:
            pass
        code = process.wait()
        with self.lock:
            if self.task and self.task.watcher is not None and self.tracker:
                self.task.watcher.finished(self.tracker)
            if self.tracker and not self.tracker.finished and not self.tracker.failed:
                if code == 0 or (self.stopping and self.task and self.task.service):
                    self._complete()
                elif self.stopping:
                    self.tracker.fail("stopped")
                else:
                    self.tracker.fail(self._why(code))

    def _sample(self, process: subprocess.Popen) -> None:
        while process.poll() is None:
            with self.lock:
                if self.task and self.task.watcher is not None and self.tracker:
                    try:
                        self.task.watcher.sample(self.tracker)
                    except OSError:
                        pass
            time.sleep(SAMPLE)

    # -- events from the child
    def _apply(self, event: dict) -> None:
        kind = event.get("event")
        tracker = self.tracker
        if tracker is None:
            return
        if kind == "plan":
            self.tracker = self._tracker(str(event.get("name", "")), event.get("estimates"))
            self.finished_at = 0.0
        elif kind == "phase":
            try:
                tracker.begin(str(event.get("name", "")), detail=str(event.get("detail", "")))
            except KeyError:
                pass
        elif kind == "progress":
            tracker.report(event.get("done", 0), event.get("total", 0),
                           detail=str(event.get("detail", "")) or None)
        elif kind == "note":
            tracker.note(str(event.get("detail", "")))
        elif kind == "state":
            self.result.update({k: v for k, v in event.items() if k != "event"})
        elif kind == "done":
            self.result.update({k: v for k, v in event.items() if k != "event"})
            self._complete()
        elif kind == "failed":
            tracker.fail(str(event.get("message", "it did not say why")))

    def _complete(self) -> None:
        tracker = self.tracker
        if tracker is None:
            return
        tracker.finish()
        self.finished_at = time.monotonic()
        plan = getattr(tracker, "plan", "")
        if plan:
            history = config.read().get("timings") or {}
            config.write(timings=progress.remember(
                history, {f"{plan}.{name}": seconds for name, seconds in tracker.timings.items()}))

    def _why(self, code: int) -> str:
        """A sentence, from the last thing the child said before it died.

        Better than "exit status 1", which tells somebody staring at a dead bar
        precisely nothing. Tracebacks put the useful line last.
        """
        for line in reversed(self.log[-25:]):
            text = line.strip()
            if text and not text.startswith(("File \"", "  ", "Traceback")):
                return text[:300]
        return f"it stopped with status {code}"

    # -- reading it
    def snapshot(self) -> dict:
        with self.lock:
            if self.tracker is None:
                return {"running": False, "task": None}
            bar = self.tracker.snapshot().as_dict()
            return {
                "running": self.busy(),
                "task": getattr(self.tracker, "plan", None),
                "title": getattr(self.tracker, "title", ""),
                "finishedAgo": (time.monotonic() - self.finished_at) if self.finished_at else None,
                "progress": bar,
                "result": dict(self.result),
                "log": "\n".join(self.log[-120:]),
            }
