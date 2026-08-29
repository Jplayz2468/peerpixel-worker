"""A small parent process that replaces unhealthy native generation runtimes."""
from __future__ import annotations

import multiprocessing
import os
import threading
import time
from dataclasses import dataclass, field

from .liveness import MemorySnapshot


RESTART_DELAYS = (2, 4, 7, 12, 21, 37, 60)
UPDATE_EXIT = 75
RUNTIME_CHILD = "PEERPIXEL_RUNTIME_CHILD"
_control = None
_control_lock = threading.Lock()


def restart_delay(failures: int) -> int:
    return RESTART_DELAYS[min(max(0, int(failures)), len(RESTART_DELAYS) - 1)]


def emit_control(event: dict) -> None:
    if _control is None:
        return
    try:
        with _control_lock:
            _control.send(dict(event))
    except Exception:
        pass


@dataclass
class RuntimeState:
    started_at: float
    idle: bool = False
    completed: int = 0
    memory: MemorySnapshot = field(default_factory=MemorySnapshot)
    phase: str | None = None
    deadline: float | None = None
    ready: bool = False

    def accept(self, event: dict, *, now: float) -> None:
        kind = event.get("type")
        if kind == "ready":
            self.ready = self.idle = True
            self.phase = None
            self.deadline = None
        elif kind == "task_started":
            self.idle = False
            self.phase = None
            self.deadline = None
        elif kind == "phase":
            phase = str(event.get("phase") or "unknown")[:40]
            if phase != self.phase:
                self.phase = phase
                self.deadline = now + max(1.0, float(event.get("timeout") or 1))
        elif kind == "idle":
            self.idle = True
            if event.get("render_completed") is True:
                self.completed += 1
            elif event.get("completed") is not None:
                self.completed = max(self.completed, int(event["completed"]))
            self.phase = None
            self.deadline = None
        try:
            self.memory = MemorySnapshot(
                int(event.get("rss_bytes", self.memory.rss_bytes)),
                int(event.get("swap_bytes", self.memory.swap_bytes)),
                int(event.get("accelerator_bytes", self.memory.accelerator_bytes)),
            )
        except (TypeError, ValueError):
            pass

    def expired(self, now: float) -> bool:
        return self.deadline is not None and now >= self.deadline


@dataclass(frozen=True)
class RuntimePolicy:
    max_renders: int = 100
    max_age_seconds: float = 86400
    shutdown_grace_seconds: float = 10
    accelerator_idle_watermark: int = 0
    max_swap_bytes: int = 512 * 1024 * 1024

    def recycle_reason(self, state: RuntimeState, *, now: float,
                       physical_memory: int) -> str | None:
        if not state.idle:
            return None
        if state.memory.swap_bytes > self.max_swap_bytes:
            return "runtime_swap"
        if physical_memory > 0 and state.memory.rss_bytes > physical_memory * .9:
            return "runtime_rss"
        if (self.accelerator_idle_watermark > 0
                and state.memory.accelerator_bytes > self.accelerator_idle_watermark):
            return "runtime_accelerator"
        if state.completed >= self.max_renders:
            return "runtime_task_limit"
        if now - state.started_at >= self.max_age_seconds:
            return "runtime_age"
        return None


def _runtime_child(connection, once: bool) -> None:
    global _control
    _control = connection
    os.environ[RUNTIME_CHILD] = "1"
    try:
        from .render import Renderer
        from .worker import run
        run(Renderer(), once=once)
    finally:
        try:
            connection.close()
        except Exception:
            pass


def _stop(process, grace: float) -> None:
    if not process.is_alive():
        return
    process.terminate()
    process.join(timeout=grace)
    if process.is_alive():
        process.kill()
        process.join(timeout=2)


def supervise(*, once: bool = False, policy: RuntimePolicy | None = None,
              clock=time.monotonic, wait=time.sleep) -> int:
    """Keep one spawn-isolated generation runtime healthy until interrupted."""
    import psutil
    from .console import DIM, OFF, say

    policy = policy or RuntimePolicy()
    context = multiprocessing.get_context("spawn")
    failures = 0
    while True:
        receiving, sending = context.Pipe(duplex=False)
        process = context.Process(target=_runtime_child, args=(sending, once),
                                  name="peerpixel-runtime")
        process.start()
        sending.close()
        state = RuntimeState(clock())
        reason = None
        try:
            while process.is_alive():
                if receiving.poll(.5):
                    state.accept(receiving.recv(), now=clock())
                    if state.ready:
                        failures = 0
                now = clock()
                if state.expired(now):
                    reason = f"phase_timeout:{state.phase}"
                    break
                reason = policy.recycle_reason(
                    state, now=now, physical_memory=int(psutil.virtual_memory().total))
                if reason:
                    break
            # Reap the just-exited child before inspecting its code. A zero-time
            # join can leave exitcode as None briefly and misclassify an update
            # request as an ordinary crash.
            process.join(timeout=1)
        except (EOFError, OSError):
            pass
        except KeyboardInterrupt:
            _stop(process, policy.shutdown_grace_seconds)
            return 0
        finally:
            receiving.close()
        if reason:
            say(f"  {DIM}restarting worker runtime ({reason}){OFF}")
            _stop(process, policy.shutdown_grace_seconds)
            if once:
                return 1
            continue
        if process.exitcode == 0 or once:
            return int(process.exitcode or 0)
        if process.exitcode == UPDATE_EXIT:
            from . import runtime
            runtime.restart()
        delay = restart_delay(failures)
        failures += 1
        say(f"  {DIM}worker runtime exited ({process.exitcode}); retrying in {delay}s{OFF}")
        wait(delay)
