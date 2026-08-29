"""Bounded, content-free liveness and cleanup for expensive worker phases."""
from __future__ import annotations

import gc
import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class MemorySnapshot:
    rss_bytes: int = 0
    swap_bytes: int = 0
    accelerator_bytes: int = 0


@dataclass(frozen=True)
class PhasePulse:
    phase: str
    elapsed_ms: int
    rss_bytes: int
    swap_bytes: int
    accelerator_bytes: int


def memory_snapshot(*, torch_module=None, psutil_module=None) -> MemorySnapshot:
    try:
        if psutil_module is None:
            import psutil as psutil_module
        process = psutil_module.Process()
        rss = int(process.memory_info().rss)
        try:
            swap = int(process.memory_full_info().swap)
        except Exception:
            swap = 0
    except Exception:
        rss = swap = 0
    accelerator = 0
    try:
        if torch_module is None:
            import torch as torch_module
        if torch_module.cuda.is_available():
            accelerator = int(torch_module.cuda.memory_allocated())
        elif hasattr(torch_module, "mps") and torch_module.backends.mps.is_available():
            accelerator = int(torch_module.mps.current_allocated_memory())
    except Exception:
        pass
    return MemorySnapshot(rss, swap, accelerator)


class PhaseLease:
    """Emit repeated liveness while preserving one immutable local deadline."""

    def __init__(self, phase, timeout, emit, *, interval=5.0,
                 clock=time.monotonic, snapshot=memory_snapshot):
        self.phase = str(phase)
        self.timeout = float(timeout)
        self.emit = emit
        self.interval = float(interval)
        self.clock = clock
        self.snapshot = snapshot
        self.started = clock()
        self.deadline = self.started + self.timeout
        self._stop = threading.Event()
        self._thread = None

    def pulse(self):
        memory = self.snapshot()
        event = PhasePulse(
            self.phase, max(0, round((self.clock() - self.started) * 1000)),
            memory.rss_bytes, memory.swap_bytes, memory.accelerator_bytes,
        )
        try:
            self.emit(event)
        except Exception:
            pass
        return event

    def expired(self) -> bool:
        return self.clock() >= self.deadline

    def _repeat(self):
        while not self._stop.wait(self.interval):
            self.pulse()

    def __enter__(self):
        self.pulse()
        self._thread = threading.Thread(target=self._repeat, daemon=True,
                                        name=f"peerpixel-{self.phase}-pulse")
        self._thread.start()
        return self

    def __exit__(self, *_error):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=min(1.0, self.interval))


def redact_reason(phase: str, error: BaseException) -> str:
    if isinstance(error, TimeoutError):
        return f"phase_timeout:{str(phase)[:40]}"
    return f"{type(error).__name__}:{str(phase)[:40]}"


def cleanup_after_task(renderer, *, torch_module=None, psutil_module=None) -> MemorySnapshot:
    """Best-effort allocator cleanup that can never replace a task outcome."""
    try:
        gc.collect()
    except Exception:
        pass
    try:
        if torch_module is None:
            import torch as torch_module
        device = str(getattr(renderer, "_device", "cpu"))
        if device.startswith("cuda") and torch_module.cuda.is_available():
            torch_module.cuda.empty_cache()
        elif device == "mps" and hasattr(torch_module, "mps"):
            torch_module.mps.empty_cache()
    except Exception:
        pass
    return memory_snapshot(torch_module=torch_module, psutil_module=psutil_module)
