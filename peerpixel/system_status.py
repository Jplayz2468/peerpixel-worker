"""A small, local-only snapshot of the machine running this worker."""
from __future__ import annotations

import subprocess
import threading
import time


GIB = 1024 ** 3


def _gb(value) -> str:
    return f"{float(value) / GIB:.1f}"


def _percent(value) -> str:
    return f"{round(float(value))}%"


class SystemStatus:
    """Sample hardware no more than once a second and format one terminal line."""

    def __init__(self, renderer, *, psutil_module=None, torch_module=None,
                 run=subprocess.run, clock=time.monotonic):
        if psutil_module is None:
            import psutil as psutil_module
        if torch_module is None:
            import torch as torch_module
        self.renderer = renderer
        self.psutil = psutil_module
        self.torch = torch_module
        self.run = run
        self.clock = clock
        self._sampled_at = float("-inf")
        self._line = ""
        self._lock = threading.Lock()

    def line(self) -> str:
        with self._lock:
            now = self.clock()
            if self._line and now - self._sampled_at < 1.0:
                return self._line
            self._line = self._sample()
            self._sampled_at = now
            return self._line

    def _sample(self) -> str:
        try:
            cpu = _percent(self.psutil.cpu_percent(interval=None))
        except Exception:  # noqa: BLE001 - status must never stop paid work
            cpu = "—"
        try:
            memory = self.psutil.virtual_memory()
            ram = f"{_gb(memory.total - memory.available)}/{_gb(memory.total)} GB"
        except Exception:  # noqa: BLE001
            ram = "—"

        parts = [f"CPU {cpu}", f"RAM {ram}"]
        device = str(getattr(self.renderer, "_device", "cpu"))
        if device.startswith("cuda"):
            parts.extend(self._cuda())
        elif device == "mps":
            parts.append(self._mps())

        mode = getattr(self.renderer, "_memory_mode", None)
        if mode:
            parts.append(str(mode))
        return " · ".join(parts)

    def _cuda(self) -> list[str]:
        utilization = temperature = vram = "—"
        try:
            answer = self.run([
                "nvidia-smi",
                "--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
                "--id=0",
            ], capture_output=True, text=True, timeout=0.8, check=True)
            values = answer.stdout.strip().splitlines()[0].split(",")
            utilization = _percent(values[0].strip())
            temperature = str(round(float(values[1].strip())))
            used = float(values[2].strip()) * 1024 ** 2
            total = float(values[3].strip()) * 1024 ** 2
            vram = f"{_gb(used)}/{_gb(total)} GB"
        except Exception:  # noqa: BLE001
            pass
        return [f"GPU {utilization}", f"VRAM {vram}", f"{temperature}°C"]

    def _mps(self) -> str:
        try:
            used = self.torch.mps.current_allocated_memory()
            total = getattr(self.renderer, "_total", 0)
            if not total:
                raise ValueError("MPS memory limit unavailable")
            return f"MPS {_gb(used)}/{_gb(total)} GB"
        except Exception:  # noqa: BLE001
            return "MPS —"
