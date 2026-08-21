"""Live local hardware status, without requiring real accelerator hardware."""
from __future__ import annotations

import subprocess
import unittest
from types import SimpleNamespace

from peerpixel.system_status import SystemStatus


GIB = 1024 ** 3


class FakeClock:
    def __init__(self):
        self.now = 10.0

    def __call__(self):
        return self.now


class SystemStatusTests(unittest.TestCase):
    def test_cuda_line_reports_machine_load_vram_temperature_and_memory_mode(self):
        memory = SimpleNamespace(total=32 * GIB, available=20 * GIB)
        psutil = SimpleNamespace(cpu_percent=lambda interval=None: 12.4,
                                 virtual_memory=lambda: memory)
        cuda = SimpleNamespace(is_available=lambda: True,
                               mem_get_info=lambda: (6 * GIB, 16 * GIB))
        torch = SimpleNamespace(cuda=cuda)
        completed = subprocess.CompletedProcess([], 0, "47, 62\n", "")
        renderer = SimpleNamespace(_device="cuda", _memory_mode="group")

        line = SystemStatus(renderer, psutil_module=psutil, torch_module=torch,
                            run=lambda *args, **kwargs: completed).line()

        self.assertEqual(
            line,
            "CPU 12% · RAM 12.0/32.0 GB · GPU 47% · VRAM 10.0/16.0 GB · 62°C · group",
        )

    def test_expensive_probes_are_sampled_at_most_once_per_second(self):
        calls = []
        clock = FakeClock()
        memory = SimpleNamespace(total=8 * GIB, available=4 * GIB)

        def cpu_percent(interval=None):
            calls.append("cpu")
            return 20

        status = SystemStatus(
            SimpleNamespace(_device="cpu", _memory_mode=None),
            psutil_module=SimpleNamespace(cpu_percent=cpu_percent,
                                           virtual_memory=lambda: memory),
            torch_module=SimpleNamespace(),
            clock=clock,
        )
        first = status.line()
        second = status.line()
        clock.now += 1.0
        third = status.line()

        self.assertEqual(first, second)
        self.assertEqual(len(calls), 2)
        self.assertEqual(third, first)

    def test_missing_platform_probes_never_break_the_worker_display(self):
        def unavailable(*args, **kwargs):
            raise OSError("not installed")

        status = SystemStatus(
            SimpleNamespace(_device="cuda", _memory_mode="sequential"),
            psutil_module=SimpleNamespace(cpu_percent=unavailable,
                                           virtual_memory=unavailable),
            torch_module=SimpleNamespace(cuda=SimpleNamespace(
                is_available=lambda: True, mem_get_info=unavailable)),
            run=unavailable,
        )

        self.assertEqual(status.line(), "CPU — · RAM — · GPU — · VRAM — · —°C · sequential")

    def test_mps_uses_shared_memory_without_claiming_gpu_utilization(self):
        memory = SimpleNamespace(total=24 * GIB, available=14 * GIB)
        psutil = SimpleNamespace(cpu_percent=lambda interval=None: 7,
                                 virtual_memory=lambda: memory)
        mps = SimpleNamespace(current_allocated_memory=lambda: 3 * GIB)
        renderer = SimpleNamespace(_device="mps", _memory_mode="resident", _total=18 * GIB)

        line = SystemStatus(renderer, psutil_module=psutil,
                            torch_module=SimpleNamespace(mps=mps)).line()

        self.assertEqual(line, "CPU 7% · RAM 10.0/24.0 GB · MPS 3.0/18.0 GB · resident")


if __name__ == "__main__":
    unittest.main()
