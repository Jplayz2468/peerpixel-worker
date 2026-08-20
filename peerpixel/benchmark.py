"""Steady-state benchmark shared by the terminal and local dashboard."""
from __future__ import annotations

import time

from . import api

JOB = {
    "id": "bench",
    "prompt": "a lighthouse made of blown glass",
    "seed": 1,
    "steps": 4,
}


def run_benchmark(renderer, *, submit=api.submit_bench, clock=time.time):
    """Warm kernels once, then time an identical steady-state render."""
    renderer.warm()
    renderer.render(JOB)
    started = clock()
    renderer.render(JOB)
    ms = round((clock() - started) * 1000)
    return ms, submit(ms, renderer.accelerator)
