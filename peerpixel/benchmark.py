"""Steady-state benchmark shared by the terminal and local dashboard."""
from __future__ import annotations

import time

from . import api

# Master resolution, but nowhere near the master step count. The size is what
# catches a card that cannot hold a real render; the steps are only there to be
# timed. A full fifty-step master takes minutes, and making somebody wait that
# long to learn whether they are allowed to join -- then failing them against an
# admission limit written when a render was four steps -- would turn working
# hardware away for no reason.
BENCH_STEPS = 4
JOB = {
    "id": "bench",
    "prompt": "a lighthouse made of blown glass",
    "seed": 1,
    "steps": BENCH_STEPS,
    "operation": "master",
}


def run_benchmark(renderer, *, submit=api.submit_bench, clock=time.time):
    """Warm kernels once, then time an identical steady-state render."""
    renderer.warm()
    renderer.render(JOB)
    started = clock()
    renderer.render(JOB)
    ms = round((clock() - started) * 1000)
    return ms, submit(ms, renderer.accelerator)
