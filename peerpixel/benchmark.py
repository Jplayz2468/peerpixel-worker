"""Steady-state benchmark shared by the terminal and the app."""
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


def run_benchmark(renderer, *, submit=api.submit_bench, clock=time.time,
                  on_step=None, between=None):
    """Warm kernels once, then time an identical steady-state render.

    Two renders, and the bar has to cross both. `between` is called when the
    first finishes so the display can move to its second phase -- otherwise the
    bar fills, resets to nothing, and fills again, which reads as the benchmark
    having failed and started over.
    """
    renderer.warm()
    renderer.render(JOB, on_step=on_step)
    if between is not None:
        between()
    started = clock()
    renderer.render(JOB, on_step=on_step)
    ms = round((clock() - started) * 1000)
    return ms, submit(ms, renderer.accelerator)
