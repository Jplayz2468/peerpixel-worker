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
#: Kept for the tests and for anybody reading; the number that actually runs
#: lives in `render.OPERATIONS["bench"]`, because step count is pinned by the
#: operation and ignored from the payload. Writing it in both places is how it
#: came to run fifty steps while claiming to run four.
from .render import OPERATIONS

BENCH_STEPS = OPERATIONS["bench"]["steps"]
GENERATION_TARGET_MS = 60_000
JOB = {
    "id": "bench",
    "prompt": "a lighthouse made of blown glass",
    "seed": 1,
    "operation": "bench",
}


def estimated_master_ms(bench_ms: int) -> int:
    """Project the short steady-state sample to a fifty-step master."""
    return round(max(0, bench_ms) * OPERATIONS["master"]["steps"] / BENCH_STEPS)


def likely_generation_work(bench_ms: int) -> bool:
    return estimated_master_ms(bench_ms) <= GENERATION_TARGET_MS


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
