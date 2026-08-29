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
    area = (OPERATIONS["master"]["width"] * OPERATIONS["master"]["height"] /
            (OPERATIONS["bench"]["width"] * OPERATIONS["bench"]["height"]))
    return round(max(0, bench_ms) * OPERATIONS["master"]["steps"] / BENCH_STEPS * area)


def likely_generation_work(bench_ms: int) -> bool:
    return estimated_master_ms(bench_ms) <= GENERATION_TARGET_MS


def generation_warning(bench_ms: int, accelerator: str) -> str:
    if likely_generation_work(bench_ms):
        return ""
    estimate = estimated_master_ms(bench_ms)
    if "Apple silicon" in accelerator or "MLX" in accelerator:
        return (f"A 1024px render is estimated at about {estimate / 1000:.0f}s. "
                "Macs are slower than current NVIDIA workers and may receive very few image jobs; "
                "this Mac remains useful for probes and verification.")
    return (f"A full high-resolution render is estimated at about {estimate / 1000:.0f}s. "
            "Faster machines are preferred, so this machine may earn few generation credits; "
            "it remains useful for probes and verification.")


def qualify_candidate(baseline_ms: int, candidate_ms: int, *, valid,
                      quality_passed: bool) -> dict:
    """Apply the release gate to one matching warm render."""
    try:
        speedup = float(baseline_ms) / float(candidate_ms)
    except (TypeError, ValueError, ZeroDivisionError):
        speedup = 0.0
    speedup = round(max(0.0, speedup), 3)
    return {
        "speedup": speedup,
        "qualified": bool(valid is True and quality_passed and speedup >= 2.0),
        "targetSpeedup": 10.0,
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
