"""Measured, monotonic generation phases shared with the browser."""
from __future__ import annotations

import time

from . import config


PHASES = (
    "preparing", "loading_flux", "enhancing_prompt", "encoding_prompt",
    "loading_style", "rendering", "decoding", "safety_check",
    "delivering", "complete",
)
_INDEX = {name: index for index, name in enumerate(PHASES)}


def valid_phase_sequence(phases) -> bool:
    try:
        indexes = [_INDEX[name] for name in phases]
    except (KeyError, TypeError):
        return False
    return all(after > before for before, after in zip(indexes, indexes[1:]))


def remember_phase(before, measured, *, alpha: float = 0.25):
    try:
        measured = float(measured)
        before = float(before) if before is not None else None
    except (TypeError, ValueError):
        return before
    if measured <= 0:
        return before
    if before is None or before <= 0:
        return round(measured)
    return round(before * (1 - alpha) + measured * alpha)


class PhaseReporter:
    def __init__(self, job_id: str, emit, *, scope: str = "generation",
                 clock=time.monotonic, wall=time.time, persist: bool = False):
        self.job_id = job_id
        self.emit = emit
        self.scope = scope
        self.clock = clock
        self.wall = wall
        self.persist_enabled = persist
        self.started = clock()
        self.current = None
        self.current_started = self.started
        self.estimates = dict((config.read().get("phaseTimings") or {}).get(scope) or {})

    def begin(self, phase: str, detail: str | None = None) -> bool:
        index = _INDEX.get(phase)
        if index is None:
            return False
        if self.current is not None and index <= _INDEX[self.current]:
            return False
        now = self.clock()
        if self.current is not None:
            elapsed = max(0, round((now - self.current_started) * 1000))
            self.estimates[self.current] = remember_phase(
                self.estimates.get(self.current), elapsed)
        self.current = phase
        self.current_started = now
        event = {
            "type": "phase", "jobId": self.job_id, "phase": phase,
            "at": round(self.wall() * 1000),
            "elapsedMs": max(0, round((now - self.started) * 1000)),
            "phaseEstimates": dict(self.estimates),
        }
        if detail:
            event["detail"] = str(detail)[:160]
        try:
            self.emit(event)
        except Exception:  # noqa: BLE001 - telemetry cannot break paid work
            pass
        if self.persist_enabled and phase == "complete":
            known = config.read().get("phaseTimings") or {}
            known[self.scope] = self.estimates
            config.write(phaseTimings=known)
        return True
