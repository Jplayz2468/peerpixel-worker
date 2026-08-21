"""One progress bar, and the rules it has to obey.

Every long thing this worker does is watched by somebody who cannot see the
work: a 3 GB dependency sync, a 15 GB download, a model load that reads
gigabytes off a disk, fifty guided diffusion steps. The rule for all of them is
the same, and it is the reason this file exists rather than each task drawing
its own bar:

    A bar moves smoothly and constantly from the moment the work starts, and it
    carries an estimate of when it will end.

Which rules out the two bars everybody has met. The one that sits at 0% for ten
seconds while something opens a connection, and then jumps -- that is a bar
which only knows how to report bytes, and reports nothing until the first byte.
And the one that reaches 99% and stops -- that is a bar whose last phase was
never in the plan, so the work it forgot has nowhere to go but the last percent.

Three ideas do all of the work here.

**Phases with weights.** A task is a list of phases declared up front, each with
a share of the bar. Nothing appears that was not planned for, so the last
percent is a phase like any other and finishing it is visible.

**Time as a floor, never a ceiling.** Every phase carries an estimate of how
long it takes, learned from the last time this machine did it. Measured
progress -- bytes, steps, files -- always wins when there is any. When there is
not yet any, or the measurement has gone quiet, the clock keeps the bar moving
at the speed the estimate implies. So a phase is never still.

**Past the estimate, slow down; never stop and never lie.** An estimate that
turns out short must not produce a frozen bar, and must not produce a bar that
claims to be finished. Past its estimate a phase decelerates toward its own
ceiling and never arrives; only the work actually finishing takes it there.

Pure and standard library only. Nothing here knows what it is measuring, and
every function is a function of its arguments and a clock that is passed in,
which is what makes the whole decision surface testable.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

#: Where a phase's bar has got to when exactly its estimated time has passed.
#: Not 1.0: an estimate is an estimate, and a phase that reaches its end on
#: schedule and then has to wait there is the frozen bar this file exists to
#: prevent. The remaining tenth is the room to be wrong in.
ON_SCHEDULE = 0.9

#: A phase decelerating past its estimate approaches this and never reaches it.
CEILING = 0.995

#: How much of the time-based estimate is allowed to run ahead of a measurement
#: that is working. Below 1.0 so that on a phase reporting real numbers the real
#: numbers are what is displayed, and the clock only shows through in the gaps.
CREEP_TRUST = 0.85

#: How long a phase runs before the point it measures its rate from is fixed.
#: Downloads ramp and the first diffusion step carries the whole graph compile,
#: so a rate taken from the very first moment is a wild number.
ANCHOR = 1.0

#: And how much has to have happened *since* that point before the rate it
#: implies is worth believing over the shipped estimate.
SETTLE = 2.0

#: An ETA is allowed to fall as fast as it likes and to rise only this much per
#: second of wall clock. A number that jumps around is one nobody reads twice.
ETA_RISE = 0.35

#: Weight given to a new timing when folding it into what this machine has
#: learned. Low, because one cold cache should not convince the worker that
#: every future model load takes four minutes.
LEARN = 0.3


def creep(elapsed: float, estimate: float) -> float:
    """Where a phase's bar is after `elapsed` seconds, going on time alone.

    Constant speed up to the estimate, which is the part somebody watching
    actually perceives as progress. After that it keeps moving and keeps
    slowing, approaching `CEILING` without ever getting there.
    """
    if elapsed <= 0:
        return 0.0
    if estimate <= 0:
        # Nothing to go on at all. Half a minute is a shape, not a claim, and
        # it is only ever used to keep the bar alive until something measures.
        estimate = 30.0
    if elapsed <= estimate:
        return ON_SCHEDULE * elapsed / estimate
    overrun = (elapsed - estimate) / estimate
    return CEILING - (CEILING - ON_SCHEDULE) / (1.0 + overrun)


@dataclass(frozen=True)
class Phase:
    """One named stretch of a task, and its share of the whole bar."""

    name: str
    label: str
    weight: float = 1.0
    #: Seconds this phase is expected to take on this machine. Overwritten from
    #: what the machine actually did, the first time it does it.
    estimate: float = 30.0


@dataclass
class _Running:
    phase: Phase
    started: float
    done: float = 0.0
    total: float = 0.0
    detail: str = ""
    #: (clock, fraction) of the first measurement worth measuring a rate from.
    anchor: tuple[float, float] | None = None


@dataclass
class Snapshot:
    """Everything a display needs, and nothing it has to work out for itself."""

    fraction: float           # 0..1 across the whole task
    percent: float            # the same, 0..100, already rounded for text
    eta_seconds: float | None # None only when there is genuinely no basis yet
    label: str                # "Downloading the model"
    detail: str               # "9.4 GB of 15.1 GB"
    phase: str                # the phase name, for a client that wants it
    elapsed: float            # seconds since the task started
    speed: float              # fraction per second, for a client to interpolate
    finished: bool
    failed: bool = False
    error: str = ""

    def as_dict(self) -> dict:
        return {
            "fraction": round(self.fraction, 5),
            "percent": round(self.percent, 1),
            "etaSeconds": None if self.eta_seconds is None else round(self.eta_seconds, 1),
            "label": self.label,
            "detail": self.detail,
            "phase": self.phase,
            "elapsedSeconds": round(self.elapsed, 2),
            "speed": round(self.speed, 6),
            "finished": self.finished,
            "failed": self.failed,
            "error": self.error,
        }


class Tracker:
    """A task in flight: which phase it is in, and where the bar should be.

    Nothing here draws anything. It answers one question -- given the clock,
    where is the bar and when does this end -- and it answers it the same way
    for a terminal, a browser and a test.
    """

    def __init__(self, phases: list[Phase], *, clock=time.monotonic):
        if not phases:
            raise ValueError("a task with no phases has no bar")
        self.phases = list(phases)
        self.clock = clock
        self.started = clock()
        self.total_weight = sum(max(p.weight, 0.0) for p in self.phases) or 1.0

        self._index = -1
        self._running: _Running | None = None
        self._completed = 0.0        # weight of phases wholly behind us
        self._floor = 0.0            # the bar never goes backwards
        self._speed = 0.0
        self._last = (self.started, 0.0)
        self._shown_eta: float | None = None
        self._eta_at = self.started
        self.finished = False
        self.failed = False
        self.error = ""
        #: Actual durations, for whoever wants to remember them for next time.
        self.timings: dict[str, float] = {}

    # -- driving it ------------------------------------------------------

    def begin(self, name: str, *, detail: str = "") -> None:
        """Enter a phase. Everything before it counts as done.

        Skipping ahead is normal and not an error: a cached model means the
        download phase never runs, and its weight should land in the bar
        immediately rather than being quietly lost.
        """
        index = self._find(name)
        if index < self._index:
            return  # a late message from a phase already left behind
        self._close_running()
        while self._index < index - 1:
            self._index += 1
            self._completed += max(self.phases[self._index].weight, 0.0)
        self._index = index
        self._running = _Running(self.phases[index], self.clock(), detail=detail)

    def report(self, done: float, total: float = 0.0, *, detail: str | None = None) -> None:
        """A measurement from inside the current phase: bytes, steps, files."""
        if self._running is None:
            return
        self._running.done = max(0.0, float(done))
        self._running.total = max(0.0, float(total))
        if detail is not None:
            self._running.detail = detail
        if self._running.anchor is None:
            elapsed = self.clock() - self._running.started
            if elapsed >= ANCHOR:
                self._running.anchor = (self.clock(), self._measured())

    def note(self, detail: str) -> None:
        """Words for the line under the bar, without touching the numbers."""
        if self._running is not None:
            self._running.detail = detail

    def finish(self) -> None:
        """The task is done. This, and only this, is what reaches 100%."""
        self._close_running()
        self._index = len(self.phases) - 1
        self._completed = self.total_weight
        self._floor = 1.0
        self.finished = True
        self._shown_eta = 0.0

    def fail(self, error: str) -> None:
        """Stopped, badly.

        The phase is left open on purpose. Banking its weight would jerk the
        bar forward at the exact moment the work stopped, which reads as the
        failure being progress. It stays where it got to, under the name of the
        thing that was happening when it broke.
        """
        self.failed = True
        self.error = error
        self._shown_eta = None

    def _close_running(self) -> None:
        if self._running is None:
            return
        self.timings[self._running.phase.name] = self.clock() - self._running.started
        self._completed += max(self._running.phase.weight, 0.0)
        self._running = None

    def _find(self, name: str) -> int:
        for index, phase in enumerate(self.phases):
            if phase.name == name:
                return index
        raise KeyError(f"no phase named {name!r} in this task")

    # -- reading it ------------------------------------------------------

    def _measured(self) -> float:
        """The current phase's own fraction, from what it reported. 0 if silent."""
        running = self._running
        if running is None or running.total <= 0:
            return 0.0
        return min(1.0, running.done / running.total)

    def _phase_fraction(self) -> float:
        """How far through the current phase, obeying the rules at the top.

        The measurement wins when there is one. The clock is a floor under it,
        so a phase that has not measured anything yet -- or has gone quiet
        mid-way -- still moves.
        """
        running = self._running
        if running is None:
            return 0.0
        elapsed = self.clock() - running.started
        by_clock = creep(elapsed, running.phase.estimate) * CREEP_TRUST
        return min(CEILING, max(self._measured(), by_clock))

    def _phase_eta(self) -> float | None:
        """Seconds left in the current phase, measured if possible."""
        running = self._running
        if running is None:
            return 0.0
        now = self.clock()
        fraction = self._measured()

        # Rate from a fixed anchor rather than the last two samples: an ETA in
        # tens of minutes must not swing with every hiccup on the wire.
        if running.anchor is not None and fraction > 0:
            since, then = running.anchor
            span = now - since
            gained = fraction - then
            if span >= SETTLE and gained > 0:
                return max(0.0, (1.0 - fraction) / (gained / span))

        # No usable measurement, so fall back to the estimate -- and past it,
        # to what the deceleration above implies, which keeps counting down
        # instead of parking on zero.
        elapsed = now - running.started
        remaining = running.phase.estimate - elapsed
        if remaining > 0:
            return remaining
        shown = creep(elapsed, running.phase.estimate)
        if shown >= CEILING:
            return None
        return elapsed * (CEILING - shown) / max(shown, 1e-6)

    def _remaining_phases(self) -> float:
        return sum(max(p.estimate, 0.0) for p in self.phases[self._index + 1:])

    def snapshot(self) -> Snapshot:
        now = self.clock()
        weight = max(self._running.phase.weight, 0.0) if self._running else 0.0
        raw = (self._completed + weight * self._phase_fraction()) / self.total_weight
        fraction = self._floor if self.failed else min(1.0, max(self._floor, raw))
        if not self.finished:
            # Nothing but finishing is allowed to say finished.
            fraction = min(fraction, CEILING)
        self._floor = fraction

        elapsed_since, previous = self._last
        span = now - elapsed_since
        if span > 0.05:
            self._speed = max(0.0, (fraction - previous) / span)
            self._last = (now, fraction)

        return Snapshot(
            fraction=fraction,
            percent=100.0 * fraction,
            eta_seconds=self._eta(now),
            label=self._running.phase.label if self._running else self._label(),
            detail=self._running.detail if self._running else "",
            phase=self._running.phase.name if self._running else "",
            elapsed=now - self.started,
            speed=self._speed,
            finished=self.finished,
            failed=self.failed,
            error=self.error,
        )

    def _label(self) -> str:
        if self.failed:
            return "Stopped"
        if self.finished:
            return "Done"
        return self.phases[0].label

    def _eta(self, now: float) -> float | None:
        if self.finished:
            return 0.0
        if self.failed:
            return None
        here = self._phase_eta()
        target = None if here is None else here + self._remaining_phases()

        # Smoothing, so the number under the bar counts down like a clock
        # rather than flickering between two guesses. Falling is free; rising
        # is rationed, because a rising ETA reads as the work going backwards.
        span = max(0.0, now - self._eta_at)
        self._eta_at = now
        if target is None:
            self._shown_eta = None
            return None
        if self._shown_eta is None:
            self._shown_eta = target
        elif target < self._shown_eta:
            self._shown_eta = target
        else:
            self._shown_eta = min(target, self._shown_eta + ETA_RISE * span + 0.05)
        return max(0.0, self._shown_eta)


def learned(phases: list[Phase], history: dict) -> list[Phase]:
    """The same phases, with estimates from what this machine actually did.

    A cold first run uses the shipped guesses, which are deliberately generous;
    from then on the bar is calibrated to the disk and the card in front of it,
    which is the whole reason the constant-speed region is usually right.
    """
    out = []
    for phase in phases:
        seconds = history.get(phase.name)
        try:
            seconds = float(seconds)
        except (TypeError, ValueError):
            seconds = 0.0
        out.append(phase if seconds <= 0 else Phase(
            phase.name, phase.label, phase.weight, seconds))
    return out


def remember(history: dict, timings: dict) -> dict:
    """Fold a finished run's real durations into the history, gently."""
    merged = dict(history)
    for name, seconds in timings.items():
        if seconds <= 0 or not math.isfinite(seconds):
            continue
        previous = merged.get(name)
        try:
            previous = float(previous)
        except (TypeError, ValueError):
            previous = 0.0
        merged[name] = seconds if previous <= 0 else previous * (1 - LEARN) + seconds * LEARN
    return merged
