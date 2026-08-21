"""The bar's rules, as tests.

Two of these are the whole point of the file they cover: a bar that has not
been told anything yet still moves, and a bar that has run past its estimate
still moves and still refuses to say it is finished.
"""
from __future__ import annotations

import unittest

from peerpixel.progress import (
    CEILING,
    ON_SCHEDULE,
    Phase,
    Tracker,
    creep,
    learned,
    remember,
)


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds
        return self.now


def tracker(clock, phases=None):
    return Tracker(phases or [
        Phase("one", "First", weight=1, estimate=10),
        Phase("two", "Second", weight=3, estimate=30),
    ], clock=clock)


class CreepTests(unittest.TestCase):
    def test_it_reaches_the_on_schedule_mark_exactly_at_the_estimate(self):
        self.assertAlmostEqual(creep(10, 10), ON_SCHEDULE)

    def test_it_is_a_constant_speed_up_to_the_estimate(self):
        # Equal slices of the expected time move the bar equal distances, which
        # is the part a person actually perceives as "it is working".
        steps = [creep(t, 10) - creep(t - 1, 10) for t in range(1, 11)]
        for step in steps:
            self.assertAlmostEqual(step, steps[0])

    def test_past_the_estimate_it_keeps_moving_and_never_arrives(self):
        overrun = [creep(t, 10) for t in (11, 20, 60, 600, 6000)]
        for earlier, later in zip(overrun, overrun[1:]):
            self.assertGreater(later, earlier, "an overrunning bar must not freeze")
        self.assertLess(overrun[-1], CEILING)

    def test_it_starts_at_zero_and_not_below(self):
        self.assertEqual(creep(0, 10), 0.0)
        self.assertEqual(creep(-5, 10), 0.0)

    def test_no_estimate_at_all_still_moves(self):
        self.assertGreater(creep(5, 0), 0.0)


class MovesFromTheStartTests(unittest.TestCase):
    def test_a_phase_that_has_measured_nothing_still_advances(self):
        clock = FakeClock()
        task = tracker(clock)
        task.begin("one")
        seen = []
        for _ in range(10):
            clock.advance(1)
            seen.append(task.snapshot().fraction)
        for earlier, later in zip(seen, seen[1:]):
            self.assertGreater(later, earlier,
                               "silence from the work is not a reason to sit at zero")

    def test_the_very_first_second_is_already_off_zero(self):
        clock = FakeClock()
        task = tracker(clock)
        task.begin("one")
        clock.advance(0.5)
        self.assertGreater(task.snapshot().fraction, 0.0)

    def test_a_measurement_overrides_the_clock_when_it_is_ahead(self):
        clock = FakeClock()
        task = tracker(clock)
        task.begin("one")
        clock.advance(1)
        task.report(9, 10)
        # One second into a ten second estimate the clock would say 9%; the
        # work says 90%, and the work is what is known rather than guessed.
        self.assertGreater(task.snapshot().fraction * 4, 0.8)

    def test_the_bar_never_goes_backwards(self):
        clock = FakeClock()
        task = tracker(clock)
        task.begin("one")
        clock.advance(5)
        high = task.snapshot().fraction
        task.report(0, 100)  # a reset, a restarted download, a retried file
        clock.advance(0.1)
        self.assertGreaterEqual(task.snapshot().fraction, high)


class NeverFinishesEarlyTests(unittest.TestCase):
    def test_only_finishing_reaches_one(self):
        clock = FakeClock()
        task = tracker(clock)
        task.begin("two")
        task.report(1, 1)
        clock.advance(10_000)
        self.assertLess(task.snapshot().fraction, 1.0)
        task.finish()
        snapshot = task.snapshot()
        self.assertEqual(snapshot.fraction, 1.0)
        self.assertTrue(snapshot.finished)
        self.assertEqual(snapshot.eta_seconds, 0.0)

    def test_a_wildly_wrong_estimate_does_not_park_the_bar(self):
        clock = FakeClock()
        task = tracker(clock)
        task.begin("one")  # estimated at ten seconds
        clock.advance(300)
        far = task.snapshot().fraction
        clock.advance(300)
        self.assertGreater(task.snapshot().fraction, far)

    def test_skipping_a_phase_banks_its_weight_immediately(self):
        clock = FakeClock()
        task = tracker(clock)
        task.begin("two")  # phase one never ran: the model was already there
        self.assertGreaterEqual(task.snapshot().fraction, 1 / 4)


class EtaTests(unittest.TestCase):
    def test_an_eta_exists_before_anything_has_been_measured(self):
        clock = FakeClock()
        task = tracker(clock)
        task.begin("one")
        clock.advance(1)
        eta = task.snapshot().eta_seconds
        self.assertIsNotNone(eta)
        # Nine seconds left of phase one plus thirty of phase two.
        self.assertAlmostEqual(eta, 39, delta=1.5)

    def test_a_measured_rate_takes_over_from_the_estimate(self):
        clock = FakeClock()
        task = tracker(clock, [Phase("only", "Only", weight=1, estimate=1000)])
        task.begin("only")
        clock.advance(4)
        task.report(10, 100)      # anchor: 10% at four seconds
        clock.advance(4)
        task.report(20, 100)      # another 10% in another four seconds
        eta = task.snapshot().eta_seconds
        self.assertIsNotNone(eta)
        # 80% left at 2.5%/s is 32 seconds, nothing like the shipped guess.
        self.assertLess(eta, 200)

    def test_the_eta_falls_freely_and_rises_slowly(self):
        clock = FakeClock()
        task = tracker(clock, [Phase("only", "Only", weight=1, estimate=100)])
        task.begin("only")
        clock.advance(5)
        task.report(50, 100)
        clock.advance(0.1)
        settled = task.snapshot().eta_seconds
        task.report(50, 1000)  # the total grew tenfold: far more work than thought
        clock.advance(1)
        risen = task.snapshot().eta_seconds
        self.assertLess(risen, settled + 1.0,
                        "an ETA that leaps upward reads as the work undoing itself")

    def test_a_failure_stops_promising_a_time(self):
        clock = FakeClock()
        task = tracker(clock)
        task.begin("one")
        clock.advance(2)
        where = task.snapshot().fraction
        task.fail("the disk is full")
        snapshot = task.snapshot()
        self.assertTrue(snapshot.failed)
        self.assertIsNone(snapshot.eta_seconds)
        self.assertAlmostEqual(snapshot.fraction, where, places=3)
        self.assertEqual(snapshot.error, "the disk is full")


class CalibrationTests(unittest.TestCase):
    def test_learned_estimates_replace_the_shipped_guesses(self):
        phases = learned([Phase("one", "First", 1, 30)], {"one": 4})
        self.assertEqual(phases[0].estimate, 4)

    def test_a_missing_or_broken_history_leaves_the_guess_alone(self):
        for history in ({}, {"one": None}, {"one": "soon"}, {"one": -3}):
            self.assertEqual(learned([Phase("one", "First", 1, 30)], history)[0].estimate, 30)

    def test_remembering_moves_toward_the_new_timing_without_leaping_to_it(self):
        merged = remember({"one": 10}, {"one": 20})
        self.assertGreater(merged["one"], 10)
        self.assertLess(merged["one"], 20)

    def test_the_first_timing_is_taken_whole(self):
        self.assertEqual(remember({}, {"one": 12})["one"], 12)

    def test_nonsense_timings_are_ignored(self):
        self.assertEqual(remember({"one": 10}, {"one": 0, "two": float("inf")}), {"one": 10})

    def test_a_run_records_what_each_phase_actually_took(self):
        clock = FakeClock()
        task = tracker(clock)
        task.begin("one")
        clock.advance(7)
        task.begin("two")
        clock.advance(11)
        task.finish()
        self.assertAlmostEqual(task.timings["one"], 7)
        self.assertAlmostEqual(task.timings["two"], 11)


if __name__ == "__main__":
    unittest.main()
