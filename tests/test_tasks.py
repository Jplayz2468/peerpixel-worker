"""The runner: one child process, one bar, and what happens when it dies.

Everything here drives a real subprocess, because the thing being tested is the
pipe. A child that buffers its output, or writes its progress to stderr, or
dies without saying why, is exactly the failure this layer exists to absorb,
and a mock of a pipe cannot reproduce any of them.
"""
from __future__ import annotations

import sys
import time
import unittest

from peerpixel import config, events
from peerpixel.tasks import Runner, Task


def child(body: str) -> Task:
    return Task("bench", [sys.executable, "-c",
                          "from peerpixel import events\n" + body])


def settle(runner: Runner, timeout: float = 20.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = runner.snapshot()
        if not snapshot["running"] and snapshot.get("task"):
            time.sleep(0.05)  # let the pump thread finish the last line
            return runner.snapshot()
        time.sleep(0.02)
    raise AssertionError("the child never finished")


class RunnerTests(unittest.TestCase):
    def test_events_move_the_bar_through_the_phases(self):
        runner = Runner()
        runner.start(child(
            "events.phase('model'); events.phase('load'); events.phase('warm')\n"
            "events.progress(2, 4, detail='step 2 of 4')\n"
            "import time; time.sleep(0.4)\n"
        ))
        seen = []
        deadline = time.monotonic() + 10
        while runner.busy() and time.monotonic() < deadline:
            seen.append(runner.snapshot()["progress"]["fraction"])
            time.sleep(0.05)
        settle(runner)
        self.assertTrue(seen, "the bar was never readable while the child ran")
        self.assertGreater(max(seen), 0.0)
        self.assertEqual(seen, sorted(seen), "the bar went backwards")

    def test_ordinary_output_becomes_the_log_and_not_progress(self):
        runner = Runner()
        runner.start(child("print('loading pipeline components')\n"
                           "events.phase('load')\n"))
        final = settle(runner)
        self.assertIn("loading pipeline components", final["log"])

    def test_a_child_that_dies_explains_itself_with_its_last_words(self):
        runner = Runner()
        runner.start(child("import sys\n"
                           "print('CUDA out of memory')\n"
                           "sys.exit(3)\n"))
        final = settle(runner)
        self.assertTrue(final["progress"]["failed"])
        self.assertIn("CUDA out of memory", final["progress"]["error"])

    def test_a_failure_leaves_the_bar_where_it_stopped(self):
        runner = Runner()
        runner.start(child("events.phase('load'); events.progress(1, 2)\n"
                           "events.failed('the disk is full')\n"
                           "import time; time.sleep(0.2)\n"))
        final = settle(runner)
        self.assertTrue(final["progress"]["failed"])
        self.assertEqual(final["progress"]["error"], "the disk is full")
        self.assertLess(final["progress"]["fraction"], 1.0,
                        "a failed run must never read as finished")

    def test_finishing_is_the_only_thing_that_reaches_a_hundred(self):
        runner = Runner()
        runner.start(child("events.phase('submit'); events.done(approved=True)\n"))
        final = settle(runner)
        self.assertEqual(final["progress"]["percent"], 100.0)
        self.assertTrue(final["progress"]["finished"])
        self.assertEqual(final["result"].get("approved"), True)

    def test_a_child_can_swap_the_plan_under_the_bar(self):
        """What the worker does every time a job arrives."""
        runner = Runner()
        runner.start(child(
            "events.emit('plan', name='job', estimates={'job.render': 12})\n"
            "events.phase('render'); events.progress(5, 50)\n"
            "import time; time.sleep(0.4)\n"))
        deadline = time.monotonic() + 10
        task = None
        while runner.busy() and time.monotonic() < deadline:
            task = runner.snapshot()["task"]
            if task == "job":
                break
            time.sleep(0.02)
        settle(runner)
        self.assertEqual(task, "job")

    def test_two_things_cannot_run_at_once(self):
        runner = Runner()
        runner.start(child("import time; time.sleep(0.5)\n"))
        with self.assertRaises(ValueError):
            runner.start(child("pass\n"))
        settle(runner)

    def test_a_finished_run_teaches_the_next_one_how_long_it_takes(self):
        runner = Runner()
        runner.start(child("import time\n"
                           "events.phase('load'); time.sleep(0.3); events.done()\n"))
        settle(runner)
        timings = config.read().get("timings") or {}
        self.assertIn("bench.load", timings)
        self.assertGreater(timings["bench.load"], 0)


class EventTests(unittest.TestCase):
    def test_an_ordinary_line_is_not_mistaken_for_an_event(self):
        self.assertIsNone(events.parse('{"event": "phase", "name": "load"}'))
        self.assertIsNone(events.parse("Downloading torch (766.4MiB)"))

    def test_a_malformed_event_is_dropped_rather_than_raising(self):
        self.assertIsNone(events.parse(events.MARK + "{not json"))
        self.assertIsNone(events.parse(events.MARK + '["a list"]'))
        self.assertIsNone(events.parse(events.MARK + '{"no": "event key"}'))

    def test_a_real_one_survives_the_round_trip(self):
        self.assertEqual(
            events.parse(events.MARK + '{"event":"progress","done":3,"total":9}'),
            {"event": "progress", "done": 3, "total": 9})


if __name__ == "__main__":
    unittest.main()
