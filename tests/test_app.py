"""The app's own surface, with no HTTP and no subprocesses in the way."""
from __future__ import annotations

import unittest
from unittest import mock

from peerpixel import config
from peerpixel.app import App, request_allowed


class LocalOnlyTests(unittest.TestCase):
    """The window's controls start and stop rendering and spend money.

    All three checks together, and the token is the one that matters: a page on
    the internet can guess a port on 127.0.0.1, and without the token it would
    be talking to somebody's worker.
    """

    def test_local_same_origin_and_the_launch_token(self):
        self.assertTrue(request_allowed("127.0.0.1:8765", "http://127.0.0.1:8765", "s", "s"))

    def test_a_remote_host_is_refused(self):
        self.assertFalse(request_allowed("attacker.example", "http://attacker.example", "s", "s"))

    def test_a_remote_origin_on_a_local_host_is_refused(self):
        self.assertFalse(request_allowed("127.0.0.1:8765", "http://attacker.example", "s", "s"))

    def test_a_wrong_or_missing_token_is_refused(self):
        self.assertFalse(request_allowed("127.0.0.1:8765", "http://127.0.0.1:8765", "no", "s"))
        self.assertFalse(request_allowed("127.0.0.1:8765", "http://127.0.0.1:8765", "", "s"))

    def test_a_port_that_does_not_match_the_origin_is_refused(self):
        self.assertFalse(request_allowed("127.0.0.1:8765", "http://127.0.0.1:9999", "s", "s"))


class SelfDrivingTests(unittest.TestCase):
    """Nobody should have to work out that libraries come before weights."""

    def setUp(self):
        self.app = App(conduct=False)

    def queue_for(self, **steps):
        with mock.patch.object(App, "steps", return_value=steps):
            self.app.catch_up()
        return list(self.app.queue)

    def test_a_fresh_machine_queues_everything_in_the_only_order_that_works(self):
        self.assertEqual(
            self.queue_for(paired=True, dependencies=False, model=False, approved=False),
            ["install", "model", "bench"])

    def test_what_is_already_done_is_not_done_again(self):
        self.assertEqual(
            self.queue_for(paired=True, dependencies=True, model=True, approved=False),
            ["bench"])

    def test_nothing_is_queued_once_the_machine_is_ready(self):
        self.assertEqual(
            self.queue_for(paired=True, dependencies=True, model=True, approved=True), [])

    def test_an_unpaired_machine_still_installs_and_downloads(self):
        # Both take a long time and neither needs an account. Only the
        # benchmark does, because it is submitted against one.
        self.assertEqual(
            self.queue_for(paired=False, dependencies=False, model=False, approved=False),
            ["install", "model"])


class OneBarTests(unittest.TestCase):
    """Which of the things going on gets the bar."""

    def setUp(self):
        self.app = App(conduct=False)

    def pretend(self, jobs=None, worker=None):
        self.app.jobs.snapshot = lambda: jobs or {"running": False, "task": None}
        self.app.worker.snapshot = lambda: worker or {"running": False, "task": None}
        return self.app._activity()

    def test_a_render_outranks_a_download(self):
        activity = self.pretend(
            jobs={"running": True, "task": "model", "progress": {"finished": False}},
            worker={"running": True, "task": "job", "finishedAgo": None,
                    "progress": {"finished": False}, "result": {"connected": True}})
        self.assertEqual(activity["kind"], "job")

    def test_a_connected_idle_worker_has_no_bar_at_all(self):
        # There is no work to be a fraction of, and a bar for waiting is the
        # kind of lie this whole design is trying not to tell.
        activity = self.pretend(worker={"running": True, "task": "startup",
                                        "finishedAgo": 30.0,
                                        "progress": {"finished": True},
                                        "result": {"connected": True}})
        self.assertIsNone(activity["kind"])

    def test_a_worker_still_loading_its_model_does_have_one(self):
        activity = self.pretend(worker={"running": True, "task": "startup",
                                        "finishedAgo": None,
                                        "progress": {"finished": False},
                                        "result": {}})
        self.assertEqual(activity["kind"], "worker")

    def test_a_finished_setup_stops_being_the_headline_after_a_moment(self):
        fresh = self.pretend(jobs={"running": False, "task": "model", "finishedAgo": 1.0,
                                   "progress": {"finished": True}})
        stale = self.pretend(jobs={"running": False, "task": "model", "finishedAgo": 60.0,
                                   "progress": {"finished": True}})
        self.assertEqual(fresh["kind"], "setup")
        self.assertIsNone(stale["kind"])

    def test_the_overall_estimate_counts_the_steps_still_queued(self):
        self.app.queue = ["bench"]
        self.app.queued_titles = ["Checking this machine is fast enough"]
        activity = self.pretend(jobs={"running": True, "task": "model", "title": "t",
                                      "progress": {"finished": False, "etaSeconds": 100}})
        self.assertEqual(activity["steps"], 2)
        self.assertGreater(activity["overallEtaSeconds"], 100)


class StoppingTests(unittest.TestCase):
    def setUp(self):
        self.app = App(conduct=False)

    def test_stopping_gently_never_kills_a_render_in_flight(self):
        killed = []
        self.app.worker.stop = lambda: killed.append(True)
        self.app.stop_worker(after_this=True)
        self.assertEqual(killed, [])
        self.assertTrue(config.read().get("stopAfterJob"))

    def test_stopping_an_idle_worker_is_immediate(self):
        killed = []
        self.app.worker.stop = lambda: killed.append(True)
        self.app.stop_worker()
        self.assertEqual(killed, [True])
        self.assertFalse(config.read().get("stopAfterJob"))

    def test_starting_again_clears_a_stop_left_over_from_last_time(self):
        config.write(stopAfterJob=True)
        with mock.patch.object(App, "steps", return_value={"a": True}):
            self.app.worker.busy = lambda: True
            self.app.start_worker()
        self.assertFalse(config.read().get("stopAfterJob"))


if __name__ == "__main__":
    unittest.main()
