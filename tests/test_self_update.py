"""Updating itself on the way in.

The launcher runs `peerpixel` and nothing else, so if this does not happen here
it does not happen at all -- and a worker on a version the network has stopped
giving work to sits connected, idle and unpaid without ever saying why.

What is tested is the decision, not the download. Whether a zip unpacks safely
is `test_updater.py`; this is about when it is allowed to run at all.
"""
from __future__ import annotations

import os
import unittest
from unittest import mock

from peerpixel import cli, config


class Recorder:
    def __init__(self, latest="v9.0.0", installed="0.1.0", updated=True):
        self.latest_value = latest
        self.installed_value = installed
        self.updated = updated
        self.applied = 0
        self.restarted = 0
        self.marker = None

    def latest(self, timeout=0):
        return self.latest_value

    def installed(self):
        return self.installed_value

    def apply(self, bar):
        self.applied += 1
        return {"updated": self.updated, "version": self.latest_value}


def run(recorder, environ=None):
    with mock.patch.multiple(cli.updater, latest=recorder.latest,
                             installed=recorder.installed, apply=recorder.apply), \
         mock.patch.object(cli.runtime, "restart",
                           lambda: setattr(recorder, "restarted", recorder.restarted + 1)), \
         mock.patch.dict(os.environ, environ or {}, clear=False):
        cli.self_update()
        # Read inside: patch.dict puts the environment back on the way out, and
        # what is being checked is what the restart would have inherited.
        recorder.marker = os.environ.get(cli.UPDATED)
    return recorder


class WhenItRunsTests(unittest.TestCase):
    def setUp(self):
        config.write(updateMode="auto")

    def test_a_newer_release_is_installed_and_restarted_into(self):
        found = run(Recorder())
        self.assertEqual(found.applied, 1)
        self.assertEqual(found.restarted, 1, "the old code is still the one running")

    def test_being_current_does_nothing_at_all(self):
        found = run(Recorder(latest="v0.1.0", installed="0.1.0"))
        self.assertEqual(found.applied, 0)
        self.assertEqual(found.restarted, 0)

    def test_an_older_release_is_never_installed_over_a_newer_install(self):
        found = run(Recorder(latest="v0.0.9", installed="0.5.0"))
        self.assertEqual(found.applied, 0)

    def test_being_offline_is_not_a_reason_to_refuse_to_render(self):
        recorder = Recorder()
        recorder.latest = lambda timeout=0: ""      # every failure answers empty
        found = run(recorder)
        self.assertEqual(found.applied, 0)
        self.assertEqual(found.restarted, 0)

    def test_a_server_required_version_updates_without_asking_github_which_is_latest(self):
        recorder = Recorder(latest="v0.13.1", installed="0.13.0")
        recorder.latest = lambda timeout=0: self.fail("server-directed updates do not poll GitHub")
        with mock.patch.multiple(cli.updater, latest=recorder.latest,
                                 installed=recorder.installed, apply=recorder.apply), \
             mock.patch.object(cli.runtime, "restart",
                               lambda: setattr(recorder, "restarted", recorder.restarted + 1)):
            cli.server_update("0.13.1")

        self.assertEqual(recorder.applied, 1)
        self.assertEqual(recorder.restarted, 1)


class LoopTests(unittest.TestCase):
    def setUp(self):
        config.write(updateMode="auto")

    def test_a_release_tagged_ahead_of_its_own_version_cannot_loop(self):
        """The failure that would turn a worker into a machine that only updates.

        If v9.0.0 ships a tree that still calls itself 0.1.0, then installing
        it, restarting and looking again finds exactly the same answer -- and
        the worker spends the rest of its life downloading and never renders.
        """
        found = run(Recorder(), environ={cli.UPDATED: "v9.0.0"})
        self.assertEqual(found.applied, 0)
        self.assertEqual(found.restarted, 0)

    def test_the_marker_is_set_before_restarting(self):
        self.assertEqual(run(Recorder()).marker, "v9.0.0")

    def test_a_different_release_is_still_installed_after_one_marked(self):
        # The guard is per version, not a one-update-per-life rule.
        found = run(Recorder(latest="v9.1.0"), environ={cli.UPDATED: "v9.0.0"})
        self.assertEqual(found.applied, 1)


class ModeTests(unittest.TestCase):
    def test_notify_says_so_and_installs_nothing(self):
        config.write(updateMode="notify")
        found = run(Recorder())
        self.assertEqual(found.applied, 0)
        self.assertEqual(found.restarted, 0)

    def test_off_does_not_even_look(self):
        config.write(updateMode="off")
        recorder = Recorder()
        looked = []
        recorder.latest = lambda timeout=0: looked.append(True) or "v9.0.0"
        run(recorder)
        self.assertEqual(looked, [], "off should not reach the network")

    def test_a_nonsense_mode_falls_back_to_updating(self):
        config.write(updateMode="sometimes")
        self.assertEqual(run(Recorder()).applied, 1)


class EntryTests(unittest.TestCase):
    def test_starting_checks_for_an_update_first(self):
        # Before onboarding, so it happens on a fresh machine too, and long
        # before any job could be claimed.
        with mock.patch.object(cli, "self_update") as checked, \
             mock.patch.object(cli, "onboard", return_value=False):
            with self.assertRaises(SystemExit):
                cli.cmd_start([])
        checked.assert_called_once()

    def test_no_update_skips_it(self):
        with mock.patch.object(cli, "self_update") as checked, \
             mock.patch.object(cli, "onboard", return_value=False):
            with self.assertRaises(SystemExit):
                cli.cmd_start(["--no-update"])
        checked.assert_not_called()


if __name__ == "__main__":
    unittest.main()
