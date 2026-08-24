"""The parts of the terminal program that can be wrong quietly."""
from __future__ import annotations

import unittest
from unittest import mock

from peerpixel import cli, config, console, settings


class SettingsTests(unittest.TestCase):
    """Every one of these writes the real config, so every one puts it back.

    Left to itself, the test that turns the model unload off turned it off for
    the rest of the suite, and two tests in a different file failed for a
    reason that had nothing to do with them.
    """

    def setUp(self):
        self.saved = config.read()
        config.write(**{key: None for key in settings.STORED.values()})

    def tearDown(self):
        config.FILE.write_text(__import__("json").dumps(self.saved, indent=2))

    def values(self):
        return {s.name: v for s, v, _ in settings.current()}

    def test_a_switch_nobody_has_touched_is_off(self):
        # The default was stored as the string "off", read back as a value and
        # tested for truth -- so every switch in the program read as on.
        self.assertEqual(self.values()["free"], "off")

    def test_setting_and_reading_a_switch_round_trips(self):
        with mock.patch.object(settings, "_sync_free", return_value="done"):
            settings.put("free", "on")
        self.assertEqual(self.values()["free"], "on")

    def test_a_value_outside_the_list_is_refused_by_name(self):
        with self.assertRaises(ValueError) as caught:
            settings.put("dtype", "float8")
        self.assertIn("dtype", str(caught.exception))

    def test_an_unknown_setting_is_refused(self):
        with self.assertRaises(ValueError):
            settings.put("nonsense", "1")

    def test_removed_preview_persistence_has_no_setting(self):
        self.assertNotIn("keep-last", settings.BY_NAME)

    def test_free_saved_without_the_account_agreeing_says_so(self):
        config.write(deviceId="dev_1", allowFreeSyncedAt=0)
        with mock.patch.object(settings.api, "set_free",
                               side_effect=settings.api.ApiError(403, "forbidden")):
            said = settings.put("free", "on")
        self.assertIn("account", said)
        self.assertEqual(self.values()["free"], "on")
        self.assertTrue(any(note for s, _, note in settings.current() if s.name == "free"))

    def test_turning_the_model_unload_off_means_never(self):
        settings.put("unload-after", "0")
        self.assertEqual(settings.unload_seconds(), 0)

    def test_every_setting_can_be_shown_and_explained(self):
        for setting in settings.SETTINGS:
            with self.subTest(setting=setting.name):
                self.assertTrue(setting.summary)
                self.assertTrue(setting.detail)
                self.assertIn(setting.name, settings.STORED)


class BarTests(unittest.TestCase):
    def test_the_bar_fills_and_never_overflows(self):
        self.assertEqual(len(console.bar(0.0, 20)), 20)
        self.assertEqual(len(console.bar(1.0, 20)), 20)
        self.assertEqual(len(console.bar(2.0, 20)), 20)
        self.assertEqual(len(console.bar(-1.0, 20)), 20)

    def test_it_moves_at_finer_than_one_character(self):
        # A twenty-character bar that could only move in whole characters would
        # sit still for five percent at a time.
        seen = {console.bar(i / 200, 20) for i in range(0, 20)}
        self.assertGreater(len(seen), 5)

    def test_a_finished_bar_is_solid(self):
        self.assertNotIn(console.TRACK, console.bar(1.0, 20))

    def test_time_is_said_in_words_somebody_can_act_on(self):
        self.assertIn("seconds", console.how_long(2))
        self.assertIn("left", console.how_long(45))
        self.assertIn("m", console.how_long(600))
        self.assertEqual(console.how_long(None), "working out how long")


class StateTests(unittest.TestCase):
    def test_a_machine_is_only_ready_when_everything_is_done(self):
        self.assertFalse(cli.ready({"libraries": True, "paired": True,
                                    "model": True, "approved": False}))
        self.assertTrue(cli.ready({"libraries": True, "paired": True,
                                   "model": True, "approved": True}))

    def test_every_command_in_the_help_actually_exists(self):
        for line, _ in cli.HELP:
            name = line.split()[1] if len(line.split()) > 1 else "start"
            if name.isupper():
                continue
            with self.subTest(command=line):
                self.assertIn(name, cli.COMMANDS)

    def test_bare_peerpixel_starts(self):
        with mock.patch.dict(cli.COMMANDS, {"start": lambda argv: seen.append(argv)}):
            seen = []
            with mock.patch.object(cli.runtime, "use_venv"):
                cli.main([])
            self.assertEqual(seen, [[]])

    def test_a_flag_with_no_command_still_starts(self):
        seen = []
        with mock.patch.dict(cli.COMMANDS, {"start": lambda argv: seen.append(argv)}):
            with mock.patch.object(cli.runtime, "use_venv"):
                cli.main(["--once"])
        self.assertEqual(seen, [["--once"]])

    def test_an_unknown_command_explains_itself_and_fails(self):
        with mock.patch.object(cli.runtime, "use_venv"):
            with self.assertRaises(SystemExit) as caught:
                cli.main(["frobnicate"])
        self.assertEqual(caught.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
