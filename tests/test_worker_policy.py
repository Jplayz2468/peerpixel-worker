import json
import unittest

from peerpixel import config, worker
from peerpixel.worker import should_unload_model


class WorkerPolicyTests(unittest.TestCase):
    def test_model_stays_loaded_until_two_idle_hours(self):
        self.assertFalse(should_unload_model(100, 100 + 7199, loaded=True))
        self.assertTrue(should_unload_model(100, 100 + 7200, loaded=True))
        self.assertFalse(should_unload_model(100, 100 + 9000, loaded=False))


if __name__ == "__main__":
    unittest.main()


class StepTimingTests(unittest.TestCase):
    """The second render's bar should be right from its first frame."""

    def setUp(self):
        self.saved = config.read()
        config.write(secondsPerStep={}, benchMs=8000)

    def tearDown(self):
        config.FILE.write_text(json.dumps(self.saved, indent=2))

    def test_a_machine_that_has_never_rendered_falls_back_to_its_benchmark(self):
        # Which is the one timed render every worker has already done.
        self.assertAlmostEqual(worker.seconds_per_step("master"), 2.0)

    def test_one_finished_render_is_taken_whole(self):
        worker.remember_step("master", 300.0, 50)
        self.assertAlmostEqual(worker.seconds_per_step("master"), 6.0)

    def test_later_renders_move_it_without_leaping_to_the_last_one(self):
        worker.remember_step("master", 300.0, 50)
        worker.remember_step("master", 100.0, 50)
        self.assertLess(worker.seconds_per_step("master"), 6.0)
        self.assertGreater(worker.seconds_per_step("master"), 2.0)

    def test_a_preview_and_a_master_are_timed_separately(self):
        # A step of a 1024px master and a step of a 256px preview differ by a
        # factor of sixteen. One number for both would be wrong for each.
        worker.remember_step("master", 300.0, 50)
        worker.remember_step("draft", 3.0, 6)
        self.assertAlmostEqual(worker.seconds_per_step("master"), 6.0)
        self.assertAlmostEqual(worker.seconds_per_step("draft"), 0.5)

    def test_it_survives_a_restart(self):
        worker.remember_step("draft", 3.0, 6)
        self.assertEqual(config.read()["secondsPerStep"]["draft"], 0.5)

    def test_nonsense_is_ignored(self):
        worker.remember_step("master", 0, 50)
        worker.remember_step("master", 10, 0)
        self.assertEqual(config.read().get("secondsPerStep"), {})
