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
        self.assertAlmostEqual(worker.seconds_per_step("master"), 8 / 9)

    def test_one_finished_render_is_taken_whole(self):
        worker.remember_step("master", 300.0, 50)
        self.assertAlmostEqual(worker.seconds_per_step("master"), 6.0)

    def test_later_renders_move_it_without_leaping_to_the_last_one(self):
        worker.remember_step("master", 300.0, 50)
        worker.remember_step("master", 100.0, 50)
        self.assertLess(worker.seconds_per_step("master"), 6.0)
        self.assertGreater(worker.seconds_per_step("master"), 2.0)

    def test_an_internal_probe_and_a_master_are_timed_separately(self):
        # Probe and master steps have different spatial costs, so one number
        # would make both progress estimates wrong.
        worker.remember_step("master", 300.0, 50)
        worker.remember_step("probe", 25.0, 50)
        self.assertAlmostEqual(worker.seconds_per_step("master"), 6.0)
        self.assertAlmostEqual(worker.seconds_per_step("probe"), 0.5)

    def test_it_survives_a_restart(self):
        worker.remember_step("probe", 25.0, 50)
        self.assertEqual(config.read()["secondsPerStep"]["probe"], 0.5)

    def test_nonsense_is_ignored(self):
        worker.remember_step("master", 0, 50)
        worker.remember_step("master", 10, 0)
        self.assertEqual(config.read().get("secondsPerStep"), {})


class JobPlanTests(unittest.TestCase):
    """Nothing a job does may happen outside a phase of its plan."""

    def test_decoding_has_a_phase_of_its_own(self):
        from peerpixel.plans import PLANS

        names = [phase.name for phase in PLANS["job"].phases]
        self.assertIn("decode", names)
        self.assertLess(names.index("render"), names.index("decode"))

    def test_nothing_waits_for_a_reference_any_more(self):
        from peerpixel.plans import PLANS

        self.assertNotIn("wait", [phase.name for phase in PLANS["job"].phases])


if __name__ == "__main__":
    unittest.main()
