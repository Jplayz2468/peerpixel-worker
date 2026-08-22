import unittest

from peerpixel.job_phases import EXPORT_PHASES, PhaseReporter, remember_phase, valid_phase_sequence


class PhaseReporterTests(unittest.TestCase):
    def test_the_generation_phase_order_is_valid(self):
        self.assertTrue(valid_phase_sequence([
            "preparing", "loading_flux", "enhancing_prompt", "encoding_prompt",
            "loading_style", "rendering", "decoding", "safety_check",
            "delivering", "complete",
        ]))
        self.assertFalse(valid_phase_sequence(["rendering", "encoding_prompt"]))

    def test_export_has_its_own_monotonic_phase_order(self):
        self.assertTrue(valid_phase_sequence([
            "preparing", "loading_upscaler", "upscaling", "encoding_export",
            "delivering", "complete",
        ], allowed=EXPORT_PHASES))

    def test_phase_history_uses_a_bounded_ema(self):
        self.assertEqual(remember_phase(1000, 3000, alpha=0.25), 1500)
        self.assertEqual(remember_phase(None, 3000), 3000)
        self.assertEqual(remember_phase(1000, -1), 1000)

    def test_events_are_job_scoped_and_regressions_are_ignored(self):
        sent = []
        times = iter([10.0, 10.0, 11.25, 11.5])
        reporter = PhaseReporter("job-7", sent.append, clock=lambda: next(times),
                                 wall=lambda: 1234.5)
        self.assertTrue(reporter.begin("preparing"))
        self.assertTrue(reporter.begin("rendering"))
        self.assertFalse(reporter.begin("encoding_prompt"))
        self.assertEqual([event["phase"] for event in sent], ["preparing", "rendering"])
        self.assertTrue(all(event["jobId"] == "job-7" for event in sent))
        self.assertEqual(sent[1]["elapsedMs"], 1250)


if __name__ == "__main__":
    unittest.main()
