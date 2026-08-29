import unittest

from scripts.evaluate_bootstrap_lora import evaluate_outputs


class BootstrapEvaluationTests(unittest.TestCase):
    def test_clean_single_responses_pass_structure_and_constraints(self):
        cases = [
            {"prompt": 'red fox beside a sign reading "OPEN ALL NIGHT"',
             "target": 'A red fox beside a weathered sign reading "OPEN ALL NIGHT".'},
            {"prompt": "three blue birds", "target": "Three blue birds perched on a branch."},
        ]
        report = evaluate_outputs(cases, [
            'A red fox stands beside a weathered sign reading "OPEN ALL NIGHT".',
            "Three blue birds perch along a narrow branch under soft daylight.",
        ])
        self.assertEqual(report["structureCompliance"], 1.0)
        self.assertEqual(report["promptFidelity"], 1.0)
        self.assertTrue(report["passed"])

    def test_wrappers_missing_copy_and_concentrated_phrases_fail(self):
        cases = [
            {"prompt": 'sign reading "KEEP OUT"', "target": 'A sign reading "KEEP OUT".'},
            {"prompt": "two cats", "target": "Two cats on a wall."},
            {"prompt": "green bicycle", "target": "A green bicycle near a curb."},
        ]
        report = evaluate_outputs(cases, [
            '{"prompt":"A cinematic sign."}',
            "Here is your prompt: cinematic volumetric lighting cat.",
            "Cinematic volumetric lighting bicycle.",
        ])
        self.assertLess(report["structureCompliance"], 1.0)
        self.assertLess(report["promptFidelity"], 1.0)
        self.assertGreater(report["phraseConcentration"], 0.5)
        self.assertFalse(report["passed"])

    def test_copies_and_overlong_outputs_are_reported(self):
        cases = [{"prompt": "a fox", "target": "A detailed red fox in a meadow."}]
        report = evaluate_outputs(cases, ["a fox " * 200])
        self.assertEqual(report["withinTokenLimit"], 0.0)
        self.assertEqual(report["normalizedCopyRate"], 0.0)
        self.assertFalse(report["passed"])


if __name__ == "__main__":
    unittest.main()
