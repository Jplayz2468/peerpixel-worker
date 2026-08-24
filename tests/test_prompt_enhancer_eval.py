import io
import unittest

from scripts.evaluate_prompt_enhancer import (
    PROMPTS, EvaluationPromptEnhancer, build_cases, run_evaluation,
)


class PromptEnhancerEvaluationTests(unittest.TestCase):
    def test_test_runner_can_request_a_gpu_without_changing_production(self):
        enhancer = EvaluationPromptEnhancer(device="mps")
        self.assertEqual(enhancer.evaluation_device, "mps")

    def test_seeded_evaluation_runs_twenty_varied_prompts_and_prints_results(self):
        cases = build_cases(seed=20260824)
        self.assertEqual(len(PROMPTS), 20)
        self.assertEqual(len(cases), 20)
        self.assertEqual(cases, build_cases(seed=20260824))
        self.assertNotEqual(cases, build_cases(seed=7))
        self.assertIn("auto", {case.style for case in cases})
        self.assertGreaterEqual(len({case.style for case in cases}), 6)

        class Enhancer:
            def enhance_pair(self, prompt, style):
                result = {
                    "prompt": f"Enhanced: {prompt}",
                    "negativePrompt": f"Negative for {style}",
                }
                if style == "auto":
                    result["style"] = "cinematic"
                return result

        output = io.StringIO()
        failures = run_evaluation(
            Enhancer(), cases=cases, stream=output, clock=lambda: 10.0,
        )
        text = output.getvalue()

        self.assertEqual(failures, 0)
        self.assertEqual(text.count("\nINPUT: "), 20)
        self.assertEqual(text.count("\nENHANCED: "), 20)
        self.assertEqual(text.count("\nNEGATIVE: "), 20)
        self.assertIn("REQUESTED STYLE: auto", text)
        self.assertIn("CHOSEN STYLE: cinematic", text)
        self.assertIn("Completed 20 prompts with 0 errors.", text)


if __name__ == "__main__":
    unittest.main()
