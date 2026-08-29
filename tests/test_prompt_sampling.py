import unittest

from peerpixel.prompt_enhancer import PerRowTemperature


class PromptSamplingTests(unittest.TestCase):
    def test_each_batched_prompt_row_uses_its_own_temperature(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        scores = torch.tensor([[2.0, 4.0], [2.0, 4.0]])
        scaled = PerRowTemperature([.5, 2.0])(None, scores)
        self.assertEqual(scaled.tolist(), [[4.0, 8.0], [1.0, 2.0]])

    def test_temperature_count_must_match_the_batch(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        with self.assertRaisesRegex(ValueError, "temperature_batch_mismatch"):
            PerRowTemperature([.7])(None, torch.ones((2, 3)))


if __name__ == "__main__":
    unittest.main()
