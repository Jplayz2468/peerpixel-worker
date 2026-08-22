import unittest

from peerpixel.precision import Probe, select_precision


class PrecisionPolicyTests(unittest.TestCase):
    def test_a_16gb_blackwell_card_prefers_resident_eight_bit(self):
        plan = select_precision(Probe(
            cuda=True, capability=(12, 0), total=16_000_000_000,
            free=15_000_000_000, bitsandbytes=True,
        ))
        self.assertEqual((plan.mode, plan.resident, plan.adapters),
                         ("int8", True, False))

    def test_an_8gb_card_prefers_resident_four_bit(self):
        plan = select_precision(Probe(
            cuda=True, capability=(8, 9), total=8_000_000_000,
            free=7_000_000_000, bitsandbytes=True,
        ))
        self.assertEqual((plan.mode, plan.resident), ("int4", True))

    def test_missing_backend_falls_back_without_host_specific_rules(self):
        plan = select_precision(Probe(
            cuda=True, capability=(8, 9), total=12_000_000_000,
            free=10_000_000_000, bitsandbytes=False,
        ))
        self.assertEqual((plan.mode, plan.resident), ("bfloat16", False))
        self.assertIn("bitsandbytes", plan.reason)

    def test_non_cuda_keeps_the_native_resident_path(self):
        plan = select_precision(Probe(cuda=False))
        self.assertEqual((plan.mode, plan.resident), ("native", True))

    def test_too_little_free_memory_uses_the_safe_fallback(self):
        plan = select_precision(Probe(
            cuda=True, capability=(12, 0), total=16_000_000_000,
            free=4_000_000_000, bitsandbytes=True,
        ))
        self.assertEqual((plan.mode, plan.resident), ("bfloat16", False))


if __name__ == "__main__":
    unittest.main()
