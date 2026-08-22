import unittest

from peerpixel.precision import Probe, select_precision


class PrecisionPolicyTests(unittest.TestCase):
    def test_all_eligible_cuda_cards_use_the_same_resident_nf4_precision(self):
        plan = select_precision(Probe(
            cuda=True, capability=(12, 0), total=16_000_000_000,
            free=15_000_000_000, bitsandbytes=True,
        ))
        self.assertEqual((plan.mode, plan.resident, plan.adapters),
                         ("int4", True, False))

    def test_an_8gb_card_uses_the_same_nf4_image_precision(self):
        plan = select_precision(Probe(
            cuda=True, capability=(8, 9), total=8_000_000_000,
            free=7_000_000_000, bitsandbytes=True,
        ))
        self.assertEqual((plan.mode, plan.resident), ("int4", True))

    def test_missing_backend_rejects_image_generation_instead_of_changing_precision(self):
        plan = select_precision(Probe(
            cuda=True, capability=(8, 9), total=12_000_000_000,
            free=10_000_000_000, bitsandbytes=False,
        ))
        self.assertEqual((plan.mode, plan.resident), ("unavailable", False))
        self.assertIn("bitsandbytes", plan.reason)

    def test_non_cuda_keeps_the_native_resident_path(self):
        plan = select_precision(Probe(cuda=False))
        self.assertEqual((plan.mode, plan.resident), ("native", True))

    def test_too_little_free_memory_rejects_instead_of_changing_precision(self):
        plan = select_precision(Probe(
            cuda=True, capability=(12, 0), total=16_000_000_000,
            free=4_000_000_000, bitsandbytes=True,
        ))
        self.assertEqual((plan.mode, plan.resident), ("unavailable", False))

    def test_operator_cannot_change_network_image_precision(self):
        plan = select_precision(Probe(
            cuda=True, capability=(12, 0), total=16_000_000_000,
            free=15_000_000_000, bitsandbytes=True,
        ), requested="bfloat16")
        self.assertEqual((plan.mode, plan.resident), ("int4", True))


if __name__ == "__main__":
    unittest.main()
