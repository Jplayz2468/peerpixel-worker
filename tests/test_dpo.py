import math
import unittest

from peerpixel.dpo import weighted_dpo_loss


class WeightedDpoLossTests(unittest.TestCase):
    def test_upscale_contributes_twice_variation_and_reduction_uses_total_weight(self):
        # The two per-example losses are softplus(-2) and softplus(0).
        got = weighted_dpo_loss(
            [2.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
            beta=1.0, weight=[1.0, 0.5],
        )
        expected = (
            math.log1p(math.exp(-2.0))
            + 0.5 * math.log(2.0)
        ) / 1.5
        self.assertAlmostEqual(got, expected, places=12)

        # Uniformly scaling weights must not scale gradients or loss.
        scaled = weighted_dpo_loss(
            [2.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
            beta=1.0, weight=[2.0, 1.0],
        )
        self.assertAlmostEqual(scaled, got, places=12)

    def test_every_weight_must_be_positive_and_finite(self):
        for invalid in (0.0, -0.5, math.nan, math.inf, -math.inf):
            with self.subTest(weight=invalid), self.assertRaisesRegex(
                    ValueError, "positive and finite"):
                weighted_dpo_loss(
                    [1.0], [0.0], [0.0], [0.0], beta=0.1,
                    weight=[invalid],
                )

    def test_policy_reference_and_weight_values_must_stay_aligned(self):
        cases = (
            ([1.0], [0.0, 0.0], [0.0], [0.0], [1.0]),
            ([1.0], [0.0], [0.0, 0.0], [0.0], [1.0]),
            ([1.0], [0.0], [0.0], [0.0], [1.0, 0.5]),
        )
        for policy_chosen, policy_rejected, reference_chosen, reference_rejected, weight in cases:
            with self.subTest(case=cases.index((policy_chosen, policy_rejected,
                                                reference_chosen, reference_rejected,
                                                weight))), self.assertRaisesRegex(
                    ValueError, "aligned"):
                weighted_dpo_loss(
                    policy_chosen, policy_rejected,
                    reference_chosen, reference_rejected,
                    beta=0.1, weight=weight,
                )

    def test_beta_must_be_positive_and_finite(self):
        for invalid in (0.0, -0.1, math.nan, math.inf):
            with self.subTest(beta=invalid), self.assertRaisesRegex(
                    ValueError, "beta"):
                weighted_dpo_loss(
                    [1.0], [0.0], [0.0], [0.0],
                    beta=invalid, weight=[1.0],
                )


if __name__ == "__main__":
    unittest.main()
