"""The noise a seed names, and why a preview's is an average and not a blow-up.

A preview and its final are tied together by nothing but the seed. That only
works if the tensor each one starts from is (a) derived from the same draw and
(b) still a valid sample of the distribution the model was trained to start
from. Both halves are tested here, because getting either wrong produces
pictures that look plausible and are quietly wrong.
"""
from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover - a machine with no rendering stack
    torch = None

from peerpixel.render import OPERATIONS, latent_grid, seeded_latents


class FakePipe:
    """Enough of a pipeline to ask about shapes. No model, no card, no wait."""

    vae_scale_factor = 8

    class transformer:  # noqa: N801 - mirrors the attribute it stands in for
        class config:
            in_channels = 128


PIPE = FakePipe()
FLOAT32 = getattr(torch, "float32", None)


@unittest.skipIf(torch is None, "torch is not installed")
class ShapeTests(unittest.TestCase):
    def test_each_operation_gets_the_shape_its_pixels_imply(self):
        for name, spec in OPERATIONS.items():
            with self.subTest(operation=name):
                noise = seeded_latents(PIPE, {**spec, "name": name}, 7, FLOAT32)
                side = spec["width"] // 16
                self.assertEqual(tuple(noise.shape), (1, 128, side, side))

    def test_the_final_and_the_preview_are_the_documented_sizes(self):
        # 512/16 = 32 and 128/16 = 8, which is the exact factor of four the
        # averaging depends on.
        self.assertEqual(latent_grid(PIPE, OPERATIONS["master"]["width"]), 32)
        self.assertEqual(latent_grid(PIPE, OPERATIONS["draft"]["width"]), 8)

    def test_a_size_that_does_not_divide_the_final_is_refused(self):
        odd = {"name": "odd", "width": 384, "height": 384, "steps": 4, "guidance": 4.0}
        with self.assertRaises(ValueError):
            seeded_latents(PIPE, odd, 7, FLOAT32)


@unittest.skipIf(torch is None, "torch is not installed")
class DistributionTests(unittest.TestCase):
    """The averaged tensor has to be indistinguishable from a fresh draw."""

    def preview(self, seed=7):
        return seeded_latents(PIPE, {**OPERATIONS["draft"], "name": "draft"}, seed, FLOAT32)

    def test_it_still_has_unit_variance(self):
        # A plain mean would have variance 1/16 and the model would be started
        # from noise a quarter the amplitude it expects.
        self.assertAlmostEqual(float(self.preview().var()), 1.0, delta=0.05)

    def test_neighbouring_values_are_still_independent(self):
        """The whole reason this is an average and not an upscale.

        Upscaling a smaller noise to a larger shape leaves each value strongly
        correlated with its neighbour -- measured at +0.76 for nearest and
        +0.93 for bilinear -- and the model reads that correlation as structure
        that nobody put in the picture.
        """
        noise = self.preview()
        left, right = noise[..., :, :-1].flatten(), noise[..., :, 1:].flatten()
        centred = ((left - left.mean()) * (right - right.mean())).mean()
        correlation = float(centred / (left.std() * right.std()))
        self.assertLess(abs(correlation), 0.05, f"correlated at {correlation:+.3f}")

    def test_an_upscale_would_have_failed_that(self):
        """Guards the claim above rather than trusting the comment.

        If some future edit reaches for interpolation, this is the test that
        says what it costs.
        """
        small = self.preview()
        blown = small.repeat_interleave(4, dim=2).repeat_interleave(4, dim=3)
        left, right = blown[..., :, :-1].flatten(), blown[..., :, 1:].flatten()
        centred = ((left - left.mean()) * (right - right.mean())).mean()
        self.assertGreater(float(centred / (left.std() * right.std())), 0.5)


@unittest.skipIf(torch is None, "torch is not installed")
class SeedTests(unittest.TestCase):
    def test_the_same_seed_is_the_same_noise_every_time(self):
        # Two machines rendering the same job have to start from the same
        # tensor or the network cannot check one against the other.
        self.assertTrue(torch.equal(
            seeded_latents(PIPE, {**OPERATIONS["master"], "name": "master"}, 42, FLOAT32),
            seeded_latents(PIPE, {**OPERATIONS["master"], "name": "master"}, 42, FLOAT32)))

    def test_different_seeds_are_different_noise(self):
        self.assertFalse(torch.equal(
            seeded_latents(PIPE, {**OPERATIONS["master"], "name": "master"}, 1, FLOAT32),
            seeded_latents(PIPE, {**OPERATIONS["master"], "name": "master"}, 2, FLOAT32)))

    def test_the_preview_is_the_final_averaged_and_not_a_separate_draw(self):
        """The link between the two pictures, stated as arithmetic."""
        final = seeded_latents(PIPE, {**OPERATIONS["master"], "name": "master"}, 7, FLOAT32)
        preview = seeded_latents(PIPE, {**OPERATIONS["draft"], "name": "draft"}, 7, FLOAT32)
        expected = final.reshape(1, 128, 8, 4, 8, 4).mean(dim=(3, 5)) * 4
        self.assertTrue(torch.allclose(preview, expected, atol=1e-5))

    def test_a_verification_reproduces_the_final_exactly(self):
        # A check re-renders somebody's finished job on the operator's machine
        # and compares. With no reference image involved, the seed is the whole
        # of what has to be reproduced.
        self.assertTrue(torch.equal(
            seeded_latents(PIPE, {**OPERATIONS["master"], "name": "master"}, 99, FLOAT32),
            seeded_latents(PIPE, {**OPERATIONS["verify"], "name": "verify"}, 99, FLOAT32)))

    def test_an_out_of_range_seed_is_wrapped_rather_than_refused(self):
        # Seeds arrive from a server as unsigned 32-bit numbers; torch wants a
        # narrower range, and a job must not fail over the difference.
        seeded_latents(PIPE, {**OPERATIONS["draft"], "name": "draft"}, 0xFFFFFFFF, FLOAT32)


if __name__ == "__main__":
    unittest.main()
