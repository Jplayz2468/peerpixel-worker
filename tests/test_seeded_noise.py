"""The deterministic noise each direct render's seed names."""
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

    def test_master_and_probe_have_their_documented_latent_sizes(self):
        self.assertEqual(latent_grid(PIPE, OPERATIONS["master"]["width"]), 64)
        self.assertEqual(latent_grid(PIPE, OPERATIONS["probe"]["width"]), 8)


@unittest.skipIf(torch is None, "torch is not installed")
class SeedTests(unittest.TestCase):
    def noise(self, operation: str, seed: int):
        return seeded_latents(
            PIPE, {**OPERATIONS[operation], "name": operation}, seed, FLOAT32,
        )

    def test_the_same_seed_is_the_same_noise_every_time(self):
        for operation in ("master", "verify", "probe", "bench"):
            with self.subTest(operation=operation):
                self.assertTrue(torch.equal(
                    self.noise(operation, 42), self.noise(operation, 42),
                ))

    def test_different_seeds_are_different_noise(self):
        self.assertFalse(torch.equal(self.noise("master", 1), self.noise("master", 2)))

    def test_every_operation_draws_directly_at_its_own_size(self):
        for operation in ("master", "probe", "bench"):
            with self.subTest(operation=operation):
                spec = OPERATIONS[operation]
                tall = latent_grid(PIPE, spec["height"])
                wide = latent_grid(PIPE, spec["width"])
                expected = torch.randn(
                    (1, 128, tall, wide), dtype=torch.float32,
                    generator=torch.Generator("cpu").manual_seed(7),
                )
                self.assertTrue(torch.equal(self.noise(operation, 7), expected))

    def test_a_verification_reproduces_the_master_exactly(self):
        self.assertTrue(torch.equal(self.noise("master", 99), self.noise("verify", 99)))

    def test_an_out_of_range_seed_is_wrapped_rather_than_refused(self):
        self.noise("master", 0xFFFFFFFF)
        self.noise("probe", 0xFFFFFFFF)


if __name__ == "__main__":
    unittest.main()
