import unittest
from unittest.mock import patch

from peerpixel import render, worker


class DiscordActionTests(unittest.TestCase):
    def test_action_seeds_are_deterministic_and_distinct(self):
        self.assertEqual(worker._action_seeds(1234, 4), worker._action_seeds(1234, 4))
        self.assertEqual(len(set(worker._action_seeds(1234, 4))), 4)

    def test_grid_and_refinement_specs_stay_inside_safe_compute_bounds(self):
        self.assertEqual(render.operation_of({"operation": "grid", "width": 512,
                         "height": 512, "steps": 16})["steps"], 16)
        self.assertEqual(render.operation_of({"operation": "refine", "width": 768,
                         "height": 1024, "steps": 50})["steps"], 50)
        with self.assertRaises(ValueError):
            render.operation_of({"operation": "grid", "width": 1024,
                                 "height": 1024, "steps": 16})

    def test_coordinator_can_tune_discord_dimensions_and_steps_within_worker_bounds(self):
        tuned = render.operation_of({"operation": "vary", "width": 448,
                                     "height": 512, "steps": 24})
        self.assertEqual((tuned["width"], tuned["height"], tuned["steps"]), (448, 512, 24))
        for bad in ({"width": 520, "height": 512, "steps": 16},
                    {"width": 512, "height": 512, "steps": 40}):
            with self.assertRaisesRegex(ValueError, "untrusted_generation_spec"):
                render.operation_of({"operation": "vary", **bad})

    def test_refine_uses_coordinator_denoise_inside_a_detail_safe_envelope(self):
        self.assertEqual(render.edit_spec({"editMode": "refine", "editStrength": .55,
                         "sourceImageId": "image-1"}), {"mode": "refine", "strength": .55})
        with self.assertRaises(ValueError):
            render.edit_spec({"editMode": "refine", "editStrength": .76,
                              "sourceImageId": "image-1"})

    def test_refine_schedules_fifty_actual_denoising_steps_after_strength_truncation(self):
        self.assertEqual(render.scheduled_edit_steps(50, .42), 120)

    def test_refine_uses_native_reference_conditioning_not_a_full_inpaint_mask(self):
        self.assertEqual(render.edit_backend("refine"), "reference")
        self.assertEqual(render.edit_backend("vary"), "inpaint")


if __name__ == "__main__":
    unittest.main()
