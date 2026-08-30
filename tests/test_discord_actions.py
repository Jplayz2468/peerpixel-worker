import unittest
from unittest.mock import patch

from peerpixel import render, worker


class DiscordActionTests(unittest.TestCase):
    def test_action_seeds_are_deterministic_and_distinct(self):
        self.assertEqual(worker._action_seeds(1234, 4), worker._action_seeds(1234, 4))
        self.assertEqual(len(set(worker._action_seeds(1234, 4))), 4)

    def test_grid_and_refinement_specs_stay_inside_safe_compute_bounds(self):
        self.assertEqual(render.operation_of({"operation": "grid", "width": 512,
                         "height": 512, "steps": 9})["steps"], 9)
        self.assertEqual(render.operation_of({"operation": "refine", "width": 768,
                         "height": 1024, "steps": 9})["steps"], 9)
        with self.assertRaises(ValueError):
            render.operation_of({"operation": "grid", "width": 1024,
                                 "height": 1024, "steps": 9})

    def test_coordinator_can_tune_discord_dimensions_and_steps_within_worker_bounds(self):
        tuned = render.operation_of({"operation": "vary", "width": 448,
                                     "height": 512, "steps": 9})
        self.assertEqual((tuned["width"], tuned["height"], tuned["steps"]), (448, 512, 9))
        for bad in ({"width": 520, "height": 512, "steps": 9},
                    {"width": 512, "height": 512, "steps": 40}):
            with self.assertRaisesRegex(ValueError, "untrusted_generation_spec"):
                render.operation_of({"operation": "vary", **bad})

    def test_refine_uses_coordinator_denoise_inside_a_detail_safe_envelope(self):
        self.assertEqual(render.edit_spec({"editMode": "refine", "editStrength": .20,
                         "sourceImageId": "image-1"}), {"mode": "refine", "strength": .20})
        with self.assertRaises(ValueError):
            render.edit_spec({"editMode": "refine", "editStrength": .76,
                              "sourceImageId": "image-1"})

    def test_refine_compensates_strength_to_keep_nine_actual_denoising_steps(self):
        self.assertEqual(render.scheduled_edit_steps(9, .20), 45)
        self.assertEqual(render.scheduled_edit_steps(9, .42), 22)

    def test_refine_uses_img2img_conditioning_to_preserve_the_selected_composition(self):
        self.assertEqual(render.edit_backend("refine"), "inpaint")
        self.assertEqual(render.edit_backend("vary"), "inpaint")


if __name__ == "__main__":
    unittest.main()
