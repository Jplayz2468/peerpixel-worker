import unittest
from unittest.mock import patch

from peerpixel import render, worker


class DiscordActionTests(unittest.TestCase):
    def test_action_seeds_are_deterministic_and_distinct(self):
        self.assertEqual(worker._action_seeds(1234, 4), worker._action_seeds(1234, 4))
        self.assertEqual(len(set(worker._action_seeds(1234, 4))), 4)

    def test_grid_and_refinement_specs_are_pinned(self):
        self.assertEqual(render.operation_of({"operation": "grid", "width": 512,
                         "height": 512, "steps": 16})["steps"], 16)
        self.assertEqual(render.operation_of({"operation": "refine", "width": 816,
                         "height": 1024, "steps": 50})["steps"], 50)
        with self.assertRaises(ValueError):
            render.operation_of({"operation": "grid", "width": 1024,
                                 "height": 1024, "steps": 16})

    def test_refine_is_source_conditioned_at_fixed_strength(self):
        self.assertEqual(render.edit_spec({"editMode": "refine", "editStrength": .30,
                         "sourceImageId": "image-1"}), {"mode": "refine", "strength": .30})
        with self.assertRaises(ValueError):
            render.edit_spec({"editMode": "refine", "editStrength": .31,
                              "sourceImageId": "image-1"})


if __name__ == "__main__":
    unittest.main()
