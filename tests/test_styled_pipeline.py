import unittest
from unittest import mock

from peerpixel.prompt_enhancer import PromptEnhancer, SYSTEM_INSTRUCTION
from peerpixel.safety import THRESHOLD
from peerpixel.render import Renderer


class StyledPipelineTests(unittest.TestCase):
    def test_exact_prompt_instruction_and_bypass_paths(self):
        self.assertIn("Output ONLY the raw optimized descriptive prompt (1–2 sentences). No preamble, no quotes.", SYSTEM_INSTRUCTION)
        enhancer = PromptEnhancer()
        self.assertEqual(enhancer.enhance(" raw prompt ", "photoreal", enabled=False), "raw prompt")
        self.assertEqual(enhancer.enhance("raw", "anime", resolved=" exact draft prompt "), "exact draft prompt")

    def test_safety_threshold_is_strictly_greater_than_point_65(self):
        self.assertEqual(THRESHOLD, 0.65)

    def test_exact_lora_stacks_are_activated_per_style(self):
        class Pipe:
            def __init__(self): self.loaded, self.active = [], None
            def load_lora_weights(self, folder, *, weight_name, adapter_name):
                self.loaded.append(adapter_name)
            def set_adapters(self, names, *, adapter_weights):
                self.active = (names, adapter_weights)
        renderer = object.__new__(Renderer)
        renderer.pipe, renderer._loaded_adapters = Pipe(), set()
        with mock.patch("peerpixel.model_cache.ensure", side_effect=lambda name: __import__("pathlib").Path(f"/cache/{name}.safetensors")):
            renderer.apply_style({"style": "anime", "recipeId": "anime-v1",
                                  "manifestVersion": "2026-08-21.1"})
            self.assertEqual(renderer.pipe.active,
                             (["rebelmidjourney", "flux-klein-art"], [0.20, 0.85]))
            renderer.apply_style({"style": "vector", "recipeId": "vector-v1",
                                  "manifestVersion": "2026-08-21.1"})
            self.assertEqual(renderer.pipe.active, (["simplefinevector"], [1.0]))
