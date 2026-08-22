import inspect
import sys
import types
import unittest
from unittest import mock

from peerpixel.prompt_enhancer import PromptEnhancer, SYSTEM_INSTRUCTION, enhancement_messages
from peerpixel.safety import SafetyClassifier, THRESHOLD
from peerpixel.render import Renderer


class StyledPipelineTests(unittest.TestCase):
    def test_auxiliary_models_stay_on_cpu_instead_of_competing_with_flux_on_mps(self):
        enhancer = PromptEnhancer(model_path="/models/qwen")
        fake_model = object()
        with mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=object()), \
             mock.patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=fake_model) as load:
            enhancer.warm()
        self.assertEqual(load.call_args.kwargs["device_map"], "cpu")

        safety = SafetyClassifier(model_path="/models/safety")
        pipeline = mock.Mock(return_value=object())
        fake_transformers = types.SimpleNamespace(pipeline=pipeline)
        with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
            safety.warm()
        self.assertEqual(pipeline.call_args.kwargs["device"], -1)

    def test_prompt_model_is_released_before_flux_uses_the_accelerator(self):
        events = []

        class Enhancer:
            def enhance(self, *_args, **_kwargs):
                events.append("enhance")
                return "polished"
            def unload(self):
                events.append("release_prompt_model")

        class Safety:
            def classify(self, _jpeg):
                events.append("moderate")
                return {"label": "normal", "nsfwScore": 0.0}

        renderer = object.__new__(Renderer)
        renderer._enhancer, renderer._safety = Enhancer(), Safety()
        renderer.render = lambda *_args, **_kwargs: events.append("render") or b"jpeg"
        renderer.generate_job({
            "prompt": "fox", "style": "anime", "operation": "draft", "seed": 7,
            "recipeId": "anime-v1", "manifestVersion": "2026-08-21.1",
        })
        self.assertEqual(events, ["enhance", "release_prompt_model", "render", "moderate"])

    def test_prompt_enhancer_accepts_a_per_draft_variation(self):
        parameters = inspect.signature(PromptEnhancer.enhance).parameters
        self.assertIn("variation", parameters)
        first = enhancement_messages("fox", "anime", variation=101)
        second = enhancement_messages("fox", "anime", variation=202)
        self.assertNotEqual(first[1]["content"], second[1]["content"])
        self.assertIn("Draft variation seed: 101", first[1]["content"])

    def test_exact_prompt_instruction_and_bypass_paths(self):
        self.assertIn("Output ONLY the raw optimized descriptive prompt (1–2 sentences). No preamble, no quotes.", SYSTEM_INSTRUCTION)
        enhancer = PromptEnhancer()
        self.assertEqual(enhancer.enhance(" raw prompt ", "photoreal", enabled=False), "raw prompt")
        self.assertEqual(enhancer.enhance("raw", "anime", resolved=" exact draft prompt "), "exact draft prompt")

    def test_safety_threshold_is_strictly_greater_than_point_65(self):
        self.assertEqual(THRESHOLD, 0.65)

    @mock.patch("peerpixel.render.lora_uses_raw_peft_keys",
                side_effect=lambda path: "flux-klein-art" in str(path))
    def test_exact_lora_stacks_are_activated_per_style(self, _raw_format):
        class Pipe:
            def __init__(self):
                self.loaded, self.active = [], None
                self.transformer = mock.Mock()
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
            renderer.pipe.transformer.load_lora_adapter.assert_called_once_with(
                "/cache", weight_name="flux-klein-art.safetensors",
                adapter_name="flux-klein-art", prefix=None,
            )
            renderer.apply_style({"style": "vector", "recipeId": "vector-v1",
                                  "manifestVersion": "2026-08-21.1"})
            self.assertEqual(renderer.pipe.active, (["simplefinevector"], [1.0]))
