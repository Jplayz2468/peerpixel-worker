import inspect
import sys
import types
import unittest
from unittest import mock

from peerpixel.prompt_enhancer import (
    PromptEnhancer, SYSTEM_INSTRUCTION, enhancement_messages,
    negative_template, parse_enhancement, sampling_seed,
    concept_messages, needs_concept,
)
from peerpixel.safety import SafetyClassifier, THRESHOLD
from peerpixel.render import Renderer, STYLE_RECIPES


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
            "recipeId": "anime-v1", "manifestVersion": "2026-08-23.1",
        })
        self.assertEqual(events, ["enhance", "release_prompt_model", "render", "moderate"])

    def test_prompt_enhancer_accepts_a_per_draft_variation(self):
        parameters = inspect.signature(PromptEnhancer.enhance).parameters
        self.assertIn("variation", parameters)
        first = enhancement_messages("fox", "anime", variation=101)
        second = enhancement_messages("fox", "anime", variation=202)
        self.assertNotEqual(first[1]["content"], second[1]["content"])
        self.assertIn("Draft variation seed: 101", first[1]["content"])

    def test_prompt_enhancer_downloads_qwen3_1_7b(self):
        enhancer = PromptEnhancer()
        tokenizer = mock.Mock()
        model = mock.Mock()
        with mock.patch("peerpixel.model_hub.ensure", return_value="/models/qwen") as ensure, \
             mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), \
             mock.patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model):
            enhancer.warm()
        ensure.assert_called_once_with("qwen3-1.7b")

    def test_safety_model_downloads_from_hugging_face(self):
        classifier = SafetyClassifier()
        pipeline = mock.Mock(return_value=object())
        fake_transformers = types.SimpleNamespace(pipeline=pipeline)
        with mock.patch("peerpixel.model_hub.ensure", return_value="/hf/safety") as ensure, \
             mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
            classifier.warm()
        ensure.assert_called_once_with("nsfw-image-detection")
        self.assertEqual(pipeline.call_args.kwargs["model"], "/hf/safety")

    def test_positive_prompt_is_hard_limited_to_two_sentences(self):
        parsed = parse_enhancement(
            '{"prompt":"First sentence. Second sentence! Third sentence must go.",'
            '"negative_prompt":"watermark"}',
            fallback_prompt="fallback", fallback_negative="blur",
        )
        self.assertEqual(parsed["prompt"], "First sentence. Second sentence!")

    def test_exact_prompt_instruction_and_bypass_paths(self):
        self.assertIn('exactly two string fields: "prompt" and "negative_prompt"', SYSTEM_INSTRUCTION)
        enhancer = PromptEnhancer()
        self.assertEqual(enhancer.enhance(" raw prompt ", "photoreal", enabled=False), "raw prompt")
        self.assertEqual(enhancer.enhance("raw", "anime", resolved=" exact draft prompt "), "exact draft prompt")

    def test_vague_prompts_request_a_complete_original_visual_concept(self):
        messages = enhancement_messages("a man", "cinematic", variation=101)
        instruction = messages[0]["content"]
        self.assertIn("invent a coherent, original visual concept", instruction)
        self.assertIn("specific identity or appearance", instruction)
        self.assertIn("action or pose", instruction)
        self.assertIn("setting", instruction)
        self.assertIn("storytelling detail", instruction)
        self.assertIn("Do not merely restate the subject and append the style directive", instruction)
        self.assertIn("Do not add extra people", instruction)

    def test_structured_enhancement_preserves_positive_and_negative_prompts(self):
        parsed = parse_enhancement(
            '{"prompt":"A rain-soaked detective under neon.",'
            '"negative_prompt":"watermark, duplicate people"}',
            fallback_prompt="a man", fallback_negative="watermark",
        )
        self.assertEqual(parsed, {
            "prompt": "A rain-soaked detective under neon.",
            "negativePrompt": "watermark, duplicate people",
        })

    def test_malformed_structured_output_falls_back_without_losing_the_job(self):
        self.assertEqual(parse_enhancement(
            "A vivid red coupe on a coastal road.", fallback_prompt="a red car",
            fallback_negative="blur, watermark",
        ), {
            "prompt": "A vivid red coupe on a coastal road.",
            "negativePrompt": "blur, watermark",
        })

    def test_negative_templates_are_common_quality_rules_plus_style_rules(self):
        photoreal = negative_template("photoreal")
        vector = negative_template("vector")
        self.assertIn("watermark", photoreal)
        self.assertIn("plastic skin", photoreal)
        self.assertIn("gradients", vector)
        self.assertNotEqual(photoreal, vector)

    def test_creative_sampling_is_reproducible_but_varies_per_draft(self):
        first = sampling_seed("a man", "cinematic", 101)
        self.assertEqual(first, sampling_seed("a man", "cinematic", 101))
        self.assertNotEqual(first, sampling_seed("a man", "cinematic", 202))
        self.assertNotEqual(first, sampling_seed("a man", "anime", 101))

    def test_only_vague_prompts_get_a_separate_scene_concept_pass(self):
        self.assertTrue(needs_concept("a man"))
        self.assertTrue(needs_concept("red car"))
        self.assertFalse(needs_concept(
            "A red coupe drifts around a wet mountain hairpin at blue hour"))
        messages = concept_messages("a man", variation=101)
        self.assertIn("identity or design, action, place, time or weather", messages[0]["content"])
        self.assertIn("Variation seed: 101", messages[1]["content"])

    def test_all_seven_styles_have_distinct_prompt_directives(self):
        expected = {"photoreal", "anime", "vector", "cinematic", "watercolor",
                    "illustration", "pixel_art"}
        self.assertEqual(set(STYLE_RECIPES), expected)
        for style in expected:
            message = enhancement_messages("subject", style)[1]["content"]
            self.assertIn("Required style directive:", message)

    def test_each_request_repeats_only_its_selected_style_as_mandatory(self):
        content = enhancement_messages("a red car", "illustration", variation=303)[1]["content"]
        self.assertIn("Required style directive: Direct polished editorial illustration", content)
        self.assertIn("Apply this chosen style and no other style", content)
        self.assertNotIn("1990s retro 2D anime", content)

    def test_safety_threshold_is_strictly_greater_than_point_65(self):
        self.assertEqual(THRESHOLD, 0.65)

    def test_every_style_is_prompt_only_and_never_downloads_an_adapter(self):
        renderer = object.__new__(Renderer)
        renderer.pipe = mock.Mock()
        with mock.patch("peerpixel.model_hub.ensure") as ensure:
            for style, (recipe_id, adapters) in STYLE_RECIPES.items():
                self.assertEqual(adapters, ())
                mode = renderer.apply_style({
                    "style": style, "recipeId": recipe_id,
                    "manifestVersion": "2026-08-23.1",
                })
                self.assertEqual(mode, "prompt_only")
        ensure.assert_not_called()
        renderer.pipe.set_adapters.assert_not_called()

    def test_generation_evidence_reports_precision_memory_and_style_mode(self):
        class Enhancer:
            def enhance_pair(self, *_args, **_kwargs):
                return {"prompt": "polished", "negativePrompt": "watermark, blur"}
            def unload(self): pass
        class Safety:
            def classify(self, _jpeg): return {"label": "normal", "nsfwScore": 0.0}

        renderer = object.__new__(Renderer)
        renderer._enhancer, renderer._safety = Enhancer(), Safety()
        renderer._precision_mode = "int8"
        renderer._memory_mode = "resident"
        renderer._style_mode = "prompt_only"
        renderer.render = lambda *_args, **_kwargs: b"jpeg"
        _, evidence = renderer.generate_job({
            "prompt": "fox", "style": "anime", "operation": "draft", "seed": 7,
            "recipeId": "anime-v1", "manifestVersion": "2026-08-23.1",
        })
        self.assertEqual({key: evidence[key] for key in
                          ("precision", "memoryMode", "styleMode",
                           "negativePrompt", "negativeConditioning")}, {
            "precision": "int8", "memoryMode": "resident",
            "styleMode": "prompt_only", "negativePrompt": "watermark, blur",
            "negativeConditioning": "native",
        })
