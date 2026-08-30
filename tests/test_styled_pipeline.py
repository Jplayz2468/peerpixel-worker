import inspect
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from peerpixel.prompt_enhancer import (
    PromptEnhancer, SYSTEM_INSTRUCTION, bootstrap_messages, enhancement_messages,
    enforce_visible_text, extract_visible_text, negative_template, parse_enhancement, sampling_seed,
    requested_visible_text, with_style_suffix,
)
from peerpixel.safety import SafetyClassifier, THRESHOLD
from peerpixel.render import Renderer, STYLE_RECIPES


class StyledPipelineTests(unittest.TestCase):
    def test_unload_releases_cuda_cache_before_the_image_model_loads(self):
        enhancer = PromptEnhancer()
        enhancer.model = object()
        enhancer.tokenizer = object()
        with (mock.patch("torch.cuda.is_available", return_value=True),
              mock.patch("torch.cuda.empty_cache") as empty_cache):
            enhancer.unload()
        self.assertIsNone(enhancer.model)
        self.assertIsNone(enhancer.tokenizer)
        empty_cache.assert_called_once_with()

    def test_bootstrap_adapter_uses_the_plain_training_template(self):
        self.assertEqual(bootstrap_messages("  fox  "), [
            {"role": "user", "content": "fox"},
        ])

    def test_adapter_path_is_injected_explicitly_instead_of_leaking_from_global_config(self):
        with mock.patch("peerpixel.config.read", return_value={"promptAdapter": "/wrong/global/path"}):
            enhancer = PromptEnhancer(adapter_path="/models/bootstrap")
        self.assertEqual(enhancer.adapter_path, Path("/models/bootstrap"))

    def test_validated_adapter_loads_and_reports_its_version(self):
        from peerpixel.lora_manifest import write_manifest
        with tempfile.TemporaryDirectory() as folder:
            adapter = Path(folder)
            (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
            write_manifest(adapter, {
                "schemaVersion": 1, "version": "bootstrap-0001", "kind": "bootstrap",
                "baseModel": "Qwen/Qwen3-1.7B", "parentVersion": None,
                "dataset": {}, "training": {}, "evaluation": {"passed": True},
                "createdAt": "2026-08-27T00:00:00Z",
            })
            enhancer = PromptEnhancer(model_path="/models/qwen", adapter_path=adapter)
            tokenizer, base, adapted = mock.Mock(), mock.Mock(), mock.Mock()
            fake_peft = types.SimpleNamespace(PeftModel=types.SimpleNamespace(
                from_pretrained=mock.Mock(return_value=adapted)))
            with mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), \
                 mock.patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=base), \
                 mock.patch.dict(sys.modules, {"peft": fake_peft}):
                enhancer.warm()
            self.assertIs(enhancer.model, adapted)
            self.assertEqual(enhancer.provenance, "bootstrap-0001")

    def test_unvalidated_or_tampered_adapter_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            enhancer = PromptEnhancer(model_path="/models/qwen", adapter_path=folder)
            with self.assertRaises((FileNotFoundError, ValueError)):
                enhancer.warm()

    def test_bootstrap_adapter_returns_plain_prompt_without_a_style_suffix(self):
        enhancer = PromptEnhancer(adapter_path="/adapter")
        enhancer._adapter_manifest = {"version": "bootstrap-0001"}
        enhancer.warm = lambda: None
        enhancer._generate_text = lambda *_args, **_kwargs: "A red fox beneath cedar trees."
        result = enhancer.enhance_pair("fox", "auto")
        self.assertEqual(result["prompt"], "A red fox beneath cedar trees.")
        self.assertNotIn("style", result)
        self.assertEqual(enhancer.provenance, "bootstrap-0001")

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

    def test_prompt_model_uses_cuda_when_a_cuda_worker_is_available(self):
        enhancer = PromptEnhancer(model_path="/models/qwen")
        fake_model = object()
        with mock.patch("torch.cuda.is_available", return_value=True), \
             mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=object()), \
             mock.patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=fake_model) as load:
            enhancer.warm()
        self.assertEqual(load.call_args.kwargs["device_map"], "cuda")

    def test_batched_lora_json_preserves_aligned_positive_and_negative_prompts(self):
        import torch
        from transformers import BatchEncoding

        enhancer = PromptEnhancer()
        enhancer.warm = lambda: None
        enhancer.tokenizer = mock.Mock()
        enhancer.tokenizer.apply_chat_template.return_value = "chat"
        enhancer.tokenizer.return_value = BatchEncoding({
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        })
        enhancer.tokenizer.decode.side_effect = [
            json.dumps({"prompt": f"scene {index}", "negative_prompt": f"defect {index}"})
            for index in range(4)
        ]
        enhancer.model = mock.Mock(device=torch.device("cpu"))
        enhancer.model.generate.side_effect = [
            torch.tensor([[1, 2, token]]) for token in (3, 4, 5, 6)
        ]
        progress = []

        pairs = enhancer.enhance_pairs_batch("fox", count=4,
                                             on_progress=progress.append)

        self.assertEqual(pairs, [
            {"prompt": f"scene {index}", "negativePrompt": f"defect {index}"}
            for index in range(4)
        ])
        self.assertEqual(progress, [1, 2, 3, 4])
        self.assertEqual(enhancer.model.generate.call_count, 4)

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
            "prompt": "fox", "style": "anime", "operation": "master", "seed": 7,
            "recipeId": "anime-v1", "manifestVersion": "2026-08-23.1",
        })
        self.assertEqual(events, ["enhance", "release_prompt_model", "render", "moderate"])

    def test_prompt_enhancement_has_no_candidate_variation_instruction(self):
        for callable_ in (
            sampling_seed, enhancement_messages,
            PromptEnhancer.enhance_pair, PromptEnhancer.enhance,
        ):
            with self.subTest(callable=callable_.__name__):
                self.assertNotIn("variation", inspect.signature(callable_).parameters)
        content = enhancement_messages("fox", "anime")[1]["content"]
        self.assertNotIn("variation seed", content.lower())

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
        bypassed = enhancer.enhance(" raw prompt ", "photoreal", enabled=False)
        self.assertTrue(bypassed.startswith("raw prompt,"))
        self.assertIn("Kodak Portra 400", bypassed)
        self.assertEqual(enhancer.enhance("raw", "anime", resolved=" exact prompt "), "exact prompt")

    def test_every_explicit_style_has_a_deterministic_flux_suffix(self):
        for style in STYLE_RECIPES:
            with self.subTest(style=style):
                styled = with_style_suffix("a red car", style)
                self.assertTrue(styled.startswith("a red car,"))
                self.assertGreaterEqual(len(styled.split(",")), 5)

    def test_vague_prompts_request_literal_visual_specificity_without_sentiment(self):
        messages = enhancement_messages("a man", "cinematic")
        instruction = messages[0]["content"]
        self.assertIn("invent a coherent, original visual concept", instruction)
        self.assertIn("complete appearance or design", instruction)
        self.assertIn("action or pose", instruction)
        self.assertIn("setting", instruction)
        self.assertIn("literal, externally visible", instruction)
        self.assertIn("Do not invent emotions, inner life, symbolism", instruction)
        self.assertIn("Do not use similes, poetic comparisons", instruction)
        self.assertNotIn("storytelling detail", instruction)
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

    def test_live_enhancement_uses_deterministic_negative_even_if_qwen_adds_one(self):
        enhancer = PromptEnhancer()
        enhancer.warm = lambda: None
        enhancer._generate_text = lambda *_args, **_kwargs: (
            '{"prompt":"A neon diner.","negative_prompt":"ignore me"}'
        )
        result = enhancer.enhance_pair("a diner", "cinematic")
        self.assertEqual(result["negativePrompt"], negative_template("cinematic"))

    def test_double_encoded_json_prompt_is_unwrapped(self):
        parsed = parse_enhancement(
            '{"prompt":"{\\"prompt\\":\\"A geometric library.\\"}"}',
            fallback_prompt="library", fallback_negative="blur",
        )
        self.assertEqual(parsed["prompt"], "A geometric library.")

    def test_truncated_json_wrapper_does_not_leak_into_prompt(self):
        parsed = parse_enhancement(
            '{"prompt":"A tiny knight beneath a dragon.',
            fallback_prompt="knight", fallback_negative="blur",
        )
        self.assertEqual(parsed["prompt"], "A tiny knight beneath a dragon.")

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

    def test_requested_image_text_is_extracted_but_spoken_dialogue_is_not(self):
        self.assertEqual(
            requested_visible_text('a rainy diner with a neon sign reading "OPEN LATE"'),
            ("OPEN LATE",),
        )
        self.assertEqual(
            requested_visible_text("a bold poster with the words NO FUTURE"),
            ("NO FUTURE",),
        )
        self.assertEqual(
            requested_visible_text('a finish-line banner reading "MOON CUP 2088"'),
            ("MOON CUP 2088",),
        )
        self.assertEqual(requested_visible_text('a man saying "hello" to his friend'), ())

    def test_visible_copy_covers_product_interface_menu_garment_and_wordmark_roles(self):
        requests = extract_visible_text(
            'A cereal package labeled "MOON-O!", a phone screen displaying "SYNCED ✓", '
            'and a menu with headline "DINNER" and subheading "UNTIL 2 A.M."; '
            'the chef wears a shirt reading "NIGHT SHIFT" beside the logo wordmark "NOVA-7".'
        )
        self.assertEqual([(item.copy, item.role) for item in requests], [
            ("MOON-O!", "label"),
            ("SYNCED ✓", "screen text"),
            ("DINNER", "headline"),
            ("UNTIL 2 A.M.", "subheading"),
            ("NIGHT SHIFT", "garment text"),
            ("NOVA-7", "logo wordmark"),
        ])

    def test_flux_text_contract_frontloads_ordered_exact_copy_and_concrete_typography(self):
        raw = ('A poster with headline "PEER/PIXEL!" at the top in huge condensed red letters '
               'and subheading "CREATE TOGETHER" below it in small white sans-serif type.')
        enforced = enforce_visible_text(
            "A graphic community poster with a balanced vertical layout.",
            extract_visible_text(raw),
        )
        self.assertLess(enforced.index('"PEER/PIXEL!"'), enforced.index('"CREATE TOGETHER"'))
        self.assertTrue(enforced.startswith('The headline displays the exact text "PEER/PIXEL!"'))
        self.assertIn("top", enforced)
        self.assertIn("huge condensed red letters", enforced)
        self.assertIn("below", enforced)
        self.assertIn("small white sans-serif", enforced)

    def test_visible_dialogue_requires_an_explicit_image_text_role(self):
        self.assertEqual(extract_visible_text('a woman says "WAIT!" to a cyclist'), ())
        self.assertEqual(
            [(item.copy, item.role) for item in extract_visible_text(
                'a speech bubble saying "WAIT!" above the woman')],
            [("WAIT!", "speech bubble")],
        )

    def test_all_enhancement_results_receive_the_same_exact_text_contract(self):
        enhancer = PromptEnhancer()
        enhancer.warm = lambda: None
        enhancer.tokenizer = mock.Mock()
        enhancer.tokenizer.apply_chat_template.return_value = "chat"
        enhancer.tokenizer.return_value = types.SimpleNamespace(
            input_ids=types.SimpleNamespace(shape=(1, 2)), to=lambda _device: None)
        enhancer.model = mock.Mock(device="cpu")
        with mock.patch.object(enhancer, "_generate_text", return_value="A clean package on a shelf."):
            result = enhancer.enhance_pair(
                'a package labeled "NOVA-7"', "photoreal", enabled=True)
        self.assertTrue(result["prompt"].startswith(
            'The label displays the exact text "NOVA-7"'))
        self.assertNotIn("unintended text", result["negativePrompt"])
        self.assertNotIn(", letters,", f', {result["negativePrompt"]},')

    def test_text_requests_change_qwens_instruction_and_negative_template(self):
        content = enhancement_messages(
            'a book cover titled "THE LONG WAY HOME"', "illustration",
        )[1]["content"]
        self.assertIn('Exact visible text requested: "THE LONG WAY HOME"', content)
        self.assertIn("typography, placement, material, and legibility", content)
        self.assertNotIn("unintended text, letters", content)
        self.assertIn("misspelled requested text", content)

    def test_postprocessing_restores_exact_copy_if_qwen_drops_or_rewrites_it(self):
        parsed = parse_enhancement(
            '{"prompt":"A glowing neighborhood cinema marquee.",'
            '"negative_prompt":"blurry, unintended text, letters, watermark"}',
            fallback_prompt='a marquee reading "MOON PALACE"',
            fallback_negative="blur",
            visible_text=("MOON PALACE",),
        )
        self.assertIn('exact text "MOON PALACE"', parsed["prompt"])
        self.assertNotIn("unintended text", parsed["negativePrompt"])
        self.assertNotIn("letters", parsed["negativePrompt"])
        self.assertIn("misspelled requested text", parsed["negativePrompt"])

    def test_requested_copy_is_normalized_to_double_quotes(self):
        parsed = parse_enhancement(
            '{"prompt":"A neon sign reading \'OPEN LATE\' above the diner.",'
            '"negative_prompt":"blur"}',
            fallback_prompt='a sign reading "OPEN LATE"',
            fallback_negative="blur",
            visible_text=("OPEN LATE",),
        )
        self.assertIn('reading "OPEN LATE"', parsed["prompt"])
        self.assertNotIn("'OPEN LATE'", parsed["prompt"])

    def test_creative_sampling_is_reproducible_for_the_direct_request(self):
        first = sampling_seed("a man", "cinematic")
        self.assertEqual(first, sampling_seed("a man", "cinematic"))
        self.assertNotEqual(first, sampling_seed("another man", "cinematic"))
        self.assertNotEqual(first, sampling_seed("a man", "anime"))

    def test_generation_uses_greedy_decoding_to_avoid_language_corruption(self):
        import torch
        from transformers import BatchEncoding

        enhancer = PromptEnhancer()
        enhancer.tokenizer = mock.Mock()
        enhancer.tokenizer.apply_chat_template.return_value = "chat"
        enhancer.tokenizer.return_value = BatchEncoding({
            "input_ids": torch.tensor([[1, 2]]),
        })
        enhancer.tokenizer.decode.return_value = '{"prompt":"clean"}'
        enhancer.model = mock.Mock(device=torch.device("cpu"))
        enhancer.model.generate.return_value = torch.tensor([[1, 2, 3]])

        enhancer._generate_text([], max_new_tokens=10, seed=7)

        self.assertFalse(enhancer.model.generate.call_args.kwargs["do_sample"])

    def test_vague_prompts_use_one_style_aware_generation_pass(self):
        enhancer = PromptEnhancer()
        enhancer.warm = lambda: None
        calls = []
        enhancer._generate_text = lambda messages, **kwargs: (
            calls.append(messages) or '{"prompt":"A complete literal scene."}'
        )
        result = enhancer.enhance_pair("a man", "illustration")
        self.assertEqual(len(calls), 1)
        self.assertTrue(result["prompt"].startswith("A complete literal scene,"))
        self.assertIn("charcoal and graphite", result["prompt"])
        self.assertIn("Required style directive:", calls[0][1]["content"])

    def test_all_seven_styles_have_distinct_prompt_directives(self):
        expected = {"photoreal", "anime", "vector", "cinematic", "watercolor",
                    "illustration", "pixel_art"}
        self.assertEqual(set(STYLE_RECIPES), expected)
        for style in expected:
            message = enhancement_messages("subject", style)[1]["content"]
            self.assertIn("Required style directive:", message)

    def test_styles_require_concrete_medium_and_production_details(self):
        photoreal = enhancement_messages("a chef", "photoreal")[1]["content"]
        cinematic = enhancement_messages("a chef", "cinematic")[1]["content"]
        anime = enhancement_messages("a chef", "anime")[1]["content"]
        illustration = enhancement_messages("a chef", "illustration")[1]["content"]
        vector = enhancement_messages("a chef", "vector")[1]["content"]
        self.assertIn("camera body or film stock", photoreal)
        self.assertIn("focal length", photoreal)
        self.assertIn("aspect ratio", cinematic)
        self.assertIn("shot size", cinematic)
        self.assertIn("luminous gradient eyes", anime)
        self.assertIn("soft layered shading", anime)
        self.assertIn("high-key bloom", anime)
        self.assertIn("Never add a face, person, hair, or eyes", anime)
        self.assertIn("charcoal", illustration)
        self.assertIn("scribbled contour", illustration)
        self.assertIn("stroke weight", vector)

    def test_each_request_repeats_only_its_selected_style_as_mandatory(self):
        content = enhancement_messages("a red car", "illustration")[1]["content"]
        self.assertIn("Required style directive: Direct an expressive hand-drawn charcoal", content)
        self.assertIn("Include at least four applicable, concrete style-technique phrases", content)
        self.assertIn("Apply this chosen style and no other style", content)
        self.assertNotIn("1990s retro 2D anime", content)

    def test_auto_style_asks_qwen_to_choose_one_supported_style(self):
        messages = enhancement_messages("a rain-soaked city", "auto")
        content = messages[1]["content"]
        self.assertIn("Choose exactly one", content)
        for style in STYLE_RECIPES:
            self.assertIn(style, content)

    def test_auto_style_is_returned_with_the_enhanced_prompt(self):
        enhancer = PromptEnhancer()
        enhancer.warm = lambda: None
        enhancer._generate_text = lambda *_args, **_kwargs: (
            '{"style":"cinematic","prompt":"A detective crosses a rain-bright street.",'
            '"negative_prompt":"watermark, malformed anatomy"}'
        )
        result = enhancer.enhance_pair("a detective", "auto")
        self.assertEqual(result["style"], "cinematic")
        self.assertTrue(result["prompt"].startswith("A detective crosses a rain-bright street,"))
        self.assertIn("40mm anamorphic lens", result["prompt"])

    def test_renderer_uses_qwens_auto_style_but_never_overrides_an_explicit_style(self):
        class Enhancer:
            def enhance_pair(self, _prompt, style, **_kwargs):
                return {"prompt": "polished", "negativePrompt": "blur",
                        "style": "cinematic" if style == "auto" else style}
            def unload(self): pass
        class Safety:
            def classify(self, _jpeg): return {"label": "normal", "nsfwScore": 0.0}

        renderer = object.__new__(Renderer)
        renderer._enhancer, renderer._safety = Enhancer(), Safety()
        renderer._precision_mode = "native"
        renderer._memory_mode = "resident"
        renderer._style_mode = "prompt_only"
        seen = []
        renderer.render = lambda job, **_kwargs: seen.append(job) or b"jpeg"
        _, auto = renderer.generate_job({
            "prompt": "detective", "style": "auto", "operation": "master", "seed": 7,
            "recipeId": "auto-v1", "manifestVersion": "2026-08-23.1",
        })
        _, explicit = renderer.generate_job({
            "prompt": "detective", "style": "anime", "operation": "master", "seed": 8,
            "recipeId": "anime-v1", "manifestVersion": "2026-08-23.1",
        })
        self.assertEqual((seen[0]["style"], auto["style"], auto["recipeId"]),
                         ("cinematic", "cinematic", STYLE_RECIPES["cinematic"][0]))
        self.assertEqual((seen[1]["style"], explicit["style"]), ("anime", "anime"))

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
            "prompt": "fox", "style": "anime", "operation": "master", "seed": 7,
            "recipeId": "anime-v1", "manifestVersion": "2026-08-23.1",
        })
        self.assertEqual({key: evidence[key] for key in
                          ("precision", "memoryMode", "styleMode",
                           "negativePrompt", "negativeConditioning")}, {
            "precision": "int8", "memoryMode": "resident",
            "styleMode": "prompt_only", "negativePrompt": "watermark, blur",
            "negativeConditioning": "none",
        })
