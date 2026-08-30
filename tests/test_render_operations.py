"""What each operation actually asks the pipeline for.

A fake pipeline stands in for Klein, so these run on a laptop with no GPU and
no model download. What is being checked is the contract the network depends
on: sizes, step counts, and that every render is handed nothing but its prompt
and the noise its seed names.
"""
import io
import unittest
from unittest import mock

from peerpixel import render


class FakePipeline:
    """Enough of Klein to record what it was asked for."""

    vae_scale_factor = 8

    class transformer:  # noqa: N801 - mirrors the attribute it stands in for
        class config:
            in_channels = 128

    def __init__(self):
        self.calls = []

    def set_progress_bar_config(self, **_):
        pass

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        hook = kwargs.get("callback_on_step_end")
        if hook is not None:
            self._num_timesteps = kwargs["num_inference_steps"]
            for index in range(kwargs["num_inference_steps"]):
                hook(self, index, 0, {})

        from PIL import Image
        image = Image.new("RGB", (kwargs["width"], kwargs["height"]), (30, 40, 50))
        return type("Result", (), {"images": [image]})()


class FakeGenerator:
    """Stands in for torch.Generator, which is the only thing `render` needs an
    accelerator stack for. Keeping it out lets the size and conditioning
    contract be checked on any machine."""

    def __init__(self, seed):
        self._seed = seed

    def initial_seed(self):
        return self._seed


def renderer_with(pipe):
    renderer = render.Renderer.__new__(render.Renderer)
    renderer.pipe = pipe
    renderer._device = "cpu"
    renderer.accelerator = "test"
    renderer.warm = lambda: None
    renderer.seed_generator = lambda seed: FakeGenerator(int(seed))
    import torch

    renderer._dtype = torch.float32
    return renderer


def a_jpeg(size=(128, 128), colour=(200, 60, 20)):
    from PIL import Image
    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, "JPEG")
    return buffer.getvalue()


class OperationTableTests(unittest.TestCase):
    def test_network_operations_have_no_draft(self):
        self.assertNotIn("draft", render.NETWORK_OPERATIONS)
        self.assertIn("master", render.NETWORK_OPERATIONS)
        self.assertIn("probe", render.NETWORK_OPERATIONS)

    def test_a_draft_payload_is_refused(self):
        with self.assertRaises(ValueError):
            render.operation_of({"operation": "draft"})

    def test_public_master_and_internal_probe_have_their_pinned_contracts(self):
        self.assertEqual(
            render.operation_of({"operation": "master"}),
            {"name": "master", "width": 1024, "height": 1024, "steps": 9, "guidance": 0.0})
        self.assertEqual(
            render.operation_of({"operation": "probe"}),
            {"name": "probe", "width": 128, "height": 128, "steps": 9, "guidance": 0.0})

    def test_a_check_is_exactly_a_master(self):
        master = dict(render.operation_of({"operation": "master"}), name="check")
        check = dict(render.operation_of({"operation": "verify"}), name="check")
        self.assertEqual(check, master, "a check that differs from the job it audits proves nothing")

    def test_an_unknown_operation_is_refused_rather_than_guessed(self):
        with self.assertRaises(ValueError):
            render.operation_of({"operation": "best"})

    def test_a_job_cannot_talk_the_worker_into_a_size_it_was_not_priced_at(self):
        with self.assertRaises(ValueError):
            render.operation_of({"operation": "probe", "width": 1024, "height": 1024})
        with self.assertRaises(ValueError):
            render.operation_of({"operation": "master", "width": 256, "height": 256})
        with self.assertRaises(ValueError):
            render.operation_of({"operation": "master", "width": 2048, "height": 2048})

    def test_a_master_accepts_each_network_priced_aspect_ratio(self):
        for width, height in (
            (1024, 1024), (896, 1120), (1120, 896), (1344, 768), (768, 1344),
        ):
            with self.subTest(width=width, height=height):
                spec = render.operation_of({
                    "operation": "master", "width": width, "height": height,
                })
                self.assertEqual((spec["width"], spec["height"]), (width, height))

    def test_refine_accepts_each_512px_upscale_aspect_at_nine_steps(self):
        for width, height in (
            (512, 512), (512, 384), (384, 512), (512, 288), (288, 512),
        ):
            with self.subTest(width=width, height=height):
                spec = render.operation_of({
                    "operation": "refine", "width": width, "height": height, "steps": 9,
                })
                self.assertEqual((spec["width"], spec["height"], spec["steps"]),
                                 (width, height, 9))

    def test_a_master_still_refuses_an_arbitrary_near_megapixel_shape(self):
        with self.assertRaises(ValueError):
            render.operation_of({"operation": "master", "width": 1024, "height": 768})

    def test_a_payload_with_no_operation_is_a_master(self):
        self.assertEqual(render.operation_of({})["name"], "master")


class ProbeTests(unittest.TestCase):
    def test_a_probe_is_128px_at_nine_steps_from_the_prompt_alone(self):
        pipe = FakePipeline()
        jpeg = renderer_with(pipe).render(
            {"prompt": "a quiet harbour", "seed": 7, "operation": "probe"})

        (call,) = pipe.calls
        self.assertEqual(call["width"], 128)
        self.assertEqual(call["height"], 128)
        self.assertEqual(call["num_inference_steps"], 9)
        self.assertEqual(call["guidance_scale"], 0.0)
        self.assertNotIn("image", call, "a probe is a complete prompt-and-seed render")
        self.assertEqual(call["generator"].initial_seed(), 7)
        self.assertTrue(jpeg.startswith(b"\xff\xd8\xff"))

    def test_a_probe_reports_every_step_it_runs(self):
        # Read from the table rather than written out, because the point of
        # this test is that nothing is skipped -- not what the number is. The
        # number is pinned once, above, against the prices on the server.
        steps = render.OPERATIONS["probe"]["steps"]
        seen = []
        renderer_with(FakePipeline()).render(
            {"prompt": "x", "seed": 1, "operation": "probe"},
            on_step=lambda done, total: seen.append((done, total)))
        self.assertEqual(seen, [(n, steps) for n in range(1, steps + 1)])


class MasterTests(unittest.TestCase):
    def test_edit_pipeline_shares_components_without_from_pipe_recasting_them(self):
        class Component:
            def to(self, **_kwargs):
                return self

        class Source:
            scheduler = Component()
            vae = Component()
            text_encoder = Component()
            tokenizer = Component()
            transformer = Component()

        class EditPipeline:
            def __init__(self, **components):
                self.__dict__.update(components)

            @classmethod
            def from_pipe(cls, _pipe):
                raise AssertionError("from_pipe recasts the FP8 encoder to float32")

            def set_progress_bar_config(self, **_kwargs):
                pass

        renderer = render.Renderer.__new__(render.Renderer)
        renderer.pipe = Source()
        renderer._device = "cuda"
        renderer._dtype = __import__("torch").bfloat16
        renderer._edit_pipe = None
        with mock.patch("diffusers.ZImageInpaintPipeline", EditPipeline):
            edit = renderer.edit_pipeline()
        self.assertIs(edit.transformer, renderer.pipe.transformer)
        self.assertIs(edit.text_encoder, renderer.pipe.text_encoder)

    def test_vary_uses_only_the_generator_named_by_its_seed(self):
        pipe = FakePipeline()
        renderer_with(pipe).render({"prompt": "x", "seed": 7, "operation": "vary",
            "noiseBlendSeed": 19, "noiseBlendStrength": .35})
        call = pipe.calls[0]
        self.assertNotIn("latents", call)
        self.assertNotIn("image", call)
        self.assertNotIn("strength", call)

    def test_a_final_is_1024px_at_nine_turbo_steps_from_prompt_and_seed(self):
        pipe = FakePipeline()
        renderer_with(pipe).render(
            {"prompt": "a quiet harbour", "seed": 7, "operation": "master"})

        (call,) = pipe.calls
        self.assertEqual(call["width"], 1024)
        self.assertEqual(call["height"], 1024)
        self.assertEqual(call["num_inference_steps"], 9)
        self.assertEqual(call["generator"].initial_seed(), 7)
        # Not img2img and not reference conditioning. A final is a native
        # render at its own resolution with nothing else in its context, which
        # is the only way it can be as good as one.
        for absent in ("image", "strength", "mask_image"):
            self.assertNotIn(absent, call)

    def test_a_final_passes_the_seeded_generator_to_the_pipeline(self):
        pipe = FakePipeline()
        renderer_with(pipe).render({"prompt": "x", "seed": 7, "operation": "master"})
        self.assertEqual(pipe.calls[0]["generator"].initial_seed(), 7)
        self.assertNotIn("latents", pipe.calls[0])

    def test_refine_compensates_low_strength_to_keep_nine_actual_steps(self):
        pipe, edit_pipe = FakePipeline(), FakePipeline()
        renderer = renderer_with(pipe)
        renderer.edit_pipeline = lambda: edit_pipe
        source = a_jpeg((512, 512))
        renderer.render({"prompt": "a quiet harbour", "seed": 7,
            "operation": "refine", "width": 512, "height": 512,
            "steps": 9, "editMode": "refine", "editStrength": .20,
            "sourceImageId": "source", "_editSource": source})
        call = edit_pipe.calls[0]
        self.assertEqual(call["image"].size, (512, 512))
        self.assertEqual(call["strength"], .20)
        self.assertEqual(call["num_inference_steps"], 45)
        self.assertEqual(call["mask_image"].getextrema(), (255, 255))

    def test_the_decode_is_announced_so_the_bar_does_not_sit_at_the_last_step(self):
        """The freeze this exists to prevent, as a test.

        Everything after the final step is the VAE building a 1024px picture,
        and on a modest machine that is a minute or two. Inside the render
        phase it has nowhere to go: the phase has already measured itself
        complete, so the bar stops dead at step 50 of 50 and stays there.
        """
        pipe = FakePipeline()
        order = []
        renderer_with(pipe).render(
            {"prompt": "x", "seed": 1, "operation": "master"},
            on_step=lambda done, total: order.append(("step", done)),
            on_decode=lambda: order.append(("decode", None)))

        self.assertEqual(order[-1], ("decode", None), "decoding is announced last")
        self.assertEqual(order[-2], ("step", 9), "and only after the final step")
        self.assertEqual(sum(1 for kind, _ in order if kind == "decode"), 1)

    def test_cuda_oom_does_not_reload_the_resident_int8_model(self):
        import torch

        renderer = object.__new__(render.Renderer)
        renderer._device = "cuda"
        renderer._memory_mode = "resident"
        attempts, fallback = [], []

        def attempt(*_args, **_kwargs):
            attempts.append(True)
            raise torch.OutOfMemoryError("CUDA out of memory")

        renderer._render = attempt
        renderer._retry_low_memory = lambda: fallback.append(True)
        with self.assertRaises(torch.OutOfMemoryError):
            renderer.render({"operation": "master"})
        self.assertEqual(len(attempts), 1)
        self.assertEqual(fallback, [])


class TurboConditioningTests(unittest.TestCase):

    def test_every_operation_uses_turbo_guidance(self):
        for operation in ("master", "verify", "probe"):
            with self.subTest(operation=operation):
                self.assertEqual(render.operation_of({"operation": operation})["guidance"], 0.0)

    def test_the_scale_reaches_the_pipeline(self):
        for operation in ("master", "probe"):
            pipe = FakePipeline()
            renderer_with(pipe).render({"prompt": "x", "seed": 1, "operation": operation})
            self.assertEqual(pipe.calls[0]["guidance_scale"], 0.0)

    def test_turbo_does_not_receive_a_negative_prompt(self):
        pipe = FakePipeline()
        renderer_with(pipe).render({
            "prompt": "a fox", "negativePrompt": "watermark, duplicate animals",
            "seed": 1, "operation": "master",
        })
        self.assertNotIn("negative_prompt", pipe.calls[0])

    def test_the_payload_cannot_retune_turbo_guidance(self):
        pipe = FakePipeline()
        renderer_with(pipe).render(
            {"prompt": "x", "seed": 1, "operation": "master", "guidance": 6.5})
        self.assertEqual(pipe.calls[0]["guidance_scale"], 0.0)

    def test_a_nonsensical_scale_falls_back_rather_than_wasting_a_render(self):
        # Below 1.0 the pipeline switches guidance off entirely, which is the
        # distilled behaviour this checkpoint exists to avoid.
        for asked in (0, 1.0, -3, 999, "high", None, [5]):
            with self.subTest(asked=asked):
                spec = render.operation_of(
                    {"operation": "master", "guidance": asked})
                self.assertEqual(spec["guidance"], 0.0)

    def test_size_and_steps_are_still_pinned_against_the_payload(self):
        # These decide what a job costs, so they are not negotiable.
        spec = render.operation_of({"operation": "master", "steps": 4})
        self.assertEqual(spec["steps"], 9)
        with self.assertRaises(ValueError):
            render.operation_of({"operation": "master", "width": 2048, "height": 2048})

    def test_the_checkpoint_is_z_image_turbo_and_is_pinned(self):
        self.assertEqual(render.MODEL, "Tongyi-MAI/Z-Image-Turbo")
        self.assertTrue(render.REVISION,
                        "an unpinned repo means two machines can run different weights")
