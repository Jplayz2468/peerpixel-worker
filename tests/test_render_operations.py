"""What each operation actually asks the pipeline for.

A fake pipeline stands in for Klein, so these run on a laptop with no GPU and
no model download. What is being checked is the contract the network depends
on: sizes, step counts, and that every render is handed nothing but its prompt
and the noise its seed names.
"""
import io
import unittest

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
            {"name": "master", "width": 1024, "height": 1024, "steps": 50, "guidance": 4.0})
        self.assertEqual(
            render.operation_of({"operation": "probe"}),
            {"name": "probe", "width": 128, "height": 128, "steps": 50, "guidance": 4.0})

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

    def test_a_master_still_refuses_an_arbitrary_near_megapixel_shape(self):
        with self.assertRaises(ValueError):
            render.operation_of({"operation": "master", "width": 1024, "height": 768})

    def test_a_payload_with_no_operation_is_a_master(self):
        self.assertEqual(render.operation_of({})["name"], "master")


class ProbeTests(unittest.TestCase):
    def test_a_probe_is_128px_at_fifty_steps_from_the_prompt_alone(self):
        pipe = FakePipeline()
        jpeg = renderer_with(pipe).render(
            {"prompt": "a quiet harbour", "seed": 7, "operation": "probe"})

        (call,) = pipe.calls
        self.assertEqual(call["width"], 128)
        self.assertEqual(call["height"], 128)
        self.assertEqual(call["num_inference_steps"], 50)
        self.assertEqual(call["guidance_scale"], 4.0)
        self.assertNotIn("image", call, "a probe is a complete prompt-and-seed render")
        self.assertEqual(call["generator"].initial_seed(), 7)
        self.assertTrue(jpeg.startswith(b"\xff\xd8\xff"))

    def test_apple_silicon_renderer_preserves_the_network_contract_through_mlx(self):
        class Backend:
            def render(self, **kwargs):
                self.kwargs = kwargs
                kwargs["on_step"](50, 50)
                return a_jpeg((512, 512))

        backend = Backend()
        renderer = render.Renderer.__new__(render.Renderer)
        renderer._mlx_backend = backend
        renderer._device = "mps"
        renderer._style_mode = "prompt_only"
        renderer.warm = lambda: None
        renderer.apply_style = lambda _job: "prompt_only"
        steps = []
        jpeg = renderer._render({
            "prompt": "a quiet harbour", "negativePrompt": "watermark, blur",
            "seed": 7, "operation": "master",
            "style": "photoreal", "recipeId": "photoreal-v2",
            "manifestVersion": render.MANIFEST_VERSION,
        }, on_step=lambda done, total: steps.append((done, total)))
        self.assertTrue(jpeg.startswith(b"\xff\xd8\xff"))
        self.assertEqual(backend.kwargs["width"], 1024)
        self.assertEqual(backend.kwargs["steps"], 50)
        self.assertEqual(backend.kwargs["guidance"], 4.0)
        self.assertEqual(backend.kwargs["seed"], 7)
        self.assertEqual(backend.kwargs["negative_prompt"], "watermark, blur")
        self.assertEqual(steps, [(50, 50)])

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
    def test_a_final_is_1024px_at_fifty_guided_steps_from_prompt_and_seed(self):
        pipe = FakePipeline()
        renderer_with(pipe).render(
            {"prompt": "a quiet harbour", "seed": 7, "operation": "master"})

        (call,) = pipe.calls
        self.assertEqual(call["width"], 1024)
        self.assertEqual(call["height"], 1024)
        self.assertEqual(call["num_inference_steps"], 50)
        self.assertEqual(call["generator"].initial_seed(), 7)
        # Not img2img and not reference conditioning. A final is a native
        # render at its own resolution with nothing else in its context, which
        # is the only way it can be as good as one.
        for absent in ("image", "strength", "mask_image"):
            self.assertNotIn(absent, call)

    def test_a_final_starts_from_the_noise_its_seed_names(self):
        pipe = FakePipeline()
        renderer_with(pipe).render({"prompt": "x", "seed": 7, "operation": "master"})
        latents = pipe.calls[0]["latents"]
        self.assertEqual(tuple(latents.shape), (1, 128, 64, 64))

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
        self.assertEqual(order[-2], ("step", 50), "and only after the final step")
        self.assertEqual(sum(1 for kind, _ in order if kind == "decode"), 1)

    def test_cuda_oom_reloads_once_in_the_guaranteed_low_memory_mode(self):
        import torch

        renderer = object.__new__(render.Renderer)
        renderer._device = "cuda"
        renderer._memory_mode = "group"
        attempts, fallback = [], []

        def attempt(*_args, **_kwargs):
            attempts.append(True)
            if len(attempts) == 1:
                raise torch.OutOfMemoryError("CUDA out of memory")
            return b"jpeg"

        renderer._render = attempt
        renderer._retry_low_memory = lambda: fallback.append(True)
        self.assertEqual(renderer.render({"operation": "master"}), b"jpeg")
        self.assertEqual(len(attempts), 2)
        self.assertEqual(fallback, [True])


class GuidanceTests(unittest.TestCase):
    """The whole point of the base checkpoint.

    The pipeline turns classifier-free guidance on only when the checkpoint is
    not distilled, and then runs a second pass per step against an empty prompt.
    Passing a scale the checkpoint ignores is how the distilled model quietly
    disregarded every spatial and negative instruction it was given.
    """

    def test_every_operation_is_guided(self):
        for operation in ("master", "verify", "probe"):
            with self.subTest(operation=operation):
                self.assertEqual(render.operation_of({"operation": operation})["guidance"], 4.0)

    def test_the_scale_reaches_the_pipeline(self):
        for operation in ("master", "probe"):
            pipe = FakePipeline()
            renderer_with(pipe).render({"prompt": "x", "seed": 1, "operation": operation})
            self.assertEqual(pipe.calls[0]["guidance_scale"], 4.0)

    def test_diffusers_receives_the_generated_negative_prompt(self):
        pipe = FakePipeline()
        renderer_with(pipe).render({
            "prompt": "a fox", "negativePrompt": "watermark, duplicate animals",
            "seed": 1, "operation": "master",
        })
        self.assertEqual(pipe.calls[0]["negative_prompt"],
                         "watermark, duplicate animals")

    def test_the_server_may_retune_guidance_without_a_worker_release(self):
        """The reason guidance is read from the payload at all."""
        pipe = FakePipeline()
        renderer_with(pipe).render(
            {"prompt": "x", "seed": 1, "operation": "master", "guidance": 6.5})
        self.assertEqual(pipe.calls[0]["guidance_scale"], 6.5)

    def test_a_nonsensical_scale_falls_back_rather_than_wasting_a_render(self):
        # Below 1.0 the pipeline switches guidance off entirely, which is the
        # distilled behaviour this checkpoint exists to avoid.
        for asked in (0, 1.0, -3, 999, "high", None, [5]):
            with self.subTest(asked=asked):
                spec = render.operation_of(
                    {"operation": "master", "guidance": asked})
                self.assertEqual(spec["guidance"], 4.0)

    def test_size_and_steps_are_still_pinned_against_the_payload(self):
        # These decide what a job costs, so they are not negotiable.
        spec = render.operation_of({"operation": "master", "steps": 4})
        self.assertEqual(spec["steps"], 50)
        with self.assertRaises(ValueError):
            render.operation_of({"operation": "master", "width": 2048, "height": 2048})

    def test_the_checkpoint_is_the_base_one_and_is_pinned(self):
        self.assertIn("base", render.MODEL,
                      "the distilled checkpoint ignores guidance entirely")
        self.assertTrue(render.REVISION,
                        "an unpinned repo means two machines can run different weights")
