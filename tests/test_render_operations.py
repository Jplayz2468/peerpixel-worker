"""What each operation actually asks the pipeline for.

A fake pipeline stands in for Klein, so these run on a laptop with no GPU and
no model download. What is being checked is the contract the network depends
on: sizes, step counts, the seed, and that a master is conditioned on the draft
somebody chose rather than started over from it.
"""
import io
import unittest

from peerpixel import render


class FakePipeline:
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
    return renderer


def a_jpeg(size=(128, 128), colour=(200, 60, 20)):
    from PIL import Image
    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, "JPEG")
    return buffer.getvalue()


class OperationTableTests(unittest.TestCase):
    def test_the_two_operations_have_the_sizes_the_network_prices(self):
        self.assertEqual(render.operation_of({"operation": "draft"}),
                         {"name": "draft", "width": 128, "height": 128, "steps": 4})
        self.assertEqual(render.operation_of({"operation": "master"}),
                         {"name": "master", "width": 1024, "height": 1024, "steps": 4})

    def test_an_unknown_operation_is_refused_rather_than_guessed(self):
        with self.assertRaises(ValueError):
            render.operation_of({"operation": "best"})

    def test_a_job_cannot_talk_the_worker_into_a_size_it_was_not_priced_at(self):
        with self.assertRaises(ValueError):
            render.operation_of({"operation": "draft", "width": 1024, "height": 1024})
        with self.assertRaises(ValueError):
            render.operation_of({"operation": "master", "width": 128, "height": 128})

    def test_a_payload_with_no_operation_is_a_master(self):
        self.assertEqual(render.operation_of({})["name"], "master")


class DraftTests(unittest.TestCase):
    def test_a_draft_is_128px_at_four_steps_from_the_prompt_alone(self):
        pipe = FakePipeline()
        jpeg = renderer_with(pipe).render(
            {"prompt": "a quiet harbour", "seed": 7, "operation": "draft"})

        (call,) = pipe.calls
        self.assertEqual(call["width"], 128)
        self.assertEqual(call["height"], 128)
        self.assertEqual(call["num_inference_steps"], 4)
        self.assertNotIn("image", call, "a draft has nothing to be conditioned on")
        self.assertEqual(call["generator"].initial_seed(), 7)
        self.assertTrue(jpeg.startswith(b"\xff\xd8\xff"))

    def test_a_draft_reports_all_four_steps(self):
        seen = []
        renderer_with(FakePipeline()).render(
            {"prompt": "x", "seed": 1, "operation": "draft"},
            on_step=lambda done, total: seen.append((done, total)))
        self.assertEqual(seen, [(1, 4), (2, 4), (3, 4), (4, 4)])


class MasterTests(unittest.TestCase):
    def test_a_master_is_1024px_at_four_full_steps_conditioned_on_the_draft(self):
        pipe = FakePipeline()
        renderer_with(pipe).render(
            {"prompt": "a quiet harbour", "seed": 7, "operation": "master"},
            reference=a_jpeg((128, 128)))

        (call,) = pipe.calls
        self.assertEqual(call["width"], 1024)
        self.assertEqual(call["height"], 1024)
        self.assertEqual(call["num_inference_steps"], 4)
        self.assertEqual(call["generator"].initial_seed(), 7,
                         "the master keeps the draft's seed")
        # Reference conditioning, not img2img: strength would throw away the
        # first part of a four-step schedule and leave one or two real steps.
        self.assertNotIn("strength", call)
        self.assertNotIn("mask_image", call)

    def test_the_chosen_draft_is_upscaled_to_the_output_size_before_conditioning(self):
        pipe = FakePipeline()
        renderer_with(pipe).render(
            {"prompt": "x", "seed": 1, "operation": "master"},
            reference=a_jpeg((128, 128)))
        (reference,) = pipe.calls[0]["image"]
        self.assertEqual(reference.size, (1024, 1024))
        self.assertEqual(reference.mode, "RGB")

    def test_upscaling_uses_lanczos_and_leaves_a_matching_size_alone(self):
        from PIL import Image
        source = Image.new("RGB", (1024, 1024), (10, 20, 30))
        self.assertIs(render.upscale_reference(source, (1024, 1024)), source)

        small = Image.new("RGB", (128, 128), (255, 0, 0))
        grown = render.upscale_reference(small, (1024, 1024))
        self.assertEqual(grown.size, (1024, 1024))
        self.assertEqual(grown.getpixel((512, 512)), (255, 0, 0))

    def test_a_master_with_no_reference_still_renders_at_master_size(self):
        # A probe has no browser behind it, and a browser can lose its copy.
        # Neither is a reason to hand back nothing.
        pipe = FakePipeline()
        renderer_with(pipe).render({"prompt": "x", "seed": 1, "operation": "master"})
        self.assertEqual(pipe.calls[0]["width"], 1024)
        self.assertNotIn("image", pipe.calls[0])


if __name__ == "__main__":
    unittest.main()
