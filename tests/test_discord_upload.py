import unittest
import io
import json
from unittest import mock

from PIL import Image, JpegImagePlugin

from peerpixel import api, worker


class Link:
    def __init__(self):
        self.sent = []

    def send(self, message):
        self.sent.append(message)


class DiscordUploadTests(unittest.TestCase):
    def task(self):
        return {"id": "job-1", "stage": "render", "assignmentToken": "lease-1",
                "prompt": "a fox", "seed": 7, "outputCount": 1, "steps": 2,
                "operation": "grid", "strength": 0, "width": 512, "height": 512}

    def renderer(self):
        renderer = mock.Mock()
        renderer.render.return_value = b"jpeg"
        return renderer

    def enhance(self, renderer, enhancer_type):
        enhancer_type.return_value.enhance_pairs_batch.return_value = [
            {"prompt": str(index), "negativePrompt": "bad"} for index in range(4)
        ]
        enhancer_type.return_value.provenance = "bootstrap-0002"
        worker._discord_task(Link(), {"id": "job", "stage": "enhance", "count": 4,
            "mode": "broad", "assignmentToken": "lease", "prompt": "fox"},
            renderer, "device")

    @mock.patch("peerpixel.render.has_cuda_headroom", return_value=False)
    @mock.patch("peerpixel.prompt_enhancer.PromptEnhancer")
    @mock.patch("peerpixel.config.read", return_value={})
    def test_a_full_card_gives_the_pipeline_back_rather_than_failing_to_enhance(
            self, _read, enhancer_type, _headroom):
        """A resident pipeline is most of the card, and enhancement needs room.

        Without this an enhancement fails with an out-of-memory error on a
        machine that was rendering happily a moment earlier, and keeps failing
        until somebody restarts the worker.
        """
        renderer = self.renderer()
        renderer._enhancer = None

        self.enhance(renderer, enhancer_type)

        renderer.unload.assert_called_once_with()
        # Releasing the pipeline must not throw away the enhancer with it.
        self.assertFalse(enhancer_type.return_value.unload.called)

    @mock.patch("peerpixel.render.has_cuda_headroom", return_value=True)
    @mock.patch("peerpixel.prompt_enhancer.PromptEnhancer")
    @mock.patch("peerpixel.config.read", return_value={})
    def test_a_card_with_room_is_left_alone_so_enhancement_does_not_stall(
            self, _read, enhancer_type, _headroom):
        """Collecting a large pipeline is slow and users saw it as a pause at 2%."""
        renderer = self.renderer()
        renderer._enhancer = None

        self.enhance(renderer, enhancer_type)

        renderer.unload.assert_not_called()

    def test_a_failed_render_still_gives_its_pipeline_back(self):
        """The success path releases after upload; a failure never reaches it.

        A pipeline left resident wedges every later enhancement on this machine.
        """
        renderer = self.renderer()
        renderer.render.side_effect = RuntimeError("CUDA out of memory")

        with self.assertRaises(RuntimeError):
            worker._discord_task(Link(), self.task(), renderer, "device")

        renderer.unload.assert_called_once_with()

    @mock.patch("peerpixel.prompt_enhancer.PromptEnhancer")
    @mock.patch("peerpixel.config.read", return_value={"promptAdapter": "/models/bootstrap"})
    def test_enhancement_injects_the_machine_local_adapter(self, _read, enhancer_type):
        renderer = self.renderer()
        renderer._enhancer = None
        enhancer_type.return_value.enhance_pairs_batch.return_value = [
            {"prompt": "one", "negativePrompt": "bad one"},
            {"prompt": "two", "negativePrompt": "bad two"},
            {"prompt": "three", "negativePrompt": "bad three"},
            {"prompt": "four", "negativePrompt": "bad four"},
        ]
        enhancer_type.return_value.provenance = "bootstrap-0002"
        link = Link()
        worker._discord_task(link, {"id": "job", "stage": "enhance", "count": 4,
            "mode": "broad", "sampling": {"temperature": .9},
            "assignmentToken": "lease", "prompt": "fox"}, renderer, "device")
        enhancer_type.assert_called_once_with(adapter_path="/models/bootstrap")
        result = next(json.loads(message) for message in link.sent
                      if json.loads(message).get("type") == "task_result")
        self.assertEqual(result["provenance"], "bootstrap-0002")
        self.assertEqual(result["prompts"], ["one", "two", "three", "four"])
        self.assertEqual(result["negativePrompts"], ["bad one", "bad two", "bad three", "bad four"])
        renderer.unload.assert_not_called()
        progress = [json.loads(message)["progress"] for message in link.sent
                    if json.loads(message).get("type") == "progress"]
        self.assertGreaterEqual(len(progress), 2)

    @mock.patch("peerpixel.safety.SafetyClassifier")
    @mock.patch.object(api, "submit_discord_result")
    def test_initial_grid_uses_coordinator_seeds_for_four_prompts(self, submit, safety_type):
        output = io.BytesIO()
        Image.new("RGB", (16, 16), "blue").save(output, "JPEG")
        renderer = self.renderer()
        renderer.render.return_value = output.getvalue()
        renderer._safety = None
        safety_type.return_value.classify.return_value = {"label": "normal", "nsfwScore": 0.01}
        task = {**self.task(), "outputCount": 4, "prompts": ["one", "two", "three", "four"],
                "seeds": [11, 22, 33, 44]}
        worker._discord_task(Link(), task, renderer, "device")
        self.assertEqual([call.args[0]["prompt"] for call in renderer.render.call_args_list], task["prompts"])
        self.assertEqual([call.args[0]["seed"] for call in renderer.render.call_args_list], [11, 22, 33, 44])
        renderer.unload.assert_called_once_with()

    @mock.patch("peerpixel.safety.SafetyClassifier")
    @mock.patch.object(api, "submit_discord_result")
    def test_variations_and_upscale_use_noise_continuation_without_source_images(self, submit, safety_type):
        renderer = self.renderer()
        output = io.BytesIO()
        Image.new("RGB", (16, 16), "blue").save(output, "JPEG")
        renderer.render.return_value = output.getvalue()
        renderer._safety = None
        safety_type.return_value.classify.return_value = {"label": "normal", "nsfwScore": 0.01}
        vary = {**self.task(), "operation": "vary", "outputCount": 4, "baseSeed": 7,
                "prompts": ["one", "two", "three", "four"], "strength": .55,
                "seeds": [11, 22, 33, 44]}
        worker._discord_task(Link(), vary, renderer, "device")
        self.assertEqual([call.args[0]["seed"] for call in renderer.render.call_args_list], [7] * 4)
        self.assertEqual([call.args[0]["noiseBlendSeed"] for call in renderer.render.call_args_list], [11, 22, 33, 44])

        renderer.reset_mock()
        refine = {**self.task(), "operation": "refine", "width": 1024, "height": 1024,
                  "steps": 50, "baseSeed": 7, "noiseBlendStrength": .12,
                  "noiseBaseWidth": 512, "noiseBaseHeight": 512}
        worker._discord_task(Link(), refine, renderer, "device")
        job = renderer.render.call_args.args[0]
        self.assertEqual((job["operation"], job["width"], job["height"], job["steps"]),
                         ("refine", 1024, 1024, 50))
        self.assertEqual((job["seed"], job["noiseBaseWidth"]), (7, 512))
        self.assertNotEqual(job["noiseBlendSeed"], job["seed"])
        self.assertNotIn("_editSource", job)

    def test_four_cells_are_composed_into_one_two_by_two_grid(self):
        cells = []
        for color in ("red", "green", "blue", "yellow"):
            output = io.BytesIO()
            Image.new("RGB", (512, 512), color).save(output, "JPEG")
            cells.append(output.getvalue())

        grid = worker.compose_grid(cells)

        image = Image.open(io.BytesIO(grid))
        self.assertEqual(image.size, (1024, 1024))
        self.assertEqual(image.format, "JPEG")
        self.assertEqual(JpegImagePlugin.get_sampling(image), 0)

    def test_composite_keeps_a_1024_pixel_long_edge_for_supported_aspects(self):
        for cell_size, expected in (((408, 512), (816, 1024)), ((512, 408), (1024, 816))):
            output = io.BytesIO()
            Image.new("RGB", cell_size, "navy").save(output, "JPEG")
            image = Image.open(io.BytesIO(worker.compose_grid([output.getvalue()] * 4)))
            self.assertEqual(image.size, expected)

    @mock.patch("peerpixel.safety.SafetyClassifier")
    @mock.patch.object(api, "submit_discord_result")
    def test_four_output_task_uploads_cells_and_composite(self, submit, safety_type):
        output = io.BytesIO()
        Image.new("RGB", (32, 24), "purple").save(output, "JPEG")
        renderer = self.renderer()
        renderer.render.return_value = output.getvalue()
        renderer._safety = None
        safety_type.return_value.classify.return_value = {"label": "normal", "nsfwScore": 0.01}
        task = {**self.task(), "outputCount": 4}

        worker._discord_task(Link(), task, renderer, "device")

        self.assertEqual(renderer.render.call_count, 4)
        cells, grid = submit.call_args.args[2:4]
        self.assertEqual(len(cells), 4)
        self.assertEqual(Image.open(io.BytesIO(grid[0])).size, (64, 48))

    @mock.patch("peerpixel.safety.SafetyClassifier")
    @mock.patch.object(worker.time, "sleep")
    @mock.patch.object(api, "submit_discord_result")
    def test_transient_upload_retries_only_the_saved_result(self, submit, _sleep, safety_type):
        submit.side_effect = [api.ApiError(503, "safety_check_unavailable"),
                              api.ApiError(503, "safety_check_unavailable"), {"imageIds": ["image"]}]
        safety_type.return_value.classify.return_value = {"label": "normal", "nsfwScore": 0.01}
        renderer = self.renderer()

        worker._discord_task(Link(), self.task(), renderer, "device")

        self.assertEqual(renderer.render.call_count, 1)
        self.assertEqual(submit.call_count, 3)

    @mock.patch("peerpixel.safety.SafetyClassifier")
    @mock.patch.object(worker.time, "sleep")
    @mock.patch.object(api, "report_discord_result_failure", create=True)
    @mock.patch.object(api, "submit_discord_result")
    def test_exhausted_upload_is_reported_terminal_without_rerender(self, submit, report, _sleep, safety_type):
        submit.side_effect = api.ApiError(503, "safety_check_unavailable")
        report.return_value = {"failed": True}
        safety_type.return_value.classify.return_value = {"label": "normal", "nsfwScore": 0.01}
        renderer = self.renderer()

        worker._discord_task(Link(), self.task(), renderer, "device")

        self.assertEqual(renderer.render.call_count, 1)
        self.assertEqual(submit.call_count, 5)
        report.assert_called_once_with(self.task(), "device", "safety_check_unavailable")


if __name__ == "__main__":
    unittest.main()
