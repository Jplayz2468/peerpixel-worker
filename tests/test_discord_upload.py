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

    @mock.patch("peerpixel.prompt_enhancer.PromptEnhancer")
    @mock.patch("peerpixel.config.read", return_value={"promptAdapter": "/models/bootstrap"})
    def test_enhancement_injects_the_machine_local_adapter(self, _read, enhancer_type):
        renderer = self.renderer()
        renderer._enhancer = None
        enhancer_type.return_value.enhance_batch.return_value = ["one", "two", "three", "four"]
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

    @mock.patch("peerpixel.safety.SafetyClassifier")
    @mock.patch.object(api, "submit_discord_result")
    def test_variations_share_the_seed_and_upscale_runs_true_50_step_refinement(self, submit, safety_type):
        renderer = self.renderer()
        output = io.BytesIO()
        Image.new("RGB", (16, 16), "blue").save(output, "JPEG")
        renderer.render.return_value = output.getvalue()
        renderer._safety = None
        safety_type.return_value.classify.return_value = {"label": "normal", "nsfwScore": 0.01}
        vary = {**self.task(), "operation": "vary", "outputCount": 4,
                "prompts": ["one", "two", "three", "four"], "strength": .88,
                "sourceUrl": "/source", "sourceImageId": "image"}
        with mock.patch.object(api, "source_image", return_value=b"source"):
            worker._discord_task(Link(), vary, renderer, "device")
        self.assertEqual({call.args[0]["seed"] for call in renderer.render.call_args_list}, {7})

        renderer.reset_mock()
        refine = {**self.task(), "operation": "refine", "width": 1024, "height": 1024,
                  "steps": 50, "strength": .42, "sourceUrl": "/source",
                  "sourceImageId": "image"}
        with mock.patch.object(api, "source_image", return_value=b"source"):
            worker._discord_task(Link(), refine, renderer, "device")
        job = renderer.render.call_args.args[0]
        self.assertEqual((job["operation"], job["width"], job["height"], job["steps"]),
                         ("refine", 1024, 1024, 50))
        self.assertEqual((job["editStrength"], job["_editSource"]), (.42, b"source"))

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
