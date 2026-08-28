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

    def test_enhancement_returns_four_aligned_prompt_pairs(self):
        renderer = self.renderer()
        renderer._enhancer = mock.Mock()
        renderer._enhancer.enhance_pairs.return_value = [
            {"prompt": f"positive {index}", "negativePrompt": f"negative {index}"}
            for index in range(1, 5)
        ]
        link = Link()
        task = {"id": "job-1", "stage": "enhance", "assignmentToken": "lease-1",
                "prompt": "city", "count": 4,
                "sampling": {"temperatures": [0.4, 0.6, 0.8, 1.0]}}

        worker._discord_task(link, task, renderer, "device")

        result = json.loads(link.sent[-1])
        self.assertEqual(result["prompts"],
                         ["positive 1", "positive 2", "positive 3", "positive 4"])
        self.assertEqual(result["negativePrompts"],
                         ["negative 1", "negative 2", "negative 3", "negative 4"])

    @mock.patch("peerpixel.safety.SafetyClassifier")
    @mock.patch.object(api, "submit_discord_result")
    def test_render_indexes_positive_and_negative_arrays_together(self, _submit, safety_type):
        safety_type.return_value.classify.return_value = {"label": "normal", "nsfwScore": 0.01}
        renderer = self.renderer()
        output = io.BytesIO()
        Image.new("RGB", (32, 24), "purple").save(output, "JPEG")
        renderer.render.return_value = output.getvalue()
        renderer._safety = None
        task = {**self.task(), "outputCount": 4,
                "prompts": [f"positive {index}" for index in range(1, 5)],
                "negativePrompts": [f"negative {index}" for index in range(1, 5)]}

        worker._discord_task(Link(), task, renderer, "device")

        jobs = [call.args[0] for call in renderer.render.call_args_list]
        self.assertEqual([(job["prompt"], job["negativePrompt"]) for job in jobs], [
            ("positive 1", "negative 1"), ("positive 2", "negative 2"),
            ("positive 3", "negative 3"), ("positive 4", "negative 4"),
        ])

    @mock.patch("peerpixel.safety.SafetyClassifier")
    @mock.patch.object(api, "submit_discord_result")
    def test_single_render_uses_scalar_prompt_pair(self, _submit, safety_type):
        safety_type.return_value.classify.return_value = {"label": "normal", "nsfwScore": 0.01}
        renderer = self.renderer()
        renderer._safety = None
        task = {**self.task(), "negativePrompt": "negative fox"}

        worker._discord_task(Link(), task, renderer, "device")

        job = renderer.render.call_args.args[0]
        self.assertEqual((job["prompt"], job["negativePrompt"]),
                         ("a fox", "negative fox"))

    def test_render_rejects_non_string_prompt_pair_entries(self):
        renderer = self.renderer()
        task = {**self.task(), "prompts": [123], "negativePrompts": ["negative fox"]}
        with self.assertRaisesRegex(ValueError, "prompt_pair_count_mismatch"):
            worker._discord_task(Link(), task, renderer, "device")
        renderer.render.assert_not_called()

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
