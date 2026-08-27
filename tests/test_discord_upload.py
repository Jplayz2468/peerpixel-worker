import unittest
import io
from unittest import mock

from PIL import Image

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

    def test_four_cells_are_composed_into_one_two_by_two_grid(self):
        cells = []
        for color in ("red", "green", "blue", "yellow"):
            output = io.BytesIO()
            Image.new("RGB", (32, 24), color).save(output, "JPEG")
            cells.append(output.getvalue())

        grid = worker.compose_grid(cells)

        image = Image.open(io.BytesIO(grid))
        self.assertEqual(image.size, (64, 48))

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
