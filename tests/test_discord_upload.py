import unittest
from unittest import mock

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
