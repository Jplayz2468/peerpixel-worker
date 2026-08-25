import unittest
from pathlib import Path
from unittest import mock

from peerpixel import config, settings, worker


class ContentPrivacyTests(unittest.TestCase):
    def setUp(self):
        self.saved = config.read()
        config.write(allowPrivate=None, allowPrivateSyncedAt=None, deviceId="device-1")

    def tearDown(self):
        config.FILE.write_text(__import__("json").dumps(self.saved, indent=2))

    def test_private_generation_defaults_off(self):
        self.assertFalse(settings.allow_private())
        self.assertEqual(dict((s.name, value) for s, value, _ in settings.current())["private"], "off")

    def test_private_setting_is_synchronized_with_the_device(self):
        with mock.patch.object(settings.api, "set_private") as sent:
            settings.put("private", "on")
        sent.assert_called_once_with("device-1", True)
        self.assertTrue(settings.allow_private())

    def test_normal_worker_presentation_hides_prompts_and_private_previews(self):
        source = Path(worker.__file__).read_text()
        self.assertNotIn("heading = f\"{operation}  {DIM}{job['prompt']", source)
        self.assertNotIn('bar.begin("render", detail=job["prompt"]', source)
        self.assertNotIn("preview.save", source)


if __name__ == "__main__":
    unittest.main()
