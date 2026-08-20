import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from peerpixel import dashboard_state


class DashboardStateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.home = Path(self.temp.name)
        self.home_patch = patch("peerpixel.dashboard_state.config.HOME", self.home)
        self.home_patch.start()

    def tearDown(self):
        self.home_patch.stop()
        self.temp.cleanup()

    def test_publish_merges_patches_and_adds_timestamp(self):
        first = dashboard_state.publish({"phase": "connecting", "images": 1})
        second = dashboard_state.publish({"phase": "online"})

        self.assertIsInstance(first["updatedAt"], int)
        self.assertEqual(second["phase"], "online")
        self.assertEqual(second["images"], 1)
        self.assertEqual(dashboard_state.read(), second)

    def test_missing_or_malformed_state_reads_empty(self):
        self.assertEqual(dashboard_state.read(), {})
        dashboard_state.state_path().parent.mkdir(parents=True, exist_ok=True)
        dashboard_state.state_path().write_text("not json")
        self.assertEqual(dashboard_state.read(), {})

    def test_concurrent_publish_never_loses_a_patch(self):
        threads = [
            threading.Thread(target=dashboard_state.publish, args=({f"key{i}": i},))
            for i in range(20)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        state = dashboard_state.read()
        for i in range(20):
            self.assertEqual(state[f"key{i}"], i)

    def test_state_replacement_leaves_no_temporary_file(self):
        dashboard_state.publish({"phase": "online"})
        self.assertEqual(json.loads(dashboard_state.state_path().read_text())["phase"], "online")
        self.assertEqual(list(self.home.glob("*.tmp")), [])

    def test_preview_replacement_returns_final_path(self):
        final = dashboard_state.save_preview(b"jpeg bytes")
        self.assertEqual(final, dashboard_state.preview_path())
        self.assertEqual(final.read_bytes(), b"jpeg bytes")
        self.assertEqual(list(self.home.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
