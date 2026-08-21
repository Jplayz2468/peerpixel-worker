"""Terminal rendering details that protect the worker's live display."""
from __future__ import annotations

import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from peerpixel import console


class LiveStatusTests(unittest.TestCase):
    def test_rendering_bar_includes_the_live_system_status_footer(self):
        snapshot = SimpleNamespace(
            percent=50, fraction=0.5, label="Rendering", detail="step 2 of 4",
            failed=False, finished=False, eta_seconds=10,
        )
        tracker = SimpleNamespace(snapshot=lambda: snapshot)
        output = io.StringIO()

        with patch.object(console, "TTY", True), patch.object(console.sys, "stdout", output):
            live = console.Live(tracker, footer=lambda: "CPU 12% · VRAM 10/16 GB")
            live._paint()

        self.assertIn("CPU 12% · VRAM 10/16 GB", output.getvalue())
        self.assertEqual(live.lines, 4)


if __name__ == "__main__":
    unittest.main()
