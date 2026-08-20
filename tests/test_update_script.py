from pathlib import Path
import unittest


class UpdateScriptTests(unittest.TestCase):
    def test_both_updaters_are_fast_forward_only_and_reopen_dashboard(self):
        root = Path(__file__).parents[1]
        shell = (root / "update.sh").read_text()
        powershell = (root / "update.ps1").read_text()
        for source in (shell, powershell):
            self.assertIn("git pull --ff-only", source)
            self.assertIn("uv sync --python 3.12", source)
            self.assertIn("uv run peerpixel dashboard", source)
            self.assertLess(source.index("git pull --ff-only"), source.index("uv sync --python 3.12"))
