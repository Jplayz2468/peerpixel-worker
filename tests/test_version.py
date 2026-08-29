import tomllib
import unittest
from pathlib import Path

from peerpixel import api
from peerpixel.version import RUNTIME_VERSION, VERSION


class VersionTests(unittest.TestCase):
    def test_every_public_version_uses_the_release_version(self):
        project = tomllib.loads(
            (Path(__file__).parents[1] / "pyproject.toml").read_text())
        self.assertEqual(project["project"]["version"], VERSION)
        self.assertEqual(RUNTIME_VERSION, f"peerpixel-worker/{VERSION}")
        self.assertTrue(api.USER_AGENT.startswith(f"{RUNTIME_VERSION} "))


if __name__ == "__main__":
    unittest.main()
