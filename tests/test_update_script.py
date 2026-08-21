"""The launcher scripts, checked as text because they cannot be imported.

These exist because of a real failure: update.sh died with "uv: command not
found" on a machine where uv was installed perfectly well, just not on the PATH
of a shell that had been open since before setup ran.
"""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).parents[1]


def read(name):
    return (ROOT / name).read_text()


class UpdateScriptTests(unittest.TestCase):
    def test_both_updaters_pull_fast_forward_then_sync_then_launch(self):
        for name in ("update.sh", "update.ps1"):
            source = read(name)
            with self.subTest(script=name):
                self.assertIn("git pull --ff-only", source)
                self.assertRegex(source, r"sync --python 3\.12")
                self.assertRegex(source, r"run peerpixel dashboard")
                self.assertLess(
                    source.index("git pull --ff-only"),
                    source.index("sync --python 3.12"),
                    "the pull has to happen before the sync it feeds",
                )


class UvResolutionTests(unittest.TestCase):
    """Every script must look for uv rather than assume it is on PATH."""

    SHELL_SCRIPTS = ("update.sh", "setup.sh")
    POWERSHELL_SCRIPTS = ("update.ps1", "setup.ps1")

    def test_no_script_calls_a_bare_uv(self):
        # A bare `uv ...` is the bug. Every real invocation goes through the
        # resolved path.
        for name in self.SHELL_SCRIPTS:
            with self.subTest(script=name):
                for line in read(name).splitlines():
                    stripped = line.strip()
                    if stripped.startswith("#") or "install.sh" in stripped:
                        continue
                    self.assertNotRegex(
                        stripped, r"^uv\s", f"{name} calls uv without resolving it: {stripped}")
        for name in self.POWERSHELL_SCRIPTS:
            with self.subTest(script=name):
                for line in read(name).splitlines():
                    stripped = line.strip()
                    if stripped.startswith("#") or "install.ps1" in stripped:
                        continue
                    self.assertNotRegex(
                        stripped, r"^uv\s", f"{name} calls uv without resolving it: {stripped}")

    def test_every_script_searches_the_place_uv_installs_itself(self):
        for name in self.SHELL_SCRIPTS:
            with self.subTest(script=name):
                self.assertIn(".local/bin/uv", read(name))
        for name in self.POWERSHELL_SCRIPTS:
            with self.subTest(script=name):
                self.assertIn(".local\\bin\\uv.exe", read(name))

    def test_every_script_honours_an_explicit_install_dir(self):
        for name in self.SHELL_SCRIPTS + self.POWERSHELL_SCRIPTS:
            with self.subTest(script=name):
                self.assertIn("UV_INSTALL_DIR", read(name))

    def test_every_script_installs_uv_when_it_is_genuinely_missing(self):
        for name in self.SHELL_SCRIPTS:
            with self.subTest(script=name):
                self.assertIn("astral.sh/uv/install.sh", read(name))
        for name in self.POWERSHELL_SCRIPTS:
            with self.subTest(script=name):
                self.assertIn("astral.sh/uv/install.ps1", read(name))

    def test_a_still_missing_uv_explains_itself_instead_of_dying_bare(self):
        for name in self.SHELL_SCRIPTS + self.POWERSHELL_SCRIPTS:
            with self.subTest(script=name):
                source = read(name)
                self.assertIn("docs.astral.sh/uv", source,
                              "the give-up path should point somewhere useful")
                self.assertIn("exit 1", source)


if __name__ == "__main__":
    unittest.main()
