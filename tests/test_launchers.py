"""The launchers, checked as text because they cannot be imported.

These exist because of a real failure: the old updater died with "uv: command
not found" on a machine where uv was installed perfectly well, just not on the
PATH of a shell that had been open since before setup ran. The bootstrap
scripts are the only code in PeerPixel that runs before anything is installed,
so they are also the only code that cannot be covered any other way.
"""
from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
SHELL = "launch/bootstrap.sh"
POWERSHELL = "launch/bootstrap.ps1"
CLICKABLE = ("PeerPixel.command", "PeerPixel.sh", "PeerPixel.cmd", "PeerPixel.desktop")


def read(name):
    return (ROOT / name).read_text()


class ClickableTests(unittest.TestCase):
    """One file to double-click, on each of the three platforms."""

    def test_every_platform_has_something_to_click(self):
        for name in CLICKABLE:
            with self.subTest(file=name):
                self.assertTrue((ROOT / name).is_file(), f"{name} is missing")

    def test_the_unix_ones_are_executable_in_the_zip(self):
        # A .command without the executable bit does not open when you
        # double-click it; it opens a text editor.
        for name in ("PeerPixel.command", "PeerPixel.sh", SHELL):
            with self.subTest(file=name):
                self.assertTrue((ROOT / name).stat().st_mode & 0o111,
                                f"{name} is not executable")

    def test_each_one_only_delegates(self):
        """The logic lives in one place, so the three cannot drift apart."""
        for name in ("PeerPixel.command", "PeerPixel.sh"):
            self.assertIn("launch/bootstrap.sh", read(name))
        self.assertIn(r"launch\bootstrap.ps1", read("PeerPixel.cmd"))

    def test_the_windows_one_survives_a_double_click_that_fails(self):
        # Without the pause the console window vanishes with the error in it.
        self.assertIn("pause", read("PeerPixel.cmd"))


class UvResolutionTests(unittest.TestCase):
    def test_neither_bootstrap_calls_a_bare_uv(self):
        for line in read(SHELL).splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or "install.sh" in stripped:
                continue
            self.assertNotRegex(stripped, r"^uv\s",
                                f"bootstrap.sh calls uv without resolving it: {stripped}")
        for line in read(POWERSHELL).splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or "install.ps1" in stripped:
                continue
            self.assertNotRegex(stripped, r"^uv\s",
                                f"bootstrap.ps1 calls uv without resolving it: {stripped}")

    def test_both_search_where_uv_actually_installs_itself(self):
        self.assertIn(".local/bin/uv", read(SHELL))
        self.assertIn(r".local\bin\uv.exe", read(POWERSHELL))

    def test_both_honour_an_explicit_install_dir(self):
        for name in (SHELL, POWERSHELL):
            self.assertIn("UV_INSTALL_DIR", read(name))

    def test_both_install_uv_when_it_is_genuinely_missing(self):
        self.assertIn("astral.sh/uv/install.sh", read(SHELL))
        self.assertIn("astral.sh/uv/install.ps1", read(POWERSHELL))

    def test_giving_up_says_something_useful(self):
        for name in (SHELL, POWERSHELL):
            with self.subTest(script=name):
                self.assertIn("docs.astral.sh/uv", read(name))


class HandoverTests(unittest.TestCase):
    def test_the_app_runs_on_a_bare_interpreter(self):
        """--no-project, or the window could not exist until torch did.

        The whole reason the dependency install has a progress bar is that the
        thing drawing it does not depend on what it is installing.
        """
        for name in (SHELL, POWERSHELL):
            with self.subTest(script=name):
                self.assertIn("--no-project", read(name))
                self.assertIn("-m peerpixel", read(name))

    def test_the_launcher_hands_over_to_the_cli_and_nothing_else(self):
        # There is no window and no localhost server any more. A launcher that
        # still tried to open one would fail in a console somebody is watching.
        for name in (SHELL, POWERSHELL):
            with self.subTest(script=name):
                source = read(name)
                self.assertNotIn("pywebview", source)
                self.assertNotIn("webview", source)

    def test_the_resolved_uv_is_handed_to_the_app(self):
        # The app installs dependencies and updates itself, and both need uv.
        # Making it search again from a different environment is how the old
        # "uv: command not found" bug would come back.
        self.assertIn("PEERPIXEL_UV", read(SHELL))
        self.assertIn("PEERPIXEL_UV", read(POWERSHELL))


class BarTests(unittest.TestCase):
    """The bootstrap draws a bar too. It is the wait most likely to look hung."""

    def test_the_shell_bar_obeys_the_same_curve_as_the_app(self):
        source = read(SHELL)
        self.assertIn("0.9 * t / e", source, "constant speed up to the estimate")
        self.assertIn("0.995", source, "and a ceiling it never reaches")

    def test_the_shell_bar_actually_moves_and_never_arrives(self):
        """Run the awk that draws it, rather than trusting the source reads right."""
        formula = ('BEGIN{ if (t <= e) { p = 0.9 * t / e } '
                   'else { o = (t - e) / e; p = 0.995 - 0.095 / (1 + o) } printf "%.5f", p }')
        def at(t, e=10):
            out = subprocess.run(["awk", "-v", f"t={t}", "-v", f"e={e}", formula],
                                 capture_output=True, text=True, check=True)
            return float(out.stdout)

        self.assertAlmostEqual(at(10), 0.9, places=4)
        for earlier, later in ((1, 2), (5, 6), (11, 30), (30, 300), (300, 3000)):
            self.assertGreater(at(later), at(earlier),
                               f"the bar stopped between {earlier}s and {later}s")
        self.assertLess(at(100000), 0.995)

    def test_powershell_uses_the_same_numbers(self):
        source = read(POWERSHELL)
        self.assertIn("0.9 * $t / $Estimate", source)
        self.assertIn("0.995", source)
        self.assertRegex(source, r"Write-Progress")


class OldNamesTests(unittest.TestCase):
    """setup.sh and update.sh are in a README somewhere and in muscle memory."""

    def test_they_still_work_and_only_delegate(self):
        for name in ("setup.sh", "update.sh"):
            with self.subTest(script=name):
                source = read(name)
                self.assertIn("launch/bootstrap.sh", source)
                self.assertTrue((ROOT / name).stat().st_mode & 0o111)

    def test_update_asks_for_the_update_rather_than_the_window(self):
        self.assertRegex(read("update.sh"), r"PEERPIXEL_COMMAND=update")


if __name__ == "__main__":
    unittest.main()


class NoWindowTests(unittest.TestCase):
    """PeerPixel is a terminal program. Nothing should suggest otherwise."""

    def test_nothing_starts_a_browser_or_binds_a_port(self):
        for path in (ROOT / "peerpixel").glob("*.py"):
            source = path.read_text()
            with self.subTest(module=path.name):
                self.assertNotIn("webbrowser", source)
                self.assertNotIn("http.server", source)
                self.assertNotIn("ThreadingHTTPServer", source)

    def test_the_terminal_stays_open_for_the_person_using_it(self):
        # A launcher that closed its console would take the program with it.
        self.assertIn("Terminal=true", read("PeerPixel.desktop"))
