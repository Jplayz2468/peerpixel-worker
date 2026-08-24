"""Updating, which means running code fetched from the internet.

So the parts worth testing are the ones that decide whether to, and the one
that unpacks somebody else's zip onto this disk.
"""
from __future__ import annotations

import re
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from peerpixel import updater


class VersionTests(unittest.TestCase):
    def test_tags_and_versions_sort_against_each_other(self):
        self.assertTrue(updater.newer("v0.2.0", "0.1.0"))
        self.assertTrue(updater.newer("0.1.1", "0.1.0"))
        self.assertFalse(updater.newer("0.1.0", "0.1.0"))
        self.assertFalse(updater.newer("v0.1", "0.1.0"))

    def test_a_release_that_is_older_is_never_offered(self):
        self.assertFalse(updater.newer("0.0.9", "0.1.0"))

    def test_nothing_is_not_newer_than_something(self):
        self.assertFalse(updater.newer("", "0.1.0"))


class InstalledVersionTests(unittest.TestCase):
    def test_the_version_comes_from_the_file_an_update_replaces(self):
        """pyproject, not the venv's package metadata.

        An update swaps the files in this folder; the metadata in .venv only
        changes if the project is reinstalled, which uv sync has no reason to
        do when the dependencies have not moved. Reading the stale one had a
        0.6.0 checkout reporting 0.1.0 -- so every worker would install an
        update it already had, once, on every clean start.
        """
        from peerpixel import updater as module

        pyproject = (module.ROOT / "pyproject.toml").read_text()
        declared = re.search(r'(?m)^version\s*=\s*"([^"]+)"', pyproject).group(1)
        self.assertEqual(module.installed(), declared)

    def test_this_checkout_does_not_think_it_needs_its_own_release(self):
        # The loop guard would catch it after one cycle; not doing it at all is
        # better than being rescued from it.
        self.assertFalse(updater.newer(f"v{updater.installed()}", updater.installed()))


class AssetTests(unittest.TestCase):
    def test_a_published_zip_wins_over_the_automatic_source_one(self):
        self.assertEqual(updater.asset_url({
            "assets": [{"name": "peerpixel-worker.zip",
                        "browser_download_url": "https://example/built.zip"}],
            "zipball_url": "https://example/source.zip",
        }), "https://example/built.zip")

    def test_the_source_zip_is_the_fallback(self):
        # Which is what makes an update work from the very first tag, before
        # anybody has got around to attaching a built asset to a release.
        self.assertEqual(updater.asset_url({"zipball_url": "https://example/source.zip"}),
                         "https://example/source.zip")

    def test_a_release_with_nothing_to_download_says_so(self):
        self.assertEqual(updater.asset_url({}), "")


class UnpackTests(unittest.TestCase):
    def zipped(self, members, root):
        archive = Path(root) / "update.zip"
        with zipfile.ZipFile(archive, "w") as out:
            for name, body in members.items():
                out.writestr(name, body)
        return archive

    def test_a_github_source_zip_unwraps_to_the_folder_that_holds_the_worker(self):
        with tempfile.TemporaryDirectory() as root:
            archive = self.zipped({
                "Jplayz2468-peerpixel-worker-abc123/pyproject.toml": 'version = "0.2.0"',
                "Jplayz2468-peerpixel-worker-abc123/peerpixel/__init__.py": "",
            }, root)
            found = updater.unpack(archive, Path(root) / "tree")
            self.assertTrue((found / "pyproject.toml").is_file())

    def test_a_zip_that_tries_to_write_outside_the_folder_is_refused(self):
        # Remote input, and "../" in a member name is the oldest trick there
        # is. Here it would be writing into somebody's home directory.
        with tempfile.TemporaryDirectory() as root:
            archive = self.zipped({"../escaped.txt": "no"}, root)
            with self.assertRaises(ValueError):
                updater.unpack(archive, Path(root) / "tree")

    def test_something_that_is_not_a_worker_is_refused(self):
        with tempfile.TemporaryDirectory() as root:
            archive = self.zipped({"readme.txt": "hello"}, root)
            with self.assertRaises(ValueError):
                updater.unpack(archive, Path(root) / "tree")

    def test_unpacking_reports_progress_per_member(self):
        with tempfile.TemporaryDirectory() as root:
            archive = self.zipped({"pyproject.toml": "x", "a.py": "", "b.py": ""}, root)
            seen = []
            updater.unpack(archive, Path(root) / "tree",
                           on_progress=lambda done, total: seen.append((done, total)))
            self.assertEqual(seen[-1], (3, 3))


class SwapTests(unittest.TestCase):
    def test_the_environment_and_local_git_survive_an_update(self):
        # Re-syncing costs gigabytes and minutes, and the Git metadata belongs
        # to whoever cloned this rather than to the release.
        for name in (".venv", ".git", "uv.lock"):
            self.assertIn(name, updater.KEEP)

    def test_swap_replaces_files_and_leaves_the_kept_ones_alone(self):
        with tempfile.TemporaryDirectory() as root:
            source, target = Path(root) / "new", Path(root) / "here"
            (source / "peerpixel").mkdir(parents=True)
            (source / "peerpixel" / "render.py").write_text("new")
            (source / ".venv").mkdir()
            (source / ".venv" / "junk").write_text("should not travel")
            (target / "peerpixel").mkdir(parents=True)
            (target / "peerpixel" / "render.py").write_text("old")
            (target / ".venv").mkdir()
            (target / ".venv" / "torch").write_text("expensive")

            updater.swap(source, target)

            self.assertEqual((target / "peerpixel" / "render.py").read_text(), "new")
            self.assertEqual((target / ".venv" / "torch").read_text(), "expensive")
            self.assertFalse((target / ".venv" / "junk").exists())


class CloneUpdateTests(unittest.TestCase):
    def test_a_clone_on_an_old_feature_branch_fast_forwards_to_origin_main(self):
        class Bar:
            def begin(self, *_args, **_kwargs):
                pass

        def git(*args, cwd):
            subprocess.run(["git", *args], cwd=cwd, check=True,
                           capture_output=True, text=True)

        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            remote, source, clone = root / "remote.git", root / "source", root / "clone"
            git("init", "--bare", "--initial-branch=main", str(remote), cwd=root)
            git("init", "--initial-branch=main", str(source), cwd=root)
            git("config", "user.email", "test@peerpixel.invalid", cwd=source)
            git("config", "user.name", "PeerPixel Test", cwd=source)
            (source / "pyproject.toml").write_text('version = "0.8.7"\n')
            git("add", "pyproject.toml", cwd=source)
            git("commit", "-m", "old", cwd=source)
            git("remote", "add", "origin", str(remote), cwd=source)
            git("push", "-u", "origin", "main", cwd=source)
            git("branch", "codex/old", cwd=source)
            git("push", "-u", "origin", "codex/old", cwd=source)
            git("clone", "--branch", "codex/old", str(remote), str(clone), cwd=root)
            (source / "pyproject.toml").write_text('version = "0.8.8"\n')
            git("commit", "-am", "new", cwd=source)
            git("push", "origin", "main", cwd=source)

            with patch.object(updater, "ROOT", clone), patch.object(updater, "_sync"):
                result = updater._git_update(Bar(), "0.8.7")

            self.assertTrue(result["updated"])
            self.assertEqual((clone / "pyproject.toml").read_text(), 'version = "0.8.8"\n')


if __name__ == "__main__":
    unittest.main()
