from __future__ import annotations

import unittest
from unittest import mock

from peerpixel import model_hub


class ModelHubTests(unittest.TestCase):
    def test_every_auxiliary_model_is_revision_pinned_on_hugging_face(self):
        self.assertEqual(set(model_hub.MODELS), {"qwen3-1.7b", "nsfw-image-detection", "aurasr-v2"})
        for repo, revision in model_hub.MODELS.values():
            self.assertIn("/", repo)
            self.assertRegex(revision, r"^[0-9a-f]{40}$")

    def test_cached_snapshot_is_used_without_network_access(self):
        with mock.patch("huggingface_hub.snapshot_download", return_value="/hf/cache/model") as download:
            self.assertEqual(model_hub.ensure("qwen3-1.7b"), "/hf/cache/model")
        download.assert_called_once_with(
            repo_id="Qwen/Qwen3-1.7B",
            revision=model_hub.MODELS["qwen3-1.7b"][1],
            local_files_only=True,
        )

    def test_cache_miss_downloads_the_same_pinned_snapshot_resumably(self):
        from huggingface_hub.errors import LocalEntryNotFoundError

        with mock.patch("huggingface_hub.snapshot_download", side_effect=[
            LocalEntryNotFoundError("not cached"), "/hf/cache/downloaded",
        ]) as download:
            self.assertEqual(model_hub.ensure("aurasr-v2"), "/hf/cache/downloaded")
        self.assertEqual(download.call_args_list[1], mock.call(
            repo_id="fal/AuraSR-v2",
            revision=model_hub.MODELS["aurasr-v2"][1],
            max_workers=4,
        ))

    def test_unknown_model_fails_before_contacting_hugging_face(self):
        with mock.patch("huggingface_hub.snapshot_download") as download, \
             self.assertRaisesRegex(ValueError, "unknown_hf_model"):
            model_hub.ensure("mystery")
        download.assert_not_called()
