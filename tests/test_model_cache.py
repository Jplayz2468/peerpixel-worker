from __future__ import annotations

import hashlib
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from peerpixel import model_cache


class FakeResponse:
    status = 200
    def __init__(self, payload): self.payload = payload
    def __enter__(self): return self
    def __exit__(self, *_): return None
    def read(self, _size=-1):
        payload, self.payload = self.payload, b""
        return payload


class ModelCacheTests(unittest.TestCase):
    def test_download_is_hash_checked_and_atomically_named(self):
        payload = b"pinned model bytes"
        manifest = {"version": model_cache.MANIFEST_VERSION, "artifacts": [{
            "name": "qwen", "key": "models/m1/qwen.bin", "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }]}
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(model_cache, "root", return_value=Path(directory)), \
             mock.patch("urllib.request.urlopen", return_value=FakeResponse(payload)):
            path = model_cache.ensure("qwen", manifest)
            self.assertEqual(path.read_bytes(), payload)
            self.assertFalse(path.with_suffix(".bin.part").exists())

    def test_bad_hash_is_deleted_and_never_accepted(self):
        manifest = {"version": model_cache.MANIFEST_VERSION, "artifacts": [{
            "name": "qwen", "key": "models/m1/qwen.bin", "size": 3, "sha256": "0" * 64,
        }]}
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(model_cache, "root", return_value=Path(directory)), \
             mock.patch("urllib.request.urlopen", return_value=FakeResponse(b"bad")):
            with self.assertRaisesRegex(RuntimeError, "model_integrity_failed"):
                model_cache.ensure("qwen", manifest)
            self.assertEqual(list(Path(directory).rglob("*part")), [])

    def test_two_callers_share_one_atomic_download(self):
        payload = b"one shared model archive"
        manifest = {"version": model_cache.MANIFEST_VERSION, "artifacts": [{
            "name": "qwen", "key": "models/m1/qwen.tar.zst", "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }]}
        calls = []

        def download(*_args, **_kwargs):
            calls.append(True)
            time.sleep(0.05)
            return FakeResponse(payload)

        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(model_cache, "root", return_value=Path(directory)), \
             mock.patch("urllib.request.urlopen", side_effect=download):
            results, errors = [], []

            def ensure():
                try:
                    results.append(model_cache.ensure("qwen", manifest))
                except Exception as error:  # captured so both threads join
                    errors.append(error)

            threads = [threading.Thread(target=ensure) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            self.assertEqual(errors, [])
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0], results[1])
            self.assertEqual(len(calls), 1, "only the lock owner downloads")
            self.assertEqual(results[0].read_bytes(), payload)
