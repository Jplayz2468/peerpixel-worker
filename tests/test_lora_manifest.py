import json
import tempfile
import unittest
from pathlib import Path

from peerpixel.lora_manifest import load_manifest, safe_version, write_manifest


class LoraManifestTests(unittest.TestCase):
    def values(self):
        return {
            "schemaVersion": 1,
            "version": "bootstrap-0001",
            "kind": "bootstrap",
            "baseModel": "Qwen/Qwen3-1.7B",
            "parentVersion": None,
            "dataset": {"trainDigest": "a" * 64, "validationDigest": "b" * 64,
                        "trainRecords": 1167, "validationRecords": 129},
            "training": {"rank": 16, "alpha": 32, "dropout": 0.05,
                         "learningRate": 5e-5, "epochs": 1, "seed": 2468},
            "evaluation": {"status": "pending"},
            "createdAt": "2026-08-27T00:00:00Z",
        }

    def test_manifest_round_trips_and_binds_artifact_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "adapter_model.safetensors").write_bytes(b"adapter")
            path = write_manifest(root, self.values())
            loaded = load_manifest(path)
            self.assertEqual(loaded["version"], "bootstrap-0001")
            self.assertEqual(loaded["artifactFiles"], {"adapter_model.safetensors":
                "ae1eae1d76e5b7c865c4122ce366a08025842566d2d96c75cc13e6353a73db0d"})

    def test_missing_required_values_and_tampering_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "adapter_model.safetensors").write_bytes(b"adapter")
            values = self.values()
            del values["baseModel"]
            with self.assertRaisesRegex(ValueError, "baseModel"):
                write_manifest(root, values)
            values = self.values()
            path = write_manifest(root, values)
            (root / "adapter_model.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "digest"):
                load_manifest(path)

    def test_versions_cannot_escape_the_run_directory(self):
        self.assertEqual(safe_version("bootstrap-0001"), "bootstrap-0001")
        for value in ("../adapter", "a/b", "", "."):
            with self.subTest(value=value), self.assertRaises(ValueError):
                safe_version(value)


if __name__ == "__main__":
    unittest.main()
