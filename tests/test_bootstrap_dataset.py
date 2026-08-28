import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_bootstrap_dataset import build_split


class BootstrapDatasetTests(unittest.TestCase):
    def source(self, root: Path, sources: int = 20) -> Path:
        path = root / "source.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for source in range(sources):
                for variant in ("terse", "casual", "noisy"):
                    record = {
                        "messages": [
                            {"role": "user", "content": f"raw {source} {variant}"},
                            {"role": "assistant", "content": f"enhanced {source}"},
                        ],
                        "source_file": f"{source:03}.jpg",
                        "variant": variant,
                    }
                    handle.write(json.dumps(record) + "\n")
        return path

    def test_split_is_grouped_deterministic_and_manifested(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = self.source(root)
            first = build_split(source, root / "first", 0.10, 2468)
            second = build_split(source, root / "second", 0.10, 2468)

            train = [json.loads(line) for line in (root / "first/train.jsonl").read_text().splitlines()]
            validation = [json.loads(line) for line in (root / "first/validation.jsonl").read_text().splitlines()]
            train_sources = {row["source_file"] for row in train}
            validation_sources = {row["source_file"] for row in validation}

            self.assertFalse(train_sources & validation_sources)
            self.assertEqual({3}, {sum(row["source_file"] == name for row in train + validation)
                                   for name in train_sources | validation_sources})
            self.assertEqual((root / "first/train.jsonl").read_bytes(),
                             (root / "second/train.jsonl").read_bytes())
            self.assertEqual(first, second)
            self.assertEqual(first["records"], {"all": 60, "train": 54, "validation": 6})
            self.assertEqual(first["sources"], {"all": 20, "train": 18, "validation": 2})
            self.assertEqual(first["digests"]["train"], hashlib.sha256(
                (root / "first/train.jsonl").read_bytes()).hexdigest())

    def test_malformed_or_incomplete_sources_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = self.source(root, sources=2)
            lines = source.read_text().splitlines()
            source.write_text("\n".join(lines[:-1]) + "\n")
            with self.assertRaisesRegex(ValueError, "exactly three"):
                build_split(source, root / "out", 0.10, 2468)


if __name__ == "__main__":
    unittest.main()
