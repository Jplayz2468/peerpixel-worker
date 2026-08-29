"""Build a source-grouped, reproducible bootstrap SFT split."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


VARIANTS = {"terse", "casual", "noisy"}


def _canonical(record: dict) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate(record: dict) -> None:
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError("each record needs exactly two messages")
    if [item.get("role") for item in messages] != ["user", "assistant"]:
        raise ValueError("message roles must be user then assistant")
    if any(not isinstance(item.get("content"), str) or not item["content"].strip()
           for item in messages):
        raise ValueError("message content cannot be empty")
    if not isinstance(record.get("source_file"), str) or not record["source_file"]:
        raise ValueError("source_file is required")
    if record.get("variant") not in VARIANTS:
        raise ValueError("variant must be terse, casual, or noisy")


def build_split(source: Path, output: Path, validation_fraction: float = 0.10,
                seed: int = 2468) -> dict:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    grouped = defaultdict(list)
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON on line {line_number}") from error
        _validate(record)
        grouped[record["source_file"]].append(record)
    if not grouped:
        raise ValueError("dataset is empty")
    for name, records in grouped.items():
        if len(records) != 3 or {record["variant"] for record in records} != VARIANTS:
            raise ValueError(f"source {name} must contain exactly three distinct variants")
        targets = {record["messages"][1]["content"] for record in records}
        if len(targets) != 1:
            raise ValueError(f"source {name} has inconsistent assistant targets")

    names = sorted(grouped)
    random.Random(seed).shuffle(names)
    validation_count = max(1, round(len(names) * validation_fraction))
    validation_names = set(names[:validation_count])
    splits = {
        "train": [record for name in sorted(set(names) - validation_names)
                  for record in sorted(grouped[name], key=lambda item: item["variant"])],
        "validation": [record for name in sorted(validation_names)
                       for record in sorted(grouped[name], key=lambda item: item["variant"])],
    }
    output.mkdir(parents=True, exist_ok=True)
    for split, records in splits.items():
        (output / f"{split}.jsonl").write_text(
            "".join(_canonical(record) for record in records), encoding="utf-8")
    manifest = {
        "schemaVersion": 1,
        "seed": seed,
        "validationFraction": validation_fraction,
        "records": {"all": sum(map(len, splits.values())),
                    "train": len(splits["train"]), "validation": len(splits["validation"])},
        "sources": {"all": len(names), "train": len(names) - validation_count,
                    "validation": validation_count},
        "digests": {split: _digest(output / f"{split}.jsonl") for split in splits},
    }
    (output / "dataset-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=2468)
    args = parser.parse_args()
    print(json.dumps(build_split(args.source, args.output, args.validation_fraction, args.seed),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
