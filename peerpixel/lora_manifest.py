"""Small immutable manifests for prompt-adapter artifacts."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


REQUIRED = ("schemaVersion", "version", "kind", "baseModel", "parentVersion",
            "dataset", "training", "evaluation", "createdAt")
VERSION = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


def safe_version(value: str) -> str:
    value = str(value or "")
    if not VERSION.fullmatch(value):
        raise ValueError("unsafe adapter version")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate(values: dict) -> None:
    for name in REQUIRED:
        if name not in values:
            raise ValueError(f"manifest is missing {name}")
    safe_version(values["version"])
    if values["kind"] not in {"bootstrap", "preference"}:
        raise ValueError("manifest kind must be bootstrap or preference")
    if not isinstance(values["dataset"], dict) or not isinstance(values["training"], dict):
        raise ValueError("dataset and training must be objects")


def write_manifest(adapter_dir: Path, values: dict) -> Path:
    adapter_dir = Path(adapter_dir)
    _validate(values)
    files = {}
    for path in sorted(adapter_dir.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            files[path.name] = _sha256(path)
    if not files:
        raise ValueError("adapter contains no artifact files")
    complete = {**values, "artifactFiles": files}
    path = adapter_dir / "manifest.json"
    path.write_text(json.dumps(complete, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_manifest(path: Path) -> dict:
    path = Path(path)
    values = json.loads(path.read_text(encoding="utf-8"))
    _validate(values)
    files = values.get("artifactFiles")
    if not isinstance(files, dict) or not files:
        raise ValueError("manifest is missing artifactFiles")
    for name, expected in files.items():
        artifact = path.parent / name
        if not artifact.is_file() or _sha256(artifact) != expected:
            raise ValueError(f"artifact digest mismatch: {name}")
    return values
