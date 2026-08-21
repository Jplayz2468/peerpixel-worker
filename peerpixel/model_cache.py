"""Authenticated, hash-pinned on-demand downloads from PeerPixel's private R2 registry."""
from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
import tarfile
from pathlib import Path

from . import api, config

MANIFEST_VERSION = "2026-08-21.1"


def root() -> Path:
    return config.HOME / "models"


def _request(path: str, *, headers: dict | None = None):
    token = config.read().get("token", "")
    return urllib.request.Request(
        f"{config.API}{path}",
        headers={"user-agent": api.USER_AGENT, "authorization": f"Bearer {token}", **(headers or {})},
    )


def fetch_manifest() -> dict:
    with urllib.request.urlopen(_request("/api/device/models/manifest"), timeout=60) as response:
        manifest = json.load(response)
    if manifest.get("version") != MANIFEST_VERSION or not isinstance(manifest.get("artifacts"), list):
        raise RuntimeError("unsupported_model_manifest")
    return manifest


def artifact(manifest: dict, name: str) -> dict:
    found = next((item for item in manifest["artifacts"] if item.get("name") == name), None)
    if not found or not isinstance(found.get("size"), int) or found["size"] <= 0:
        raise RuntimeError(f"unavailable_model:{name}")
    digest = found.get("sha256", "")
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise RuntimeError(f"unpinned_model:{name}")
    return found


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure(name: str, manifest: dict | None = None) -> Path:
    """Return the exact cached artifact, resuming an interrupted download safely."""
    manifest = manifest or fetch_manifest()
    item = artifact(manifest, name)
    folder = root() / manifest["version"]
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / Path(item["key"]).name
    if target.is_file() and target.stat().st_size == item["size"] and sha256(target) == item["sha256"]:
        return target
    target.unlink(missing_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > item["size"]:
        partial.unlink()
        offset = 0
    headers = {"range": f"bytes={offset}-"} if offset else {}
    try:
        with urllib.request.urlopen(
            _request(f"/api/device/models/{name}", headers=headers), timeout=300,
        ) as response, partial.open("ab" if offset else "wb") as output:
            if offset and response.status != 206:
                output.seek(0)
                output.truncate()
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"model_download_failed:{name}:{error.code}") from None
    if partial.stat().st_size != item["size"] or sha256(partial) != item["sha256"]:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"model_integrity_failed:{name}")
    os.replace(partial, target)
    return target


def ensure_directory(name: str, manifest: dict | None = None) -> Path:
    """Materialize a pinned tar.zst snapshot once, without path traversal."""
    manifest = manifest or fetch_manifest()
    archive = ensure(name, manifest)
    destination = archive.with_suffix("").with_suffix("")
    marker = destination / ".peerpixel-sha256"
    expected = artifact(manifest, name)["sha256"]
    if marker.is_file() and marker.read_text() == expected:
        return destination
    import shutil
    import zstandard

    shutil.rmtree(destination, ignore_errors=True)
    staging = destination.with_name(destination.name + ".extracting")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    with archive.open("rb") as compressed, zstandard.ZstdDecompressor().stream_reader(compressed) as stream:
        with tarfile.open(fileobj=stream, mode="r|") as bundle:
            for member in bundle:
                resolved = (staging / member.name).resolve()
                if staging.resolve() not in resolved.parents and resolved != staging.resolve():
                    raise RuntimeError(f"unsafe_model_archive:{name}")
                bundle.extract(member, staging, filter="data")
    marker = staging / ".peerpixel-sha256"
    marker.write_text(expected)
    os.replace(staging, destination)
    return destination
