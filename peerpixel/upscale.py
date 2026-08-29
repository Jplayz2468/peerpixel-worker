"""Optional AuraSR-v2 inference behind a small, testable boundary."""
from __future__ import annotations

import hashlib
import importlib.util
import io
from dataclasses import dataclass
from typing import Callable

from PIL import Image


@dataclass(frozen=True)
class UpscaleSupport:
    available: bool
    reason: str
    estimate_ms: int | None


def probe_upscale_support() -> UpscaleSupport:
    if importlib.util.find_spec("aurasr") is None:
        return UpscaleSupport(False, "AuraSR runtime is not installed", None)
    try:
        import torch
        if not torch.cuda.is_available():
            return UpscaleSupport(False, "AuraSR-v2 requires CUDA", None)
    except ImportError:
        return UpscaleSupport(False, "PyTorch is not installed", None)
    return UpscaleSupport(True, "ready", 120_000)


@dataclass(frozen=True)
class UpscaleJob:
    width: int
    height: int
    source_width: int
    source_height: int
    source_digest: str = ""
    model: str = "aurasr-v2"

    @classmethod
    def from_payload(cls, payload: dict) -> "UpscaleJob":
        if payload.get("operation") != "upscale" or payload.get("model") != "aurasr-v2":
            raise ValueError("unsupported upscale model")
        width, height = int(payload.get("width", 0)), int(payload.get("height", 0))
        source_width = int(payload.get("sourceWidth", width // 4))
        source_height = int(payload.get("sourceHeight", height // 4))
        if min(width, height, source_width, source_height) <= 0 \
                or width != source_width * 4 or height != source_height * 4 \
                or max(width, height) > 2048:
            raise ValueError("invalid upscale dimensions")
        digest = str(payload.get("sourceDigest") or "")
        if digest and (len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower())):
            raise ValueError("invalid source digest")
        return cls(width, height, source_width, source_height, digest.lower())


class _AuraBackend:
    def __init__(self):
        from aurasr import AuraSR
        self.model = AuraSR.from_pretrained("fal-ai/AuraSR-v2")

    def upscale(self, image: Image.Image, on_tile=None) -> Image.Image:
        if on_tile:
            on_tile(0, 1)
        result = self.model.upscale_4x_overlapped(image)
        if on_tile:
            on_tile(1, 1)
        return result

    def unload(self) -> None:
        self.model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


class Upscaler:
    def __init__(self, backend_factory: Callable = _AuraBackend):
        self.backend_factory = backend_factory

    def run(self, job: UpscaleJob, source: bytes, on_tile=None) -> bytes:
        if job.source_digest and hashlib.sha256(source).hexdigest() != job.source_digest:
            raise ValueError("source digest mismatch")
        image = Image.open(io.BytesIO(source))
        image.load()
        image = image.convert("RGB")
        if image.size != (job.source_width, job.source_height):
            raise ValueError("source dimensions mismatch")
        backend = self.backend_factory()
        try:
            result = backend.upscale(image, on_tile=on_tile)
            if result.size != (job.width, job.height):
                raise ValueError("invalid upscale result dimensions")
            output = io.BytesIO()
            result.convert("RGB").save(output, "JPEG", quality=95, subsampling=0, optimize=True)
            return output.getvalue()
        finally:
            backend.unload()
