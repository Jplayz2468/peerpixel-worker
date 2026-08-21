"""Transient AuraSR-v2 4x export. The returned bytes are never cached."""
from __future__ import annotations

import io
from PIL import Image


class Upscaler:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.model = None

    def warm(self):
        if self.model is not None:
            return
        from transformers import AutoModel
        from . import model_cache

        path = self.model_path or model_cache.ensure_directory("aurasr-v2")
        self.model = AutoModel.from_pretrained(
            path, local_files_only=True, trust_remote_code=True,
            dtype="auto", device_map="auto",
        )

    def upscale(self, jpeg: bytes) -> bytes:
        self.warm()
        source = Image.open(io.BytesIO(jpeg)).convert("RGB")
        image = self.model.upscale_4x_overlapped(source)
        output = io.BytesIO()
        image.save(output, "JPEG", quality=95, optimize=True)
        return output.getvalue()

    def unload(self):
        self.model = None
