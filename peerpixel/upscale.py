"""Transient AuraSR-v2 4x export. The returned bytes are never cached."""
from __future__ import annotations

import io
import json
from pathlib import Path
from PIL import Image


class Upscaler:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.model = None

    def warm(self):
        if self.model is not None:
            return
        import torch
        from aura_sr import AuraSR
        from safetensors.torch import load_file
        from . import model_hub

        path = Path(self.model_path) if self.model_path else Path(model_hub.ensure("aurasr-v2"))
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
        )
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        self.model = AuraSR(config, device=device)
        checkpoint = load_file(path / "model.safetensors", device="cpu")
        self.model.upsampler.load_state_dict(checkpoint, strict=True)

    def upscale(self, jpeg: bytes, on_phase=None, on_progress=None) -> bytes:
        self.warm()
        source = Image.open(io.BytesIO(jpeg)).convert("RGB")
        if on_phase is not None:
            on_phase("upscaling")
        if on_progress is not None:
            on_progress(0, 1)
        image = self.model.upscale_4x_overlapped(source)
        if on_progress is not None:
            on_progress(1, 1)
        if on_phase is not None:
            on_phase("encoding_export")
        output = io.BytesIO()
        image.save(output, "JPEG", quality=95, optimize=True)
        return output.getvalue()

    def unload(self):
        self.model = None
