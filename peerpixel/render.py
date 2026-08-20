"""Rendering.

FLUX.2 Klein run the way it was meant to run: diffusers on PyTorch, using
whatever accelerator this machine has. CUDA on most Windows and Linux boxes,
MPS on Apple silicon, CPU as a last resort.

One pipeline covers every job the network sends. Flux2KleinInpaintPipeline is
the one that exposes `strength`, and with a fully white mask that is ordinary
image to image, which is what variations and refines need. With a blank source
at full strength it is ordinary text to image.

This file is deliberately plain and short. If a render goes wrong, this is where
to look, and you can edit it and restart the worker without rebuilding anything.
"""
from __future__ import annotations

import base64
import io
import os
import subprocess
import time

MODEL = os.environ.get("PEERPIXEL_MODEL", "black-forest-labs/FLUX.2-klein-4B")


def pick_device():
    import torch

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        _, total = torch.cuda.mem_get_info()
        return "cuda", torch.bfloat16, f"{name} ({total / 1e9:.0f} GB)", total
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        try:
            total = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]))
        except Exception:  # noqa: BLE001
            total = 0
        return "mps", torch.bfloat16, f"Apple silicon ({total / 1e9:.0f} GB unified)", total
    return "cpu", torch.float32, "CPU", 0


def _reporter(on_step, asked: int):
    """Turn the diffusers step hook into (done, total) for the display.

    The total is not `asked`: with an init image the scheduler throws away the
    first (1 - strength) of the schedule, so a 24 step refine at 0.35 really
    runs 8. The pipeline knows the true count once it has built its timesteps,
    so take it from there and only fall back to what we asked for.
    """

    def hook(pipe, index, timestep, kwargs):
        on_step(index + 1, getattr(pipe, "_num_timesteps", 0) or asked)
        return kwargs

    return hook


class Renderer:
    def __init__(self):
        self.pipe = None
        self._device, _, self.accelerator, self._total = pick_device()

    def warm(self):
        """Load once and keep it. A 4B model takes tens of seconds to load."""
        if self.pipe is not None:
            return
        import torch  # noqa: F401
        from diffusers import Flux2KleinInpaintPipeline

        device, dtype, label, total = pick_device()
        self.accelerator = label
        print(f"loading {MODEL} on {label}...", flush=True)
        started = time.time()
        pipe = Flux2KleinInpaintPipeline.from_pretrained(MODEL, torch_dtype=dtype)

        # Under roughly 24 GB the transformer and the text encoder cannot both
        # sit on the accelerator. Handing them over a layer at a time is slower
        # but it is the difference between running and not running at all.
        if device == "cuda" and total and total < 24e9:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe
        print(f"ready in {time.time() - started:.0f}s", flush=True)

    def render(self, job: dict, on_step=None) -> bytes:
        import torch
        from PIL import Image

        self.warm()
        width = int(job.get("width", 512))
        height = int(job.get("height", 512))
        steps = int(job.get("steps", 4))

        if job.get("init"):
            # Variation and refine. `strength` is how much of the original to
            # throw away: ~0.55 varies noticeably, ~0.35 sharpens without
            # moving anything.
            source = Image.open(io.BytesIO(base64.b64decode(job["init"]))).convert("RGB")
            if source.size != (width, height):
                source = source.resize((width, height), Image.LANCZOS)
            strength = float(job.get("strength", 0.55))
        else:
            source = Image.new("RGB", (width, height), (0, 0, 0))
            strength = 1.0

        # Klein is step-distilled, so guidance_scale is ignored. Not passing it
        # keeps the logs honest.
        watch = {}
        if on_step is not None:
            watch["callback_on_step_end"] = _reporter(on_step, steps)

        image = self.pipe(
            prompt=job["prompt"],
            image=source,
            mask_image=Image.new("L", (width, height), 255),
            strength=strength,
            num_inference_steps=steps,
            height=height,
            width=width,
            generator=torch.Generator("cpu").manual_seed(int(job.get("seed", 0))),
            **watch,
        ).images[0]

        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=92)
        return buffer.getvalue()
