"""Rendering.

FLUX.2 Klein run the way it was meant to run: diffusers on PyTorch, using
whatever accelerator this machine has. CUDA on most Windows and Linux boxes,
MPS on Apple silicon, CPU as a last resort.

This is the **base** checkpoint, not the step-distilled one, and that choice is
the reason everything here is slower than it used to be. Distilled Klein
renders in four steps and is several times faster, but it ignores guidance
completely -- the pipeline disables classifier-free guidance outright when the
checkpoint reports `is_distilled` -- and compared side by side on the same
prompts it drifted on spatial instructions and on negative ones, and turned
backgrounds into featureless blur. Fifty guided steps is worth the wait.

Guidance costs double on top of the step count: a guided step runs the
transformer twice, once for the prompt and once for an empty one, and mixes
them. Fifty steps is a hundred forward passes.

A **draft** is 128x128 from the prompt alone. It exists to answer whether the
composition is the one somebody wanted. Sixteen steps rather than four, because
a draft that is not a fair preview of the master is worse than no draft at all.
It goes straight back down the socket rather than being uploaded anywhere.

A **master** is 512x512 conditioned on the draft that was chosen. The draft
arrives over the socket, is upscaled to the output size with Lanczos, and is
handed to Klein as a reference image. It is deliberately not img2img: an
img2img `strength` throws away the first part of the schedule, so a conditioned
render would silently run fewer steps than it was paid for. Reference-image
conditioning keeps the whole schedule and still keeps the framing, palette and
pose of the draft.

This file is deliberately plain and short. If a render goes wrong, this is where
to look, and you can edit it and restart the worker without rebuilding anything.
"""
from __future__ import annotations

import io
import os
import subprocess
import time

MODEL = os.environ.get("PEERPIXEL_MODEL", "black-forest-labs/FLUX.2-klein-base-4B")

#: Pinned, because an unpinned repo means two machines that downloaded on
#: different days are running different weights. That is invisible until it is
#: not: the network re-renders a fraction of jobs on a machine the operator owns
#: and compares them, and honest machines disagreeing about weights looks
#: exactly like fraud. Override only if you know why you are doing it.
REVISION = os.environ.get("PEERPIXEL_MODEL_REVISION", "a3b4f4849157f664bdbc776fd7453c2783562f4d")
if os.environ.get("PEERPIXEL_MODEL") and not os.environ.get("PEERPIXEL_MODEL_REVISION"):
    # A custom model with the stock revision pin would fail to resolve.
    REVISION = None

#: Everything the network sends. The worker refuses anything else rather than
#: guessing, because guessing wrong means charging for the wrong picture. These
#: numbers must match public/generation-policy.mjs on the server.
GUIDANCE = 5.0
OPERATIONS = {
    "draft": {"width": 128, "height": 128, "steps": 16, "guidance": GUIDANCE},
    "master": {"width": 512, "height": 512, "steps": 50, "guidance": GUIDANCE},
    # A check is a master rendered a second time on a machine the operator
    # owns, so it is the same work with the same inputs. Only what happens to
    # the result differs: it is compared rather than delivered.
    "verify": {"width": 512, "height": 512, "steps": 50, "guidance": GUIDANCE},
}


def pick_device():
    """Which accelerator this machine has, and how much of it there is.

    Every probe here is guarded. Asking CUDA anything creates a context on the
    card, and that can fail outright when something else already holds the
    memory -- most often this machine's own worker, sitting on a 4B model while
    the dashboard asks what GPU it has. A card that is busy is still the card
    this machine renders on, so an unreadable probe becomes an unknown size
    rather than an exception, and nothing that merely wants to name the
    hardware is allowed to fall over because of it.

    An unknown size is treated as small later on, in `warm`, which is the safe
    direction to be wrong in.
    """
    import torch

    if _cuda_present(torch):
        name = _tried(lambda: torch.cuda.get_device_name(0), "NVIDIA GPU")
        # mem_get_info returns (free, total).
        total = _tried(lambda: torch.cuda.mem_get_info()[1], 0)
        label = f"{name} ({total / 1e9:.0f} GB)" if total else f"{name} (size unknown)"
        return "cuda", torch.bfloat16, label, total
    if _mps_present(torch):
        total = _tried(lambda: int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])), 0)
        return "mps", torch.bfloat16, f"Apple silicon ({total / 1e9:.0f} GB unified)", total
    return "cpu", torch.float32, "CPU", 0


def _tried(probe, fallback):
    """Run a hardware probe, or answer with the fallback.

    Deliberately broad. A driver can raise anything at all -- torch's own
    AcceleratorError, an OSError from a missing library, a RuntimeError from a
    context that would not initialise -- and none of them are worth more than a
    less precise answer about the hardware.
    """
    try:
        return probe()
    except Exception:  # noqa: BLE001 - see above
        return fallback


def _cuda_present(torch) -> bool:
    return bool(_tried(torch.cuda.is_available, False))


def _mps_present(torch) -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend) and bool(_tried(backend.is_available, False))


def describe_accelerator() -> str:
    """What to call this machine's hardware, for pairing and for the display.

    Never raises and never needs a model. Pairing a machine should not depend
    on the card being free, and a box whose GPU is busy rendering is exactly
    the box somebody is most likely to be pairing from.
    """
    try:
        return pick_device()[2]
    except Exception:  # noqa: BLE001 - torch itself may be missing or broken
        return "unknown"


def operation_of(job: dict) -> dict:
    """The size and step count for a job, from its operation and nothing else.

    A job may not talk the worker into a different resolution than the one it
    was priced at, so the numbers come from this table rather than from the
    payload. The payload's width/height are checked against it and a mismatch
    is refused.
    """
    name = job.get("operation", "master")
    spec = OPERATIONS.get(name)
    if spec is None:
        raise ValueError(f"unknown operation: {name}")
    for axis in ("width", "height"):
        asked = job.get(axis)
        if asked is not None and int(asked) != spec[axis]:
            raise ValueError(f"{name} is {spec[axis]}px, not {asked}px")
    return {"name": name, **spec}


def upscale_reference(image, size):
    """The chosen draft at the master's resolution.

    Lanczos, because the draft is being handed to the model as a description of
    a composition and a nearest-neighbour blow-up would hand it 128px of blocks
    to reproduce faithfully.
    """
    from PIL import Image

    reference = image if image.mode == "RGB" else image.convert("RGB")
    if reference.size == size:
        return reference
    return reference.resize(size, Image.LANCZOS)


def seeded_generator(seed: int):
    """The CPU generator every render is seeded from.

    A named seam rather than an inline import: it is the one line of `render`
    that needs torch, and pulling it out lets the operation contract above be
    tested on a machine with no accelerator and no model.
    """
    import torch

    return torch.Generator("cpu").manual_seed(int(seed))


def _reporter(on_step, asked: int):
    """Turn the diffusers step hook into (done, total) for the display.

    The total comes from the pipeline once it has built its timesteps, and
    falls back to what was asked for. With reference conditioning the two agree,
    which is the point of not using img2img here.
    """

    def hook(pipe, index, timestep, kwargs):
        on_step(index + 1, getattr(pipe, "_num_timesteps", 0) or asked)
        return kwargs

    return hook


class Renderer:
    seed_generator = staticmethod(seeded_generator)

    def __init__(self):
        self.pipe = None
        self._device, _, self.accelerator, self._total = pick_device()

    def warm(self):
        """Load once and keep it. A 4B model takes tens of seconds to load."""
        if self.pipe is not None:
            return
        import torch  # noqa: F401
        from diffusers import Flux2KleinPipeline

        device, dtype, label, total = pick_device()
        self.accelerator = label
        print(f"loading {MODEL} on {label}...", flush=True)
        started = time.time()
        pipe = Flux2KleinPipeline.from_pretrained(MODEL, revision=REVISION, torch_dtype=dtype)

        # Under roughly 24 GB the transformer and the text encoder cannot both
        # sit on the accelerator. Handing them over a layer at a time is slower
        # but it is the difference between running and not running at all.
        #
        # An unknown size counts as small. If the card would not say how much
        # memory it has, something is already using it, and putting the whole
        # model on it is the way to turn that into an out-of-memory crash.
        if device == "cuda" and (not total or total < 24e9):
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe
        print(f"ready in {time.time() - started:.0f}s", flush=True)

    def unload(self):
        """Release the loaded pipeline after a very long idle spell."""
        if self.pipe is None:
            return
        self.pipe = None
        try:
            import torch
            if self._device == "cuda":
                torch.cuda.empty_cache()
            elif self._device == "mps":
                torch.mps.empty_cache()
        except Exception:  # noqa: BLE001 - cleanup is best effort
            pass

    def render(self, job: dict, on_step=None, reference: bytes | None = None) -> bytes:
        from PIL import Image

        self.warm()
        spec = operation_of(job)
        width, height, steps = spec["width"], spec["height"], spec["steps"]

        # A master renders the composition somebody picked, so the draft they
        # picked is handed to the model as a reference. Without one it is an
        # ordinary render at the master's size, which is what a probe is and
        # what a browser that lost its own copy falls back to.
        conditioning = {}
        if spec["name"] in ("master", "verify") and reference:
            source = Image.open(io.BytesIO(reference))
            conditioning["image"] = [upscale_reference(source, (width, height))]

        watch = {}
        if on_step is not None:
            watch["callback_on_step_end"] = _reporter(on_step, steps)

        image = self.pipe(
            prompt=job["prompt"],
            num_inference_steps=steps,
            # Real classifier-free guidance, which the pipeline enables only
            # because this checkpoint is not distilled. The negative side is an
            # empty prompt, which is what it is compared against.
            guidance_scale=spec["guidance"],
            height=height,
            width=width,
            generator=self.seed_generator(job.get("seed", 0)),
            **conditioning,
            **watch,
        ).images[0]

        buffer = io.BytesIO()
        # A draft is thrown away in a minute and travels over a socket with a
        # hard size ceiling; a master is the thing somebody keeps.
        image.save(buffer, "JPEG", quality=80 if spec["name"] == "draft" else 92)
        return buffer.getvalue()
