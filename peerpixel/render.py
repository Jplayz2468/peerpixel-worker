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

from . import config

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
#:
#: Size and step count are pinned here and a payload that disagrees is refused:
#: they decide what a job costs, and a job may not talk a machine into rendering
#: four times the pixels it was priced at. Guidance is different -- it is a
#: quality knob that does not change the amount of work -- so it is taken from
#: the payload when one is given, and can be retuned on the server without every
#: machine on the network needing an update.
GUIDANCE = 4.0

#: A guidance scale outside this is not a tuning choice, it is a mistake or a
#: corrupted payload, and it would waste a real render. At or below 1.0 the
#: pipeline turns guidance off entirely, which is the distilled behaviour this
#: checkpoint was chosen to get away from.
GUIDANCE_RANGE = (1.5, 12.0)
OPERATIONS = {
    "draft": {"width": 128, "height": 128, "steps": 16, "guidance": GUIDANCE},
    "master": {"width": 512, "height": 512, "steps": 50, "guidance": GUIDANCE},
    # A check is a master rendered a second time on a machine the operator
    # owns, so it is the same work with the same inputs. Only what happens to
    # the result differs: it is compared rather than delivered.
    "verify": {"width": 512, "height": 512, "steps": 50, "guidance": GUIDANCE},
    # The admission test, and the only operation the network never sends. It is
    # master resolution -- that is what catches a card which cannot hold a real
    # render -- at a step count chosen to be timed rather than looked at.
    #
    # It needs its own row because step count is pinned by the operation and
    # ignored from the payload, deliberately: a job may not talk a machine into
    # rendering fewer steps than it was paid for. That rule quietly applied to
    # the benchmark too, so a test written to run four steps ran fifty, took
    # twelve times as long as it was meant to, and was then judged against a
    # limit written for four.
    "bench": {"width": 512, "height": 512, "steps": 4, "guidance": GUIDANCE},
}

#: What may arrive over the wire. `bench` is local, and a job claiming to be one
#: would be four steps of work submitted for a fifty-step price.
NETWORK_OPERATIONS = ("draft", "master", "verify")


#: Precision, by name, so a machine that renders badly in one has somewhere to
#: go. bfloat16 is the default everywhere it works and is what the checkpoint
#: was trained in; float32 is twice the memory and slower, and is the answer
#: when a card or a driver produces arithmetic nobody can use.
def dtype_named(name: str):
    """A precision by name, or None for "whatever this device would pick".

    Guarded rather than a dict of torch attributes, because this is reached
    from `pick_device`, which is the one function here that has to survive a
    torch that is broken, stubbed or half-there.
    """
    if name not in ("bfloat16", "float16", "float32"):
        return None
    try:
        import torch

        return getattr(torch, name, None)
    except Exception:  # noqa: BLE001 - no precision is a fine answer
        return None


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

    asked = dtype_named(os.environ.get("PEERPIXEL_DTYPE") or config.read().get("dtype", ""))

    if _cuda_present(torch):
        name = _tried(lambda: torch.cuda.get_device_name(0), "NVIDIA GPU")
        # mem_get_info returns (free, total).
        total = _tried(lambda: torch.cuda.mem_get_info()[1], 0)
        label = f"{name} ({total / 1e9:.0f} GB)" if total else f"{name} (size unknown)"
        return "cuda", asked or torch.bfloat16, label, total
    if _mps_present(torch):
        total = _tried(lambda: int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])), 0)
        return "mps", asked or torch.bfloat16, f"Apple silicon ({total / 1e9:.0f} GB unified)", total
    return "cpu", asked or torch.float32, "CPU", 0


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
    """What to render, from the operation and the parts of the payload it trusts.

    A job may not talk the worker into a different resolution or step count than
    the one it was priced at, so those come from the table above and a payload
    that disagrees is refused. Guidance is taken from the payload when it is
    given and sane, because it costs nothing extra and being able to retune it
    from the server is worth more than pinning it.
    """
    name = job.get("operation", "master")
    spec = OPERATIONS.get(name)
    if spec is None:
        raise ValueError(f"unknown operation: {name}")
    for axis in ("width", "height"):
        asked = job.get(axis)
        if asked is not None and int(asked) != spec[axis]:
            raise ValueError(f"{name} is {spec[axis]}px, not {asked}px")

    guidance = spec["guidance"]
    asked = job.get("guidance")
    if asked is not None:
        try:
            asked = float(asked)
        except (TypeError, ValueError):
            asked = None
        low, high = GUIDANCE_RANGE
        if asked is not None and low <= asked <= high:
            guidance = asked
    return {"name": name, **spec, "guidance": guidance}


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


class _Nonsense(Exception):
    """The latents came back as NaN. Internal: `render` turns it into an answer."""


class _Watch:
    """The per-step hook: reports progress, and checks the arithmetic survived.

    The check is the important half. A card, a driver or a precision that this
    model does not get on with does not raise -- it quietly fills the latents
    with NaN or infinity, and the decoder turns that into the grey noise with
    black blotches that looks, to anybody watching, exactly like a bad model.
    A worker that delivers one of those is paid for a picture nobody can use,
    and the network has no way to tell it from fraud.

    So it is looked at, once, on the last step, where the damage has certainly
    accumulated if it is going to. One reduction over the latents costs nothing
    next to the render that produced them.
    """

    def __init__(self, on_step, asked: int):
        self.on_step = on_step
        self.asked = asked
        self.broken = False

    def __call__(self, pipe, index, timestep, kwargs):
        total = getattr(pipe, "_num_timesteps", 0) or self.asked
        if index + 1 >= total:
            latents = kwargs.get("latents")
            if latents is not None:
                import torch

                self.broken = not bool(torch.isfinite(latents).all())
        if self.on_step is not None:
            self.on_step(index + 1, total)
        return kwargs


#: What to say when it happens and there is nothing left to try.
BROKEN = (
    "this render came out as nan on every precision this machine has. The "
    "arithmetic is broken rather than the model; run `peerpixel doctor` and "
    "send what it prints."
)

#: Precisions to try, in order, when one of them produces nonsense.
#:
#: bfloat16 is what the checkpoint was trained in and is right nearly
#: everywhere. Nearly: bfloat16 on Metal is newer than the machines people run
#: it on, and on some of them it silently returns NaN instead of numbers --
#: which decodes to flat grey with a few black specks and looks, to anybody
#: watching, exactly like a broken model. float16 has been on Metal for years.
#: float32 is twice the memory and slower and is always correct.
LADDER = ("bfloat16", "float16", "float32")


class Renderer:
    seed_generator = staticmethod(seeded_generator)

    def __init__(self):
        self.pipe = None
        self._device, self._dtype, self.accelerator, self._total = pick_device()

    def warm(self):
        """Load once and keep it. A 4B model takes tens of seconds to load."""
        if self.pipe is not None:
            return
        import torch  # noqa: F401
        from diffusers import Flux2KleinPipeline

        device, dtype, label, total = pick_device()
        self._device, self._dtype, self.accelerator = device, dtype, label
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

    def demote(self) -> str | None:
        """Give up on this precision and take the next one down. Returns its name.

        Called only when a render came back as NaN, and remembered, because a
        machine whose bfloat16 is broken has broken bfloat16 tomorrow as well
        and nobody should have to lose a render a day to rediscover it.
        """
        import torch

        names = {torch.bfloat16: "bfloat16", torch.float16: "float16",
                 torch.float32: "float32"}
        here = names.get(self._dtype, "bfloat16")
        rest = LADDER[LADDER.index(here) + 1:] if here in LADDER else ()
        if not rest:
            return None
        chosen = rest[0]
        config.write(dtype=chosen)
        os.environ["PEERPIXEL_DTYPE"] = chosen
        self.unload()
        return chosen

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

    def render(self, job: dict, on_step=None, reference: bytes | None = None,
               on_demote=None) -> bytes:
        """Render, and never hand back something that is not a picture.

        One retry, on the next precision down, because the failure this catches
        is a property of the machine rather than of the job: if it happens once
        it happens every time, and the only useful thing to do about it is stop
        using the precision that caused it.
        """
        try:
            return self._render(job, on_step=on_step, reference=reference)
        except _Nonsense:
            chosen = self.demote()
            if chosen is None:
                raise RuntimeError(BROKEN) from None
            if on_demote is not None:
                on_demote(chosen)
            print(f"that render came out as nan; retrying in {chosen}", flush=True)
        try:
            return self._render(job, on_step=on_step, reference=reference)
        except _Nonsense:
            raise RuntimeError(BROKEN) from None

    def _render(self, job: dict, on_step=None, reference: bytes | None = None) -> bytes:
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

        watch = _Watch(on_step, steps)

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
            callback_on_step_end=watch,
        ).images[0]

        if watch.broken:
            raise _Nonsense()

        buffer = io.BytesIO()
        # A draft is thrown away in a minute and travels over a socket with a
        # hard size ceiling; a master is the thing somebody keeps.
        image.save(buffer, "JPEG", quality=80 if spec["name"] == "draft" else 92)
        return buffer.getvalue()
