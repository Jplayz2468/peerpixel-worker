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

A **preview** is 128x128 from the prompt alone, in six steps. It exists to
answer whether the composition is the one somebody wanted, and it is asked
several times before anything is chosen, so being cheap is most of its job. It
goes straight back down the socket rather than being uploaded anywhere.

A **master** is 512x512 at the full fifty guided steps, from the
prompt and a seed. Nothing else. It is not conditioned on the preview that was
chosen, does not receive it, and does not know it exists.

What ties the two together is the noise. A seed names one 512px noise tensor;
a preview renders that same tensor averaged down to its own smaller shape, so
the two share the low-frequency structure that decides where things end up in
the frame. See `seeded_latents`, which is also where the reason it must be an
average rather than an upscale is written down.

This replaced handing the chosen preview back as a reference image. That worked
-- it kept the framing -- but it could not promise that a final looked like a
native render at its own resolution, because it was not one: it was a render
with somebody's 256px picture blown up 4x sitting in its context. It also meant
a final could not start until the browser that asked for it sent the bytes
back, so closing a tab lost a paid render. None of that is true any more.

This file is deliberately plain and short. If a render goes wrong, this is where
to look, and you can edit it and restart the worker without rebuilding anything.
"""
from __future__ import annotations

import io
import hashlib
import json
import os
import subprocess
import time
import xml.etree.ElementTree as ET

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
    # A preview exists to answer one question -- is this the composition I
    # wanted -- and it is asked several times before anything is chosen, so
    # what matters about it is that it is cheap. Six steps at 128px is enough
    # for this checkpoint to settle a composition.
    "draft": {"width": 128, "height": 128, "steps": 6, "guidance": GUIDANCE},
    # The master retains the full fifty guided steps at 512px.
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
MANIFEST_VERSION = "2026-08-23.1"
STYLE_RECIPES = {
    name: (f"{name}-v2", ()) for name in (
        "photoreal", "anime", "vector", "cinematic", "watercolor",
        "illustration", "pixel_art",
    )
}


def _digest(value) -> str:
    if isinstance(value, bytes):
        payload = value
    else:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


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
        _, total = nvidia_memory()
        label = f"{name} ({total / 1e9:.0f} GB)" if total else f"{name} (size unknown)"
        return "cuda", asked or torch.bfloat16, label, total
    if _mps_present(torch):
        total = _tried(lambda: int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])), 0)
        return "mps", asked or torch.bfloat16, f"Apple silicon ({total / 1e9:.0f} GB unified)", total
    return "cpu", asked or torch.float32, "CPU", 0


def nvidia_memory() -> tuple[int, int]:
    """Free and total bytes without creating or entering a CUDA context."""
    try:
        answer = subprocess.run([
            "nvidia-smi",
            "--query-gpu=memory.free,memory.total",
            "--format=csv,noheader,nounits",
            "--id=0",
        ], capture_output=True, text=True, timeout=2, check=True)
        free, total = answer.stdout.strip().splitlines()[0].split(",")
        return int(float(free.strip()) * 1024 ** 2), int(float(total.strip()) * 1024 ** 2)
    except Exception:  # noqa: BLE001 - an unknown size selects safe offload
        return 0, 0


def nvidia_processes() -> list[tuple[str, int, int]]:
    """GPU consumers as (app, pid, bytes), including graphics processes."""
    try:
        answer = subprocess.run(
            ["nvidia-smi", "-q", "-x"], capture_output=True, text=True,
            timeout=3, check=True,
        )
        root = ET.fromstring(answer.stdout)
    except Exception:  # noqa: BLE001 - diagnostics are allowed to be partial
        return []

    found = []
    for process in root.findall(".//process_info"):
        try:
            pid = int(process.findtext("pid", "0"))
            raw_name = process.findtext("process_name", "unknown")
            name = os.path.basename(raw_name.replace("\\", "/")) or raw_name
            used = process.findtext("used_memory", "0 MiB").split()[0]
            found.append((name, pid, int(float(used) * 1024 ** 2)))
        except (TypeError, ValueError):
            continue
    return sorted(found, key=lambda item: item[2], reverse=True)


MINIMUM_CUDA_HEADROOM = 2 * 1024 ** 3


def _display_memory(size: int) -> str:
    if size >= 1024 ** 3:
        return f"{size / 1024 ** 3:.1f} GB"
    return f"{round(size / 1024 ** 2)} MB"


def require_cuda_headroom(free: int, total: int, *, processes=None) -> None:
    """Fail early with an actionable app list instead of a CUDA traceback."""
    if not total or free >= MINIMUM_CUDA_HEADROOM:
        return
    processes = nvidia_processes() if processes is None else processes
    lines = [
        f"Only {_display_memory(free)} of {_display_memory(total)} VRAM is free.",
        "GPU memory is currently being used by:",
    ]
    if processes:
        lines.extend(f"- {name} (PID {pid}): {_display_memory(used)}"
                     for name, pid, used in processes)
    else:
        lines.append("- another application (run nvidia-smi to identify it)")
    lines.append("Close or restart those apps, then run PeerPixel again.")
    raise RuntimeError("\n".join(lines))


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


def cuda_memory_mode(*, total: int, free: int) -> str:
    """Fastest placement that leaves useful activation headroom.

    Consumer cards use streamed block groups. Whole-model offload still has to
    place the complete 4B transformer on the card at once, which is why a 16GB
    5080 can OOM despite the weights themselves being smaller than the card.
    """
    if total >= 24e9 and free >= 20e9:
        return "resident"
    return "group"


def cuda_group_options(*, total: int, free: int) -> dict:
    """Transfers for the constrained-card path.

    Stream prefetch deliberately stays off: it overlaps compute nicely, but
    holds the current and next groups together and can OOM a 16GB card before
    its first step. One synchronous block is still much faster than moving
    every leaf for every call in sequential offload.
    """
    # The pipeline's VAE receives CUDA latents after the diffusion loop. Group
    # offloading it leaves its convolution weights on CPU at that boundary,
    # which crashes only after every denoising step has completed. Keeping the
    # comparatively small decoder resident avoids the device mismatch.
    return {
        "non_blocking": False,
        "use_stream": False,
        "record_stream": False,
        "exclude_modules": ["vae"],
        # Four adjacent blocks amortize PCIe transfers on 16 GB cards. Smaller
        # cards retain the proven one-block path and its lower peak.
        "num_blocks_per_group": 4 if total >= 15e9 and free >= 13e9 else 1,
    }


def _cuda_oom(error: BaseException) -> bool:
    try:
        import torch
        if isinstance(error, torch.OutOfMemoryError):
            return True
    except Exception:
        pass
    return "cuda" in str(error).lower() and "out of memory" in str(error).lower()


def _quieten() -> None:
    """Stop the libraries drawing over the bar.

    diffusers, transformers and the Hub each keep their own tqdm and their own
    logger, and all three print while a model loads -- straight through the
    lines this program is repainting, which tears the drawing apart and leaves
    fragments of somebody else's percentage on the screen. None of it is
    information the person watching asked for; the bar already says what is
    happening. Every call is guarded because a missing one of these is not a
    reason to fail to render.
    """
    import logging
    import os
    import warnings

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")
    for call in (
        lambda: __import__("diffusers").utils.logging.disable_progress_bar(),
        lambda: __import__("diffusers").utils.logging.set_verbosity_error(),
        lambda: __import__("transformers").utils.logging.disable_progress_bar(),
        lambda: __import__("transformers").utils.logging.set_verbosity_error(),
        lambda: __import__("huggingface_hub").utils.logging.set_verbosity_error(),
    ):
        try:
            call()
        except Exception:  # noqa: BLE001 - a quieter log is never worth a crash
            pass
    logging.getLogger("diffusers").setLevel(logging.ERROR)


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


def latent_grid(pipe, pixels: int) -> int:
    """How many latent positions a side of `pixels` becomes.

    The VAE compresses by `vae_scale_factor` and the transformer packs 2x2 of
    what is left, which is where the doubling comes from. Asked of the pipeline
    rather than written down, so a checkpoint with a different VAE does not
    silently produce noise of the wrong shape.
    """
    return int(pixels) // (pipe.vae_scale_factor * 2)


def seeded_latents(pipe, spec: dict, seed: int, dtype):
    """The noise this render starts from, derived from the final's noise.

    This is what replaces conditioning a final on the preview that was picked.
    A seed names one 1024px noise tensor and nothing else. A preview renders
    the same tensor with each block of it averaged down to its own smaller
    shape, so the two share their low-frequency structure -- which is the part
    of the noise that decides where things end up in the frame -- while the
    final stays an ordinary render at its own resolution, with nothing upscaled
    into it and no reference image anywhere.

    The averaging is the only direction that works. Scaling noise *up* is what
    it is tempting to do and it destroys the thing being scaled: a 4x
    nearest-neighbour blow-up leaves neighbouring values 0.76 correlated and
    bilinear leaves them 0.93, where real noise is 0.00, and the model reads
    that correlation as structure that was never in the picture. Averaging
    down survives it exactly. Each output is a sum of f*f independent unit
    normals divided by f, so its variance is f*f/f*f = 1 and the blocks do not
    overlap, which leaves the result an exact sample of the same distribution
    the model was trained to start from. Measured: 0.001 correlation, variance
    0.996.
    """
    import torch

    master = OPERATIONS["master"]
    channels = pipe.transformer.config.in_channels
    tall = latent_grid(pipe, master["height"])
    wide = latent_grid(pipe, master["width"])
    # Always drawn on the CPU at full precision. The seed has to name the same
    # tensor on every machine in the network, and neither a GPU's generator nor
    # bfloat16 rounding is portable enough to promise that.
    noise = torch.randn((1, channels, tall, wide), dtype=torch.float32,
                        generator=torch.Generator("cpu").manual_seed(int(seed) & 0xFFFFFFFF))

    here_tall = latent_grid(pipe, spec["height"])
    here_wide = latent_grid(pipe, spec["width"])
    if (here_tall, here_wide) != (tall, wide):
        down = tall // here_tall
        if down < 1 or tall % here_tall or wide % here_wide or wide // here_wide != down:
            raise ValueError(
                f"{spec['name']} at {spec['width']}px does not divide the "
                f"{master['width']}px final, so its noise cannot be derived from it")
        noise = noise.reshape(1, channels, here_tall, down, here_wide, down)
        noise = noise.mean(dim=(3, 5)) * down
    return noise.to(dtype)


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

    def __init__(self, on_step, asked: int, on_last=None):
        self.on_step = on_step
        self.asked = asked
        self.on_last = on_last
        self.broken = False

    def __call__(self, pipe, index, timestep, kwargs):
        total = getattr(pipe, "_num_timesteps", 0) or self.asked
        last = index + 1 >= total
        if last:
            latents = kwargs.get("latents")
            if latents is not None:
                import torch

                self.broken = not bool(torch.isfinite(latents).all())
        if self.on_step is not None:
            self.on_step(index + 1, total)
        if last and self.on_last is not None:
            # Everything after this point is the VAE turning latents into a
            # 1024px picture, which on a memory-tight machine is the longest
            # single thing a render does. Whatever is drawing needs to be told,
            # or its bar sits at the end of the render phase with a full
            # measurement behind it and nothing left to move it.
            self.on_last()
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
        self._enhancer = None
        self._safety = None
        self._device, self._dtype, self.accelerator, self._total = pick_device()
        self._mlx_backend = None
        if self._device == "mps":
            from .mlx_backend import MLXBackend
            self._mlx_backend = MLXBackend()
            self.accelerator = self.accelerator.replace("Apple silicon", "Apple silicon MLX q4")
        self._memory_mode = None
        self._forced_memory_mode = None
        self._precision_mode = "native"
        self._style_mode = "prompt_only"

    def warm(self):
        """Load once and keep it. A 4B model takes tens of seconds to load."""
        if self.pipe is not None:
            return
        if getattr(self, "_mlx_backend", None) is not None:
            started = time.time()
            self._mlx_backend.ensure_model()
            self.pipe = self._mlx_backend
            self._precision_mode = "mlx-q4"
            self._memory_mode = "resident"
            self.load_seconds = time.time() - started
            return
        import torch  # noqa: F401
        from diffusers import Flux2KleinPipeline

        device, dtype, label, total = pick_device()
        self._device, self._dtype, self.accelerator = device, dtype, label
        free = 0
        if device == "cuda":
            free, current_total = nvidia_memory()
            total = total or current_total
            require_cuda_headroom(free, total)
        _quieten()
        started = time.time()
        from .precision import pipeline_quantization_config, runtime_probe, select_precision

        requested_precision = os.environ.get("PEERPIXEL_DTYPE") or config.read().get("dtype")
        plan = select_precision(runtime_probe(total=total, free=free), requested_precision) if device == "cuda" else None
        self._precision_mode = plan.mode if plan is not None else str(dtype).split(".")[-1]
        if plan is not None and plan.mode == "unavailable":
            raise RuntimeError(plan.reason)

        # `dtype`, not `torch_dtype`: the old spelling is deprecated and prints
        # a warning across whatever is being drawn at the time.
        load_options = {"revision": REVISION, "dtype": dtype}
        if plan is not None and plan.mode in {"int8", "int4"}:
            load_options.update(
                quantization_config=pipeline_quantization_config(plan.mode),
                device_map="cuda",
            )
        try:
            pipe = Flux2KleinPipeline.from_pretrained(MODEL, **load_options)
        except Exception:
            # A worker must never silently produce a different-quality image.
            # Quantization failures make image rendering unavailable while the
            # machine can continue serving its auxiliary model capabilities.
            raise

        # Under roughly 24 GB the transformer and the text encoder cannot both
        # sit on the accelerator. Handing them over a layer at a time is slower
        # but it is the difference between running and not running at all.
        #
        # An unknown size counts as small. If the card would not say how much
        # memory it has, something is already using it, and putting the whole
        # model on it is the way to turn that into an out-of-memory crash.
        if device == "cuda":
            mode = self._forced_memory_mode or (
                "resident" if plan is not None and plan.resident
                else cuda_memory_mode(total=total, free=free))
            if mode == "sequential":
                pipe.enable_sequential_cpu_offload()
            elif mode == "group":
                # One block per group keeps even 12GB cards below their limit.
                # A separate CUDA stream overlaps the next transfer with the
                # current block's compute, avoiding sequential-offload speed.
                group_options = cuda_group_options(total=total, free=free)
                pipe.enable_group_offload(
                    onload_device=torch.device("cuda"),
                    offload_device=torch.device("cpu"),
                    offload_type="block_level",
                    **group_options,
                )
            elif plan is None or plan.mode not in {"int8", "int4"}:
                pipe.to(device)
            self._memory_mode = mode
        else:
            pipe.to(device)
            self._memory_mode = "resident"
        pipe.set_progress_bar_config(disable=True)

        # Decoding is one unbounded call over the whole picture, and at 1024px
        # that is where a machine without memory to spare starts swapping --
        # a minute or two after the last step, or never. Tiling bounds the peak
        # instead of the total, which is the number that hurts.
        _tried(pipe.vae.enable_tiling, None)

        # `force_upcast` runs the decoder in float32 because float16 overflows
        # in it. bfloat16 has float32's exponent range and does not, so paying
        # for the upcast there is paying twice the memory for nothing.
        import torch as _torch

        if dtype is not _torch.float16:
            _tried(lambda: pipe.vae.config.__setattr__("force_upcast", False), None)

        self.pipe = pipe
        self.load_seconds = time.time() - started

    def apply_style(self, job: dict) -> str:
        """Validate the prompt-only style recipe without mutating the pipeline."""
        style = job.get("style", "photoreal")
        recipe = STYLE_RECIPES.get(style)
        if recipe is None:
            raise ValueError(f"unknown_style:{style}")
        recipe_id, _adapters = recipe
        if job.get("recipeId", recipe_id) != recipe_id:
            raise ValueError("wrong_style_recipe")
        if job.get("manifestVersion", MANIFEST_VERSION) != MANIFEST_VERSION:
            raise ValueError("wrong_model_manifest")
        return "prompt_only"

    def generate_job(self, job: dict, **render_options):
        """Polish, style, render, then classify locally in that exact order."""
        from .prompt_enhancer import PromptEnhancer
        from .safety import SafetyClassifier

        on_phase = render_options.get("on_phase")
        if on_phase is not None:
            on_phase("enhancing_prompt")
        if self._enhancer is None:
            self._enhancer = PromptEnhancer()
        enhancement_options = {
            "enabled": job.get("enhance", True),
            "resolved": job.get("enhancedPrompt"),
            "variation": job.get("seed") if job.get("operation") == "draft" else None,
        }
        if hasattr(self._enhancer, "enhance_pair"):
            pair = self._enhancer.enhance_pair(
                job["prompt"], job.get("style", "photoreal"),
                resolved_negative=job.get("negativePrompt"), **enhancement_options,
            )
        else:  # Compatibility with a worker finishing a job across an update.
            pair = {"prompt": self._enhancer.enhance(
                job["prompt"], job.get("style", "photoreal"), **enhancement_options,
            ), "negativePrompt": str(job.get("negativePrompt") or "")}
        effective = pair["prompt"]
        negative = pair["negativePrompt"]
        # Enhancement is complete before inference. Do not retain Qwen's
        # tensors while the much larger FLUX pipeline and VAE need the memory.
        self._enhancer.unload()
        negative_mode = ("none" if not negative else
                         "inline" if getattr(self, "_mlx_backend", None) is not None
                         else "native")
        resolved = {**job, "prompt": effective, "negativePrompt": negative}
        jpeg = self.render(resolved, **render_options)
        if on_phase is not None:
            on_phase("safety_check")
        if self._safety is None:
            self._safety = SafetyClassifier()
        moderation = self._safety.classify(jpeg)
        runtime = "peerpixel-worker/0.8.6"
        return jpeg, {
            "enhancedPrompt": effective,
            "negativePrompt": negative,
            "negativeConditioning": negative_mode,
            "moderation": moderation,
            "manifestVersion": job.get("manifestVersion", MANIFEST_VERSION),
            "recipeId": STYLE_RECIPES[job.get("style", "photoreal")][0],
            "precision": getattr(self, "_precision_mode", "native"),
            "memoryMode": getattr(self, "_memory_mode", "resident"),
            "styleMode": getattr(self, "_style_mode", "prompt_only"),
            "attestations": [
                {"operation": "prompt", "inputDigest": _digest({
                    "prompt": job["prompt"], "style": job.get("style", "photoreal"),
                    "enhance": job.get("enhance", True),
                    "variation": job.get("seed") if job.get("operation") == "draft" else None,
                }), "outputDigest": _digest({
                    "prompt": effective, "negativePrompt": negative,
                }), "runtimeVersion": runtime},
                {"operation": "render", "inputDigest": _digest({
                    "prompt": effective, "seed": job.get("seed", 0),
                    "negativePrompt": negative, "negativeConditioning": negative_mode,
                    "recipe": STYLE_RECIPES[job.get("style", "photoreal")][0],
                    "operation": job.get("operation", "master"),
                }), "outputDigest": _digest(jpeg), "runtimeVersion": runtime},
                {"operation": "moderation", "inputDigest": _digest(jpeg),
                 "outputDigest": _digest(moderation), "runtimeVersion": runtime},
            ],
        }

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
        old = self.pipe
        self.pipe = None
        del old
        try:
            import gc
            import torch
            gc.collect()
            if self._device == "cuda":
                torch.cuda.empty_cache()
            elif self._device == "mps":
                torch.mps.empty_cache()
        except Exception:  # noqa: BLE001 - cleanup is best effort
            pass

    def _retry_low_memory(self) -> None:
        """Reload with the guaranteed-fit CUDA fallback after one real OOM."""
        self._forced_memory_mode = "sequential"
        self.unload()
        self.warm()

    def render(self, job: dict, on_step=None, reference: bytes | None = None,
               on_demote=None, on_decode=None, on_phase=None) -> bytes:
        """Render, and never hand back something that is not a picture.

        One retry, on the next precision down, because the failure this catches
        is a property of the machine rather than of the job: if it happens once
        it happens every time, and the only useful thing to do about it is stop
        using the precision that caused it.
        """
        try:
            return self._render(job, on_step=on_step, on_decode=on_decode,
                                on_phase=on_phase)
        except RuntimeError as error:
            if self._device != "cuda" or self._memory_mode == "sequential" or not _cuda_oom(error):
                raise
            if on_demote is not None:
                on_demote("low-memory CUDA")
            self._retry_low_memory()
            return self._render(job, on_step=on_step, on_decode=on_decode,
                                on_phase=on_phase)
        except _Nonsense:
            chosen = self.demote()
            if chosen is None:
                raise RuntimeError(BROKEN) from None
            if on_demote is not None:
                on_demote(chosen)
            print(f"that render came out as nan; retrying in {chosen}", flush=True)
        try:
            return self._render(job, on_step=on_step, on_decode=on_decode,
                                on_phase=on_phase)
        except _Nonsense:
            raise RuntimeError(BROKEN) from None

    def _render(self, job: dict, on_step=None, on_decode=None, on_phase=None) -> bytes:
        self.warm()
        spec = operation_of(job)
        if getattr(self, "_mlx_backend", None) is not None:
            if on_phase is not None:
                on_phase("encoding_prompt")
            if spec["name"] != "bench" and ("style" in job or "recipeId" in job):
                if on_phase is not None:
                    on_phase("loading_style")
                self._style_mode = self.apply_style(job)
            if on_phase is not None:
                on_phase("rendering")
            jpeg = self._mlx_backend.render(
                prompt=job["prompt"], width=spec["width"], height=spec["height"],
                steps=spec["steps"], guidance=spec["guidance"],
                seed=job.get("seed", 0), negative_prompt=job.get("negativePrompt", ""),
                on_step=on_step,
            )
            if on_phase is not None:
                on_phase("decoding")
            if on_decode is not None:
                on_decode()
            return jpeg
        prompt_embeds = negative_prompt_embeds = None
        if on_phase is not None:
            on_phase("encoding_prompt")
        if hasattr(self.pipe, "encode_prompt"):
            prompt_embeds, _ = self.pipe.encode_prompt(job["prompt"])
            if spec["guidance"] > 1:
                negative_prompt_embeds, _ = self.pipe.encode_prompt(
                    job.get("negativePrompt", ""))
        if spec["name"] != "bench" and ("style" in job or "recipeId" in job):
            if on_phase is not None:
                on_phase("loading_style")
            self._style_mode = self.apply_style(job)
        width, height, steps = spec["width"], spec["height"], spec["steps"]

        # The seed is the whole of the relationship between a preview and its
        # final. Nothing is handed to the model but the prompt and the noise
        # that seed names; `reference` is accepted and ignored so that a server
        # still sending one gets a correct render rather than an error.
        latents = seeded_latents(self.pipe, spec, job.get("seed", 0), self._dtype)

        def decoding():
            if on_phase is not None:
                on_phase("decoding")
            if on_decode is not None:
                on_decode()

        watch = _Watch(on_step, steps, on_last=decoding)

        if on_phase is not None:
            on_phase("rendering")

        pipeline_args = dict(
            prompt=None if prompt_embeds is not None else job["prompt"],
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            num_inference_steps=steps,
            # Real classifier-free guidance, which the pipeline enables only
            # because this checkpoint is not distilled. The negative side is an
            # empty prompt, which is what it is compared against.
            guidance_scale=spec["guidance"],
            height=height,
            width=width,
            generator=self.seed_generator(job.get("seed", 0)),
            latents=latents,
            callback_on_step_end=watch,
        )
        if prompt_embeds is None:
            pipeline_args["negative_prompt"] = job.get("negativePrompt", "")
        image = self.pipe(**pipeline_args).images[0]

        if watch.broken:
            raise _Nonsense()

        buffer = io.BytesIO()
        # A draft is thrown away in a minute and travels over a socket with a
        # hard size ceiling; a master is the thing somebody keeps.
        image.save(buffer, "JPEG", quality=80 if spec["name"] == "draft" else 92)
        return buffer.getvalue()
