#!/usr/bin/env python3
"""The renderer.

Runs FLUX.2 Klein the way it was meant to run: diffusers on PyTorch, using
whatever accelerator this machine has. CUDA on most Windows and Linux boxes,
MPS on Apple silicon, CPU as a last resort.

It speaks line-delimited JSON on stdin and stdout so the Node worker can drive
it as a long-lived subprocess. Loading a 4B model takes long enough that doing
it once and keeping it warm is the whole point.

    {"cmd":"render","id":"...","prompt":"...","seed":1,"steps":4}
    {"cmd":"render", ..., "init":"<base64 png>", "strength":0.55}

Replies:
    {"type":"progress","id":...,"step":n,"total":m}
    {"type":"image","id":...,"jpeg":"<base64>","ms":1234}
    {"type":"error","id":...,"message":"..."}
"""
import base64, io, json, os, sys, time

MODEL = os.environ.get("PEERPIXEL_MODEL", "black-forest-labs/FLUX.2-klein-4B")


def out(payload):
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def pick_device():
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info()
        return "cuda", torch.bfloat16, f"{name} ({total / 1e9:.0f} GB)", total
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        import subprocess
        total = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]))
        return "mps", torch.bfloat16, f"Apple silicon ({total / 1e9:.0f} GB unified)", total
    return "cpu", torch.float32, "CPU", 0


class Renderer:
    def __init__(self):
        self.pipe = None
        self.device = None
        self.label = None

    def load(self):
        if self.pipe is not None:
            return
        import torch
        from diffusers import Flux2KleinPipeline

        device, dtype, label, total = pick_device()
        self.device, self.label = device, label
        out({"type": "loading", "device": device, "accelerator": label})

        pipe = Flux2KleinPipeline.from_pretrained(MODEL, torch_dtype=dtype)
        # Under roughly 24 GB the transformer and the text encoder cannot both
        # sit on the accelerator, so hand them over a layer at a time. Slower,
        # but it is the difference between running and not running at all.
        if device == "cuda" and total and total < 24e9:
            pipe.enable_model_cpu_offload()
        elif device == "cuda":
            pipe.to(device)
        else:
            pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe
        out({"type": "ready", "device": device, "accelerator": label})

    def render(self, request):
        import torch
        self.load()
        started = time.time()
        job_id = request.get("id")
        steps = int(request.get("steps", 4))
        generator = torch.Generator(device="cpu").manual_seed(int(request.get("seed", 0)))

        def on_step(pipe, index, timestep, kwargs):
            out({"type": "progress", "id": job_id, "step": index + 1, "total": steps})
            return kwargs

        call = {
            "prompt": request["prompt"],
            "num_inference_steps": steps,
            "guidance_scale": float(request.get("guidance", 4)),
            "generator": generator,
            "height": int(request.get("height", 512)),
            "width": int(request.get("width", 512)),
            "callback_on_step_end": on_step,
        }

        # Variations and refines are image to image: keep the composition, rework
        # the detail. strength is how much of the original to throw away.
        if request.get("init"):
            from PIL import Image
            call["image"] = Image.open(io.BytesIO(base64.b64decode(request["init"]))).convert("RGB")
            call["strength"] = float(request.get("strength", 0.55))

        image = self.pipe(**call).images[0]
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=92)
        out({
            "type": "image",
            "id": job_id,
            "jpeg": base64.b64encode(buffer.getvalue()).decode(),
            "ms": int((time.time() - started) * 1000),
            "accelerator": self.label,
        })


def main():
    renderer = Renderer()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue
        try:
            if request.get("cmd") == "render":
                renderer.render(request)
            elif request.get("cmd") == "load":
                renderer.load()
            elif request.get("cmd") == "quit":
                return
        except Exception as error:  # noqa: BLE001 - report, never die
            out({"type": "error", "id": request.get("id"), "message": f"{type(error).__name__}: {error}"})


if __name__ == "__main__":
    main()
