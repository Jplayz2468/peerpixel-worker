"""Apple-Silicon FLUX.2 Klein Base rendering through the native MLX runtime."""
from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess
import tempfile


MODEL = "AbstractFramework/flux.2-klein-base-4b-4bit"
_STEP = re.compile(r"(\d+)\s*/\s*(\d+)")


def progress_steps(text: str, total: int) -> list[int]:
    """Extract unique monotonic step counts from tqdm's CR-delimited output."""
    found = []
    for done, reported_total in _STEP.findall(text):
        step = min(total, int(done))
        if int(reported_total) == total and step and (not found or step > found[-1]):
            found.append(step)
    return found


class MLXBackend:
    def __init__(self, executable: str | None = None):
        self.executable = executable or shutil.which("mlxgen") or "mlxgen"

    def ensure_model(self) -> None:
        subprocess.run(
            [self.executable, "download", "--model", MODEL],
            check=True, text=True,
        )

    def render(self, *, prompt: str, width: int, height: int, steps: int,
               guidance: float, seed: int, on_step=None) -> bytes:
        with tempfile.TemporaryDirectory(prefix="peerpixel-mlx-") as folder:
            output = Path(folder) / "render.jpg"
            command = [
                self.executable, "generate", "--model", MODEL,
                "--prompt", prompt, "--width", str(width), "--height", str(height),
                "--steps", str(steps), "--guidance", str(guidance),
                "--seed", str(int(seed)), "--output", str(output),
            ]
            process = subprocess.Popen(
                command, text=True, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE, bufsize=1,
            )
            transcript = ""
            reported = 0
            while True:
                character = process.stderr.read(1)
                if not character:
                    break
                transcript += character
                if character not in "\r\n":
                    continue
                for done in progress_steps(transcript, steps):
                    if done > reported and on_step is not None:
                        on_step(done, steps)
                    reported = max(reported, done)
                transcript = ""
            code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, command, stderr=transcript)
            data = output.read_bytes()
        if not data.startswith(b"\xff\xd8\xff"):
            raise RuntimeError("mlx_generation_invalid_jpeg")
        if on_step is not None and reported < steps:
            on_step(steps, steps)
        return data
