"""Owner-only deterministic prompt-LoRA image comparisons."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import shutil
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw

from . import api, config
from .prompt_enhancer import PromptEnhancer, bootstrap_messages


class ComparisonClient:
    def __init__(self, device_id: str):
        self.device_id = str(device_id)

    def lease(self):
        return api._call("/api/worker/comparison/lease", method="POST",
                         payload={"deviceId": self.device_id}, timeout=30)

    def _download(self, path: str, lease: dict) -> bytes:
        request = urllib.request.Request(f"{config.API}{path}", headers={
            "user-agent": api.USER_AGENT,
            "authorization": f"Bearer {config.read().get('token', '')}",
            "x-peerpixel-device": self.device_id,
            "x-peerpixel-comparison-lease": lease["leaseToken"],
        })
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return response.read(256 * 1024 * 1024)
        except urllib.error.HTTPError as error:
            raise api.ApiError(error.code, "comparison_download_failed") from None

    def benchmark(self, lease):
        return self._download(f"/api/worker/comparison/benchmark/{lease['runId']}", lease)

    def adapter(self, lease, side):
        return self._download(f"/api/worker/comparison/adapter/{lease['runId']}/{side}", lease)

    def progress(self, lease, phase, completed, total):
        for attempt in range(3):
            try:
                return api._call(f"/api/worker/comparison/progress/{lease['runId']}", method="POST",
                    payload={"deviceId": self.device_id, "leaseToken": lease["leaseToken"],
                             "phase": phase, "completed": completed, "total": total})
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == 2:
                    raise
                time.sleep(0.5 * (2 ** attempt))

    def submit(self, lease, artifact: bytes, manifest: dict):
        return api._call(f"/api/worker/comparison/result/{lease['runId']}", method="PUT",
            payload={"deviceId": self.device_id, "leaseToken": lease["leaseToken"],
                     "artifactBase64": base64.b64encode(artifact).decode(),
                     "artifactDigest": hashlib.sha256(artifact).hexdigest(), "manifest": manifest},
            timeout=1800)

    def fail(self, lease, reason):
        return api._call(f"/api/worker/comparison/report/{lease['runId']}", method="POST",
            payload={"deviceId": self.device_id, "leaseToken": lease["leaseToken"],
                     "reason": str(reason)[:160]}, timeout=30)


def _extract_adapter(body: bytes, target: Path, expected_digest: str) -> Path:
    if hashlib.sha256(body).hexdigest() != expected_digest:
        raise ValueError("adapter_digest_mismatch")
    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        for member in archive.infolist():
            destination = (target / member.filename).resolve()
            if not str(destination).startswith(str(target.resolve())):
                raise ValueError("unsafe_adapter_archive")
        archive.extractall(target)
    manifests = list(target.rglob("manifest.json"))
    if len(manifests) != 1:
        raise ValueError("adapter_manifest_missing")
    return manifests[0].parent


def _enhance(adapter: Path, prompts: list[dict], progress) -> list[str]:
    enhancer = PromptEnhancer(adapter_path=adapter)
    results = []
    try:
        enhancer.warm()
        for prompt in prompts:
            generated = enhancer._generate_text(bootstrap_messages(prompt["rawPrompt"]),
                max_new_tokens=192, seed=int(prompt["llmSeed"]))
            results.append(generated.strip() or prompt["rawPrompt"])
            progress()
    finally:
        enhancer.unload()
    return results


def _pair(left: bytes, right: bytes, number: int, labels: tuple[str, str]) -> bytes:
    left_image = Image.open(io.BytesIO(left)).convert("RGB")
    right_image = Image.open(io.BytesIO(right)).convert("RGB")
    canvas = Image.new("RGB", (1024, 512), "black")
    canvas.paste(left_image.resize((512, 512)), (0, 0))
    canvas.paste(right_image.resize((512, 512)), (512, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 1024, 28), fill=(0, 0, 0))
    draw.text((10, 8), f"{number:02d} · {labels[0]}", fill="white")
    draw.text((522, 8), labels[1], fill="white")
    out = io.BytesIO()
    canvas.save(out, "JPEG", quality=94, subsampling=0, optimize=True)
    return out.getvalue()


def _archive(manifest: dict, pairs: list[bytes]) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, body in [("manifest.json", json.dumps(manifest, separators=(",", ":")).encode()),
                           *[(f"pairs/{index:02d}.jpg", body) for index, body in enumerate(pairs, 1)]]:
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, body)
    return out.getvalue()


def run_comparison(client: ComparisonClient, lease: dict, renderer) -> bool:
    root = Path(tempfile.mkdtemp(prefix="peerpixel-comparison-"))
    try:
        benchmark_body = client.benchmark(lease)
        if hashlib.sha256(benchmark_body).hexdigest() != lease["benchmarkDigest"]:
            raise ValueError("benchmark_digest_mismatch")
        benchmark = json.loads(benchmark_body)
        renderer.unload()
        comparison_path = _extract_adapter(client.adapter(lease, "comparison"), root / "comparison",
                                            lease["comparisonAdapter"]["artifactDigest"])
        current_path = _extract_adapter(client.adapter(lease, "current"), root / "current",
                                         lease["currentAdapter"]["artifactDigest"])
        done = 0
        def enhanced():
            nonlocal done
            done += 1; client.progress(lease, "enhancing", done, 40)
        comparison_prompts = _enhance(comparison_path, benchmark["prompts"], enhanced)
        current_prompts = _enhance(current_path, benchmark["prompts"], enhanced)
        rendered = [[], []]
        done = 0
        for side, prompts in enumerate((comparison_prompts, current_prompts)):
            for prompt, item in zip(prompts, benchmark["prompts"]):
                rendered[side].append(renderer.render({"operation": "grid", "prompt": prompt,
                    "seed": int(item["diffusionSeed"]), **benchmark["render"]}))
                done += 1; client.progress(lease, "rendering", done, 40)
        pairs = []
        outputs = []
        for index, item in enumerate(benchmark["prompts"]):
            pairs.append(_pair(rendered[0][index], rendered[1][index], index + 1,
                (lease["comparisonAdapter"]["version"], lease["currentAdapter"]["version"])))
            outputs.append({**item, "comparisonPrompt": comparison_prompts[index],
                "currentPrompt": current_prompts[index], "file": f"pairs/{index + 1:02d}.jpg",
                "contentType": "image/jpeg", "width": 1024, "height": 512})
            client.progress(lease, "composing", index + 1, 20)
        manifest = {"version": 1, "runId": lease["runId"],
            "benchmarkVersion": lease["benchmarkVersion"], "benchmarkDigest": lease["benchmarkDigest"],
            "comparisonAdapter": lease["comparisonAdapter"], "currentAdapter": lease["currentAdapter"],
            "render": benchmark["render"], "promptCount": 20, "imageCount": 20, "outputs": outputs}
        client.submit(lease, _archive(manifest, pairs), manifest)
        return True
    except Exception as error:
        try: client.fail(lease, str(error))
        except Exception: pass
        return False
    finally:
        try:
            renderer.unload()
        except Exception:
            pass
        shutil.rmtree(root, ignore_errors=True)
