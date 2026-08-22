#!/usr/bin/env python3
"""Short real-hardware qualification for the resident CUDA path."""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageStat  # noqa: E402

from peerpixel import config  # noqa: E402
from peerpixel.benchmark import qualify_candidate  # noqa: E402
from peerpixel.render import MANIFEST_VERSION, Renderer, STYLE_RECIPES  # noqa: E402


PROMPTS = {
    "photoreal": "a red bicycle beside a rain-wet brick wall at blue hour",
    "anime": "a young astronomer looking through a brass telescope under a starry sky",
    "vector": "a cheerful orange fox carrying a green leaf on a white background",
    "cinematic": "an explorer crossing a glass bridge above a misty canyon at dawn",
    "watercolor": "a small blue cottage surrounded by wildflowers in spring rain",
    "illustration": "a curious badger arranging books in a moonlit library",
    "pixel_art": "a tiny knight beside a glowing crystal cave entrance",
}


def valid_image(data: bytes, expected: tuple[int, int]) -> bool:
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        spread = sum(ImageStat.Stat(image.resize((32, 32))).var) / 3
        return image.size == expected and spread > 4
    except Exception:
        return False


def render_one(renderer, operation: str, style: str, output: Path):
    job = {
        "id": f"qualify-{operation}-{style}", "prompt": PROMPTS[style],
        "seed": 24681357, "operation": operation, "style": style,
        "enhance": False, "recipeId": STYLE_RECIPES[style][0],
        "manifestVersion": MANIFEST_VERSION,
    }
    started = time.perf_counter()
    jpeg, evidence = renderer.generate_job(job)
    elapsed = round((time.perf_counter() - started) * 1000)
    output.write_bytes(jpeg)
    expected = (128, 128) if operation == "draft" else (512, 512)
    valid = valid_image(jpeg, expected) and "moderation" in evidence
    return elapsed, valid, evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="drafts only")
    parser.add_argument("--style", choices=tuple(PROMPTS), action="append",
                        help="render only this style; may be repeated")
    parser.add_argument("--baseline-draft-ms", type=int, default=0)
    parser.add_argument("--baseline-master-ms", type=int, default=0)
    parser.add_argument("--approve-quality", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("qualification-output"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    os.environ.pop("PEERPIXEL_DTYPE", None)
    saved = config.read()
    if saved.get("dtype"):
        config.write(dtype="")

    renderer = Renderer()
    renderer.warm()
    # Compare the base model in both precisions. Adapters are a separate,
    # optional optimization dimension and may not decide whether residency works.
    try:
        import torch
        if renderer._device == "cuda":
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

    draft_times = []
    valid = True
    evidence = []
    styles = args.style or list(PROMPTS)
    for style in styles:
        elapsed, good, proof = render_one(
            renderer, "draft", style,
            args.output / f"{renderer._precision_mode}-draft-{style}.jpg")
        draft_times.append(elapsed)
        valid = valid and good
        evidence.append(proof)

    master_ms = 0
    if not args.quick:
        master_style = styles[0]
        master_ms, good, proof = render_one(
            renderer, "master", master_style,
            args.output / f"{renderer._precision_mode}-master-{master_style}.jpg")
        valid = valid and good
        evidence.append(proof)

    try:
        import torch
        peak = int(torch.cuda.max_memory_allocated()) if renderer._device == "cuda" else 0
    except Exception:
        peak = 0

    draft_ms = round(sum(draft_times) / len(draft_times))
    draft_gate = qualify_candidate(
        args.baseline_draft_ms, draft_ms, valid=valid,
        quality_passed=args.approve_quality) if args.baseline_draft_ms else None
    master_gate = qualify_candidate(
        args.baseline_master_ms, master_ms, valid=valid,
        quality_passed=args.approve_quality) if args.baseline_master_ms and master_ms else None
    qualified = bool(draft_gate and draft_gate["qualified"] and
                     (args.quick or (master_gate and master_gate["qualified"])))
    result = {
        "precision": renderer._precision_mode,
        "memoryMode": renderer._memory_mode,
        "styleMode": "prompt_only",
        "draftMs": draft_ms,
        "draftSamplesMs": draft_times,
        "masterMs": master_ms or None,
        "peakVram": peak,
        "valid": valid,
        "qualityPassed": bool(args.approve_quality),
        "draftGate": draft_gate,
        "masterGate": master_gate,
        "qualified": qualified,
        "output": str(args.output.resolve()),
    }
    if qualified:
        config.write(qualification={**result, "qualifiedAt": int(time.time() * 1000)})
    print(json.dumps(result, indent=2))
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
