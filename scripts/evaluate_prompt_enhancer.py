#!/usr/bin/env python3
"""Run a varied, human-readable evaluation of the real Qwen enhancer."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import random
import sys
import time
from typing import Callable, Iterable, TextIO


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from peerpixel.prompt_enhancer import PromptEnhancer, STYLES  # noqa: E402


DEFAULT_SEED = 20260824
PROMPTS = (
    "a man",
    "three red foxes",
    "A tired marine biologist repairing a yellow research drone on black volcanic sand at dawn, with rough winter surf behind her.",
    "an impossible library folded inside a glass marble",
    "A cheerful bakery logo shaped like a smiling croissant on a plain cream background.",
    'A rain-soaked roadside diner with a flickering neon sign reading "OPEN ALL NIGHT".',
    "top-down cutaway diagram of a self-sustaining lunar greenhouse, clearly labeled rooms but no written text",
    "a tiny knight facing an enormous sleeping dragon",
    "Portrait of an 82-year-old Japanese ceramicist, clay-covered hands, indigo apron, north-facing workshop light, no smile.",
    "the feeling of remembering a place that never existed",
    "A crowded 1990s commuter train crossing Tokyo at sunset; exactly one passenger notices a small blue bird inside the carriage.",
    "four seasonal icons for a weather app, consistent proportions, transparent-looking white backdrop",
    "underwater chess match between two octopuses in the ruins of a ballroom",
    "A bowl of ramen photographed for a cheap neighborhood menu, not luxury food styling.",
    "isometric cyberpunk laundromat game environment with readable machines, two NPCs, and a stray cat",
    "a loose botanical study of dandelions at every stage from bud to seed head",
    'Children racing homemade spaceships, with a cardboard finish-line banner reading "MOON CUP 2088".',
    "A brutalist concert poster composition using only cobalt blue, safety orange, black, and off-white; no words or letters.",
    "snow leopard mother and one cub crossing a narrow Himalayan ridge during a whiteout",
    "macro view inside a mechanical wristwatch where miniature gardeners prune moss growing between the gears",
)


class EvaluationPromptEnhancer(PromptEnhancer):
    """Allow accelerator-only test runs without changing production placement."""

    def __init__(self, model_path=None, *, device="cpu"):
        super().__init__(model_path=model_path)
        self.evaluation_device = device

    def warm(self):
        if self.model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peerpixel import model_hub

        path = self.model_path or model_hub.ensure("qwen3-1.7b")
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True, device_map=self.evaluation_device,
        )
        print(f"Qwen test device: {self.evaluation_device}", file=sys.stderr)


@dataclass(frozen=True)
class Case:
    number: int
    prompt: str
    style: str


def build_cases(seed: int = DEFAULT_SEED) -> list[Case]:
    """Pair every prompt with a shuffled, broadly balanced style sample."""
    rng = random.Random(seed)
    available = (*STYLES, "auto")
    assignments = [available[index % len(available)] for index in range(len(PROMPTS))]
    rng.shuffle(assignments)
    return [
        Case(number=index, prompt=prompt, style=style)
        for index, (prompt, style) in enumerate(zip(PROMPTS, assignments), start=1)
    ]


def run_evaluation(
    enhancer: PromptEnhancer,
    *,
    cases: Iterable[Case],
    stream: TextIO = sys.stdout,
    clock: Callable[[], float] = time.perf_counter,
) -> int:
    cases = list(cases)
    failures = 0
    print(f"Qwen prompt enhancer evaluation: {len(cases)} prompts", file=stream)
    for case in cases:
        print(f"\n{'=' * 80}", file=stream)
        print(f"[{case.number:02d}/{len(cases):02d}]", file=stream)
        print(f"INPUT: {case.prompt}", file=stream)
        print(f"REQUESTED STYLE: {case.style}", file=stream)
        started = clock()
        try:
            result = enhancer.enhance_pair(case.prompt, case.style)
            elapsed = clock() - started
            chosen = result.get("style", case.style)
            print(f"CHOSEN STYLE: {chosen}", file=stream)
            print(f"ENHANCED: {result['prompt']}", file=stream)
            print(f"NEGATIVE: {result['negativePrompt']}", file=stream)
            print(f"ELAPSED: {elapsed:.2f}s", file=stream)
        except Exception as error:
            failures += 1
            elapsed = clock() - started
            print(f"ERROR: {type(error).__name__}: {error}", file=stream)
            print(f"ELAPSED: {elapsed:.2f}s", file=stream)
    print(f"\nCompleted {len(cases)} prompts with {failures} errors.", file=stream)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help="seed used to shuffle style assignments",
    )
    parser.add_argument("--model-path", help="optional local Qwen model directory")
    parser.add_argument(
        "--device", choices=("cpu", "mps", "cuda"), default="cpu",
        help="test-only model device; production placement is unchanged",
    )
    args = parser.parse_args()
    enhancer = EvaluationPromptEnhancer(
        model_path=args.model_path, device=args.device,
    )
    try:
        return 1 if run_evaluation(enhancer, cases=build_cases(args.seed)) else 0
    finally:
        enhancer.unload()


if __name__ == "__main__":
    raise SystemExit(main())
