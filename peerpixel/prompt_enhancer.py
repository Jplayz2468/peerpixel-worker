"""Qwen3 1.7B prompt polishing with the network's exact instruction."""
from __future__ import annotations

import json
import hashlib
import re

MAX_NEW_TOKENS = 192
CONCEPT_TOKENS = 80

COMMON_NEGATIVE = (
    "blurry, low detail, malformed anatomy, distorted hands, extra limbs, "
    "duplicate subjects, unintended text, letters, logos, signatures, "
    "watermarks, borders, compression artifacts"
)
STYLE_NEGATIVES = {
    "photoreal": "plastic skin, waxy faces, illustration, anime, CGI, oversmoothing",
    "anime": "photorealism, 3D CGI, muddy colors, soft airbrushed outlines",
    "vector": "3D rendering, gradients, depth shading, painterly texture, rough edges",
    "cinematic": "flat lighting, amateur framing, oversaturated HDR, artificial plastic texture",
    "watercolor": "3D CGI, glossy digital rendering, hard vector edges, opaque plastic color",
    "illustration": "photographic realism, generic stock imagery, muddy hierarchy, unfinished sketch",
    "pixel_art": "antialiasing, smooth gradients, vector curves, 3D rendering, inconsistent pixel scale",
}

SYSTEM_INSTRUCTION = """You are an expert prompt optimizer for FLUX image models.
Output ONLY one valid compact JSON object with exactly two string fields: "prompt" and "negative_prompt". No markdown, preamble, or additional keys. Keep the positive prompt to 1–2 rich sentences and the negative prompt to one concise comma-separated phrase.

Behavior:
- If the user's prompt is simple, short, or underspecified: invent a coherent, original visual concept rather than only decorating the given words. Choose a specific identity or appearance, action or pose, setting, composition, lighting, atmosphere, textures, and one memorable storytelling detail suited to the chosen style. Make bold but plausible choices that turn inputs such as "a man" into a complete scene. Do not merely restate the subject and append the style directive.
- For an underspecified prompt, silently decide at least four concrete non-style facts before writing: WHO or WHAT specifically, doing WHAT, WHERE, WHEN or under what conditions, and which prop or visual clue implies a story. Generic adjectives and medium markers do not count. Never copy a stock scene; invent decisions that fit this particular subject and variation seed.
- If the user's prompt is already detailed: Retain all core subjects, colors, and actions, refining flow and adding subtle medium markers.
- Preserve explicit constraints and subject count. Do not add extra people, text, logos, brands, or named characters unless requested, and never contradict specified traits. If a draft variation seed is provided, use it to choose a genuinely distinct concept, not as visible text in the output.
- Develop the scene concept before applying the one required style supplied with the request. Never borrow characteristics from a different style."""

STYLES = ("photoreal", "anime", "vector", "cinematic", "watercolor",
          "illustration", "pixel_art")
STYLE_DIRECTIVES = {
    "photoreal": "Direct candid 35mm photography, natural lighting, sharp depth of field, fine skin pores, authentic textures, unretouched film look.",
    "anime": "Direct 1990s retro 2D anime cel aesthetic, crisp dark ink line-art, flat vibrant color fills, painted watercolor backgrounds.",
    "vector": "Direct flat 2D graphic vector art, clean geometric silhouettes, solid color planes, pure solid background, zero 3D gradients or depth.",
    "cinematic": "Direct dramatic widescreen cinema still, motivated lighting, intentional lens choice, atmospheric depth, rich filmic color, natural production design.",
    "watercolor": "Direct traditional watercolor painting on textured cold-press paper, translucent pigment washes, soft blooms, expressive edges, visible granulation.",
    "illustration": "Direct polished editorial illustration, confident shapes, expressive linework, layered color, clear visual hierarchy, handcrafted print texture.",
    "pixel_art": "Direct handcrafted pixel art, crisp pixel clusters, limited color palette, deliberate dithering, readable silhouettes, zero smoothing or antialiasing.",
}


def negative_template(style: str) -> str:
    if style not in STYLES:
        raise ValueError(f"unknown_style:{style}")
    return f"{COMMON_NEGATIVE}, {STYLE_NEGATIVES[style]}"


def sampling_seed(prompt: str, style: str, variation=None) -> int:
    material = f"{style}\0{prompt.strip()}\0{variation if variation is not None else ''}"
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big") % (2 ** 31)


def parse_enhancement(text: str, *, fallback_prompt: str,
                      fallback_negative: str) -> dict[str, str]:
    cleaned = str(text or "").strip().strip("`").strip()
    try:
        value = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        value = None
    if isinstance(value, dict):
        prompt = str(value.get("prompt") or "").strip().strip('"')
        negative = str(value.get("negative_prompt") or "").strip().strip('"')
    else:
        prompt, negative = cleaned.strip('"'), ""
    prompt = prompt or fallback_prompt.strip()
    # Model instructions are not a dependable output boundary. Keep the first
    # two complete sentences so an over-creative response cannot bloat every
    # diffusion request; preserve a trailing fragment when there are fewer.
    ends = list(re.finditer(r"[.!?](?=\s|$)", prompt))
    if len(ends) >= 2:
        prompt = prompt[:ends[1].end()].strip()
    return {
        "prompt": prompt,
        "negativePrompt": negative or fallback_negative.strip(),
    }


def needs_concept(prompt: str) -> bool:
    return len(re.findall(r"[\w'-]+", str(prompt))) <= 6


def concept_messages(prompt: str, variation=None) -> list[dict[str, str]]:
    variation_line = "" if variation is None else f"\nVariation seed: {variation}"
    return [
        {"role": "system", "content": (
            "You invent concrete visual scenes from vague subjects. Output exactly one "
            "plain sentence and no preamble. Specify identity or design, action, place, "
            "time or weather, and one meaningful prop or story clue. Use no medium, style, "
            "camera, quality, or generic mood language. Preserve subject count and constraints."
        )},
        {"role": "user", "content": f"Subject: {prompt.strip()}{variation_line}"},
    ]


def enhancement_messages(prompt: str, style: str, variation=None,
                         concept: str = "") -> list[dict[str, str]]:
    variation_line = "" if variation is None else f"\nDraft variation seed: {variation}"
    template = negative_template(style)
    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": (
            f"Chosen style: {style.upper()}{variation_line}\nUser prompt: {prompt.strip()}\n"
            + (f"Creative scene concept: {concept.strip()}\n"
               "Use every compatible concrete fact from this concept.\n" if concept else "") +
            f"Required style directive: {STYLE_DIRECTIVES[style]}\n"
            "Apply this chosen style and no other style.\n"
            f"Negative prompt template: {template}\n"
            "Keep every applicable template item and add only scene-specific failures "
            "that do not negate anything the user requested."
        )},
    ]


class PromptEnhancer:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None

    def warm(self):
        if self.model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from . import model_hub

        path = self.model_path or model_hub.ensure("qwen3-1.7b")
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        # Keep the small language model on CPU. On Apple silicon, "auto" puts
        # it in unified GPU memory beside FLUX; the resulting pressure makes
        # the last diffusion step and VAE decode swap for minutes.
        self.model = AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True, device_map="cpu")

    def _generate_text(self, messages, *, max_new_tokens: int, seed: int) -> str:
        import torch

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            output = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=True,
                temperature=0.7, top_p=0.85,
            )
        generated = output[0][inputs.input_ids.shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def enhance_pair(self, prompt: str, style: str, *, enabled=True, resolved=None,
                     resolved_negative=None, variation=None) -> dict[str, str]:
        if resolved:
            return {"prompt": str(resolved).strip(),
                    "negativePrompt": str(resolved_negative or "").strip()}
        if not enabled:
            return {"prompt": prompt.strip(), "negativePrompt": ""}
        if style not in STYLES:
            raise ValueError(f"unknown_style:{style}")
        self.warm()
        concept = ""
        if needs_concept(prompt):
            concept = self._generate_text(
                concept_messages(prompt, variation), max_new_tokens=CONCEPT_TOKENS,
                seed=sampling_seed(prompt, "concept", variation),
            ).strip().strip('"')
        messages = enhancement_messages(prompt, style, variation, concept)
        generated_text = self._generate_text(
            messages, max_new_tokens=MAX_NEW_TOKENS,
            seed=sampling_seed(prompt, style, variation),
        )
        if not generated_text:
            raise RuntimeError("prompt_enhancement_empty")
        return parse_enhancement(
            generated_text, fallback_prompt=prompt,
            fallback_negative=negative_template(style),
        )

    def enhance(self, prompt: str, style: str, *, enabled=True, resolved=None,
                variation=None) -> str:
        """Compatibility API for callers that only need the positive prompt."""
        return self.enhance_pair(
            prompt, style, enabled=enabled, resolved=resolved, variation=variation,
        )["prompt"]

    def unload(self):
        self.tokenizer = None
        self.model = None
        # Drop large tensor storage before FLUX starts. This matters most on
        # unified-memory Macs, where CPU and MPS allocations share one pool.
        import gc
        gc.collect()
