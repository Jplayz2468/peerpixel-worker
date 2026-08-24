"""Qwen3 1.7B prompt polishing with the network's exact instruction."""
from __future__ import annotations

import json
import hashlib
import re

MAX_NEW_TOKENS = 192
CONCEPT_TOKENS = 80

COMMON_NEGATIVE = (
    "blurry, low detail, malformed anatomy, distorted hands, extra limbs, "
    "duplicate subjects, logos, signatures, "
    "watermarks, borders, compression artifacts"
)
NO_TEXT_NEGATIVE = "unintended text, letters"
REQUESTED_TEXT_NEGATIVE = (
    "misspelled requested text, gibberish text, duplicated text, "
    "missing text, cropped typography"
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
- For an underspecified prompt, silently decide at least four concrete non-style facts before writing: WHO or WHAT specifically, doing WHAT, WHERE, WHEN or under what conditions, and which prop or visual clue implies a story. Generic adjectives and medium markers do not count. Never copy a stock scene; invent decisions that fit this particular subject.
- If the user's prompt is already detailed: Retain all core subjects, colors, and actions, refining flow and adding subtle medium markers.
- Preserve explicit constraints and subject count. Do not add extra people, text, logos, brands, or named characters unless requested, and never contradict specified traits.
- When visible text is explicitly requested on a sign, poster, cover, label, screen, garment, title, caption, or similar surface: reproduce its exact spelling and capitalization in double quotes. Describe the physical surface plus typography, placement, material, contrast, and legibility so FLUX treats the words as part of the composition. Never paraphrase, translate, extend, or invent copy. Spoken dialogue is not visible image text unless the user requests a speech bubble, subtitle, or caption.
- Develop the scene concept before applying the one required style supplied with the request. Never borrow characteristics from a different style."""

AUTO_SYSTEM_INSTRUCTION = SYSTEM_INSTRUCTION.replace(
    'exactly two string fields: "prompt" and "negative_prompt"',
    'exactly three string fields: "style", "prompt", and "negative_prompt"',
) + """
- Choose exactly one style from: photoreal, anime, vector, cinematic, watercolor, illustration, pixel_art. Put its lowercase name in "style". Choose the style that best serves the subject and concept; do not default mechanically to photoreal."""

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


_TEXT_SURFACE = (
    r"(?:sign|poster|billboard|marquee|label|caption|subtitle|speech bubble|"
    r"book cover|album cover|cover|title|headline|menu|screen|shirt|t-shirt|garment)"
)
_TEXT_VERB = r"(?:reading|reads|saying|says|bearing|titled|with\s+(?:the\s+)?(?:text|words))"


def requested_visible_text(prompt: str) -> tuple[str, ...]:
    """Extract copy that must survive enhancement byte-for-byte.

    Deliberately require a visual text surface, so quoted dialogue does not
    accidentally become typography. Quoted copy is preferred; an all-caps
    unquoted phrase is accepted because it is unambiguous in natural prompts.
    """
    source = str(prompt or "")
    found = []
    quoted = re.compile(
        rf"{_TEXT_SURFACE}\b[^.!?\n]{{0,80}}?\b{_TEXT_VERB}\s*[\"“']([^\"”']+)[\"”']",
        re.IGNORECASE,
    )
    capitals = re.compile(
        rf"{_TEXT_SURFACE}\b[^.!?\n]{{0,80}}?\b{_TEXT_VERB}\s+"
        r"([A-Z0-9][A-Z0-9 '&-]{1,60}?)(?=$|[,.;!?])",
    )
    for pattern in (quoted, capitals):
        for match in pattern.finditer(source):
            value = re.sub(r"\s+", " ", match.group(1)).strip()
            if value and value not in found:
                found.append(value)
    return tuple(found)


def negative_template(style: str, visible_text: tuple[str, ...] = ()) -> str:
    if style not in STYLES:
        raise ValueError(f"unknown_style:{style}")
    text_rules = REQUESTED_TEXT_NEGATIVE if visible_text else NO_TEXT_NEGATIVE
    return f"{COMMON_NEGATIVE}, {text_rules}, {STYLE_NEGATIVES[style]}"


def sampling_seed(prompt: str, style: str) -> int:
    material = f"{style}\0{prompt.strip()}"
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big") % (2 ** 31)


def parse_enhancement(text: str, *, fallback_prompt: str,
                      fallback_negative: str,
                      visible_text: tuple[str, ...] = (),
                      requested_style: str | None = None) -> dict[str, str]:
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
    for copy in visible_text:
        for left, right in (("'", "'"), ("‘", "’"), ("“", "”")):
            prompt = prompt.replace(f"{left}{copy}{right}", f'"{copy}"')
        prompt = re.sub(
            rf'(?<!["\']){re.escape(copy)}(?!["\'])', f'"{copy}"', prompt,
        )
    missing = [copy for copy in visible_text if copy not in prompt]
    if missing:
        exact = ", ".join(f'"{copy}"' for copy in missing)
        prompt = prompt.rstrip().rstrip(".!?") + (
            f', featuring exact visible text {exact} with clear, correctly spelled typography.'
        )
    # Model instructions are not a dependable output boundary. Keep the first
    # two complete sentences so an over-creative response cannot bloat every
    # diffusion request; preserve a trailing fragment when there are fewer.
    ends = list(re.finditer(r"[.!?](?=\s|$)", prompt))
    if len(ends) >= 2:
        prompt = prompt[:ends[1].end()].strip()
    if visible_text:
        parts = [part.strip() for part in negative.split(",") if part.strip()]
        parts = [part for part in parts if part.lower() not in {"unintended text", "letters"}]
        for rule in REQUESTED_TEXT_NEGATIVE.split(", "):
            if rule not in parts:
                parts.append(rule)
        negative = ", ".join(parts)
    result = {
        "prompt": prompt,
        "negativePrompt": negative or fallback_negative.strip(),
    }
    if requested_style == "auto":
        chosen = str(value.get("style") if isinstance(value, dict) else "").strip().lower()
        result["style"] = chosen if chosen in STYLES else "photoreal"
        if not negative:
            result["negativePrompt"] = negative_template(result["style"], visible_text)
    return result


def needs_concept(prompt: str) -> bool:
    return len(re.findall(r"[\w'-]+", str(prompt))) <= 6


def concept_messages(prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": (
            "You invent concrete visual scenes from vague subjects. Output exactly one "
            "plain sentence and no preamble. Specify identity or design, action, place, "
            "time or weather, and one meaningful prop or story clue. Use no medium, style, "
            "camera, quality, or generic mood language. Preserve subject count and constraints."
        )},
        {"role": "user", "content": f"Subject: {prompt.strip()}"},
    ]


def enhancement_messages(prompt: str, style: str, concept: str = "") -> list[dict[str, str]]:
    visible_text = requested_visible_text(prompt)
    if style != "auto" and style not in STYLES:
        raise ValueError(f"unknown_style:{style}")
    template = negative_template(style, visible_text) if style != "auto" else COMMON_NEGATIVE
    text_instruction = ""
    if visible_text:
        copies = ", ".join(f'"{copy}"' for copy in visible_text)
        text_instruction = (
            f"Exact visible text requested: {copies}\n"
            "Preserve that copy exactly and describe its typography, placement, material, and legibility.\n"
        )
    return [
        {"role": "system", "content": AUTO_SYSTEM_INSTRUCTION if style == "auto" else SYSTEM_INSTRUCTION},
        {"role": "user", "content": (
            (("Choose exactly one of these styles: " + ", ".join(STYLES)) if style == "auto"
             else f"Chosen style: {style.upper()}") +
            f"\nUser prompt: {prompt.strip()}\n"
            + (f"Creative scene concept: {concept.strip()}\n"
               "Use every compatible concrete fact from this concept.\n" if concept else "") +
            text_instruction +
            (("After choosing, apply its matching style directive and no other style.\n")
             if style == "auto" else
             f"Required style directive: {STYLE_DIRECTIVES[style]}\nApply this chosen style and no other style.\n") +
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
                     resolved_negative=None) -> dict[str, str]:
        if resolved and style != "auto":
            return {"prompt": str(resolved).strip(),
                    "negativePrompt": str(resolved_negative or "").strip()}
        if not enabled and style != "auto":
            return {"prompt": prompt.strip(), "negativePrompt": ""}
        if style not in STYLES and style != "auto":
            raise ValueError(f"unknown_style:{style}")
        self.warm()
        visible_text = requested_visible_text(prompt)
        concept = ""
        if needs_concept(prompt):
            concept = self._generate_text(
                concept_messages(prompt), max_new_tokens=CONCEPT_TOKENS,
                seed=sampling_seed(prompt, "concept"),
            ).strip().strip('"')
        messages = enhancement_messages(prompt, style, concept)
        generated_text = self._generate_text(
            messages, max_new_tokens=MAX_NEW_TOKENS,
            seed=sampling_seed(prompt, style),
        )
        if not generated_text:
            raise RuntimeError("prompt_enhancement_empty")
        parsed = parse_enhancement(
            generated_text, fallback_prompt=prompt,
            fallback_negative=(negative_template(style, visible_text)
                               if style != "auto" else ""),
            visible_text=visible_text,
            requested_style=style,
        )
        if not enabled:
            parsed["prompt"] = prompt.strip()
        elif resolved:
            parsed["prompt"] = str(resolved).strip()
            if resolved_negative is not None:
                parsed["negativePrompt"] = str(resolved_negative).strip()
        return parsed

    def enhance(self, prompt: str, style: str, *, enabled=True, resolved=None) -> str:
        """Compatibility API for callers that only need the positive prompt."""
        return self.enhance_pair(
            prompt, style, enabled=enabled, resolved=resolved,
        )["prompt"]

    def unload(self):
        self.tokenizer = None
        self.model = None
        # Drop large tensor storage before FLUX starts. This matters most on
        # unified-memory Macs, where CPU and MPS allocations share one pool.
        import gc
        gc.collect()
