"""Qwen3 1.7B prompt polishing with the network's exact instruction."""
from __future__ import annotations

import json
import hashlib
import os
import re
from pathlib import Path

MAX_NEW_TOKENS = 192

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
    "anime": "photorealism, 3D CGI, muddy colors, coarse ink, flat retro cel shading, dull eyes",
    "vector": "3D rendering, gradients, depth shading, painterly texture, rough edges",
    "cinematic": "flat lighting, amateur framing, oversaturated HDR, artificial plastic texture",
    "watercolor": "3D CGI, glossy digital rendering, hard vector edges, opaque plastic color",
    "illustration": "photorealism, anime cel shading, glossy digital painting, vector art, clean geometric fills",
    "pixel_art": "antialiasing, smooth gradients, vector curves, 3D rendering, inconsistent pixel scale",
}


class PerRowTemperature:
    """Apply independent sampling temperatures to one batched generation."""

    def __init__(self, temperatures):
        self.temperatures = [max(0.05, min(2.0, float(value))) for value in temperatures]

    def __call__(self, _input_ids, scores):
        import torch
        if scores.shape[0] != len(self.temperatures):
            raise ValueError("temperature_batch_mismatch")
        scale = torch.tensor(self.temperatures, device=scores.device,
                             dtype=scores.dtype).unsqueeze(1)
        return scores / scale

SYSTEM_INSTRUCTION = """You are an expert prompt optimizer for FLUX image models.
Output ONLY one valid compact JSON object with exactly two string fields: "prompt" and "negative_prompt". No markdown, preamble, or additional keys. Write the prompt as 1–2 dense sentences of concrete image-model instructions. Write negative_prompt as a concise comma-separated list of concrete visual defects or unwanted elements specifically relevant to this image; never negate requested subjects or visible text.

Behavior:
- If the user's prompt is simple, short, or underspecified: invent a coherent, original visual concept rather than only decorating the given words. Describe literal, externally visible facts: the subject's complete appearance or design, clothing and accessories when applicable, exact action or pose, setting and background elements, spatial composition, lighting direction and quality, weather or time, colors, materials, and surface textures. Make bold but plausible visual choices that turn inputs such as "a man" into a fully specified scene. Do not merely restate the subject and append the style directive.
- For an underspecified prompt, silently decide concrete non-style facts for WHO or WHAT specifically, doing WHAT, WHERE, WHEN or under what conditions, and how every major visible part looks. Generic adjectives and medium markers do not count. Add props only when they clarify the requested action or composition. Never add a symbolic keepsake, tragic history, nostalgic detail, or implied personal drama just to make the image feel meaningful.
- If the user's prompt is already detailed: Retain every subject, count, color, action, object, relationship, and constraint, improve visual organization, and fill only genuinely unspecified visible details.
- Be visually literal and concrete. Do not invent emotions, inner life, symbolism, allegory, nostalgia, sentiment, psychological interpretation, or themes such as hope, loneliness, memory, mystery, resilience, or "the essence of" anything unless the user explicitly requests them. Never explain what the image represents or what emotional weight it carries.
- Do not use similes, poetic comparisons, figurative language, or phrases such as "captures," "evokes," "symbolizes," "shimmers like," or "a sense of." State only what can be seen.
- Prefer dense, useful image-model tokens over prose. Name observable geometry, placement, materials, colors, light, camera or medium, and production technique. Describe the image itself; do not review, interpret, or praise it.
- Preserve explicit constraints and subject count. Do not add extra people, text, logos, brands, or named characters unless requested, and never contradict specified traits.
- When visible text is explicitly requested on a sign, poster, cover, label, screen, garment, title, caption, or similar surface: reproduce its exact spelling and capitalization in double quotes. Describe the physical surface plus typography, placement, material, contrast, and legibility so FLUX treats the words as part of the composition. Never paraphrase, translate, extend, or invent copy. Spoken dialogue is not visible image text unless the user requests a speech bubble, subtitle, or caption.
- Develop the scene concept before applying the one required style supplied with the request. Never borrow characteristics from a different style."""

AUTO_SYSTEM_INSTRUCTION = SYSTEM_INSTRUCTION.replace(
    'exactly one string field: "prompt"',
    'exactly two string fields: "style" and "prompt"',
) + """
- Choose exactly one style from: photoreal, anime, vector, cinematic, watercolor, illustration, pixel_art. Put its lowercase name in "style". Choose the style that best serves the subject and concept; do not default mechanically to photoreal."""

STYLES = ("photoreal", "anime", "vector", "cinematic", "watercolor",
          "illustration", "pixel_art")
STYLE_DIRECTIVES = {
    "photoreal": "Direct a believable photographic capture. Choose and state an appropriate camera body or film stock, focal length, aperture or depth of field, shot distance, camera angle, composition, light source and direction, exposure character, and color response. Specify skin, fabric, hair, material, weather, and background textures with unretouched natural detail; avoid generic 'high quality' language.",
    "anime": "Direct a polished contemporary digital anime illustration. Always specify fine tapered linework, soft layered shading, pastel color harmony, bright atmospheric rim light, high-key bloom, and crisp forms beneath the soft light. For an existing character or face only, add clean facial construction, large luminous gradient eyes with layered iris reflections, detailed hair strands with glossy shaped highlights, and gentle skin blush. Use shallow-depth foreground and background bokeh only where compositionally useful. Never add a face, person, hair, or eyes merely to demonstrate the style; zero photorealism or 3D CGI.",
    "vector": "Direct flat 2D graphic vector art. Specify geometric silhouette construction, consistent stroke weight or no-stroke treatment, exact limited color palette, solid color planes, negative-space design, alignment and spacing, icon or poster composition, and a pure solid background; zero gradients, volumetric light, texture, 3D depth, or painterly effects.",
    "cinematic": "Direct a dramatic live-action cinema still. State aspect ratio, shot size, camera angle and movement implication, an appropriate cinema camera or film stock, focal length and lens character, aperture or depth of field, motivated key and practical lighting, production design, atmospheric depth, exposure, and film color grade. Keep every detail physically filmable and visually literal.",
    "watercolor": "Direct a traditional watercolor painting on a named cold-press or rough paper texture. Specify transparent pigment palette, wet-on-wet washes, wet-on-dry edges, granulation, blooms, glazing, reserved paper highlights, brush scale, value structure, and where edges dissolve or stay crisp; avoid digital, photographic, vector, or 3D terms.",
    "illustration": "Direct an expressive hand-drawn charcoal, graphite, or dry-ink illustration. Specify loose scribbled contour lines, varied hand pressure, gestural construction marks, broad charcoal blocks, smudged tonal shading, cross-hatching, toothy paper grain, dust and broken edges, erased highlights, and selective dark accents. Keep it visibly handmade, imperfect, and tactile; zero anime cel, glossy digital paint, vector, photoreal, or 3D language.",
    "pixel_art": "Direct handcrafted pixel art at a stated sprite or scene resolution and pixel scale. Specify crisp pixel clusters, readable silhouette, limited indexed color palette, tile or sprite proportions, selective outlines, deliberate dithering pattern, one-pixel highlights, and layer separation; zero smoothing, antialiasing, vector curves, or subpixel detail.",
}

STYLE_SUFFIXES = {
    "photoreal": "Kodak Portra 400 film response, 85mm photographic lens, f/2.8 natural depth of field, directional available light, unretouched authentic material texture",
    "anime": "contemporary digital anime, fine tapered linework, soft layered shading, pastel color harmony, atmospheric rim light, crisp forms with high-key bloom",
    "vector": "flat 2D vector construction, uniform geometric silhouettes, limited solid-color palette, controlled negative space, pure untextured background, zero gradients",
    "cinematic": "2.39:1 live-action cinema still, ARRI Alexa 35, 40mm anamorphic lens, motivated practical lighting, atmospheric depth, restrained filmic color grade",
    "watercolor": "traditional transparent watercolor, cold-press paper tooth, wet-on-wet washes, pigment granulation and blooms, reserved paper highlights, expressive edge variation",
    "illustration": "expressive charcoal and graphite drawing, loose scribbled contour lines, gestural construction marks, smudged tonal blocks, toothy paper grain, erased highlights",
    "pixel_art": "handcrafted low-resolution pixel art, crisp pixel clusters, limited indexed palette, selective one-pixel outlines, deliberate dithering, zero antialiasing",
}


def with_style_suffix(prompt: str, style: str) -> str:
    try:
        suffix = STYLE_SUFFIXES[style]
    except KeyError:
        raise ValueError(f"unknown_style:{style}") from None
    base = str(prompt or "").strip().rstrip(".!?,;:")
    return f"{base}, {suffix}."


_TEXT_SURFACE = (
    r"(?:sign|banner|poster|billboard|marquee|label|caption|subtitle|speech bubble|"
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
        wrapper = re.match(r'^\s*\{\s*"prompt"\s*:\s*"', prompt)
        if wrapper:
            prompt = prompt[wrapper.end():]
            prompt = re.sub(r'"\s*\}\s*$', "", prompt)
            prompt = prompt.replace(r'\"', '"').replace(r'\n', ' ')
    if prompt.startswith("{"):
        try:
            nested = json.loads(prompt)
            if isinstance(nested, dict) and nested.get("prompt"):
                prompt = str(nested["prompt"]).strip().strip('"')
        except (json.JSONDecodeError, TypeError):
            pass
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


def enhancement_messages(prompt: str, style: str) -> list[dict[str, str]]:
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
    if style == "auto":
        style_instruction = (
            "Choose exactly one supported style and put its lowercase name in the style field.\n"
            "Style directives:\n" + "\n".join(
                f"- {name}: {directive}" for name, directive in STYLE_DIRECTIVES.items()
            ) + "\nApply only the chosen directive. Include at least four applicable, concrete "
            "style-technique phrases in the positive prompt. Never invent a subject or object just "
            "to demonstrate a technique."
        )
        chosen = "Choose one style from: " + ", ".join(STYLES)
    else:
        chosen = f"Chosen style: {style.upper()}"
        style_instruction = (
            f"Required style directive: {STYLE_DIRECTIVES[style]}\n"
            "Include at least four applicable, concrete style-technique phrases from this directive "
            "in the positive prompt; for photoreal or cinematic, camera or film stock and focal "
            "length are mandatory. Never invent a subject or object just to demonstrate a technique. "
            "Apply this chosen style and no other style."
        )
    return [
        {"role": "system", "content": AUTO_SYSTEM_INSTRUCTION if style == "auto" else SYSTEM_INSTRUCTION},
        {"role": "user", "content": (
            f"{chosen}\nUser prompt: {prompt.strip()}\n" +
            text_instruction +
            "Output an aligned negative_prompt for this exact image concept.\n"
            f"Never introduce anything contradicted by these exclusions: {template}\n"
            "FINAL AND HIGHEST-PRIORITY REQUIREMENT:\n" + style_instruction
        )},
    ]


def bootstrap_messages(prompt: str) -> list[dict[str, str]]:
    """The disposable SFT contract: one raw prompt in, one plain prompt out."""
    return [{"role": "user", "content": str(prompt or "").strip()}]


class PromptEnhancer:
    def __init__(self, model_path=None, adapter_path=None):
        self.model_path = model_path
        configured = adapter_path if adapter_path is not None else os.environ.get(
            "PEERPIXEL_PROMPT_ADAPTER")
        self.adapter_path = Path(configured).expanduser() if configured else None
        self._adapter_manifest = None
        self.tokenizer = None
        self.model = None

    @property
    def provenance(self) -> str:
        return str((self._adapter_manifest or {}).get("version") or "qwen3-1.7b")

    def warm(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from . import model_hub

        if self.adapter_path:
            from .lora_manifest import load_manifest
            manifest = load_manifest(self.adapter_path / "manifest.json")
            evaluation = manifest.get("evaluation", {})
            if (manifest.get("kind") not in {"bootstrap", "preference"}
                    or evaluation.get("passed") is not True):
                raise ValueError("prompt adapter has not passed evaluation")
            self._adapter_manifest = manifest
        path = self.model_path or model_hub.ensure("qwen3-1.7b")
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        # CUDA workers release FLUX before enhancement, so the small language
        # model can use the GPU without competing for VRAM. Non-CUDA workers
        # retain the portable CPU path (especially important on unified-memory
        # Apple systems, where sharing with FLUX can make decode swap).
        device_map = "cuda" if torch.cuda.is_available() else "cpu"
        base = AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True, device_map=device_map)
        if self.adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(base, self.adapter_path)
        else:
            self.model = base

    def _generate_text(self, messages, *, max_new_tokens: int, seed: int) -> str:
        import torch

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            output = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            )
        generated = output[0][inputs.input_ids.shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def enhance_pairs_batch(self, prompt: str, *, count: int = 4,
                            sampling: dict | None = None,
                            mode: str = "broad",
                            on_progress=None) -> list[dict[str, str]]:
        """Explore several prompt directions in one model pass.

        Sampling policy comes from the coordinator, so changing exploration
        strength never requires updating a worker.
        """
        import torch

        count = max(1, min(4, int(count)))
        policy = sampling or {}
        supplied = policy.get("temperatures", [policy.get("temperature", 0.9)] * count)
        if not isinstance(supplied, (list, tuple)) or len(supplied) != count:
            raise ValueError("temperature_batch_mismatch")
        temperatures = [max(0.05, min(2.0, float(value))) for value in supplied]
        top_p = max(0.1, min(1.0, float(policy.get("topP", 0.95))))
        repetition_penalty = max(1.0, min(1.3, float(policy.get("repetitionPenalty", 1.0))))
        max_tokens = max(32, min(MAX_NEW_TOKENS, int(policy.get("maxNewTokens",
                                                                policy.get("maxTokens", MAX_NEW_TOKENS)))))
        self.warm()
        # The Discord product has one learned contract: raw prompt in, enhanced
        # prompt out. There is no style classifier or style instruction layer.
        messages = bootstrap_messages(prompt)
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        width = inputs.input_ids.shape[-1]
        visible_text = requested_visible_text(prompt)
        pairs = []
        base_seed = sampling_seed(prompt, mode)
        for index, temperature in enumerate(temperatures):
            with torch.random.fork_rng():
                torch.manual_seed(base_seed + index * 104729)
                output = self.model.generate(
                    **inputs, max_new_tokens=max_tokens, do_sample=True,
                    temperature=temperature, top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )[0]
            generated = self.tokenizer.decode(output[width:], skip_special_tokens=True).strip()
            invalid = (not generated or generated.startswith(("[", "```"))
                       or generated.lower().startswith(("here is", "enhanced prompt", "prompt:")))
            if invalid:
                generated = prompt.strip()
            pairs.append(parse_enhancement(
                generated, fallback_prompt=prompt, fallback_negative=COMMON_NEGATIVE,
                visible_text=visible_text))
            if on_progress is not None:
                on_progress(index + 1)
        return pairs

    def enhance_batch(self, prompt: str, *, count: int = 4,
                      sampling: dict | None = None, mode: str = "broad") -> list[str]:
        return [pair["prompt"] for pair in self.enhance_pairs_batch(
            prompt, count=count, sampling=sampling, mode=mode)]

    def enhance_pair(self, prompt: str, style: str, *, enabled=True, resolved=None,
                     resolved_negative=None) -> dict[str, str]:
        if self.adapter_path and enabled and not resolved:
            self.warm()
            visible_text = requested_visible_text(prompt)
            generated = self._generate_text(
                bootstrap_messages(prompt), max_new_tokens=MAX_NEW_TOKENS,
                seed=sampling_seed(prompt, "bootstrap"),
            ).strip()
            invalid = (not generated or generated.startswith(("{", "[", "```"))
                       or generated.lower().startswith(("here is", "enhanced prompt", "prompt:")))
            if invalid:
                generated = prompt.strip()
            return parse_enhancement(
                generated, fallback_prompt=prompt, fallback_negative=COMMON_NEGATIVE,
                visible_text=visible_text,
            )
        if resolved and style != "auto":
            return {"prompt": str(resolved).strip(),
                    "negativePrompt": str(resolved_negative or "").strip()}
        if not enabled and style != "auto":
            return {
                "prompt": with_style_suffix(prompt, style),
                "negativePrompt": negative_template(
                    style, requested_visible_text(prompt),
                ),
            }
        if style not in STYLES and style != "auto":
            raise ValueError(f"unknown_style:{style}")
        self.warm()
        visible_text = requested_visible_text(prompt)
        messages = enhancement_messages(prompt, style)
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
        chosen_style = parsed.get("style", style)
        parsed["prompt"] = with_style_suffix(parsed["prompt"], chosen_style)
        parsed["negativePrompt"] = negative_template(chosen_style, visible_text)
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
