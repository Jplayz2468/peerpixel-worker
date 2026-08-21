"""Qwen3 0.6B prompt polishing with the network's exact instruction."""
from __future__ import annotations

SYSTEM_INSTRUCTION = """You are an expert prompt optimizer for FLUX image models.
Output ONLY the raw optimized descriptive prompt (1–2 sentences). No preamble, no quotes.

Behavior:
- If the user's prompt is simple/short: Creatively expand it with rich lighting, spatial composition, atmosphere, and textures suited to the chosen style.
- If the user's prompt is already detailed: Retain all core subjects, colors, and actions, refining flow and adding subtle medium markers.

Style Directives:
- PHOTOREAL: Direct candid 35mm photography, natural lighting, sharp depth of field, fine skin pores, authentic textures, unretouched film look.
- ANIME: Direct 1990s retro 2D anime cel aesthetic, crisp dark ink line-art, flat vibrant color fills, painted watercolor backgrounds.
- VECTOR: Direct flat 2D graphic vector art, clean geometric silhouettes, solid color planes, pure solid background, zero 3D gradients or depth."""


def enhancement_messages(prompt: str, style: str, variation=None) -> list[dict[str, str]]:
    variation_line = "" if variation is None else f"\nDraft variation seed: {variation}"
    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": (
            f"Chosen style: {style.upper()}{variation_line}\nUser prompt: {prompt.strip()}"
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
        from . import model_cache

        path = self.model_path or model_cache.ensure_directory("qwen3-0.6b")
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, device_map="auto")

    def enhance(self, prompt: str, style: str, *, enabled=True, resolved=None, variation=None) -> str:
        if resolved:
            return str(resolved).strip()
        if not enabled:
            return prompt.strip()
        if style not in ("photoreal", "anime", "vector"):
            raise ValueError(f"unknown_style:{style}")
        self.warm()
        messages = enhancement_messages(prompt, style, variation)
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=96, do_sample=False)
        generated = output[0][inputs.input_ids.shape[-1]:]
        polished = self.tokenizer.decode(generated, skip_special_tokens=True).strip().strip('"')
        if not polished:
            raise RuntimeError("prompt_enhancement_empty")
        return polished

    def unload(self):
        self.tokenizer = None
        self.model = None
