"""Pinned, restricted loader for the small Unsloth Z-Image-Turbo artifacts."""

from __future__ import annotations

import importlib
import re

BASE_MODEL = "Tongyi-MAI/Z-Image-Turbo"
BASE_REVISION = "f332072aa78be7aecdf3ee76d5c247082da564a6"
QUANT_MODEL = "unsloth/Z-Image-Turbo-FP8"
QUANT_REVISION = "055a19b7ab875e80e6463a8d458228b26ff55915"
TRANSFORMER_FILE = "Z-Image-Turbo-INT8.pt"
TEXT_ENCODER_FILE = "Z-Image-Turbo-text_encoder-FP8.pt"

SCHEDULER_STEPS = 9
DIT_FORWARDS = 8
GUIDANCE = 0.0

TRANSFORMER_FORMAT = "unsloth_prequant_transformer_state_dict_v1"
TEXT_ENCODER_FORMAT = "unsloth_prequant_text_encoder_state_dict_v1"


def validated_state_dict(checkpoint: object, *, component: str) -> dict:
    """Return only an exactly identified Z-Image checkpoint state dictionary."""
    expected_format = (TRANSFORMER_FORMAT if component == "transformer"
                       else TEXT_ENCODER_FORMAT if component == "text_encoder" else None)
    metadata = checkpoint.get("metadata", {}) if isinstance(checkpoint, dict) else {}
    expected_scheme = "int8" if component == "transformer" else "fp8"
    valid = (expected_format is not None
             and checkpoint.get("format") == expected_format
             and isinstance(checkpoint.get("state_dict"), dict)
             and bool(checkpoint["state_dict"])
             and metadata.get("scheme") == expected_scheme
             and metadata.get("family") == "z-image"
             and metadata.get("base_model_id") == BASE_MODEL)
    if component == "text_encoder":
        valid = valid and metadata.get("component") == "text_encoder"
    if not valid:
        raise RuntimeError("invalid_z_image_checkpoint")
    return checkpoint["state_dict"]


_SAFE_GLOBALS = (
    ("torchao.dtypes.affine_quantized_tensor", "AffineQuantizedTensor"),
    ("torchao.dtypes.uintx.plain_layout", "PlainAQTTensorImpl"),
    ("torchao.dtypes.utils", "PlainLayout"),
    ("torchao.quantization.linear_activation_quantized_tensor", "LinearActivationQuantizedTensor"),
    ("torchao.quantization.quant_api", "_int8_symm_per_token_reduced_range_quant"),
    ("torchao.quantization.quant_primitives", "ZeroPointDomain"),
    ("torch.torch_version", "TorchVersion"),
)


def _load_checkpoint(path: str, *, component: str) -> dict:
    """Load quantized tensors without permitting arbitrary pickle globals."""
    import torch

    allowed = []
    for module, name in _SAFE_GLOBALS:
        try:
            allowed.append((getattr(importlib.import_module(module), name), f"{module}.{name}"))
        except (ImportError, AttributeError):
            continue
    required = {f"{module}.{name}" for module, name in _SAFE_GLOBALS}
    if {name for _value, name in allowed} != required:
        raise RuntimeError("torchao_int8_loader_unavailable")
    torch.serialization.add_safe_globals(allowed)
    return validated_state_dict(
        torch.load(path, weights_only=True, map_location="cpu"), component=component)


def _has_meta_tensors(module) -> bool:
    return any(getattr(value, "is_meta", False)
               for value in tuple(module.parameters()) + tuple(module.buffers()))


def _artifact(filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(QUANT_MODEL, filename, revision=QUANT_REVISION)


def load_transformer(*, device: str, dtype):
    from accelerate import init_empty_weights
    from diffusers import ZImageTransformer2DModel

    state = _load_checkpoint(_artifact(TRANSFORMER_FILE), component="transformer")
    config = ZImageTransformer2DModel.load_config(
        BASE_MODEL, subfolder="transformer", revision=BASE_REVISION)
    with init_empty_weights():
        model = ZImageTransformer2DModel.from_config(config)
    model.load_state_dict(state, strict=True, assign=True)
    if _has_meta_tensors(model):
        model = ZImageTransformer2DModel.from_config(config)
        model.load_state_dict(state, strict=True, assign=True)
    return model.to(device).eval()


def load_text_encoder(*, dtype):
    import torch
    import transformers
    from accelerate import init_empty_weights
    from diffusers.hooks import apply_layerwise_casting
    from diffusers.hooks.layerwise_casting import DEFAULT_SKIP_MODULES_PATTERN

    path = _artifact(TEXT_ENCODER_FILE)
    checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    state = validated_state_dict(checkpoint, component="text_encoder")
    cls = getattr(transformers, checkpoint["metadata"].get("te_class", ""), None)
    if cls is None:
        raise RuntimeError("unsupported_z_image_text_encoder")
    config = transformers.AutoConfig.from_pretrained(
        BASE_MODEL, subfolder="text_encoder", revision=BASE_REVISION)
    with init_empty_weights():
        encoder = cls(config)
    encoder.load_state_dict(state, strict=True, assign=True)
    if _has_meta_tensors(encoder):
        encoder = cls(config)
        encoder.load_state_dict(state, strict=True, assign=True)
    if callable(getattr(encoder, "tie_weights", None)):
        encoder.tie_weights()
    encoder.eval()

    skip = tuple(DEFAULT_SKIP_MODULES_PATTERN)
    skip += tuple(re.escape(name) for name in
                  (getattr(encoder, "_keep_in_fp32_modules", None) or ()))
    output = encoder.get_output_embeddings()
    inputs = encoder.get_input_embeddings()
    if output is not None and inputs is not None and output.weight is inputs.weight:
        tied_name = next((name for name, module in encoder.named_modules()
                          if module is output), None)
        if tied_name:
            skip += (rf"^{re.escape(tied_name)}$",)
    apply_layerwise_casting(
        encoder, storage_dtype=torch.float8_e4m3fn, compute_dtype=dtype,
        skip_modules_pattern=skip, skip_modules_classes=(torch.nn.Embedding,),
    )
    return encoder


def load_pipeline(*, device: str = "cuda", dtype=None):
    import torch
    from diffusers import ZImagePipeline

    if device != "cuda":
        raise RuntimeError("image_rendering_requires_cuda")
    dtype = dtype or torch.bfloat16
    pipeline = ZImagePipeline.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, dtype=dtype,
        transformer=load_transformer(device=device, dtype=dtype),
        text_encoder=load_text_encoder(dtype=dtype),
    )
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline
