"""Capability-based precision selection with a safe BF16 fallback."""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util


@dataclass(frozen=True)
class Probe:
    cuda: bool
    capability: tuple[int, int] = (0, 0)
    total: int = 0
    free: int = 0
    bitsandbytes: bool = False


@dataclass(frozen=True)
class PrecisionPlan:
    mode: str
    resident: bool
    adapters: bool
    reason: str


def select_precision(probe: Probe, requested: str | None = None) -> PrecisionPlan:
    """Choose residency from capabilities, never a hostname or GPU label."""
    if not probe.cuda:
        return PrecisionPlan("native", True, True, "non-CUDA native precision")
    if not probe.bitsandbytes:
        return PrecisionPlan("unavailable", False, False,
                             "bitsandbytes unavailable; consistent NF4 images disabled")
    if probe.capability < (7, 5):
        return PrecisionPlan("unavailable", False, False,
                             "CUDA capability is below the network NF4 path")
    if probe.total >= 8_000_000_000 and probe.free >= 6_000_000_000:
        return PrecisionPlan("int4", True, False,
                             "network-standard resident NF4 weights with BF16 compute")
    return PrecisionPlan("unavailable", False, False,
                         "image generation requires 8 GB VRAM for consistent NF4 quality")


def runtime_probe(*, total: int, free: int) -> Probe:
    try:
        import torch
        capability = tuple(torch.cuda.get_device_capability(0))
    except Exception:  # noqa: BLE001 - inability to probe selects the safe path
        capability = (0, 0)
    try:
        available = importlib.util.find_spec("bitsandbytes") is not None
    except (ImportError, ValueError):
        available = False
    return Probe(cuda=True, capability=capability, total=total, free=free,
                 bitsandbytes=available)


def pipeline_quantization_config(mode: str):
    """Build matching Diffusers/Transformers configs for pipeline components."""
    import torch
    from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
    from diffusers import PipelineQuantizationConfig
    from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig

    if mode == "int8":
        transformer = DiffusersBitsAndBytesConfig(load_in_8bit=True)
        text_encoder = TransformersBitsAndBytesConfig(load_in_8bit=True)
    elif mode == "int4":
        common = dict(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                      bnb_4bit_use_double_quant=True,
                      bnb_4bit_compute_dtype=torch.bfloat16)
        transformer = DiffusersBitsAndBytesConfig(**common)
        text_encoder = TransformersBitsAndBytesConfig(**common)
    else:
        return None
    return PipelineQuantizationConfig(quant_mapping={
        "transformer": transformer,
        "text_encoder": text_encoder,
    })
