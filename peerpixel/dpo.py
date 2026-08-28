"""Small, auditable weighted Direct Preference Optimization primitives."""
from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real


def _as_floats(value, name: str) -> tuple[float, ...]:
    if isinstance(value, Real):
        values = (float(value),)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = tuple(float(item) for item in value)
    else:
        raise TypeError(f"{name} must be a number or one-dimensional sequence")
    if not values:
        raise ValueError("policy, reference, and weight values must stay aligned")
    return values


def _validate_beta(beta: float) -> float:
    beta = float(beta)
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be positive and finite")
    return beta


def _torch_loss(policy_chosen, policy_rejected, reference_chosen,
                reference_rejected, beta: float, weight):
    import torch
    import torch.nn.functional as functional

    values = (policy_chosen, policy_rejected, reference_chosen,
              reference_rejected, weight)
    if not all(torch.is_tensor(value) for value in values):
        raise TypeError("policy, reference, and weight must all be tensors")
    shape = policy_chosen.shape
    if policy_chosen.ndim != 1 or any(value.shape != shape for value in values[1:]):
        raise ValueError("policy, reference, and weight values must stay aligned")
    if policy_chosen.numel() == 0:
        raise ValueError("policy, reference, and weight values must stay aligned")
    if not torch.isfinite(weight).all().item() or not (weight > 0).all().item():
        raise ValueError("weights must be positive and finite")

    logits = beta * (
        (policy_chosen - policy_rejected)
        - (reference_chosen - reference_rejected)
    )
    losses = -functional.logsigmoid(logits)
    return (losses * weight).sum() / weight.sum()


def weighted_dpo_loss(policy_chosen, policy_rejected, reference_chosen,
                      reference_rejected, beta: float, weight):
    """Return weighted sigmoid DPO loss normalized by total example weight.

    Each argument represents aligned per-example sequence log probabilities.
    Passing tensors preserves autograd; ordinary numeric sequences are also
    accepted so the contract can be tested on installations without the
    optional trainer dependencies.
    """
    beta = _validate_beta(beta)

    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and any(torch.is_tensor(value) for value in (
            policy_chosen, policy_rejected, reference_chosen,
            reference_rejected, weight)):
        return _torch_loss(policy_chosen, policy_rejected, reference_chosen,
                           reference_rejected, beta, weight)

    chosen = _as_floats(policy_chosen, "policy_chosen")
    rejected = _as_floats(policy_rejected, "policy_rejected")
    ref_chosen = _as_floats(reference_chosen, "reference_chosen")
    ref_rejected = _as_floats(reference_rejected, "reference_rejected")
    weights = _as_floats(weight, "weight")
    if len({len(chosen), len(rejected), len(ref_chosen), len(ref_rejected),
            len(weights)}) != 1:
        raise ValueError("policy, reference, and weight values must stay aligned")
    if any(not math.isfinite(item) or item <= 0 for item in weights):
        raise ValueError("weights must be positive and finite")

    weighted = 0.0
    for policy_win, policy_loss, reference_win, reference_loss, item_weight in zip(
            chosen, rejected, ref_chosen, ref_rejected, weights):
        logit = beta * (
            (policy_win - policy_loss) - (reference_win - reference_loss)
        )
        # softplus(-logit), written this way to remain stable at large margins.
        loss = max(0.0, -logit) + math.log1p(math.exp(-abs(logit)))
        weighted += item_weight * loss
    return weighted / sum(weights)
