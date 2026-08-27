"""Revision-pinned, resumable Hugging Face downloads for auxiliary models."""
from __future__ import annotations

MODELS = {
    "qwen3-1.7b": (
        "Qwen/Qwen3-1.7B",
        "b9352fbb8ce704292730cf54b3b1dceb2a808738",
    ),
    "nsfw-image-detection": (
        "Falconsai/nsfw_image_detection",
        "04367978d3474804ab1a00a9bd6548b741764069",
    ),
}


def ensure(name: str) -> str:
    """Return a local snapshot, downloading it only when absent from HF cache."""
    try:
        repo_id, revision = MODELS[name]
    except KeyError:
        raise ValueError(f"unknown_hf_model:{name}") from None

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return snapshot_download(
            repo_id=repo_id, revision=revision, local_files_only=True,
        )
    except LocalEntryNotFoundError:
        return snapshot_download(
            repo_id=repo_id, revision=revision, max_workers=4,
        )
