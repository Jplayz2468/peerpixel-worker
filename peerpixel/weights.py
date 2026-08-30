"""Are the model files on this disk? Asked without importing the Hub.

The app has no huggingface_hub, on purpose -- it is the half that can start
before anything is installed, and it has to be able to say "the model is
already here" on the very first frame it draws. The Hub's cache layout is
stable and documented, so looking is cheaper than a subprocess and works before
there is an environment to run one in.

`download.py` uses the same answer with the real library once it exists, so
these two must agree about where the cache is.
"""
from __future__ import annotations

import os
from pathlib import Path

MODEL = "Tongyi-MAI/Z-Image-Turbo"
QUANT_MODEL = "unsloth/Z-Image-Turbo-FP8"
QUANT_FILES = ("Z-Image-Turbo-INT8.pt", "Z-Image-Turbo-text_encoder-FP8.pt")


def cache_root() -> Path:
    """The Hub cache, resolved the way the Hub resolves it."""
    told = os.environ.get("HF_HUB_CACHE")
    if told:
        return Path(told)
    home = os.environ.get("HF_HOME")
    if home:
        return Path(home) / "hub"
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "huggingface" / "hub"


def repo_dir(model: str = "") -> Path:
    return cache_root() / ("models--" + (model or MODEL).replace("/", "--"))


def cached(model: str = "") -> bool:
    """A complete-looking snapshot of this repo, rather than a half download.

    `model_index.json` is written by the Hub only for a file that finished, and
    an interrupted 15 GB fetch leaves `.incomplete` blobs and no index. That is
    the difference between "resume this" and "you already have it".
    """
    name = model or MODEL
    if os.path.isdir(name):
        return True  # PEERPIXEL_MODEL points at a local copy
    try:
        base = any((repo_dir(name) / "snapshots").glob("*/model_index.json"))
        quant = repo_dir(QUANT_MODEL) / "snapshots"
        return base and all(any(quant.glob(f"*/{filename}")) for filename in QUANT_FILES)
    except OSError:
        return False
