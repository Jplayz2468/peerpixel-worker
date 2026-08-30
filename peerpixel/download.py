"""Fetching the weights, out loud.

Left to itself diffusers pulls ~15 GB in complete silence on first run, which
is indistinguishable from a hang. So we fetch them first, with
`snapshot_download` - it resumes a killed download from the byte it stopped at
- and report progress by measuring the cache on disk. Disk is the only count
that stays true across a resume: bytes already there are never sent again, so
watching the wire would show the wrong total.

Nothing here is needed to render. With the cache warm it returns at once and
prints nothing.
"""
from __future__ import annotations

import fnmatch
import json
import os
import threading
from pathlib import Path

from .render import MODEL, REVISION
from .z_image import QUANT_MODEL, QUANT_REVISION, TEXT_ENCODER_FILE, TRANSFORMER_FILE
from .console import human
from .weights import repo_dir as _repo_dir_of

POLL = 0.5  # seconds between disk measurements


def _repo_dir() -> Path:
    """Where the Hub keeps this repo: <cache>/models--org--name/.

    Asked of the real library when it is here, because the library is the
    authority. `weights.py` works the same answer out from the environment for
    the app, which has no library; if these two ever disagree the app would
    offer to download something that is already on the disk.
    """
    try:
        from huggingface_hub import constants

        return Path(constants.HF_HUB_CACHE) / ("models--" + MODEL.replace("/", "--"))
    except ImportError:
        return _repo_dir_of(MODEL)


def _plan() -> tuple[list[str], int, str]:
    """Which files to fetch, how many bytes that is, and which revision.

    The repo also ships one big single-file checkpoint at its root for people
    using other tools. Taking that as well would nearly double the download for
    nothing, so ask model_index.json which components the pipeline actually has
    and fetch only those folders. That stays right if the pipeline changes.
    """
    from huggingface_hub import HfApi, hf_hub_download

    index = json.loads(Path(hf_hub_download(MODEL, "model_index.json", revision=REVISION)).read_text())
    wanted = ["model_index.json"] + [f"{name}/*" for name in index if not name.startswith("_")]
    info = HfApi().repo_info(MODEL, revision=REVISION, files_metadata=True)
    chosen = [
        f for f in info.siblings
        if any(fnmatch.fnmatch(f.rfilename, pattern) for pattern in wanted)
        and not (f.rfilename.startswith(("transformer/", "text_encoder/"))
                 and f.rfilename.endswith((".safetensors", ".bin", ".pt")))
    ]
    return [f.rfilename for f in chosen], sum(f.size or 0 for f in chosen), info.sha


def _measure(root: Path, sha: str, files: list[str]) -> int:
    """Bytes of this revision already on disk.

    A finished file shows up in snapshots/<sha>/ as a symlink into blobs/; the
    one in flight is a blobs/*.incomplete. Counting both is what lets a resumed
    download pick up its percentage instead of starting again from zero.
    """
    done = 0
    snapshot = root / "snapshots" / sha
    for name in files:
        path = snapshot / name
        if path.is_file():  # follows the symlink, so this is the real size
            done += path.stat().st_size
    blobs = root / "blobs"
    if blobs.is_dir():
        done += sum(p.stat().st_size for p in blobs.glob("*.incomplete"))
    return done


def ensure(on_phase=None, on_progress=None) -> str:
    """Make sure the weights are on disk. Returns the folder holding them.

    The two callbacks are how this reports to whatever is drawing. Both
    optional, because `ensure` is also called by things that are already
    drawing something else and only want the weights.
    """
    phase = on_phase or (lambda *a, **k: None)
    report = on_progress or (lambda *a, **k: None)
    from huggingface_hub import hf_hub_download, snapshot_download
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()  # its per-file bars would fight with the one bar

    phase("plan")
    try:
        files, total, sha = _plan()
    except Exception as error:  # noqa: BLE001
        try:
            return snapshot_download(MODEL, revision=REVISION, local_files_only=True)  # cached, merely offline
        except Exception:  # noqa: BLE001
            raise SystemExit(
                f"cannot fetch {MODEL} from huggingface.co: {error}\n"
                "if the repo is gated, accept its licence there and set HF_TOKEN"
            ) from None

    root = _repo_dir()
    snapshot = root / "snapshots" / sha
    # Presence, not byte count: a stray byte of difference must not make every
    # run announce a download it is not going to do.
    if all((snapshot / name).is_file() for name in files):
        phase("check")
        path = str(snapshot)
        for filename in (TRANSFORMER_FILE, TEXT_ENCODER_FILE):
            hf_hub_download(QUANT_MODEL, filename, revision=QUANT_REVISION)
        return path

    done = _measure(root, sha, files)
    phase("fetch", detail=f"{human(max(total - done, 0))} to fetch")
    outcome: dict = {}

    def fetch():
        try:
            outcome["path"] = snapshot_download(MODEL, revision=REVISION, allow_patterns=files, max_workers=4)
        except BaseException as error:  # noqa: BLE001 - re-raised on the main thread
            outcome["error"] = error

    # Daemon, so Ctrl-C ends the process instead of waiting on the fetch. The
    # partial files stay where they are and the next run continues from them.
    thread = threading.Thread(target=fetch, daemon=True)
    thread.start()
    try:
        while thread.is_alive():
            measured = _measure(root, sha, files)
            report(measured, total, detail=f"{human(measured)} of {human(total)}")
            thread.join(POLL)
    except BaseException:
        raise

    if "error" in outcome:
        raise outcome["error"]
    phase("check")
    for filename in (TRANSFORMER_FILE, TEXT_ENCODER_FILE):
        hf_hub_download(QUANT_MODEL, filename, revision=QUANT_REVISION)
    return outcome["path"]
