"""Taking back VRAM that a PeerPixel process left behind.

A training run happens in a spawned child. If the parent is killed outright --
an update, a crash, a person closing the window -- the child never runs its
cleanup and keeps whatever it had on the card. The next start then finds a GPU
that is full and no longer has anything alive to blame, and every render fails
with an out-of-memory error that names processes nobody recognises.

So a start looks for its own leftovers and ends them.

The one rule this does not break: **only ever our own.** A machine that renders
also plays games, trains somebody's unrelated model, and drives a display. A
process is ended only when every check agrees it is a PeerPixel process, owned
by this user, and not part of the run asking the question. Anything unreadable,
unrecognised, or uncertain is left alone and reported instead, because a wrong
kill costs somebody else's work and a missed one costs a restart.
"""
from __future__ import annotations

import os
from collections.abc import Iterable

#: Our own processes name themselves; a spawned trainer is `peerpixel-training`
#: and the app runs as a module or console script. Matched against the process
#: name and its full command line, lowercased.
SIGNATURES = ("peerpixel",)

#: Below this a leftover is not what is filling a card, and ending processes for
#: a few megabytes is all risk and no reward.
WORTH_RECLAIMING = 256 * 1024 ** 2

TERMINATE_GRACE_SECONDS = 5.0


def _text(value) -> str:
    if isinstance(value, (list, tuple)):
        return " ".join(str(part) for part in value).lower()
    return str(value or "").lower()


def protected_pids(psutil_module, pid: int | None = None) -> set[int]:
    """This process and everything it descends from, which we never end."""
    current = os.getpid() if pid is None else int(pid)
    keep = {current}
    try:
        process = psutil_module.Process(current)
        for parent in process.parents():
            keep.add(parent.pid)
        for child in process.children(recursive=True):
            keep.add(child.pid)
    except Exception:  # noqa: BLE001 - an unreadable tree protects more, not less
        pass
    return keep


def reclaimable(processes: Iterable[tuple[str, int, int]], *, psutil_module,
                pid: int | None = None, minimum: int = WORTH_RECLAIMING) -> list[dict]:
    """Decide, for each GPU process, whether it is our leftover.

    Returns one record per candidate holding at least `minimum` bytes, each with
    a `reclaim` flag and the `reason` behind it, so a start can say what it did
    and what it deliberately left alone.
    """
    keep = protected_pids(psutil_module, pid)
    try:
        user = psutil_module.Process(os.getpid() if pid is None else int(pid)).username()
    except Exception:  # noqa: BLE001 - without an owner to match, match nothing
        user = None
    verdicts = []
    for name, process_id, used in processes:
        if used < minimum:
            continue
        record = {"pid": process_id, "name": name, "bytes": used, "reclaim": False}
        if process_id in keep:
            record["reason"] = "part of this run"
            verdicts.append(record)
            continue
        try:
            process = psutil_module.Process(process_id)
            owner = process.username()
            signature = f"{_text(process.name())} {_text(process.cmdline())} {_text(name)}"
        except Exception:  # noqa: BLE001 - unreadable is somebody else's business
            record["reason"] = "not readable"
            verdicts.append(record)
            continue
        if user is not None and owner != user:
            record["reason"] = f"owned by {owner}"
        elif not any(mark in signature for mark in SIGNATURES):
            record["reason"] = "not a PeerPixel process"
        else:
            record["reclaim"] = True
            record["reason"] = "PeerPixel leftover"
        verdicts.append(record)
    return verdicts


def reclaim(processes: Iterable[tuple[str, int, int]], *, psutil_module,
            pid: int | None = None, minimum: int = WORTH_RECLAIMING,
            grace: float = TERMINATE_GRACE_SECONDS) -> list[dict]:
    """End our own leftovers and report every candidate that was considered."""
    verdicts = reclaimable(processes, psutil_module=psutil_module, pid=pid, minimum=minimum)
    ending = []
    for record in verdicts:
        if not record["reclaim"]:
            continue
        try:
            process = psutil_module.Process(record["pid"])
            process.terminate()
            ending.append((record, process))
        except Exception as error:  # noqa: BLE001 - it may have exited on its own
            record["reclaim"] = False
            record["reason"] = f"could not end: {type(error).__name__}"
    if ending:
        try:
            _, alive = psutil_module.wait_procs([process for _, process in ending], timeout=grace)
        except Exception:  # noqa: BLE001 - fall through to the hard stop
            alive = [process for _, process in ending]
        for process in alive:
            try:
                process.kill()
            except Exception:  # noqa: BLE001 - already gone is the outcome we wanted
                pass
    return verdicts


def describe(verdicts: Iterable[dict]) -> list[str]:
    """Lines a person can act on, for the console and the log."""
    lines = []
    for record in verdicts:
        size = record["bytes"] / 1024 ** 3
        if record["reclaim"]:
            lines.append(f"Reclaimed {size:.1f} GB from a leftover PeerPixel process "
                         f"(PID {record['pid']}).")
        else:
            lines.append(f"Left {size:.1f} GB held by {record['name']} (PID {record['pid']}) "
                         f"alone: {record['reason']}.")
    return lines
