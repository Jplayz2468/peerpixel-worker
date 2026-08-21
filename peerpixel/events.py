"""How a working child process tells the app what it is doing.

The heavy work -- resolving dependencies, pulling 15 GB of weights, loading a
4B model, rendering -- happens in a subprocess, because that is the only way to
keep the app itself free of torch and answering its socket while a card is
saturated. So progress has to cross a pipe.

It crosses as one JSON object per line, each line prefixed with a byte no
sensible library emits. Everything without the prefix is somebody else's
output -- a torch warning, a Hugging Face notice, a traceback -- and is passed
through to the log rather than being parsed or lost. That matters more than it
sounds: the reason people stare at silent installers is that the interesting
line is on stderr and nobody kept it.

Standard library, no dependency in either direction: a child that fails to
import anything else can still say why.
"""
from __future__ import annotations

import json
import os
import sys

#: ASCII record separator. Not a character any library prints by accident, and
#: unlike a "PEERPIXEL:" prefix it cannot collide with a log line about this
#: program.
MARK = "\x1e"

#: Set by the app on every child it starts. Off means this command was run by a
#: person in a terminal, where a line of machine-readable JSON prefixed with a
#: control character is not information, it is litter. The same code path then
#: simply says nothing, and the terminal display in `ui.py` is what talks.
ENABLED = os.environ.get("PEERPIXEL_EVENTS") == "1"


def emit(event: str, **fields) -> None:
    """Say something structured. Never raises: a closed pipe is not a crash."""
    if not ENABLED:
        return
    try:
        line = MARK + json.dumps({"event": event, **fields}, separators=(",", ":"))
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
    except (OSError, ValueError, TypeError):
        pass


def parse(line: str) -> dict | None:
    """The event in this line, or None if the line is ordinary output."""
    if not line.startswith(MARK):
        return None
    try:
        value = json.loads(line[len(MARK):])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) and "event" in value else None


# The vocabulary. Small on purpose: a child that has to explain itself in more
# than these has probably grown a second job.

def phase(name: str, detail: str = "") -> None:
    """Entering a named stretch of the plan the parent already knows about."""
    emit("phase", name=name, detail=detail)


def progress(done: float, total: float = 0, detail: str = "") -> None:
    """A real measurement: bytes on disk, steps taken, files written."""
    emit("progress", done=done, total=total, detail=detail)


def note(detail: str) -> None:
    """Words for the line under the bar. Does not move it."""
    emit("note", detail=detail)


def done(**result) -> None:
    emit("done", **result)


def failed(message: str) -> None:
    emit("failed", message=message)
