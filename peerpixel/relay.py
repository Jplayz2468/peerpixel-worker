"""The binary frame that carries image bytes over the worker socket.

    [uint32 big-endian header length][UTF-8 JSON header][payload]

Two things travel this way and nothing else does: a finished 128px draft going
back to the dispatcher, which relays it live to the browser that asked, and the
chosen draft coming the other way as the reference image for a master.

This is the exact same format as `public/relay-frame.mjs` on the server side.
If one end changes, both change: the header is routing information, and a
mismatch would mean bytes handed to the wrong candidate rather than a clean
error. Standard library only, like the rest of this package.
"""
from __future__ import annotations

import json
import struct

HEADER_BYTES = 4
#: A header is a type, an id and a length. Anything bigger is somebody trying
#: to smuggle a payload through the part that gets parsed.
MAX_HEADER_BYTES = 1024

#: The dispatcher refuses anything larger, so there is no point sending it.
MAX_RESULT_BYTES = 256 * 1024


def encode(header: dict, payload: bytes = b"") -> bytes:
    raw = json.dumps(header, separators=(",", ":")).encode()
    if len(raw) > MAX_HEADER_BYTES:
        raise ValueError("header_too_large")
    return struct.pack(">I", len(raw)) + raw + payload


def decode(frame: bytes):
    """Return (header, payload), or None for anything malformed.

    None rather than an exception: a garbled frame should cost one dropped
    message, not the worker's connection.
    """
    if not isinstance(frame, (bytes, bytearray, memoryview)):
        return None
    data = bytes(frame)
    if len(data) < HEADER_BYTES:
        return None
    (length,) = struct.unpack(">I", data[:HEADER_BYTES])
    if length == 0 or length > MAX_HEADER_BYTES:
        return None
    if len(data) < HEADER_BYTES + length:
        return None
    try:
        header = json.loads(data[HEADER_BYTES:HEADER_BYTES + length].decode())
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(header, dict):
        return None
    return header, data[HEADER_BYTES + length:]
