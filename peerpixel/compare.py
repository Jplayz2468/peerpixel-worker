"""Comparing two renders of the same thing.

The operator's machine renders a job a second time, with the same prompt, seed
and reference image, and this decides how far apart the two pictures are. It
reports numbers and never a verdict: the thresholds live on the server so they
can be changed without every machine on the network needing an update.

Two measurements, because each catches what the other misses.

**dHash** asks, for a small greyscale copy, whether each cell is brighter than
the one to its right, and counts how many of those 64 answers differ. It is
blind to brightness and contrast shifts and to the last-bit disagreements two
different GPUs always have, which is exactly what should be ignored here.

**RMSE** on a downscaled copy is the blunt instrument that catches what a hash
cannot: a solid colour, or a cached blob, whose hash may land near the
reference by coincidence but whose pixels are nowhere near it.

Pillow and the standard library only. No numpy, so this runs anywhere the
worker does.
"""
from __future__ import annotations

import io
import math

#: The comparison is done at a fixed size so two machines' JPEG settings and
#: any resolution difference cannot influence the numbers.
COMPARE_SIZE = 256
HASH_SIZE = 8


def _load(data: bytes):
    from PIL import Image

    return Image.open(io.BytesIO(data)).convert("RGB")


def dhash(image) -> int:
    """A 64-bit difference hash, as an integer."""
    from PIL import Image

    small = image.convert("L").resize((HASH_SIZE + 1, HASH_SIZE), Image.LANCZOS)
    pixels = list(small.getdata())
    bits = 0
    for row in range(HASH_SIZE):
        for column in range(HASH_SIZE):
            left = pixels[row * (HASH_SIZE + 1) + column]
            right = pixels[row * (HASH_SIZE + 1) + column + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def rmse(first, second) -> float:
    """Root mean squared error over a downscaled greyscale pair, 0-255."""
    from PIL import Image

    size = (COMPARE_SIZE, COMPARE_SIZE)
    left = list(first.convert("L").resize(size, Image.LANCZOS).getdata())
    right = list(second.convert("L").resize(size, Image.LANCZOS).getdata())
    total = sum((x - y) ** 2 for x, y in zip(left, right))
    return math.sqrt(total / len(left))


def compare(subject: bytes, reference: bytes) -> dict:
    """How far apart these two pictures are.

    Raises rather than guessing if either will not decode: a missing number is
    treated as "no comparison happened" by the server, which is the safe
    reading. Returning a zero here would score a broken check as a perfect
    match and wave a bad render through.
    """
    first = _load(subject)
    second = _load(reference)
    return {
        "distance": hamming(dhash(first), dhash(second)),
        "rmse": round(rmse(first, second), 3),
    }
