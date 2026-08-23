"""Expose MFLUX's existing FLUX.2 negative-CFG branch to PeerPixel.

MFLUX 0.30 has the two-pass implementation internally but its FLUX.2 CLI
rejects ``--negative-prompt`` and substitutes an empty prompt. This runner
removes that CLI-only restriction without changing MFLUX on disk.
"""
from __future__ import annotations

import sys


def extract_negative(argv: list[str]) -> tuple[str, list[str]]:
    negative = ""
    remaining: list[str] = []
    index = 0
    while index < len(argv):
        value = argv[index]
        if value in {"--negative-prompt", "--negative"}:
            if index + 1 >= len(argv):
                raise SystemExit(f"{value} requires a value")
            negative = argv[index + 1]
            index += 2
            continue
        remaining.append(value)
        index += 1
    return negative, remaining


def main() -> None:
    negative, remaining = extract_negative(sys.argv[1:])
    sys.argv = [sys.argv[0], *remaining]

    from mflux.models.flux2.cli.flux2_generate import main as mflux_main
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

    original = Flux2Klein._encode_prompt_pair

    def encode_with_negative(self, *, prompt, negative_prompt, guidance):
        return original(
            self,
            prompt=prompt,
            negative_prompt=negative if guidance is not None and guidance > 1 else None,
            guidance=guidance,
        )

    Flux2Klein._encode_prompt_pair = encode_with_negative
    mflux_main()


if __name__ == "__main__":
    main()
