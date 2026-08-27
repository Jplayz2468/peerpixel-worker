# PeerPixel Worker

Open-source volunteer compute for [PeerPixel](https://peerpixel.cc), a Discord-first community image generator. The coordinator has one public operation: direct text-to-image generation through Discord `/imagine`.

Workers can advertise two capabilities:

- Prompt enhancement with Qwen. The coordinator favors an idle CUDA worker that already has the enhancer loaded when it is close to the fastest option, and prefers weaker rendering GPUs among otherwise similar choices.
- Image rendering with FLUX. The coordinator sends the enhanced prompt to the fastest available compatible renderer. The renderer also runs its local Falconsai safety classifier before uploading the result for authoritative server moderation.

AuraSR, browser editing, inpainting, probes, reputation checks, and distributed verification are not part of the protocol. Discord generation uses four 512-scale, 16-step images. Its U buttons refine a selected source at 1024-scale/50 steps with strength 0.30; V buttons create four source-conditioned alternatives at strength 0.45.

## Install

Download the release archive, extract it, and launch:

| Platform | Launcher |
|---|---|
| Windows | `PeerPixel.cmd` |
| macOS | `PeerPixel.command` |
| Linux | `PeerPixel.sh` or `PeerPixel.desktop` |

The launcher installs a private Python environment and downloads revision-pinned models from Hugging Face when needed. It does not require administrator access.

## Ask an administrator for a worker key

Workers cannot pair themselves. Ask a PeerPixel moderator to create a permanent worker key for your Discord account with the hidden `/peerpixel-admin` command in the moderator operations channel.

The moderator sends you the key once. Store it securely in the worker configuration and do not share it. PeerPixel stores only its SHA-256 hash. The key does not expire, but a moderator can disable or replace it at any time; replacement invalidates the old key immediately.

Configure the worker with the issued key during guided setup. The old `peerpixel pair CODE` flow and temporary pairing codes are retired.

## Commands

```text
peerpixel             set up if needed, then work until stopped
peerpixel setup       configure the administrator-issued worker key
peerpixel download    fetch retained models ahead of time
peerpixel bench       measure enhancement and rendering performance
peerpixel status      show local and coordinator status
peerpixel doctor      diagnose the installation
peerpixel update      install the newest GitHub release
```

## Protocol

Only a moderator can issue a permanent worker key. Save it once with `peerpixel pair KEY`; there is no public or automatic pairing flow. Each connection registers its accelerator, enhancement/render capabilities, loaded-model state, and timing estimates. One connection performs one task at a time.

Enhancement tasks receive prompt work only and return an enhanced prompt with minimal provenance. Render tasks receive the final enhanced prompt and must not enhance it again. Every task includes an assignment token; stale or mismatched results are rejected. A render is complete only after its upload passes byte validation, worker safety evidence, authoritative server moderation, R2 persistence, and D1 persistence.

Every completed contributed render adds one bonus generation to the owner's current weekly Discord allowance. There are no pixels, credits, payouts, or reputation scores.

The official worker does not print assigned prompts or save image previews. Inference still requires prompts and pixels in worker memory, and this project is open source, so a modified worker could inspect assigned content. Contributors must not inspect, retain, or republish it.

The repository includes the Apache 2.0 license and third-party notices for distributed model components. No model weights are committed to Git.
