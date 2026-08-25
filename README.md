# PeerPixel Worker

Open-source volunteer renderer for [peerpixel.cc](https://peerpixel.cc). A
public job is one 1024×1024 FLUX.2 Klein 4B Base final at 50 guided steps.
Workers also serve internal 128px verification probes and on-demand Qwen3-1.7B
prompt enhancement, Falconsai safety classification, AuraSR-v2 upscaling, and
trusted verification work. Models are revision-pinned and downloaded from
Hugging Face only when needed.

## Install

Download the release zip, extract it, and launch:

| Platform | Launcher |
|---|---|
| Windows | `PeerPixel.cmd` |
| macOS | `PeerPixel.command` |
| Linux | `PeerPixel.sh` or `PeerPixel.desktop` |

The launcher installs a private Python environment, pairs the device, performs
a short realistic benchmark, and starts the auto-updating worker. It does not
require administrator access. On macOS, right-click the quarantined command and
choose Open once.

An NVIDIA GPU with at least 8 GB VRAM or an Apple-silicon Mac with at least
16 GB unified memory is supported. Recent NVIDIA cards receive most generation
work; Macs may receive few full renders but remain useful for verification and
transient upscaling. Benchmark output warns clearly when expected earnings are
likely to be negligible and reports competing VRAM processes after an OOM.

## Commands

```text
peerpixel             set up if needed, then work until stopped
peerpixel setup       guided setup
peerpixel pair CODE   link this machine
peerpixel download    fetch the image model ahead of time
peerpixel bench       run the short admission benchmark
peerpixel status      show local and network status
peerpixel doctor      diagnose and run a small test render
peerpixel update      install the newest GitHub release
```

Settings include separate free-work and private-generation opt-ins, arithmetic
precision, idle unloading, terminal colour, and API endpoint. Both content
opt-ins default off and are synchronized with the paired device. The worker
uses the platform's consistent
automatic precision for production work so output quality does not vary by
request. Apple silicon uses the published MLX 4-bit Klein Base package.

## Accounting and safety

Whatever a paid operation costs is what the assigned machine earns. A final is
paid only after moderation approves it and the server accepts it. Failed or
malformed probes earn zero. Upscale bytes are transient and never stored by the
server. Signed evidence accompanies rendering and every auxiliary model so a
trusted user can independently check sampled work.

The official protocol-12 worker does not print assigned prompts or save image
previews. Private jobs are dispatched only after the machine owner explicitly
enables them. Inference still requires the prompt and pixels to exist in worker
memory, and this project is open source, so a modified worker could inspect
assigned content. Contributors must not inspect, retain, or republish it.

The repository includes the Apache 2.0 license and third-party notices for all
distributed model components. No model weights are committed to Git; Hugging
Face's resumable cache stores them locally.
