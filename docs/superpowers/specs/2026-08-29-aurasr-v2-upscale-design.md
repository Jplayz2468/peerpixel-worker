# AuraSR-v2 Two-Stage Upscale Design

## Goal

Turn selection and resolution enhancement into two explicit Discord actions. `U1`–`U4` spends one weekly generation to produce a source-anchored, 50-step image at the existing 512-scale aspect dimensions. The completed refinement offers a separate `4× to 2K` action that spends one weekly generation and uses AuraSR-v2 to produce an exact four-times-larger permanent result.

This design spans the `peerpixel` coordinator and the separate `peerpixel-worker` runtime. The repositories share the operation names and wire contract defined here.

## User Experience

An initial or variation grid keeps its existing `U1`–`U4` and `V1`–`V4` controls. Selecting `U1`–`U4` creates a `refine` job with one output, 50 diffusion steps, and the trusted 512-scale dimensions for the grid aspect. It costs one weekly generation.

The completed refinement includes one owner-scoped button labeled `4× to 2K`. Exact four-times scaling means 512×512 becomes 2048×2048, 512×384 becomes 2048×1536, 384×512 becomes 1536×2048, 512×288 becomes 2048×1152, and 288×512 becomes 1152×2048. The label uses “2K” as friendly shorthand for the 2048-pixel long edge.

Pressing the button reserves another weekly generation and creates an `upscale` child job referencing the refined image. The interaction receives private progress. The accepted result is delivered using the refined image's visibility rules and has no recursive upscale control.

Buttons are signed, owner-bound, idempotent by Discord interaction ID, and expire seven days after refinement delivery. A replay or duplicate delivery cannot reserve another generation or create another job.

## Job and Worker Protocol

`upscale` is a first-class operation, not a render mode or hidden second stage of `refine`. It skips prompt enhancement and diffusion. The coordinator sends only trusted fields:

- job ID and assignment token;
- operation `upscale`;
- model `aurasr-v2`;
- assignment-scoped source URL;
- source image ID, SHA-256 digest, width, and height;
- exact output width and height, each four times its source dimension.

The worker advertises an independent `aurasr_v2` capability and an AuraSR estimate. A device may advertise any combination of `enhance`, `render`, and `aurasr_v2`. Missing AuraSR support never disables normal contribution. The worker rejects unsupported models, arbitrary scale factors, dimensions that do not match the authenticated source, oversized inputs, and output dimensions other than exact four-times scaling.

AuraSR-v2 dependencies and model artifacts are optional worker extras. Setup probes hardware and library support without importing or downloading AuraSR for workers that do not opt in. The model is loaded for an assigned upscale and may unload afterward so its residency does not displace normal diffusion rendering.

Source download, progress, result upload, and failure messages retain the existing device, assignment-token, lease, and authenticated-worker checks. The resulting image is stored as a normal permanent image rather than a transient export.

## Scheduling

Only connected, idle, unsuspended devices advertising `aurasr_v2` are eligible. The coordinator ranks them by:

1. the fewest non-upscale render assignments during the trailing seven days;
2. the earliest predicted AuraSR completion using the device's learned AuraSR duration or advertised cold estimate;
3. the oldest `last_assigned_at` value.

This sends upscale work toward capable devices that are not normally busy with diffusion. Once an upscale has waited for the existing renderer-starvation threshold, predicted completion takes priority over the seven-day quiet-device count, preventing indefinite delay.

AuraSR uses its own lease duration derived from learned or advertised execution time, bounded by trusted server minimum and maximum values. It does not consume an enhancer slot and never routes through a generic render-only worker.

## Progress and ETA

Progress has explicit phases: `waiting_for_upscaler`, `loading_upscaler`, `upscaling`, `encoding_export`, `uploading`, and `delivering`. The worker reports completed and total tiles during `upscaling` when the AuraSR implementation exposes tile boundaries. Phase transitions and tile counts must be monotonic and bound to the current assignment token.

The coordinator persists per-device exponentially weighted durations for model load, seconds per tile, encoding, and upload. A warm run may report zero model-load duration. Queue ETA uses the availability and advertised or learned estimates of compatible connected devices. Running ETA uses observed tile throughput when at least two tile samples exist, otherwise the device history, then a conservative cold default. Encoding, upload, and delivery estimates are appended as remaining phases.

Discord shows a progress bar, current plain-language phase, elapsed time, and remaining-time range. Low-confidence cold or queued estimates use a wider range; learned estimates use a narrower one. Percentage and displayed remaining work never move backward. If an estimate is exceeded, the bar continues to creep below completion and the copy becomes `finishing` instead of showing zero remaining. Only authoritative completion reaches 100%.

Normal refinement keeps its existing step-based progress but recalibrates its baseline to 50 steps at 512 scale rather than the former 1024-scale assumption.

## Accounting and Rewards

Admission for `refine` and `upscale` independently reserves exactly one weekly generation. A successful upscale rewards the owner of the assigned AuraSR device exactly `1.0` bonus weekly generation using an idempotent reward marker dedicated to that job. There is no prompt-enhancer split for an upscale.

A terminal compute, model-load, source-download, or upload failure retries on another compatible device up to the existing trusted attempt limit. Exhaustion refunds the upscale reservation exactly once. An authoritative moderation rejection after completed compute remains consumed, matching existing image-generation policy. Duplicate results, stale leases, replayed components, and delivery retries cannot double-charge, double-refund, or double-pay.

## Safety and Storage

The source is an already accepted PeerPixel image, so an upscale does not repeat prompt moderation. The worker returns local safety evidence for the output, and the coordinator performs authoritative image moderation before persistence and delivery. A blocked output is discarded and not added to the public feed.

Assignment-scoped source access ends when the lease changes or the job becomes terminal. The coordinator verifies the source digest and dimensions before assignment and validates output type, byte limit, exact dimensions, and safety evidence before storing the result in R2 and metadata in D1.

## Failure Recovery

Disconnects and expired leases return an upscale to the AuraSR queue with a new assignment token. Partial or stale progress is ignored. The same accepted output cannot be overwritten with different bytes. Delivery remains persisted and retryable independently of compute, so a Discord outage cannot rerun AuraSR or change accounting.

If no AuraSR device is online, progress explicitly says it is waiting for an AuraSR worker and gives a low-confidence estimate only when recent compatible-device history supports one. Cancellation before assignment refunds once; cancellation after compute begins follows the coordinator's existing accepted-job policy.

## Repository Responsibilities

The `peerpixel` repository owns schema migrations, component signing and parsing, weekly admission and reward settlement, source authorization, AuraSR worker selection, leases, progress persistence and ETA calculation, moderation, storage, Discord delivery, public copy, and coordinator tests.

The `peerpixel-worker` repository owns optional AuraSR-v2 dependency installation, support probing, capability and timing advertisement, authenticated source verification, model lifecycle, exact four-times inference, tiled progress, encoding, upload, local safety evidence, learned local timings, operator-facing progress, and worker tests.

Both repositories increment their protocol/minimum compatible versions together and document the deployment order: coordinator compatibility first, then AuraSR-capable workers, then enabling the Discord button.

## Verification

Coordinator tests cover fixed 512-scale refinement and exact four-times outputs for every aspect; signed button ownership, expiry, and replay; independent weekly reservations and refunds; capability filtering; trailing-seven-day quiet-device ranking; starvation; AuraSR leases; monotonic phase/tile progress; learned ETA and overrun behavior; authenticated source access; moderation; permanent storage and delivery; and exactly one `1.0` worker reward.

Worker tests cover optional dependency behavior, hardware support probing, capability advertisement, trusted payload validation, source digest and dimensions, AuraSR-v2 model selection, exact four-times output, tiled progress, learned phase timings, local safety evidence, upload/failure messages, model unloading, and rejection of arbitrary or oversized requests.

An end-to-end acceptance run uses a real 512×512 refinement, produces a valid 2048×2048 image, records cold and warm phase timings, confirms that the second ETA is calibrated by the first run, verifies one user charge and one worker bonus, and confirms that the original 512-scale result remains independently downloadable.
