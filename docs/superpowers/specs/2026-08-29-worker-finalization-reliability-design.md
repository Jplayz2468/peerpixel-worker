# Worker Finalization Reliability Design

## Goal

Make image generation recover automatically from decode, safety-classification,
composition, upload, and training failures without requiring an operator to
restart a worker.

## Evidence and Root Cause

Production initially appeared to stall at `decoding` (76%) or `safety_check`
(92%). The universal visible stall was a coordinator presentation defect: jobs
completed and their images were posted separately, but the original ephemeral
progress response was never removed. That defect is fixed in the coordinator.

The live worker inspection also exposed a separate reliability risk. One Python
process had rendered and trained for more than forty hours; its systemd cgroup
reached 22.1 GB resident memory and 1.9 GB swap. Restarting reduced the process
to 623 MB RSS and zero swap. Decode, CPU safety classification, composition,
and upload are synchronous operations that emit no liveness until they return.
Training releases known renderer references but imports its entire stack into
the long-lived generation process.

## Architecture

The installed worker becomes a supervisor plus a replaceable runtime child.
The supervisor owns update checks and process lifecycle. The runtime owns the
WebSocket, renderer, safety classifier, and one assignment at a time. A local
liveness channel carries phase pulses from runtime to supervisor. If the child
misses a fixed phase deadline, the supervisor terminates its process tree,
starts a clean runtime, and lets the coordinator's assignment-token watchdog
retry or refund the abandoned task.

LoRA training runs in a separate subprocess. Training completion, failure,
cancellation, or promotion always ends that process, releasing its CUDA
allocator, threads, datasets, and models. A promoted adapter causes a clean
generation runtime to start with only the validated adapter.

The coordinator remains the network authority for retries, refunds, and stale
result rejection. Worker deadlines restore local health; they never settle a
job themselves.

## Liveness and Deadlines

Expensive phases emit a pulse at least every five seconds containing only job
ID, phase, elapsed milliseconds, RSS, swap, and accelerator allocation. Pulses
refresh the coordinator lease without advancing progress or extending the
supervisor's fixed deadline.

Initial deadlines are intentionally above healthy measurements:

- VAE decode: 90 seconds per output.
- Safety model warm-up: 60 seconds once per runtime.
- Safety classification: 30 seconds per output.
- Four-image composition and encoding: 30 seconds.
- Result upload: 90 seconds across its existing bounded retries.
- Graceful shutdown after timeout: 10 seconds, then forced termination.

The safety classifier warms before render readiness is advertised. Warm-up
failure disables rendering for that runtime instead of accepting work that
cannot pass mandatory moderation.

## Cleanup and Recycling

Every task releases image buffers, grids, prompt embeddings, latents, and
response frames in a `finally` path, runs Python garbage collection, and clears
the active accelerator cache without unloading a healthy resident FLUX model.
Cleanup records bounded before/after memory snapshots.

The runtime recycles between assignments when swap is nonzero, RSS remains
above 90% of physical memory after cleanup, accelerator allocation remains
above the measured idle watermark, 100 renders have completed, or runtime age
reaches 24 hours. Age and task-count recycling never interrupts active work.

Training runs only after the generation child exits. The supervisor bounds the
trainer child, kills its complete process tree on timeout, and starts a fresh
generation child on success or failure.

## Diagnostics and Recovery

Structured journal lines record runtime start/stop reason, job ID, phase start
and finish, elapsed time, and bounded memory snapshots. Deadline exits use
stable reasons such as `phase_timeout:decoding`. Exceptions include only their
type and a whitespace-normalized 160-character message. Prompts, images,
credentials, assignment tokens, and configuration values are never logged.

Startup failures use exponential backoff from two to sixty seconds. A phase
timeout restarts immediately. Repeated failures leave the supervisor alive and
produce one actionable health line per attempt.

## Compatibility and Rollout

Task protocol and assignment-token semantics remain unchanged. The worker
package and runtime evidence become `1.14.8`, published as GitHub release
`v1.14.8`. After that release exists, the coordinator-required version becomes
`1.14.8`; connected 1.14.7 workers then auto-update while idle. Interactive,
desktop, and systemd launches all use the supervisor, so recovery does not
depend on an external service manager.

## Testing

- A decode child that stops pulsing is terminated and replaced.
- Safety warm-up completes before render readiness.
- Long but live finalization repeats progress without extending its deadline.
- Finalization timeouts report stable, redacted reasons.
- Training uses a distinct PID and generation resumes in a new PID after every
  terminal training outcome.
- Cleanup releases temporary references and accelerator caches.
- Swap, RSS, GPU watermark, task-count, and age thresholds recycle only between
  assignments.
- Restart backoff is bounded and resets after a healthy start.
- Worker metadata, runtime evidence, GitHub tag, and coordinator requirement
  align at `1.14.8`.
- Full worker and coordinator suites remain green.

## Non-Goals

This does not weaken moderation, change image quality, alter Discord allowance
accounting, or attempt unsafe cancellation of native threads inside a shared
process.
