# Worker Finalization Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make finalization and training self-recovering, improve exact visible-text prompting for FLUX, and release the auto-updating worker as 1.14.8.

**Architecture:** A lightweight parent supervisor runs a replaceable generation child and watches fixed-deadline liveness pulses. Training runs in its own child process, while deterministic typography normalization sits after every enhancement path so exact copy cannot be lost by Qwen or an adapter.

**Tech Stack:** Python 3.11–3.13, multiprocessing/subprocess, psutil, PyTorch, unittest, Node.js test runner, Cloudflare Workers/Wrangler.

**Spec:** `docs/superpowers/specs/2026-08-29-worker-finalization-reliability-design.md`

## Global Constraints

- Work directly on `main`; do not create a worktree or use subagents.
- Preserve `scripts/chaos_embedding_spike.py` unchanged.
- Never log prompts, images, credentials, assignment tokens, or configuration values.
- Keep task protocol 14 and assignment-token semantics unchanged.
- Publish worker `1.14.8` before requiring `1.14.8` from the coordinator.
- Do not weaken local or authoritative image moderation.

---

### Task 1: Deterministic FLUX Typography Prompts

**Files:**
- Modify: `peerpixel/prompt_enhancer.py`
- Modify: `tests/test_styled_pipeline.py`

**Interfaces:**
- Produces: `VisibleText(copy: str, role: str, surface: str | None)` and `enforce_visible_text(prompt: str, requested: tuple[VisibleText, ...]) -> str`.
- Preserves: `requested_visible_text(prompt) -> tuple[str, ...]` as a compatibility wrapper.

- [ ] **Step 1: Write failing extraction tests** for packaging, UI screens, menus, logo wordmarks, multiple ordered strings, punctuation, and dialogue exclusion. Expected literals must preserve exact case and punctuation.
- [ ] **Step 2: Run** `.venv/bin/python -m unittest tests.test_styled_pipeline.StyledPipelineTests -v` and confirm failures arise from missed roles or missing deterministic clauses.
- [ ] **Step 3: Implement `VisibleText` extraction** with explicit surface/role patterns and stable de-duplication. Normalize only surrounding whitespace; never alter copy bytes otherwise.
- [ ] **Step 4: Write failing normalization tests** proving exact quoted copy is front-loaded, each role gets placement/size/lettering/contrast language, multiple roles stay ordered, and generated paraphrases are removed rather than retained as extra copy.
- [ ] **Step 5: Implement `enforce_visible_text`** using natural FLUX clauses such as `The centered large headline displays the exact text "OPEN LATE" in high-contrast bold sans-serif lettering`. Retain supplied font, placement, material, color, and hex phrases from the source request; add no invented copy or color.
- [ ] **Step 6: Apply normalization to every path** in `enhance_pairs_batch`, `enhance_pair`, fallback parsing, resolved prompts, and enhancement-disabled output. Remove generic `text`, `letters`, and `typography` negatives only when copy is requested; retain misspelling/duplication/cropping failures.
- [ ] **Step 7: Run the focused tests**, then the entire `tests.test_styled_pipeline` and `tests.test_discord_upload` modules.
- [ ] **Step 8: Commit** with message `Improve exact FLUX typography prompts`.

### Task 2: Finalization Liveness and Cleanup Primitives

**Files:**
- Create: `peerpixel/liveness.py`
- Create: `tests/test_liveness.py`
- Modify: `peerpixel/worker.py`
- Modify: `peerpixel/safety.py`
- Modify: `tests/test_discord_upload.py`

**Interfaces:**
- Produces: `PhasePulse(job_id, phase, elapsed_ms, rss_bytes, swap_bytes, accelerator_bytes)`.
- Produces: `PhaseLease(phase, timeout, emit, clock=time.monotonic)` with fixed `deadline`, `pulse()`, and context-manager start/finish events.
- Produces: `cleanup_after_task(renderer) -> MemorySnapshot`.

- [ ] **Step 1: Write failing `PhaseLease` tests** proving pulses repeat every five seconds, redact arbitrary exception text, and never move the fixed deadline.
- [ ] **Step 2: Run** `.venv/bin/python -m unittest tests.test_liveness -v` and confirm the missing-module failure.
- [ ] **Step 3: Implement immutable pulse/snapshot values and `PhaseLease`** without a timeout-killing thread; it reports liveness to the supervisor and coordinator while the supervisor owns termination.
- [ ] **Step 4: Write failing worker tests** for phase leases around decode, safety warm/classify, composition, and upload, including 90/60/30/30/90-second literal deadlines.
- [ ] **Step 5: Integrate phase pulses** into worker milestones and warm `SafetyClassifier` before render capability is advertised.
- [ ] **Step 6: Write failing cleanup tests** proving per-job byte collections are cleared in `finally`, `gc.collect()` runs, CUDA/MPS cache cleanup is guarded, and cleanup failures cannot replace the task result.
- [ ] **Step 7: Implement `cleanup_after_task`** and call it for every terminal task outcome.
- [ ] **Step 8: Run focused tests and commit** with message `Report and clean up finalization phases`.

### Task 3: Replaceable Runtime Supervisor

**Files:**
- Create: `peerpixel/supervisor.py`
- Create: `tests/test_supervisor.py`
- Modify: `peerpixel/cli.py`
- Modify: `peerpixel/worker.py`
- Modify: `tests/test_self_update.py`
- Modify: `tests/test_worker_protocol.py`

**Interfaces:**
- Produces: `RuntimePolicy` with phase deadlines, `max_renders=100`, `max_age_seconds=86400`, `shutdown_grace_seconds=10`, and restart backoff `(2, 4, 7, 12, 21, 37, 60)`.
- Produces: `run_supervisor(spawn_runtime, policy, clock=time.monotonic, wait=time.sleep) -> int`.
- Runtime child writes newline-delimited JSON control events to an inherited pipe; supervisor commands contain no job content.

- [ ] **Step 1: Write failing supervisor tests** using real lightweight child processes: healthy exit, frozen decode, fixed deadline despite pulses, graceful then forced termination, and immediate replacement.
- [ ] **Step 2: Run** `.venv/bin/python -m unittest tests.test_supervisor -v` and confirm missing behavior.
- [ ] **Step 3: Implement process-tree lifecycle** with process groups on POSIX and new process groups plus `taskkill /T` fallback on Windows. Never attempt to cancel a native Python thread.
- [ ] **Step 4: Write failing recycling tests** for nonzero swap, RSS above 90%, accelerator watermark, 100 completed renders, and 24-hour age, proving all recycle decisions occur only after idle events.
- [ ] **Step 5: Implement policy and bounded startup backoff**; reset backoff after the runtime emits `ready`.
- [ ] **Step 6: Add an internal runtime-child CLI entry** and make normal `peerpixel` startup enter the supervisor after update/onboarding. Preserve `--once` behavior for tests and explicit operator runs.
- [ ] **Step 7: Run supervisor, CLI, update, and worker protocol tests**, then commit with message `Supervise and recycle worker runtimes`.

### Task 4: Isolate LoRA Training

**Files:**
- Create: `peerpixel/training_process.py`
- Create: `tests/test_training_process.py`
- Modify: `peerpixel/trainer.py`
- Modify: `peerpixel/worker.py`
- Modify: `tests/test_trainer.py`
- Modify: `tests/test_worker_protocol.py`

**Interfaces:**
- Produces: `TrainingRequest` containing only lease-bound paths and identifiers.
- Produces: `run_training_child(request_path: Path) -> int` with a JSON result file written atomically.
- Supervisor control events: `training_requested`, `training_finished`, and `adapter_promoted`; no prompt or snapshot content crosses the control pipe.

- [ ] **Step 1: Write failing PID-isolation tests** proving training runs outside the generation PID and generation restarts in a third PID after success, failure, interruption, and timeout.
- [ ] **Step 2: Run focused tests** and confirm training still runs in-process.
- [ ] **Step 3: Move orchestration into `training_process.py`** while retaining existing lease verification, artifact packaging, promotion authority, progress reporting, and failure reporting.
- [ ] **Step 4: Route coordinator training signals through supervisor control**, stop the idle generation child, run one bounded trainer child, then start a clean generation child.
- [ ] **Step 5: Add process-tree termination and staging cleanup tests** for timeout and interruption, ensuring active adapters remain untouched until promotion.
- [ ] **Step 6: Run trainer, supervisor, protocol, and full worker suites**, then commit with message `Isolate LoRA training from generation`.

### Task 5: Release and Automatic Rollout

**Files:**
- Modify: `pyproject.toml`
- Modify: `peerpixel/render.py`
- Modify: `tests/test_worker_protocol.py`
- Modify in coordinator: `server/worker-protocol.mjs`
- Modify in coordinator: `test/worker-protocol.test.mjs`

**Interfaces:**
- Worker release/tag: `v1.14.8`.
- Coordinator requirement: `REQUIRED_WORKER_VERSION = "1.14.8"`.

- [ ] **Step 1: Write failing alignment tests** proving package metadata and runtime evidence are `1.14.8`, while protocol remains 14.
- [ ] **Step 2: Bump worker metadata/evidence to `1.14.8`** and run the full 290+ worker test suite plus `git diff --check`.
- [ ] **Step 3: Run a local chaos loop** with lightweight runtime children covering 100 successful task cleanups, repeated timeout replacement, and training/generation alternation; require zero orphan children and bounded parent RSS.
- [ ] **Step 4: Commit and push worker main**, create GitHub release/tag `v1.14.8`, and verify the release API exposes the correct source/artifact before changing the coordinator.
- [ ] **Step 5: Bump coordinator requirement to `1.14.8`**, run `npm test`, syntax checks, `git diff --check`, and `npx wrangler deploy --dry-run`.
- [ ] **Step 6: Commit and deploy coordinator**, verify `/api/health`, then observe the RTX worker auto-update while idle and reconnect as `1.14.8`.
- [ ] **Step 7: Run fixed-seed RTX typography fixtures** for neon sign, package, headline/subheading poster, UI screen, garment, and punctuation-heavy logo. Retain images/timings under `benchmark-results/2026-08-29-typography-1.14.8/` and visually review exact copy.
- [ ] **Step 8: Submit one live Discord smoke generation** with quoted visible text; verify the result arrives, the stale progress response disappears, and the worker returns online with zero swap.

