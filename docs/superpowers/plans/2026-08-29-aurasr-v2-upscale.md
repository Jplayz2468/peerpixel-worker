# AuraSR-v2 Worker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let compatible volunteer workers opt into AuraSR-v2 jobs with exact 4× output, truthful tile/phase progress, isolated dependencies, and safe resource cleanup.

**Architecture:** AuraSR is an optional capability behind a focused `Upscaler` boundary rather than part of the FLUX renderer. The worker advertises support only after local probing, validates the coordinator's immutable job contract, reports real phases/tiles, uploads through existing authenticated APIs, and unloads AuraSR independently.

**Tech Stack:** Python 3.11+, PyTorch, Pillow, optional AuraSR runtime/model, WebSocket worker protocol, unittest.

**Spec:** `docs/superpowers/specs/2026-08-29-aurasr-v2-upscale-design.md`

## Global Constraints

- AuraSR dependencies remain optional; workers without them retain normal enhance/render capabilities.
- Accepted jobs use operation `upscale`, model `aurasr-v2`, and exact four-times dimensions.
- Never download/import AuraSR merely to advertise an unsupported capability.
- Progress phases and tile counts are monotonic and assignment-bound.
- Model memory is released after upscale without disturbing FLUX state.
- Use the project-managed `.venv/bin/python -m unittest`; the system Python lacks project test dependencies.
- Preserve unrelated working-tree changes and use `apply_patch` for edits.

---

### Task 1: Optional AuraSR support probe and dependency boundary

**Files:**
- Create: `peerpixel/upscale.py`
- Create: `tests/test_upscale.py`
- Modify: `pyproject.toml`
- Modify: `peerpixel/runtime.py`
- Modify: `tests/test_dependencies.py`

**Interfaces:**
- Produces: immutable `UpscaleSupport(available, reason, estimate_ms)`, `probe_upscale_support()`, and optional dependency group `upscale`.
- Consumes: installed modules, CUDA availability, and local configuration; performs no network request.

- [ ] **Step 1: Add failing support-probe tests**

```python
@patch("importlib.util.find_spec", return_value=None)
def test_missing_optional_runtime_disables_only_upscale(self, _find):
    self.assertEqual(probe_upscale_support(),
                     UpscaleSupport(False, "AuraSR runtime is not installed", None))

def test_optional_import_is_not_eager(self):
    self.assertNotIn("aurasr", sys.modules)
```

- [ ] **Step 2: Run `PEERPIXEL_HOME=/tmp/peerpixel-upscale-tests .venv/bin/python -m unittest tests.test_upscale tests.test_dependencies`**

Expected: FAIL because `peerpixel.upscale` and the optional group do not exist.

- [ ] **Step 3: Add the lazy boundary and opt-in dependency group**

Keep all third-party AuraSR imports inside the default backend factory. The core install and setup remain unchanged; `peerpixel setup --upscale` selects the optional group. Return an unavailable support object for missing package, unavailable CUDA, or insufficient memory rather than failing worker startup.

- [ ] **Step 4: Run focused tests and full unittest discovery**

Run the Step 2 command, then `PEERPIXEL_HOME=/tmp/peerpixel-upscale-tests .venv/bin/python -m unittest discover -s tests`.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add peerpixel/upscale.py peerpixel/runtime.py pyproject.toml tests/test_upscale.py tests/test_dependencies.py
git commit -m "feat: add optional AuraSR support boundary"
```

### Task 2: Trusted exact-4× job validation and inference

**Files:**
- Modify: `peerpixel/upscale.py`
- Modify: `peerpixel/api.py`
- Modify: `tests/test_upscale.py`
- Modify: `tests/test_discord_actions.py`

**Interfaces:**
- Consumes: assignment task fields and authenticated source download.
- Produces: `UpscaleJob.from_payload(payload)`, `Upscaler.run(job, on_phase, on_tile) -> bytes`, and output JPEG metadata.

- [ ] **Step 1: Add failing validation and backend tests**

```python
job = UpscaleJob.from_payload({"operation":"upscale", "model":"aurasr-v2",
    "sourceDigest":"ab" * 32, "sourceWidth":512, "sourceHeight":288,
    "width":2048, "height":1152, "sourceUrl":"/api/worker/source/x1"})
self.assertEqual((job.width, job.height), (2048, 1152))
with self.assertRaisesRegex(ValueError, "invalid upscale dimensions"):
    UpscaleJob.from_payload({**payload, "width":4096})
```

Use a fake backend to assert digest verification, RGB conversion, exact output dimensions, JPEG 4:4:4 encoding, and rejection of malformed/oversized sources without loading the model.

- [ ] **Step 2: Run `PEERPIXEL_HOME=/tmp/peerpixel-upscale-tests .venv/bin/python -m unittest tests.test_upscale tests.test_discord_actions`**

Expected: FAIL because job validation and inference do not exist.

- [ ] **Step 3: Implement the focused AuraSR adapter**

Validate the immutable fields before source download. Hash downloaded bytes, decode with Pillow's verification path, enforce the declared source dimensions, run the backend's AuraSR-v2 four-times method, verify the returned dimensions, and encode a high-quality 4:4:4 JPEG. Keep the backend injectable so tests never download model weights.

- [ ] **Step 4: Run focused tests and full unittest discovery**

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add peerpixel/upscale.py peerpixel/api.py tests/test_upscale.py tests/test_discord_actions.py
git commit -m "feat: run trusted AuraSR v2 four-times jobs"
```

### Task 3: Capability advertisement and upscale job routing

**Files:**
- Modify: `peerpixel/worker.py`
- Modify: `peerpixel/settings.py`
- Modify: `peerpixel/cli.py`
- Modify: `tests/test_worker_protocol.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_worker_policy.py`

**Interfaces:**
- Consumes: `probe_upscale_support()` and `Upscaler.run()`.
- Produces: connection query fields `aurasr_v2=1`, `upscaleEstimateMs`, stage `upscale` handling, and an explicit operator opt-in setting.

- [ ] **Step 1: Add failing routing tests**

```python
self.assertIn("aurasr_v2", connection_capabilities(support=UpscaleSupport(True, "ready", 90_000)))
self.assertNotIn("aurasr_v2", connection_capabilities(support=UpscaleSupport(False, "missing", None)))
self.assertEqual(route_job({"stage":"upscale"}), "upscale")
```

Also assert normal render jobs still use `Renderer`, one socket runs one job, AuraSR exceptions produce `task_failed` with stage `upscale`, and non-opted-in workers never advertise or accept upscale.

- [ ] **Step 2: Run `PEERPIXEL_HOME=/tmp/peerpixel-upscale-tests .venv/bin/python -m unittest tests.test_worker_protocol tests.test_cli tests.test_worker_policy`**

Expected: FAIL for unknown capability and stage.

- [ ] **Step 3: Add opt-in setup and isolated routing**

Persist `upscaleEnabled` only after `peerpixel setup --upscale` succeeds. Include the capability and cold estimate in the connection URL. Route stage `upscale` to the injected upscaler without warming or unloading FLUX; preserve assignment token on every progress/result/failure message.

- [ ] **Step 4: Run focused tests and full unittest discovery**

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add peerpixel/worker.py peerpixel/settings.py peerpixel/cli.py tests/test_worker_protocol.py tests/test_cli.py tests/test_worker_policy.py
git commit -m "feat: advertise and route AuraSR jobs"
```

### Task 4: Real phase/tile progress and learned timings

**Files:**
- Modify: `peerpixel/upscale.py`
- Modify: `peerpixel/job_phases.py`
- Modify: `peerpixel/plans.py`
- Modify: `peerpixel/progress.py`
- Modify: `peerpixel/worker.py`
- Modify: `tests/test_upscale.py`
- Modify: `tests/test_job_phases.py`
- Modify: `tests/test_progress.py`

**Interfaces:**
- Consumes: AuraSR backend tile callback and existing progress tracker.
- Produces: phase messages with `phase`, `completed`, `total`, `elapsedMs`; config timing keys `upscale.load`, `upscale.tile`, `upscale.encode`, `upscale.upload`.

- [ ] **Step 1: Add failing progress tests**

```python
self.assertEqual(events, [
    ("loading_upscaler", 0, 1), ("upscaling", 0, 16),
    ("upscaling", 1, 16), ("upscaling", 16, 16),
    ("encoding_export", 0, 1), ("uploading", 0, 1)])
self.assertEqual(remember_phase(1.0, 3.0, alpha=.25), 1.5)
```

Assert tile regressions are discarded, progress never reaches 100 before result acceptance, no-tile backends remain honestly time-based, and cold/warm timings are scoped to upscale.

- [ ] **Step 2: Run `PEERPIXEL_HOME=/tmp/peerpixel-upscale-tests .venv/bin/python -m unittest tests.test_upscale tests.test_job_phases tests.test_progress`**

Expected: FAIL for missing upscale plan and tile events.

- [ ] **Step 3: Add an `upscale` plan and measured callbacks**

Declare all phases before work begins. Prefer actual tile counts; otherwise use the tracker's learned time creep. Measure successful phase boundaries with `time.monotonic()`, fold them into existing timing history with alpha `.25`, and send the coordinator each monotonic sample.

- [ ] **Step 4: Run focused tests and full unittest discovery**

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add peerpixel/upscale.py peerpixel/job_phases.py peerpixel/plans.py peerpixel/progress.py peerpixel/worker.py tests/test_upscale.py tests/test_job_phases.py tests/test_progress.py
git commit -m "feat: report measured AuraSR progress"
```

### Task 5: Safety, upload, memory cleanup, and operator documentation

**Files:**
- Modify: `peerpixel/upscale.py`
- Modify: `peerpixel/safety.py`
- Modify: `peerpixel/worker.py`
- Modify: `peerpixel/system_status.py`
- Modify: `README.md`
- Modify: `THIRD_PARTY_NOTICES.txt`
- Modify: `tests/test_upscale.py`
- Modify: `tests/test_content_privacy.py`
- Modify: `tests/test_system_status.py`

**Interfaces:**
- Consumes: encoded AuraSR output and existing local image classifier/upload client.
- Produces: accepted result manifest with dimensions, digest, local safety evidence, model; guaranteed AuraSR teardown; operator status and setup documentation.

- [ ] **Step 1: Add failing completion and cleanup tests**

```python
self.assertEqual(result["model"], "aurasr-v2")
self.assertEqual((result["width"], result["height"]), (2048, 2048))
self.assertEqual(result["safety"]["label"], "normal")
self.assertTrue(fake_backend.unloaded)
```

Assert unload occurs after success, inference failure, safety failure, upload failure, and cancellation; source bytes and model internals never appear in logs or evidence.

- [ ] **Step 2: Run `PEERPIXEL_HOME=/tmp/peerpixel-upscale-tests .venv/bin/python -m unittest tests.test_upscale tests.test_content_privacy tests.test_system_status`**

Expected: FAIL for missing manifest and cleanup behavior.

- [ ] **Step 3: Finalize result and cleanup boundaries**

Run local safety before upload, attach exact dimensions/digest/model evidence, and put backend teardown plus CUDA cache release in `finally`. Document optional setup, hardware limitations reported by the probe, model download timing, progress phases, one-generation worker reward, and third-party license attribution.

- [ ] **Step 4: Run full unittest discovery and an import smoke test without upscale extras**

Run: `PEERPIXEL_HOME=/tmp/peerpixel-upscale-tests .venv/bin/python -m unittest discover -s tests`
Run: `PEERPIXEL_HOME=/tmp/peerpixel-upscale-tests .venv/bin/python -c "import peerpixel.worker; print('ok')"`
Expected: PASS and `ok`.

- [ ] **Step 5: Commit**

```bash
git add peerpixel/upscale.py peerpixel/safety.py peerpixel/worker.py peerpixel/system_status.py README.md THIRD_PARTY_NOTICES.txt tests/test_upscale.py tests/test_content_privacy.py tests/test_system_status.py
git commit -m "feat: safely complete AuraSR worker jobs"
```

