"""The worker loop.

Holds a WebSocket open to the dispatcher and renders whatever comes down it.
Jobs are pushed, not polled, so an idle machine costs nothing at either end.

What it draws depends on who is looking. On a terminal: a bar per job, and a
quiet spinning line the rest of the time, because a bar for waiting would be a
lie and a still screen looks like a crash. Piped to a file or a systemd
journal: plain timestamped lines, because escape codes in a journal are noise
nobody can read back. A box in a cupboard is a perfectly good peer.
"""
from __future__ import annotations

import base64
import hashlib
import json
import time

from . import api, compare, config, console, plans, relay, settings, updater
from .console import DIM, OFF, clock, say, step_line
from .system_status import SystemStatus

#: What this install speaks, and it must match `PROTOCOL_VERSION` in the
#: server's `public/generation-policy.mjs`.
#:
#: Protocol 12 is direct-only: public work is a native 1024px master, internal
#: fraud work is an explicit 128px probe, and only probes and upscales return
#: bytes over the socket. It adds explicit worker consent for private jobs and
#: hides job content in the official worker interface. Older installs stay
#: connected but ineligible until they update.
PROTOCOL_VERSION = 14

HEARTBEAT_SECONDS = 25
RECONNECT_MIN = 2
RECONNECT_MAX = 60
TICK = 1.0  # seconds waited for a message before the clock is redrawn


def should_unload_model(last_active: float, current: float, *, loaded: bool,
                        after: float | None = None) -> bool:
    """Has this machine been idle long enough to give the memory back?

    A loaded model is several gigabytes somebody may want for something else.
    Zero means never, which is the right answer for a machine that does nothing
    but render.
    """
    limit = settings.unload_seconds() if after is None else after
    return bool(loaded and limit > 0 and current - last_active >= limit)


def asked_to_stop() -> bool:
    """Has somebody asked this worker to stop between jobs?

    Read from the config rather than from a signal so that `peerpixel stop`
    from another terminal, or the launcher, can ask for it. A render already
    running is a picture somebody is waiting for and has been charged for, so
    the flag is only ever read between jobs.
    """
    return bool(config.read().get("stopAfterJob"))


def await_settlement(link, job_id: str, *, timeout: float = 30.0, clock=time.monotonic):
    """What the dispatcher paid for a socket-delivered probe or upscale.

    By the time this answer arrives the ledger has already moved; this only
    reads back how much for the local earnings line. A missing answer is worth
    no wait: the ledger is the truth and the next refresh will show it.
    """
    deadline = clock() + timeout
    while clock() < deadline:
        try:
            raw = link.recv(timeout=min(TICK, max(0.0, deadline - clock())))
        except TimeoutError:
            continue
        if not isinstance(raw, str):
            continue
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if message.get("jobId") != job_id:
            continue
        if message.get("type") == "result_accepted":
            return message.get("earnedCredits", 0) or 0
        if message.get("type") == "result_rejected":
            raise RuntimeError(message.get("reason", "the result was not accepted"))
    return 0


def notify_finished(link, job_id: str) -> bool:
    """Best-effort socket hint after the authoritative result was accepted.

    Permanent results settle over HTTP. A deployment can close the websocket
    in the few milliseconds between that commit and this hint; that must not
    turn a successfully stored image into a locally reported failure.
    """
    try:
        link.send(json.dumps({"type": "finished", "jobId": job_id}))
        return True
    except Exception:  # noqa: BLE001 - reconnect will discover the free slot
        return False


def handle_idle_control(message: dict, *, installed: str | None = None, update=None) -> bool:
    """Apply a coordinator-required update before accepting another job.

    Welcome and heartbeat replies arrive only while this loop is idle. The
    coordinator always states its minimum worker version, so a running worker
    learns about a release without polling GitHub and never interrupts a render.
    """
    if message.get("type") not in ("welcome", "ack"):
        return False
    required = str(message.get("requiredWorkerVersion") or "")
    here = updater.installed() if installed is None else installed
    if not required or not updater.newer(required, here):
        return False
    if update is None:
        from .cli import server_update
        update = server_update
    update(required)
    return True


#: How much a new timing counts against everything before it. Low, because a
#: job that happened to wait on a cold cache should not convince the next one
#: that this machine is slow.
LEARN = 0.35


def seconds_per_step(operation: str) -> float:
    """How long one step of this kind of job takes on this machine.

    Remembered per operation and across runs, because the whole point is that
    the bar on somebody's *second* render is right from its first frame. A step
    of a 1024px master and a step of a 128px probe are different amounts of
    work, so one number for both would be wrong for each.

    A machine that has never rendered this operation falls back to the
    benchmark, which is the one timed render every worker has already done.
    """
    remembered = (config.read().get("secondsPerStep") or {}).get(operation)
    try:
        if remembered and float(remembered) > 0:
            return float(remembered)
    except (TypeError, ValueError):
        pass
    return _bench_per_step()


def remember_step(operation: str, seconds: float, steps: int) -> None:
    if steps <= 0 or seconds <= 0:
        return
    measured = seconds / steps
    known = config.read().get("secondsPerStep") or {}
    try:
        before = float(known.get(operation) or 0)
    except (TypeError, ValueError):
        before = 0.0
    known[operation] = measured if before <= 0 else before * (1 - LEARN) + measured * LEARN
    config.write(secondsPerStep=known)


class Session:
    """What this run has done, for the line under the spinner."""

    def __init__(self):
        self.started = time.monotonic()
        self.idle_since = time.monotonic()
        self.images = 0
        self.pixels = 0.0

    def learn(self, operation: str, seconds: float, steps: int) -> None:
        remember_step(operation, seconds, steps)

    def idle(self) -> None:
        self.idle_since = time.monotonic()

    def line(self, state: str) -> str:
        parts = [state, console.plural(self.images, "image"), f"{self.pixels:g} px"]
        parts.append(f"up {clock(time.monotonic() - self.started)}")
        return f"  {DIM}·{OFF}  ".join(parts)


def _bench_per_step() -> float:
    """A first guess, from the one timed render every worker has already done."""
    from .render import OPERATIONS

    ms = config.read().get("benchMs")
    try:
        return max(0.02, float(ms) / 1000.0 / OPERATIONS["bench"]["steps"])
    except (TypeError, ValueError, ZeroDivisionError):
        return 1.0


def status_line(session: Session, state: str, hardware: SystemStatus) -> str:
    """The idle line combines worker state with local machine health."""
    return f"{hardware.line()} · {session.line(state)}"


def run(renderer, once: bool = False) -> int:
    saved = config.read()
    config.write(stopAfterJob=False)
    if not saved.get("token"):
        raise SystemExit("this machine needs a permanent worker key from a PeerPixel moderator")

    try:
        from websockets.sync.client import connect
    except ImportError:
        raise SystemExit("missing dependency: run `peerpixel setup`") from None

    session = Session()
    hardware = SystemStatus(renderer)
    state = ["connecting"]
    prompt = [""]

    # Loading a 4B model takes tens of seconds and says nothing while it does.
    # It gets a bar for the same reason everything else here does.
    start = plans.tracker("start")
    with console.Live(start, heading="Starting the worker", footer=hardware.line):
        start.begin("load")
        renderer.warm()
        start.note(renderer.accelerator)
        start.begin("connect")
        start.finish()
    plans.remember(start)

    url = (config.API.replace("http", "ws", 1)
           + f"/api/device/connect?protocol={PROTOCOL_VERSION}"
           + f"&edit={1 if getattr(renderer, 'supports_editing', False) else 0}"
           + f"&version={updater.installed()}")
    headers = {"authorization": f"Bearer {saved['token']}", "user-agent": api.USER_AGENT}
    backoff = RECONNECT_MIN
    link_ref: list = [None, ""]
    sent_at = [0.0]

    def status() -> str:
        return status_line(session, state[0], hardware)

    try:
        while True:
            try:
                state[0] = "connecting"
                with connect(url, additional_headers=headers, max_size=None,
                             # ping_interval=None on purpose. Cloudflare's
                             # hibernatable sockets do not answer protocol-level
                             # pings, so the library's own keepalive decides the
                             # connection is dead after about thirty seconds and
                             # closes it. The dispatcher has its own heartbeat
                             # message, sent below.
                             ping_interval=None, close_timeout=5) as link:
                    link_ref[0] = link
                    backoff = RECONNECT_MIN
                    state[0] = "online, waiting for work"
                    session.idle()
                    step_line(True, f"Connected to {config.API}",
                              renderer.accelerator)

                    last_beat = time.monotonic()
                    idle = console.Line(status)
                    idle.__enter__()
                    try:
                        while True:
                            try:
                                raw = link.recv(timeout=TICK)
                            except TimeoutError:
                                if asked_to_stop():
                                    idle.__exit__(None, None, None)
                                    step_line(True, "Stopped, as asked.")
                                    return session.images
                                if should_unload_model(
                                        session.idle_since, time.monotonic(),
                                        loaded=getattr(renderer, "pipe", None) is not None):
                                    renderer.unload()
                                if time.monotonic() - last_beat >= HEARTBEAT_SECONDS:
                                    link.send(json.dumps({"type": "heartbeat"}))
                                    last_beat = time.monotonic()
                                continue

                            try:
                                message = json.loads(raw)
                            except json.JSONDecodeError:
                                continue
                            if handle_idle_control(message):
                                continue
                            if message.get("type") != "job":
                                continue

                            idle.__exit__(None, None, None)
                            _do_job(link, message["job"], renderer, session,
                                    link_ref, sent_at, prompt, hardware)
                            last_beat = time.monotonic()
                            if once or asked_to_stop():
                                return session.images
                            state[0] = "online, waiting for work"
                            session.idle()
                            idle = console.Line(status)
                            idle.__enter__()
                    finally:
                        idle.__exit__(None, None, None)
            except KeyboardInterrupt:
                raise
            except Exception as error:  # noqa: BLE001 - every disconnect is temporary
                link_ref[0] = None
                state[0] = "offline"
                say(f"  {DIM}disconnected ({error}) - retrying in {backoff}s{OFF}")
                time.sleep(backoff)
                backoff = min(RECONNECT_MAX, int(backoff * 1.8) or 2)
    except KeyboardInterrupt:
        say()
        step_line(True, f"Stopped. {console.plural(session.images, 'image')}, "
                        f"{session.pixels:g} pixels this session.")
    return session.images


def _do_job(link, job: dict, renderer, session: Session, link_ref, sent_at, prompt,
            hardware: SystemStatus | None = None) -> float:
    """One job, start to finish, under one bar.

    Every exit from here goes back to the dispatcher: a finished result, or a
    failure with a reason. A job that simply stopped being mentioned would hold
    a slot until it timed out and leave somebody waiting for nothing.
    """
    operation = job.get("operation", "master")
    steps = int(job.get("steps", 0)) or 1
    link_ref[1] = job["id"]
    sent_at[0] = 0.0
    prompt[0] = "active generation"

    bar = plans.tracker("job", {"job.render": seconds_per_step(operation) * steps})
    started = time.monotonic()
    earned = 0.0
    from .job_phases import EXPORT_PHASES, PHASES, PhaseReporter

    reporter = PhaseReporter(
        job["id"], lambda event: link.send(json.dumps({
            **event,
            "precision": getattr(renderer, "_precision_mode", "native"),
            "memoryMode": getattr(renderer, "_memory_mode", "unknown"),
        })),
        scope=f"{operation}:{job.get('width', 0)}:{getattr(renderer, '_precision_mode', 'native')}:{getattr(renderer, '_memory_mode', 'unknown')}",
        persist=True,
        phases=EXPORT_PHASES if operation == "upscale" else PHASES,
    )
    reporter.begin("preparing")

    def stepped(done: int, total: int) -> None:
        bar.report(done, total, detail=f"step {done} of {total}")
        now = time.monotonic()
        if link_ref[0] and (done >= total or now - sent_at[0] >= 0.5):
            try:
                link_ref[0].send(json.dumps({
                    "type": "progress", "jobId": link_ref[1], "step": done, "steps": total,
                }))
            except Exception:  # noqa: BLE001 - telemetry cannot break a render
                pass
            sent_at[0] = now

    heading = f"{operation}  {DIM}job details hidden{OFF}"
    try:
        with console.Live(bar, heading=heading,
                          footer=hardware.line if hardware else None):
            if getattr(renderer, "pipe", None) is None:
                reporter.begin("loading_flux")
                bar.begin("load")
                renderer.warm()

            # Nothing to wait for. A final is its prompt and its seed, so it
            # starts the moment it is claimed -- and a browser that closed its
            # tab no longer costs somebody the render they paid for.
            bar.begin("render", detail="prompt hidden")
            render_started = time.monotonic()
            render_options = {
                "on_step": stepped,
                "on_decode": lambda: bar.begin("decode"),
                "on_demote": lambda name: bar.note(f"retrying in {name}"),
                "on_phase": reporter.begin,
            }
            observed_digest = None
            if operation == "auxiliary_verify":
                from .render import _digest
                auxiliary = job.get("auxiliaryOperation")
                if auxiliary == "prompt":
                    from .prompt_enhancer import PromptEnhancer
                    if getattr(renderer, "_enhancer", None) is None:
                        renderer._enhancer = PromptEnhancer()
                    output = renderer._enhancer.enhance(
                        job["prompt"], job.get("style", "photoreal"),
                        enabled=job.get("enhance", True),
                    )
                    observed_digest = _digest(output)
                elif auxiliary == "moderation":
                    from .safety import SafetyClassifier
                    if getattr(renderer, "_safety", None) is None:
                        renderer._safety = SafetyClassifier()
                    observed_digest = _digest(renderer._safety.classify(api.auxiliary_input(job["id"])))
                elif auxiliary == "upscale":
                    from .upscale import Upscaler
                    renderer.unload()
                    if not hasattr(renderer, "_upscaler") or renderer._upscaler is None:
                        renderer._upscaler = Upscaler()
                    observed_digest = hashlib.sha256(
                        renderer._upscaler.upscale(api.auxiliary_input(job["id"]))
                    ).hexdigest()
                else:
                    raise RuntimeError("unknown_auxiliary_operation")
                jpeg, evidence = b"", {}
            elif operation == "upscale":
                from .upscale import Upscaler
                reporter.begin("loading_upscaler")
                renderer.unload()
                if not hasattr(renderer, "_upscaler") or renderer._upscaler is None:
                    renderer._upscaler = Upscaler()
                source = api.upscale_source(job["id"])
                def upscale_progress(done, total):
                    try:
                        link.send(json.dumps({
                            "type": "upscale_progress", "jobId": job["id"],
                            "done": done, "total": total,
                        }))
                    except Exception:
                        pass
                jpeg = renderer._upscaler.upscale(
                    source, on_phase=reporter.begin, on_progress=upscale_progress)
                evidence = {"manifestVersion": job.get("manifestVersion", "2026-08-23.1"),
                            "attestations": [{"operation": "upscale",
                                "inputDigest": hashlib.sha256(source).hexdigest(),
                                "outputDigest": hashlib.sha256(jpeg).hexdigest(),
                                "runtimeVersion": "peerpixel-worker/0.14.0"}]}
            elif hasattr(renderer, "generate_job"):
                render_job = job
                if job.get("editMode"):
                    render_job = {**job, "_editSource": api.edit_asset(job["id"], "source")}
                    if job.get("hasMask"):
                        render_job["_editMask"] = api.edit_asset(job["id"], "mask")
                jpeg, evidence = renderer.generate_job(render_job, **render_options)
            else:  # small test doubles and third-party renderer integrations
                jpeg = renderer.render(job, **render_options)
                evidence = {
                    "enhancedPrompt": job["prompt"],
                    "moderation": {"label": "normal", "nsfwScore": 0.0},
                    "manifestVersion": job.get("manifestVersion", "2026-08-23.1"),
                    "recipeId": job.get("recipeId", "photoreal-v1"),
                }
            session.learn(operation, time.monotonic() - render_started, steps)
            reporter.begin("delivering")
            bar.begin("deliver")
            if operation == "auxiliary_verify":
                api.submit_auxiliary(job["id"], observed_digest)
                notify_finished(link, job["id"])
            elif operation == "verify":
                subject = api.verify_asset(job["id"], "subject")
                measurements = compare.compare(subject, jpeg)
                measurements["image"] = base64.b64encode(jpeg).decode()
                api.submit_verification(job["id"], measurements)
                notify_finished(link, job["id"])
            elif operation == "upscale":
                link.send(relay.encode({
                    "type": "upscale_result", "jobId": job["id"], **evidence,
                }, jpeg))
                earned = await_settlement(link, job["id"])
            elif operation == "probe":
                if len(jpeg) > relay.MAX_RESULT_BYTES:
                    raise RuntimeError(f"the probe is {len(jpeg)} bytes, over the "
                                       f"{relay.MAX_RESULT_BYTES} limit")
                link.send(relay.encode({
                    "type": "probe_result", "jobId": job["id"], **evidence,
                }, jpeg))
                earned = await_settlement(link, job["id"])
            else:
                result = api.submit_result(job["id"], relay.encode({
                    "type": "master_result", "jobId": job["id"], **evidence,
                }, jpeg))
                earned = result.get("earnedCredits", 0) or 0
                notify_finished(link, job["id"])
            reporter.begin("complete")
            bar.finish()
    except Exception as error:  # noqa: BLE001 - one bad job must not end the run
        say(f"  {console.RED}failed: {error}{OFF}")
        try:
            link.send(json.dumps({"type": "failed", "jobId": job["id"],
                                  "reason": str(error)[:200]}))
        except Exception:  # noqa: BLE001 - the socket is gone; the reconnect handles it
            pass
        return 0.0

    plans.remember(bar)
    session.images += 1
    session.pixels += earned
    took = time.monotonic() - started
    step_line(True, f"{operation} done in {clock(took)}",
              f"+{earned:g} px" if earned else "no payout")
    return earned


# Discord-first protocol. Kept at the bottom so installations updating across
# the clean-slate release resolve this implementation without a second entrypoint.
def _action_seeds(seed: int, count: int) -> list[int]:
    state = int(seed) & 0xffffffff or 0x9e3779b9
    result = []
    for _ in range(count):
        state ^= (state << 13) & 0xffffffff
        state ^= state >> 17
        state ^= (state << 5) & 0xffffffff
        state &= 0xffffffff
        result.append(state)
    return result


def _discord_task(link, task: dict, renderer, device_id: str) -> None:
    token = task["assignmentToken"]
    stage = task["stage"]
    if stage == "enhance":
        from .prompt_enhancer import PromptEnhancer
        enhancer = getattr(renderer, "_enhancer", None) or PromptEnhancer()
        renderer._enhancer = enhancer
        pair = enhancer.enhance_pair(task["prompt"], "auto")
        link.send(json.dumps({"type": "task_result", "taskId": task["id"],
            "stage": "enhance", "assignmentToken": token, "prompt": pair["prompt"],
            "provenance": "qwen"}))
        return

    from .safety import SafetyClassifier
    source = None
    if task.get("sourceUrl"):
        source = api.source_image(task["sourceUrl"], device_id=device_id,
                                  assignment_token=token)
    seeds = _action_seeds(task.get("seed", 0), int(task.get("outputCount", 1)))
    safety = getattr(renderer, "_safety", None) or SafetyClassifier()
    renderer._safety = safety
    rendered = []
    total_steps = max(1, int(task.get("steps", 1)) * len(seeds))
    completed_steps = 0
    for seed in seeds:
        job = {**task, "seed": seed, "operation": task.get("operation", "grid"),
               "enhance": False, "enhancedPrompt": task["prompt"]}
        if source is not None:
            job.update(editMode=task["operation"], editStrength=task["strength"],
                       sourceImageId=task.get("sourceImageId"), _editSource=source)
        def progress(done, _total, offset=completed_steps):
            link.send(json.dumps({"type": "progress", "taskId": task["id"],
                "stage": "render", "assignmentToken": token,
                "progress": min(1.0, (offset + done) / total_steps)}))
        image = renderer.render(job, on_step=progress)
        rendered.append((image, {"moderation": safety.classify(image)}))
        completed_steps += int(task.get("steps", 1))
    link.send(json.dumps({"type": "task_result", "taskId": task["id"],
        "stage": "render", "assignmentToken": token, "resultId": task["id"]}))
    # The socket result creates the upload lease. A very short retry handles
    # propagation without ever accepting a stale assignment.
    for attempt in range(5):
        try:
            api.submit_discord_result(task, device_id, rendered)
            return
        except api.ApiError as error:
            if error.status != 409 or attempt == 4:
                raise
            time.sleep(0.25 * (attempt + 1))


def run(renderer, once: bool = False) -> int:
    """Serve the compact Discord-first enhancement/render protocol."""
    from urllib.parse import quote
    saved = config.read()
    if not saved.get("token") or not saved.get("deviceId"):
        raise SystemExit("this worker needs the permanent key and device ID supplied by a PeerPixel admin")
    try:
        from websockets.sync.client import connect
    except ImportError:
        raise SystemExit("missing dependency: run `peerpixel setup`") from None
    capabilities = quote(json.dumps({"enhance": True, "render": True}, separators=(",", ":")))
    url = (config.API.replace("http", "ws", 1) + "/api/worker/connect"
           + f"?deviceId={quote(str(saved['deviceId']))}&capabilities={capabilities}"
           + f"&accelerator={quote(str(saved.get('accelerator', 'unknown')))}&enhancerLoaded=0"
           + f"&renderEstimateMs={int(saved.get('benchMs') or 0)}")
    headers = {"authorization": f"Bearer {saved['token']}", "user-agent": api.USER_AGENT}
    completed = 0
    backoff = RECONNECT_MIN
    while True:
        try:
            with connect(url, additional_headers=headers, max_size=None, ping_interval=None,
                         close_timeout=5) as link:
                backoff = RECONNECT_MIN
                step_line(True, f"Connected to {config.API}", saved.get("accelerator", "worker"))
                while True:
                    try:
                        raw = link.recv(timeout=HEARTBEAT_SECONDS)
                    except TimeoutError:
                        link.send(json.dumps({"type": "heartbeat"}))
                        continue
                    message = json.loads(raw) if isinstance(raw, str) else {}
                    if message.get("type") != "task":
                        continue
                    task = message.get("task") or {}
                    try:
                        _discord_task(link, task, renderer, str(saved["deviceId"]))
                        completed += 1
                    except Exception as error:  # one failed task must not kill the worker
                        link.send(json.dumps({"type": "task_failed", "taskId": task.get("id"),
                            "stage": task.get("stage"), "assignmentToken": task.get("assignmentToken"),
                            "reason": str(error)[:200]}))
                    if once:
                        return completed
        except KeyboardInterrupt:
            return completed
        except Exception as error:
            say(f"  {DIM}disconnected ({error}) - retrying in {backoff}s{OFF}")
            time.sleep(backoff)
            backoff = min(RECONNECT_MAX, int(backoff * 1.8) or 2)
