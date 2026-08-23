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

from . import api, compare, config, console, plans, preview, relay, settings
from .console import DIM, OFF, clock, say, step_line
from .system_status import SystemStatus

#: What this install speaks, and it must match `PROTOCOL_VERSION` in the
#: server's `public/generation-policy.mjs`.
#:
#: Version 1 rendered 512px and posted results over HTTP; it knew nothing of
#: operations, transient previews or reference images. Version 2 rendered the
#: step-distilled checkpoint: four steps, no guidance. Version 3 was a 128px
#: preview and a 512px final, conditioned on the preview handed back over the
#: socket. Version 5 renders a final from its seed alone: nothing is sent back,
#: and a version-4 worker would sit waiting forty-five seconds for conditioning
#: bytes that are never coming and then fail the job. Version 6 adds pinned
#: style recipes, optional Qwen enhancement and mandatory moderation evidence.
#: The server therefore
#: gives work only to the current version, and an old install sits connected,
#: idle and unpaid until it is updated.
PROTOCOL_VERSION = 7

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
    """What the dispatcher paid for a preview it has just relayed.

    Delivery to the browser is the completion boundary, so by the time this
    answer arrives the money has already moved; this only reads back how much,
    for the local earnings line. A missing answer is worth no wait: the ledger
    is the truth and the next refresh will show it.
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


#: How much a new timing counts against everything before it. Low, because a
#: job that happened to wait on a cold cache should not convince the next one
#: that this machine is slow.
LEARN = 0.35


def seconds_per_step(operation: str) -> float:
    """How long one step of this kind of job takes on this machine.

    Remembered per operation and across runs, because the whole point is that
    the bar on somebody's *second* render is right from its first frame. A step
    of a 1024px master and a step of a 256px preview are different amounts of
    work by a factor of sixteen, so one number for both would be wrong for
    each.

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
        raise SystemExit("this machine is not paired yet - run: peerpixel pair CODE")

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
           + f"/api/device/connect?protocol={PROTOCOL_VERSION}")
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
    prompt[0] = job["prompt"]

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

    heading = f"{operation}  {DIM}{job['prompt'][:60]}{OFF}"
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
            bar.begin("render", detail=job["prompt"][:80])
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
                                "runtimeVersion": "peerpixel-worker/0.8.6"}]}
            elif hasattr(renderer, "generate_job"):
                jpeg, evidence = renderer.generate_job(job, **render_options)
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
            if settings.keep_last():
                try:
                    preview.save(jpeg)
                except OSError:
                    pass  # a full disk costs a thumbnail, not a render

            if operation == "auxiliary_verify":
                api.submit_auxiliary(job["id"], observed_digest)
                link.send(json.dumps({"type": "finished", "jobId": job["id"]}))
            elif operation == "verify":
                subject = api.verify_asset(job["id"], "subject")
                measurements = compare.compare(subject, jpeg)
                measurements["image"] = base64.b64encode(jpeg).decode()
                api.submit_verification(job["id"], measurements)
                link.send(json.dumps({"type": "finished", "jobId": job["id"]}))
            elif operation == "upscale":
                link.send(relay.encode({
                    "type": "upscale_result", "jobId": job["id"], **evidence,
                }, jpeg))
                earned = await_settlement(link, job["id"])
            elif job.get("transient"):
                # A preview has no permanent home. It goes back down this
                # socket and the dispatcher relays it straight to the browser
                # waiting for it.
                if len(jpeg) > relay.MAX_RESULT_BYTES:
                    raise RuntimeError(f"the preview is {len(jpeg)} bytes, over the "
                                       f"{relay.MAX_RESULT_BYTES} limit")
                link.send(relay.encode({
                    "type": "draft_result", "draftId": job["id"], **evidence,
                }, jpeg))
                earned = await_settlement(link, job["id"])
            else:
                result = api.submit_result(job["id"], relay.encode({
                    "type": "master_result", "jobId": job["id"], **evidence,
                }, jpeg))
                earned = result.get("earnedCredits", 0) or 0
                link.send(json.dumps({"type": "finished", "jobId": job["id"]}))
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
