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
import json
import time

from . import api, compare, config, console, plans, preview, relay, settings
from .console import DIM, OFF, clock, say, step_line

#: What this install speaks. Version 1 rendered 512px and posted results over
#: HTTP; it knew nothing of operations, transient previews or reference images.
#: Version 2 rendered the step-distilled checkpoint: four steps, no guidance,
#: 1024px masters. Either would take a job priced for this version and hand
#: back a different picture at a different size, so the server gives work only
#: to the current version and an old install sits idle until it is updated.
PROTOCOL_VERSION = 3

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


def await_reference(link, job_id: str, *, timeout: float, clock=time.monotonic):
    """Block until the chosen preview arrives for this master, or give up.

    The dispatcher asks the browser for it the moment this job is claimed, so
    the usual wait is one round trip. A browser that has closed means there is
    no picture to render from, and the job fails and refunds rather than
    quietly producing a different image from the same words.
    """
    deadline = clock() + timeout
    while clock() < deadline:
        try:
            raw = link.recv(timeout=min(TICK, max(0.0, deadline - clock())))
        except TimeoutError:
            continue
        frame = relay.decode(raw) if not isinstance(raw, str) else None
        if not frame:
            continue
        header, payload = frame
        if header.get("type") == "conditioning" and header.get("jobId") == job_id:
            return payload
    return None


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


class Session:
    """What this run has done, for the line under the spinner."""

    def __init__(self, per_step: float):
        self.started = time.monotonic()
        self.idle_since = time.monotonic()
        self.images = 0
        self.pixels = 0.0
        #: Seconds a step takes on this machine. Seeded from the benchmark and
        #: re-measured after every render, so the second job's bar is right
        #: from its first frame instead of after enough steps to work it out.
        self.per_step = per_step

    def learn(self, seconds: float, steps: int) -> None:
        if steps > 0 and seconds > 0:
            measured = seconds / steps
            # Averaged rather than replaced: one job that waited on a cold
            # cache should not convince the next one it will be slow too.
            self.per_step = self.per_step * 0.6 + measured * 0.4 if self.per_step else measured

    def idle(self) -> None:
        self.idle_since = time.monotonic()

    def line(self, state: str) -> str:
        parts = [state, console.plural(self.images, "image"), f"{self.pixels:g} px"]
        parts.append(f"up {clock(time.monotonic() - self.started)}")
        return f"  {DIM}·{OFF}  ".join(parts)


def _per_step_seed() -> float:
    """A first guess at seconds per step, from the benchmark that already ran.

    The benchmark is four steps at master resolution, timed. That is exactly
    the number a master's bar wants, and it is already on disk.
    """
    from .render import OPERATIONS

    ms = config.read().get("benchMs")
    try:
        return max(0.02, float(ms) / 1000.0 / OPERATIONS["bench"]["steps"])
    except (TypeError, ValueError, ZeroDivisionError):
        return 1.0


def run(renderer, once: bool = False) -> int:
    saved = config.read()
    config.write(stopAfterJob=False)
    if not saved.get("token"):
        raise SystemExit("this machine is not paired yet - run: peerpixel pair CODE")

    try:
        from websockets.sync.client import connect
    except ImportError:
        raise SystemExit("missing dependency: run `peerpixel setup`") from None

    session = Session(_per_step_seed())
    state = ["connecting"]
    prompt = [""]

    # Loading a 4B model takes tens of seconds and says nothing while it does.
    # It gets a bar for the same reason everything else here does.
    start = plans.tracker("start")
    with console.Live(start, heading="Starting the worker"):
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
        return session.line(state[0])

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
                                    link_ref, sent_at, prompt)
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


def _do_job(link, job: dict, renderer, session: Session, link_ref, sent_at, prompt) -> float:
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

    bar = plans.tracker("job", {"job.render": session.per_step * steps})
    started = time.monotonic()
    earned = 0.0

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
        with console.Live(bar, heading=heading):
            if getattr(renderer, "pipe", None) is None:
                bar.begin("load")
                renderer.warm()

            reference = None
            if job.get("reference"):
                bar.begin("wait")
                if operation == "verify":
                    # This machine belongs to the operator and is re-rendering
                    # somebody else's finished job to check it. Both pictures
                    # are fetched rather than pushed: nobody is waiting on this
                    # and the bytes are already in storage.
                    reference = api.verify_asset(job["id"], "reference")
                else:
                    reference = await_reference(
                        link, job["id"],
                        timeout=(job.get("referenceWaitMs") or 45000) / 1000)
                    if reference is None:
                        raise RuntimeError(
                            "the browser never sent the preview to render from")

            bar.begin("render", detail=job["prompt"][:80])
            jpeg = renderer.render(job, on_step=stepped, reference=reference,
                                   on_demote=lambda name: bar.note(f"retrying in {name}"))
            session.learn(time.monotonic() - started, steps)
            bar.begin("deliver")
            if settings.keep_last():
                try:
                    preview.save(jpeg)
                except OSError:
                    pass  # a full disk costs a thumbnail, not a render

            if operation == "verify":
                subject = api.verify_asset(job["id"], "subject")
                measurements = compare.compare(subject, jpeg)
                measurements["image"] = base64.b64encode(jpeg).decode()
                api.submit_verification(job["id"], measurements)
                link.send(json.dumps({"type": "finished", "jobId": job["id"]}))
            elif job.get("transient"):
                # A preview has no permanent home. It goes back down this
                # socket and the dispatcher relays it straight to the browser
                # waiting for it.
                if len(jpeg) > relay.MAX_RESULT_BYTES:
                    raise RuntimeError(f"the preview is {len(jpeg)} bytes, over the "
                                       f"{relay.MAX_RESULT_BYTES} limit")
                link.send(relay.encode({"type": "draft_result", "draftId": job["id"]}, jpeg))
                earned = await_settlement(link, job["id"])
            else:
                result = api.submit_result(job["id"], jpeg)
                earned = result.get("earnedCredits", 0) or 0
                link.send(json.dumps({"type": "finished", "jobId": job["id"]}))
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
