"""The worker loop.

Holds a WebSocket open to the dispatcher and renders whatever comes down it.
Jobs are pushed, not polled, so an idle machine costs nothing at either end.

Headless throughout: no window, no desktop environment, no display. A box in a
cupboard is a perfectly good peer.
"""
from __future__ import annotations

import base64
import json
import socket as sockets
import time

from . import api, compare, config, events, preview, relay, ui

#: What this install speaks. Version 1 rendered 512px and posted results over
#: HTTP; it knew nothing of operations, transient drafts or reference images.
#: Version 2 rendered the step-distilled checkpoint: four steps, no guidance,
#: 1024px masters. Either would take a job priced for this version and hand
#: back a different picture at a different size, so the server gives work only
#: to the current version and an old install sits idle until it is updated.
PROTOCOL_VERSION = 3

HEARTBEAT_SECONDS = 25
RECONNECT_MIN = 2
RECONNECT_MAX = 60
TICK = 1.0  # seconds waited for a message before the clock is redrawn
MODEL_IDLE_UNLOAD_SECONDS = 2 * 60 * 60


def should_unload_model(last_active: float, current: float, *, loaded: bool) -> bool:
    return loaded and current - last_active >= MODEL_IDLE_UNLOAD_SECONDS


def await_reference(link, job_id: str, *, timeout: float, clock=time.monotonic):
    """Block until the chosen draft arrives for this master, or give up.

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
    """What the dispatcher paid for a draft it has just relayed.

    Delivery to the browser is the completion boundary, so by the time this
    answer arrives the money has already moved; this only reads back how much,
    for the local earnings display. A missing answer is worth no wait: the
    ledger is the truth and the next /api/me refresh will show it.
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


def asked_to_stop() -> bool:
    """Has somebody pressed stop while this machine was mid-render?

    A render that is already running is a picture somebody is waiting for and
    has been charged for, so stopping abandons nothing: the flag is read
    between jobs, the current one finishes and is delivered, and then the
    worker exits. Killing the process outright is still available for a worker
    that is merely idle, where there is nothing to abandon.
    """
    return bool(config.read().get("stopAfterJob"))


def run(renderer, once: bool = False) -> int:
    settings = config.read()
    config.write(stopAfterJob=False)
    if not settings.get("token"):
        raise SystemExit("this machine is not paired yet - run: peerpixel pair CODE")

    try:
        from websockets.sync.client import connect
    except ImportError:
        raise SystemExit("missing dependency: run ./setup.sh (or: uv sync)") from None

    session_started = time.monotonic()
    earned_pixels = 0.0

    def publish(**patch):
        """Everything the app shows about this worker, over the pipe.

        A no-op when nobody is listening, which is the headless case: the panel
        in `ui.py` is what a terminal reads.
        """
        events.emit("state", **patch)

    def bar(name: str, estimates: dict | None = None):
        events.emit("plan", name=name, **({"estimates": estimates} if estimates else {}))

    publish(phase="loading", connected=False, prompt="", step=0, steps=0,
            elapsedSeconds=0, images=0, earnedPixels=0, pixelsPerHour=None)
    # Loading a 4B model prints as it goes and takes tens of seconds. Get it
    # over with before the panel takes over the bottom of the screen. It is
    # also the longest unexplained wait in the whole app, so it gets a phase.
    bar("startup")
    events.phase("load")
    renderer.warm()
    events.phase("connect")
    publish(phase="connecting")

    # How long a step takes here, so a job's bar is calibrated from its very
    # first frame rather than after enough steps to measure. The benchmark
    # timed four steps at master resolution, which is exactly this number.
    bench_ms = settings.get("benchMs")
    per_step = [float(bench_ms) / 4000.0 if bench_ms else 1.5]

    status = ui.Status(
        api=config.API,
        machine=sockets.gethostname(),
        device=settings.get("deviceId", "unpaired"),
        accelerator=renderer.accelerator,
        free=bool(settings.get("allowFree")),
        free_confirmed=bool(settings.get("allowFreeSyncedAt")),
    )
    display = ui.Display(status)

    active_link = [None]
    progress_sent_at = [0.0]

    def stepped(done: int, total: int) -> None:
        status.step, status.steps = done, total
        publish(step=done, steps=total,
                elapsedSeconds=round(time.monotonic() - status.job_started, 1))
        events.progress(done, total, detail=f"step {done} of {total}")
        current = time.monotonic()
        if active_link[0] and (done >= total or current - progress_sent_at[0] >= 0.5):
            try:
                active_link[0].send(json.dumps({
                    "type": "progress", "jobId": active_link[1], "step": done, "steps": total,
                }))
            except Exception:  # noqa: BLE001 - telemetry cannot break a render
                pass
            progress_sent_at[0] = current
        display.refresh()

    url = (config.API.replace("http", "ws", 1)
           + f"/api/device/connect?protocol={PROTOCOL_VERSION}")
    headers = {"authorization": f"Bearer {settings['token']}", "user-agent": api.USER_AGENT}
    backoff = RECONNECT_MIN

    try:
        while True:
            try:
                status.state = "connecting"
                publish(phase="connecting", connected=False)
                display.refresh(force=True)
                # ping_interval=None on purpose. Cloudflare's hibernatable
                # sockets do not answer protocol-level pings, so the library's
                # own keepalive decides the connection is dead after about
                # thirty seconds and closes it. The dispatcher has its own
                # heartbeat message, sent below, which also refreshes when the
                # server last saw this machine.
                with connect(url, additional_headers=headers, max_size=None,
                             ping_interval=None, close_timeout=5) as link:
                    active_link[0] = link
                    backoff = RECONNECT_MIN
                    status.idle()
                    publish(phase="online", connected=True, prompt="", step=0, steps=0,
                            elapsedSeconds=0)
                    events.done(connected=True)
                    display.event(f"connected to {config.API}")
                    last_beat = time.monotonic()

                    while True:
                        try:
                            # A timeout rather than a plain iteration, so uptime
                            # and the idle clock keep moving while nothing comes.
                            raw = link.recv(timeout=TICK)
                        except TimeoutError:
                            if asked_to_stop():
                                display.event("stopping, as asked")
                                return status.images
                            if should_unload_model(status.idle_since, time.monotonic(),
                                                   loaded=getattr(renderer, "pipe", None) is not None):
                                renderer.unload()
                                publish(modelLoaded=False)
                            if time.monotonic() - last_beat >= HEARTBEAT_SECONDS:
                                link.send(json.dumps({"type": "heartbeat"}))
                                last_beat = time.monotonic()
                            display.refresh()
                            continue

                        try:
                            message = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        if message.get("type") != "job":
                            continue

                        job = message["job"]
                        operation = job.get("operation", "master")
                        active_link[1:] = [job["id"]]
                        progress_sent_at[0] = 0.0
                        steps_asked = int(job.get("steps", 0)) or 1
                        bar("job", {"job.render": per_step[0] * steps_asked})
                        if getattr(renderer, "pipe", None) is None:
                            events.phase("load")
                            publish(phase="loading", connected=True, prompt=job["prompt"])
                            renderer.warm()
                            publish(modelLoaded=True)
                        status.begin(job["prompt"], int(job.get("steps", 0)))
                        events.phase("render", detail=job["prompt"][:120])
                        publish(phase="rendering", connected=True, prompt=job["prompt"],
                                step=0, steps=int(job.get("steps", 0)), elapsedSeconds=0)
                        display.event(
                            f"{operation} {job['id']}  "
                            f"{job.get('width', '?')}px  {job['steps']} steps"
                        )
                        started = time.time()
                        try:
                            # A master renders what the person actually chose,
                            # so it waits for that picture to come down the
                            # socket before it starts. Nothing is stored at
                            # either end: the bytes are used and dropped.
                            reference = None
                            if job.get("reference"):
                                events.phase("wait")
                                publish(phase="rendering", connected=True,
                                        prompt=job["prompt"], step=0,
                                        steps=int(job.get("steps", 0)))
                                reference = await_reference(
                                    link, job["id"],
                                    timeout=(job.get("referenceWaitMs") or 45000) / 1000,
                                )
                                if reference is None:
                                    raise RuntimeError(
                                        "the browser never sent the draft to render from"
                                    )
                                events.phase("render", detail=job["prompt"][:120])

                            if operation == "verify":
                                # A check is preemptible: the moment real work
                                # turns up with no machine free to take it, the
                                # dispatcher takes this one back. Nobody is
                                # paid for a check, so an abandoned one costs
                                # nothing and simply runs again later.
                                # This machine belongs to the operator and is
                                # re-rendering somebody else's finished job to
                                # check it. Both pictures are fetched rather
                                # than pushed: nobody is waiting on this and
                                # the bytes are already in storage.
                                reference = None
                                if job.get("reference"):
                                    reference = api.verify_asset(job["id"], "reference")
                                jpeg = renderer.render(job, on_step=stepped, reference=reference)
                                subject = api.verify_asset(job["id"], "subject")
                                measurements = compare.compare(subject, jpeg)
                                measurements["image"] = base64.b64encode(jpeg).decode()
                                events.phase("deliver")
                                api.submit_verification(job["id"], measurements)
                                display.event(
                                    f"checked {job['id'][:6]}  "
                                    f"distance {measurements['distance']}  "
                                    f"rmse {measurements['rmse']}"
                                )
                                link.send(json.dumps({"type": "finished", "jobId": job["id"]}))
                                earned = 0.0
                                status.finish(earned)
                                publish(phase="online", connected=True, prompt="", step=0,
                                        steps=0, elapsedSeconds=0)
                                events.done(images=status.images)
                                last_beat = time.monotonic()
                                if once:
                                    return status.images
                                continue

                            jpeg = renderer.render(job, on_step=stepped, reference=reference)
                            events.phase("deliver")
                            per_step[0] = max(
                                0.05, (time.monotonic() - status.job_started) / steps_asked)
                            try:
                                preview.save(jpeg)
                            except OSError:
                                pass  # a full disk costs a thumbnail, not a render

                            if job.get("transient"):
                                # A draft has no permanent home. It goes back
                                # down this socket and the dispatcher relays it
                                # straight to the browser waiting for it.
                                if len(jpeg) > relay.MAX_RESULT_BYTES:
                                    raise RuntimeError(
                                        f"draft is {len(jpeg)} bytes, over the "
                                        f"{relay.MAX_RESULT_BYTES} limit"
                                    )
                                link.send(relay.encode(
                                    {"type": "draft_result", "draftId": job["id"]}, jpeg))
                                earned = await_settlement(link, job["id"])
                            else:
                                result = api.submit_result(job["id"], jpeg)
                                earned = result.get("earnedCredits", 0) or 0
                                link.send(json.dumps({"type": "finished", "jobId": job["id"]}))

                            earned_pixels += earned
                            status.finish(earned)
                            elapsed = time.monotonic() - session_started
                            rate = earned_pixels * 3600 / elapsed if earned_pixels > 0 and elapsed >= 60 else None
                            publish(phase="online", connected=True, prompt="", step=0, steps=0,
                                    elapsedSeconds=0, images=status.images,
                                    earnedPixels=earned_pixels, pixelsPerHour=rate,
                                    lastEarnedPixels=earned,
                                    lastImageAt=int(time.time() * 1000))
                            events.done(images=status.images, earnedPixels=earned_pixels)
                            display.event(
                                f"done in {time.time() - started:.1f}s  +{earned:g} pixels"
                            )
                            last_beat = time.monotonic()
                            if once or asked_to_stop():
                                return status.images
                        except Exception as error:  # noqa: BLE001 - one bad job must not end the run
                            status.idle()
                            publish(phase="online", connected=True, prompt="", step=0, steps=0,
                                    elapsedSeconds=0, error=str(error))
                            display.event(f"failed: {error}")
                            link.send(json.dumps({
                                "type": "failed", "jobId": job["id"], "reason": str(error)[:200],
                            }))
            except Exception as error:  # noqa: BLE001 - every disconnect is temporary
                active_link[0] = None
                status.state = "offline"
                bar("startup")
                events.phase("connect", detail=f"retrying in {backoff}s")
                publish(phase="offline", connected=False)
                display.event(f"disconnected ({error}) - retrying in {backoff}s")
                time.sleep(backoff)
                backoff = min(RECONNECT_MAX, int(backoff * 1.8) or 2)
    except KeyboardInterrupt:
        # Ctrl-C during a render, a wait or a backoff all end up here, so there
        # is one way out and it always runs the finally below.
        display.event("stopping")
    finally:
        publish(phase="stopped", connected=False)
        # However this ends, the shell gets its cursor back.
        display.close()
    return status.images
