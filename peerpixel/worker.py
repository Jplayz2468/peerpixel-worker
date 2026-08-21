"""The worker loop.

Holds a WebSocket open to the dispatcher and renders whatever comes down it.
Jobs are pushed, not polled, so an idle machine costs nothing at either end.

Headless throughout: no window, no desktop environment, no display. A box in a
cupboard is a perfectly good peer.
"""
from __future__ import annotations

import json
import socket as sockets
import time

from . import api, config, dashboard_state, relay, ui

#: What this install speaks. Version 1 rendered 512px and posted results over
#: HTTP; it knew nothing of operations, transient drafts or reference images.
#: The server hands drafts and masters only to version 2 and above, so an old
#: install sits connected and idle until it is updated.
PROTOCOL_VERSION = 2

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


def run(renderer, once: bool = False) -> int:
    settings = config.read()
    if not settings.get("token"):
        raise SystemExit("this machine is not paired yet - run: peerpixel pair CODE")

    try:
        from websockets.sync.client import connect
    except ImportError:
        raise SystemExit("missing dependency: run ./setup.sh (or: uv sync)") from None

    session_started = time.monotonic()
    earned_pixels = 0.0

    def publish(**patch):
        try:
            dashboard_state.publish(patch)
        except OSError:
            pass

    publish(phase="loading", connected=False, prompt="", step=0, steps=0,
            elapsedSeconds=0, images=0, earnedPixels=0, pixelsPerHour=None)
    # Loading a 4B model prints as it goes and takes tens of seconds. Get it
    # over with before the panel takes over the bottom of the screen.
    renderer.warm()
    publish(phase="connecting")

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
                    display.event(f"connected to {config.API}")
                    last_beat = time.monotonic()

                    while True:
                        try:
                            # A timeout rather than a plain iteration, so uptime
                            # and the idle clock keep moving while nothing comes.
                            raw = link.recv(timeout=TICK)
                        except TimeoutError:
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
                        if getattr(renderer, "pipe", None) is None:
                            publish(phase="loading", connected=True, prompt=job["prompt"])
                            renderer.warm()
                            publish(modelLoaded=True)
                        status.begin(job["prompt"], int(job.get("steps", 0)))
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

                            jpeg = renderer.render(job, on_step=stepped, reference=reference)
                            dashboard_state.save_preview(jpeg)

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
                            display.event(
                                f"done in {time.time() - started:.1f}s  +{earned:g} pixels"
                            )
                            last_beat = time.monotonic()
                            if once:
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
