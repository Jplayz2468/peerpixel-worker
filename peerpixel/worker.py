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

from . import api, config, dashboard_state, ui

HEARTBEAT_SECONDS = 25
RECONNECT_MIN = 2
RECONNECT_MAX = 60
TICK = 1.0  # seconds waited for a message before the clock is redrawn
MODEL_IDLE_UNLOAD_SECONDS = 2 * 60 * 60


def should_unload_model(last_active: float, current: float, *, loaded: bool) -> bool:
    return loaded and current - last_active >= MODEL_IDLE_UNLOAD_SECONDS


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

    url = config.API.replace("http", "ws", 1) + "/api/device/connect"
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
                        active_link[1:] = [job["id"]]
                        progress_sent_at[0] = 0.0
                        if getattr(renderer, "pipe", None) is None:
                            publish(phase="loading", connected=True, prompt=job["prompt"])
                            renderer.warm()
                            publish(modelLoaded=True)
                        status.begin(job["prompt"], int(job.get("steps", 0)))
                        publish(phase="rendering", connected=True, prompt=job["prompt"],
                                step=0, steps=int(job.get("steps", 0)), elapsedSeconds=0)
                        display.event(f"job {job['id']}  {job['steps']} steps")
                        started = time.time()
                        try:
                            jpeg = renderer.render(job, on_step=stepped)
                            dashboard_state.save_preview(jpeg)
                            result = api.submit_result(job["id"], jpeg)
                            earned = result.get("earnedCredits", 0) or 0
                            earned_pixels += earned
                            status.finish(earned)
                            elapsed = time.monotonic() - session_started
                            rate = earned_pixels * 3600 / elapsed if earned_pixels > 0 and elapsed >= 60 else None
                            publish(phase="online", connected=True, prompt="", step=0, steps=0,
                                    elapsedSeconds=0, images=status.images,
                                    earnedPixels=earned_pixels, pixelsPerHour=rate,
                                    lastImageAt=int(time.time() * 1000))
                            display.event(
                                f"done in {time.time() - started:.1f}s  +{earned:g} pixels"
                            )
                            link.send(json.dumps({"type": "finished", "jobId": job["id"]}))
                            last_beat = time.monotonic()
                            if once:
                                return status.images
                        except Exception as error:  # noqa: BLE001 - one bad job must not end the run
                            status.idle()
                            publish(phase="online", connected=True, prompt="", step=0, steps=0,
                                    elapsedSeconds=0, error=str(error))
                            display.event(f"failed: {error}")
                            link.send(json.dumps({"type": "failed", "jobId": job["id"]}))
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
