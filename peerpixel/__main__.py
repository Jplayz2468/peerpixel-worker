"""peerpixel - render for people whose machines cannot.

Most people never type any of this. They unzip the folder, double-click the
launcher, and everything below happens in a window with a progress bar on it.
These exist because a machine in a cupboard has no window, and because when
something goes wrong it is far easier to debug one command than a whole app.

    peerpixel app           the window: setup, updates and running (the default)
    peerpixel pair CODE     link this machine to your account
    peerpixel download      fetch the model (~15 GB) ahead of time
    peerpixel bench         prove it is fast enough
    peerpixel run           start rendering (Ctrl-C to stop)
    peerpixel run --free    the same, and take unpaid work too
    peerpixel free on|off   take unpaid work from people without an account
    peerpixel status        pool and device state
    peerpixel update        fetch and install a newer worker
"""
from __future__ import annotations

import platform
import socket
import sys
import time

from . import api, config, download, events, updater
from .benchmark import run_benchmark
from .render import Renderer
from .worker import run as run_worker

# The free switch is the one thing a device token cannot do: it belongs to the
# account, and the API only recognises accounts by the website session cookie.
SIGN_IN = (
    "peerpixel.cc would not take the device token for this - the free switch\n"
    "belongs to your account, not to this machine. Either flip it on the\n"
    "Contribute page, or export PEERPIXEL_SESSION=<your pp cookie> and retry."
)


def machine() -> dict:
    return {
        "name": socket.gethostname(),
        "platform": f"{platform.system().lower()}-{platform.machine()}",
    }


def cmd_app(argv):
    from .app import serve

    serve(show_window=not {"--no-window", "--no-browser"} & set(argv))


def cmd_accelerator(_argv):
    """What this machine renders on. Asked by the app, which cannot import torch."""
    from .render import describe_accelerator

    print(describe_accelerator())


def cmd_pair(argv):
    if not argv:
        raise SystemExit("usage: peerpixel pair CODE   (get one from peerpixel.cc)")
    # The name of the card, not a renderer: pairing must not depend on the GPU
    # being free, and the machine somebody is pairing is often already busy.
    from .render import describe_accelerator

    name = describe_accelerator()
    result = api.pair(argv[0].upper(), {**machine(), "accelerator": name})
    config.write(deviceId=result["deviceId"], token=result["token"], api=config.API,
                 accelerator=name)
    print(f"Paired as {machine()['name']}.")
    print(f"Saved to {config.FILE}")
    print("Next: peerpixel bench")


def cmd_download(_argv):
    print(f"Model ready at {download.ensure()}")
    events.done(model=True)


def cmd_bench(_argv):
    events.phase("model")
    download.ensure()
    events.phase("load")
    renderer = Renderer()
    print(f"Warming up {renderer.accelerator}...", flush=True)
    renderer.warm()

    # Two renders, and the bar has to cross both of them rather than filling up
    # and then doing it all again. The phase it is in decides which half.
    phases = iter(("warm", "measure"))
    current = {"name": next(phases)}
    events.phase(current["name"])

    def stepped(done, total):
        events.progress(done, total, detail=f"step {done} of {total}")

    def next_phase():
        current["name"] = next(phases, current["name"])
        events.phase(current["name"])

    try:
        ms, result = run_benchmark(renderer, on_step=stepped, between=next_phase)
    except BaseException as error:
        events.failed(str(error))
        raise
    events.phase("submit")
    config.write(benchMs=ms, approved=bool(result.get("approved")),
                 accelerator=renderer.accelerator)
    print(f"{ms / 1000:.1f}s for 4 steps (limit {result['limitMs'] / 1000:.0f}s)")
    if result["approved"]:
        events.done(approved=True, ms=ms)
        print("Approved. Run `peerpixel run` to start earning.")
    else:
        events.failed("This machine is too slow to keep people waiting.")
        raise SystemExit("Not approved: this machine is too slow to keep people waiting.")


def cmd_free(argv):
    if not argv or argv[0] not in ("on", "off"):
        raise SystemExit("usage: peerpixel free on|off")
    wanted = argv[0] == "on"

    device_id = config.read().get("deviceId")
    if not device_id:
        raise SystemExit("this machine is not paired yet - run: peerpixel pair CODE")

    # Remembered locally either way, so `run` can show what was asked for and
    # retry the switch on a machine that was offline when it was thrown. The
    # confirmation goes with it: one from an older setting would be a lie.
    config.write(allowFree=wanted, allowFreeSyncedAt=0)
    try:
        api.set_free(device_id, wanted)
    except api.ApiError as error:
        if error.status in (401, 403):
            print(f"Saved locally, but not to your account.\n{SIGN_IN}")
            return
        raise
    config.write(allowFreeSyncedAt=int(time.time()))
    print("Now taking free work as well as paid." if wanted else "Paid work only.")


def cmd_run(argv):
    if "--free" in argv:
        cmd_free(["on"])
    download.ensure()
    run_worker(Renderer(), once="--once" in argv)


def cmd_update(_argv):
    updater.apply()


def cmd_status(_argv):
    settings = config.read()
    free = bool(settings.get("allowFree"))
    confirmed = bool(settings.get("allowFreeSyncedAt"))
    pool = api.pool()

    print(f"api       {config.API}")
    print(f"device    {settings.get('deviceId', 'not paired')}")
    print(f"free work {'on' if free else 'off'}"
          f"{'' if not free or confirmed else '  (saved here, never accepted by the server)'}")
    print(f"pool      {pool.get('workersOnline', 0)} online, {pool.get('workersIdle', 0)} idle, "
          f"{pool.get('workersFree', 0)} taking free work")
    print(f"queue     {pool.get('queued', 0)} paid, {pool.get('queuedFree', 0)} free, "
          f"{pool.get('running', 0)} running")
    if free and not confirmed:
        print(SIGN_IN)


COMMANDS = {
    "app": cmd_app,
    "dashboard": cmd_app,      # what the old README called it
    "pair": cmd_pair,
    "download": cmd_download,
    "bench": cmd_bench,
    "free": cmd_free,
    "run": cmd_run,
    "status": cmd_status,
    "update": cmd_update,
    "apply-update": cmd_update,
    "accelerator": cmd_accelerator,
}

#: Commands the app runs as children. They talk over the pipe and must not be
#: interrupted by a "there is a newer version" line addressed to a person.
QUIET = {"app", "dashboard", "accelerator", "apply-update"}


def notice() -> None:
    """Tell a terminal user that a newer worker exists. Never installs anything.

    Silent on every failure. Being offline, behind a proxy or rate limited by
    GitHub is not a reason to delay somebody's render by a single second.
    """
    latest = updater.latest(timeout=2.0)
    here = updater.installed()
    if latest and updater.newer(latest, here):
        print(f"Update available: {latest} (you have {here}) - run: peerpixel update")


def main():
    argv = sys.argv[1:]
    # No arguments is somebody who double-clicked something. Open the window.
    name = argv[0] if argv else "app"
    handler = COMMANDS.get(name)
    if not handler:
        print(__doc__)
        raise SystemExit(1)
    if name not in QUIET and not events.ENABLED:
        notice()
    try:
        handler(argv[1:])
    except api.ApiError as error:
        events.failed(str(error))
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    main()
