"""peerpixel - render for people whose machines cannot.

    peerpixel pair CODE     link this machine to your account
    peerpixel download      fetch the model (~15 GB) ahead of time
    peerpixel bench         prove it is fast enough
    peerpixel run           start rendering (Ctrl-C to stop)
    peerpixel run --free    the same, and take unpaid work too
    peerpixel free on|off   take unpaid work from people without an account
    peerpixel status        pool and device state
"""
from __future__ import annotations

import platform
import socket
import sys
import time

from . import api, config, download, update
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


def cmd_pair(argv):
    if not argv:
        raise SystemExit("usage: peerpixel pair CODE   (get one from peerpixel.cc)")
    renderer = Renderer()
    result = api.pair(argv[0].upper(), {**machine(), "accelerator": renderer.accelerator})
    config.write(deviceId=result["deviceId"], token=result["token"], api=config.API)
    print(f"Paired as {machine()['name']}.")
    print(f"Saved to {config.FILE}")
    print("Next: peerpixel bench")


def cmd_download(_argv):
    print(f"Model ready at {download.ensure()}")


def cmd_bench(_argv):
    download.ensure()
    renderer = Renderer()
    print(f"Benchmarking on {renderer.accelerator} - 4 steps...")
    renderer.warm()
    started = time.time()
    renderer.render({"id": "bench", "prompt": "a lighthouse made of blown glass", "seed": 1, "steps": 4})
    ms = int((time.time() - started) * 1000)
    result = api.submit_bench(ms, renderer.accelerator)
    print(f"{ms / 1000:.1f}s for 4 steps (limit {result['limitMs'] / 1000:.0f}s)")
    if result["approved"]:
        print("Approved. Run `peerpixel run` to start earning.")
    else:
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
    "pair": cmd_pair,
    "download": cmd_download,
    "bench": cmd_bench,
    "free": cmd_free,
    "run": cmd_run,
    "status": cmd_status,
}


def main():
    argv = sys.argv[1:]
    handler = COMMANDS.get(argv[0]) if argv else None
    if not handler:
        print(__doc__)
        raise SystemExit(0 if not argv else 1)
    update.check()
    try:
        handler(argv[1:])
    except api.ApiError as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    main()
