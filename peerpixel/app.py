"""The app: a window in the browser, served from this machine and nowhere else.

Double-clicking one file gets somebody here. What they should find is a thing
that is already working -- installing what it needs, telling them how long it
will take, and asking for the one piece it cannot work out on its own, which is
the pairing code.

This process is deliberately the light half. Standard library only, no torch,
no diffusers, no Hub. It starts in under a second on an install where nothing
has been downloaded yet, which is the only way the dependency install itself
can have a progress bar. Everything heavy is a child process reporting back
over a pipe; see `events.py` and `tasks.py`.

Bound to 127.0.0.1, and every request has to prove it came from this machine
and carry the token minted when the app started. The page is served with the
token baked in, so nothing on the internet can reach these controls even if it
guesses the port.
"""
from __future__ import annotations

import json
import mimetypes
import platform
import secrets
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from . import api, config, preview, runtime, updater, window
from .tasks import PLANS, Runner, Task, command_task, install_task

UI = Path(__file__).resolve().parent / "ui"
PORTS = (8765, 8766, 8767, 8768, 0)

#: How often the app looks for a newer release. Rarely: this is a notice and an
#: offer, never something that happens to somebody mid-render.
UPDATE_EVERY = 6 * 3600


def machine() -> dict:
    return {
        "name": socket.gethostname(),
        "platform": f"{platform.system().lower()}-{platform.machine()}",
    }


def accelerator() -> str:
    """What to call this machine's card, asked of a child that has torch.

    In-process would mean importing torch into the app, which would mean the
    app could not start until torch was installed -- and the app is what
    installs torch. A child that is missing, slow or broken simply yields an
    honest "unknown"; a name is a nicety and pairing must not hang on it.
    """
    argv = runtime.child(["accelerator"])
    if not argv:
        return "unknown"
    try:
        out = subprocess.run(argv, cwd=str(runtime.ROOT), env=runtime.environment(),
                             capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    name = (out.stdout or "").strip().splitlines()[-1:] or ["unknown"]
    return name[0][:120] or "unknown"


class App:
    """Everything the window can ask for, with no HTTP in it.

    Kept apart from the request handler so the whole surface can be driven from
    a test without opening a socket.
    """

    def __init__(self, *, conduct: bool = True):
        self.jobs = Runner()      # one long task at a time: install, model, bench
        self.worker = Runner()    # the renderer, which is a service
        self.queue: list[str] = []
        self.queued_titles: list[str] = []
        self.update = {"checked": 0.0, "available": False,
                       "current": updater.installed(), "latest": ""}
        self.message = ""
        self.stop_flag = threading.Event()
        #: Set when the window closes, or when somebody presses Quit.
        self.closed = threading.Event()
        # Off in tests, where a thread that starts installing things and asks
        # GitHub for releases is not what is being tested.
        if conduct:
            threading.Thread(target=self._conductor, daemon=True).start()

    # -- the standing state ------------------------------------------------

    def steps(self) -> dict:
        settings = config.read()
        return {
            "paired": bool(settings.get("token")),
            "dependencies": runtime.dependencies_ready(),
            "model": self.model_ready(),
            "approved": bool(settings.get("approved")),
        }

    def model_ready(self) -> bool:
        """Are the weights on this disk?

        Asked without importing the Hub, because the app has no Hub: the cache
        layout is a documented directory, and looking at it is cheaper than a
        subprocess and works before anything is installed.
        """
        try:
            from .weights import cached
            return cached()
        except Exception:  # noqa: BLE001 - a missing cache is simply not ready
            return False

    def state(self) -> dict:
        settings = config.read()
        steps = self.steps()
        activity = self._activity()
        return {
            "version": self.update["current"],
            "api": config.API,
            "machine": machine()["name"],
            "accelerator": settings.get("accelerator") or "",
            "deviceId": settings.get("deviceId"),
            "steps": steps,
            "ready": all(steps.values()),
            "allowFree": bool(settings.get("allowFree")),
            "allowFreeConfirmed": bool(settings.get("allowFreeSyncedAt")),
            "benchMs": settings.get("benchMs"),
            "stopping": bool(settings.get("stopAfterJob")) and self.worker.busy(),
            "activity": activity,
            "worker": self._worker_state(),
            "update": {k: v for k, v in self.update.items() if k != "checked"},
            "message": self.message,
            "log": (self.jobs.snapshot().get("log") or ""),
        }

    #: A finished bar is left up this long, so "100%, delivered, plus one pixel"
    #: is a thing somebody sees rather than a frame the poll happened to miss.
    LINGER = 4.0

    def _activity(self) -> dict:
        """The one bar, and what it is a bar for.

        A render in progress outranks everything: it is the thing the machine is
        actually doing and the thing somebody is waiting on at the other end.
        Next is the worker starting up, because loading the model is a minute
        of nothing. A worker that is connected and merely waiting has no bar --
        there is no work to be a fraction of, and a bar for waiting is the kind
        of lie this app is trying not to tell.
        """
        worker = self.worker.snapshot()
        if worker.get("running"):
            bar = worker.get("progress") or {}
            since = worker.get("finishedAgo")
            fresh = not bar.get("finished") or (since is not None and since < self.LINGER)
            starting = (worker.get("task") == "startup"
                        and not (worker.get("result") or {}).get("connected"))
            if fresh and (worker.get("task") == "job" or starting):
                kind = "job" if worker.get("task") == "job" else "worker"
                return {"kind": kind, "step": 0, "steps": 0, "queued": [], **worker}
        jobs = self.jobs.snapshot()
        since = jobs.get("finishedAgo")
        stale = (not jobs.get("running")
                 and (jobs.get("progress") or {}).get("finished")
                 and (since is None or since > self.LINGER))
        if jobs.get("task") and not stale:
            total = 1 + len(self.queue)
            bar = jobs.get("progress") or {}
            eta = bar.get("etaSeconds")
            remaining = sum(sum(p.estimate for p in PLANS[name].phases)
                            for name in self.queue if name in PLANS)
            return {
                "kind": "setup",
                "step": 1,
                "steps": total,
                "queued": list(self.queued_titles),
                "overallEtaSeconds": None if eta is None else round(eta + remaining, 1),
                **jobs,
            }
        return {"kind": None, "running": False, "queued": []}

    def _worker_state(self) -> dict:
        snapshot = self.worker.snapshot()
        result = snapshot.get("result") or {}
        return {
            "running": snapshot.get("running", False),
            "connected": bool(result.get("connected")),
            "state": result.get("phase", "stopped"),
            "prompt": result.get("prompt", ""),
            "images": result.get("images", 0),
            "earnedPixels": result.get("earnedPixels", 0),
            "pixelsPerHour": result.get("pixelsPerHour"),
            "lastImageAt": result.get("lastImageAt"),
            "lastEarnedPixels": result.get("lastEarnedPixels"),
            "error": result.get("error", ""),
        }

    # -- the things a button does -------------------------------------------

    def pair(self, code: str) -> dict:
        if not code:
            raise ValueError("Paste the pairing code from peerpixel.cc first")
        name = accelerator()
        result = api.pair(code, {**machine(), "accelerator": name})
        config.write(deviceId=result["deviceId"], token=result["token"],
                     api=config.API, accelerator=name)
        self.message = "Paired."
        self.catch_up()
        return self.state()

    def catch_up(self) -> None:
        """Queue whatever is still missing, in the order it has to happen.

        This is what makes the app self-driving. Nobody should have to work out
        that libraries come before weights and weights before a benchmark, and
        nobody should have to press three buttons to say yes to a sequence that
        has only one possible order.
        """
        steps = self.steps()
        wanted = [name for name, ok in (
            ("install", steps["dependencies"]),
            ("model", steps["model"]),
            ("bench", steps["approved"] and steps["paired"]),
        ) if not ok]
        if "bench" in wanted and not steps["paired"]:
            wanted.remove("bench")  # a benchmark is submitted against an account
        with self.jobs.lock:
            running = self.jobs.snapshot().get("task") if self.jobs.busy() else None
            self.queue = [name for name in wanted if name != running]
            self.queued_titles = [PLANS[name].title for name in self.queue]

    def run_task(self, name: str) -> dict:
        if self.jobs.busy():
            raise ValueError("Something is already running")
        task = self._task(name)
        if task is None:
            raise ValueError("Cannot find uv, which is what installs everything. "
                             "Close this and run the launcher again.")
        self.jobs.start(task)
        return self.state()

    def _task(self, name: str) -> Task | None:
        if name == "install":
            return install_task()
        if name == "model":
            return command_task("model", ["download"])
        if name == "bench":
            return command_task("bench", ["bench"])
        if name == "update":
            return command_task("update", ["apply-update"])
        raise ValueError(f"Unknown task {name}")

    def start_worker(self) -> dict:
        if not all(self.steps().values()):
            raise ValueError("Finish setting up first")
        config.write(stopAfterJob=False)
        if self.worker.busy():
            return self.state()
        task = command_task("startup", ["run"], service=True)
        if task is None:
            raise ValueError("Cannot find the Python environment. Run the launcher again.")
        self.worker.start(task)
        config.write(autoStart=True)
        return self.state()

    def stop_worker(self, *, after_this: bool = False) -> dict:
        """Stop rendering. Gently if there is a picture in flight.

        `after_this` sets a flag the worker reads between jobs. The alternative
        -- terminating a process that is forty steps into a master somebody
        paid for -- fails the job, refunds them, and wastes every second of the
        render. Waiting a minute is cheaper for everybody.
        """
        config.write(autoStart=False, stopAfterJob=bool(after_this))
        if not after_this:
            self.worker.stop()
        return self.state()

    def cancel(self) -> dict:
        self.queue = []
        self.queued_titles = []
        self.jobs.stop()
        return self.state()

    def set_free(self, allow: bool) -> dict:
        settings = config.read()
        config.write(allowFree=bool(allow), allowFreeSyncedAt=0)
        try:
            api.set_free(settings.get("deviceId"), bool(allow))
            config.write(allowFreeSyncedAt=int(time.time()))
            self.message = ""
        except api.ApiError as error:
            if error.status in (401, 403):
                self.message = ("Saved on this machine, but the free-work switch belongs to "
                                "your account. Flip it on the Contribute page at peerpixel.cc.")
            else:
                raise
        if self.worker.busy():
            self.worker.stop()
            time.sleep(0.2)
            self.start_worker()
        return self.state()

    def check_update(self, force: bool = False) -> dict:
        if not force and time.monotonic() - self.update["checked"] < UPDATE_EVERY:
            return self.update
        self.update["checked"] = time.monotonic()
        latest = updater.latest()
        current = updater.installed()
        self.update.update({
            "current": current,
            "latest": latest or "",
            "available": bool(latest and updater.newer(latest, current)),
        })
        return self.update

    # -- the thread that keeps things moving --------------------------------

    def _conductor(self) -> None:
        """Starts the next queued step, and the worker once everything is done.

        A loop rather than a callback chain because a child can die at any
        moment and the recovery is the same either way: look at what is true
        now, and do the next thing that is not.
        """
        self.check_update(force=True)
        last_check = time.monotonic()
        while not self.stop_flag.is_set():
            try:
                if not self.jobs.busy() and self.queue:
                    nxt = self.queue.pop(0)
                    self.queued_titles = [PLANS[n].title for n in self.queue]
                    self.jobs.start(self._task(nxt))
                elif (not self.jobs.busy() and not self.queue
                        and not self.worker.busy()
                        and all(self.steps().values())
                        and config.read().get("autoStart")):
                    self.worker.start(command_task("startup", ["run"], service=True))
                if time.monotonic() - last_check > UPDATE_EVERY:
                    last_check = time.monotonic()
                    self.check_update(force=True)
            except Exception as error:  # noqa: BLE001 - the loop must survive
                self.message = f"{type(error).__name__}: {error}"
            self.stop_flag.wait(1.0)

    def shutdown(self) -> None:
        self.stop_flag.set()
        self.worker.stop()
        self.jobs.stop()


# -- HTTP ---------------------------------------------------------------------

def request_allowed(host, origin, token, expected) -> bool:
    """Local, same-origin and carrying this launch's token. All three."""
    try:
        local_host = urlsplit("//" + (host or "")).hostname in ("127.0.0.1", "localhost")
        local_origin = not origin or (
            urlsplit(origin).hostname in ("127.0.0.1", "localhost")
            and urlsplit(origin).netloc == host
        )
        good = secrets.compare_digest(token or "", expected or "")
        return bool(local_host and local_origin and good)
    except (TypeError, ValueError):
        return False


ACTIONS = {"install", "model", "bench", "update"}


class Handler(BaseHTTPRequestHandler):
    app: App = None  # type: ignore[assignment]
    token = ""
    server_version = "PeerPixel"

    # -- plumbing
    def _send(self, status, body: bytes, kind: str) -> None:
        self.send_response(status)
        self.send_header("content-type", kind)
        self.send_header("cache-control", "no-store")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except OSError:
            pass  # the window was closed mid-poll

    def _json(self, status, payload) -> None:
        self._send(status, json.dumps(payload).encode(), "application/json")

    def _local(self) -> bool:
        host = self.headers.get("host", "")
        return urlsplit("//" + host).hostname in ("127.0.0.1", "localhost")

    def _authorised(self, token=None) -> bool:
        return request_allowed(self.headers.get("host", ""), self.headers.get("origin", ""),
                               token if token is not None
                               else self.headers.get("x-peerpixel-token", ""),
                               self.token)

    # -- routes
    def do_GET(self):
        path = urlsplit(self.path).path
        if path in ("/", "/index.html"):
            if not self._local():
                self.send_error(403)
                return
            page = (UI / "index.html").read_text(encoding="utf-8")
            self._send(200, page.replace("__TOKEN__", self.token).encode(),
                       "text/html; charset=utf-8")
            return
        if path.startswith("/ui/"):
            if not self._local():
                self.send_error(403)
                return
            name = Path(path).name
            target = UI / name
            if not target.is_file() or target.parent != UI:
                self.send_error(404)
                return
            kind = mimetypes.guess_type(name)[0] or "application/octet-stream"
            self._send(200, target.read_bytes(), kind)
            return
        if path == "/api/preview":
            token = parse_qs(urlsplit(self.path).query).get("token", [""])[0]
            if not self._authorised(token):
                self.send_error(403)
                return
            data = preview.read()
            if data is None:
                self.send_error(404)
                return
            self._send(200, data, "image/jpeg")
            return
        self._api("GET", path)

    def do_POST(self):
        self._api("POST", urlsplit(self.path).path)

    def _api(self, method, path):
        if not self._authorised():
            self.send_error(403)
            return
        try:
            length = int(self.headers.get("content-length", "0") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            body = {}
        if not isinstance(body, dict):
            body = {}
        try:
            self._json(*self._route(method, path, body))
        except (api.ApiError, ValueError) as error:
            self._json(400, {"error": str(error)})
        except Exception as error:  # noqa: BLE001 - the socket must be answered
            # A driver that will not initialise, a disk that will not write. The
            # person who pressed the button gets a sentence rather than a dead
            # connection and a traceback in a console they may have closed.
            self._json(500, {"error": f"{type(error).__name__}: {error}"})

    def _route(self, method, path, body):
        app = self.app
        if method == "GET" and path == "/api/state":
            return 200, app.state()
        if method != "POST":
            return 404, {"error": "Not found"}
        if path == "/api/pair":
            return 200, app.pair(str(body.get("code", "")).strip().upper())
        if path == "/api/run":
            name = body.get("task")
            if name not in ACTIONS:
                return 400, {"error": "Unknown task"}
            return 200, app.run_task(name)
        if path == "/api/setup":
            app.catch_up()
            return 200, app.state()
        if path == "/api/cancel":
            return 200, app.cancel()
        if path == "/api/worker":
            if body.get("running"):
                return 200, app.start_worker()
            return 200, app.stop_worker(after_this=bool(body.get("afterThis")))
        if path == "/api/free":
            return 200, app.set_free(bool(body.get("allowFree")))
        if path == "/api/check-update":
            app.check_update(force=True)
            return 200, app.state()
        if path == "/api/quit":
            threading.Timer(0.2, app.closed.set).start()
            return 200, {"ok": True}
        return 404, {"error": "Not found"}

    def log_message(self, *_args):
        pass


def serve(*, show_window=True, ports=PORTS) -> None:
    """Start the app, put a window in front of it, and wait for that to close.

    The server runs in a thread and the window owns the main one. That is not a
    preference: every native webview toolkit refuses to be driven from anywhere
    but the main thread, and macOS enforces it by crashing rather than raising.
    """
    app = App()
    Handler.app = app
    Handler.token = secrets.token_urlsafe(24)
    server = None
    for candidate in ports:
        try:
            server = ThreadingHTTPServer(("127.0.0.1", candidate), Handler)
            break
        except OSError:
            continue
    if server is None:
        raise SystemExit("could not open a local port for the PeerPixel window")

    url = f"http://127.0.0.1:{server.server_port}"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    app.catch_up()

    try:
        if not show_window:
            print(f"PeerPixel is serving at {url} with no window.", flush=True)
            app.closed.wait()
        else:
            # Quit in the page sets the same event the window's close button
            # does, so this is what makes the button close a native window.
            threading.Thread(target=_watch_for_quit, args=(app,), daemon=True).start()
            kind = window.open_window(url, config.HOME / "window", app.closed.set)
            # A native window blocks until it is shut, so reaching here means it
            # already is. Anything else went off on its own and has to be waited
            # for -- and watched, so Quit in the page closes it too.
            if kind != "native":
                app.closed.wait()
            else:
                app.closed.set()
    except KeyboardInterrupt:
        pass
    finally:
        window.stop()
        app.shutdown()
        server.shutdown()
        server.server_close()


def _watch_for_quit(app: App) -> None:
    app.closed.wait()
    window.stop()
