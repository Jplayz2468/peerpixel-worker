"""A localhost dashboard for pairing, benchmarking and running the worker."""
from __future__ import annotations

import json
import platform
import secrets
import socket
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from . import api, config


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PeerPixel Worker</title><style>
:root{color-scheme:dark;font:16px/1.45 system-ui,sans-serif;background:#0a0a0b;color:#f4f2ed}*{box-sizing:border-box}
body{margin:0;min-height:100vh;background:radial-gradient(circle at 20% 0,#28201a 0,transparent 34%),#0a0a0b}
main{width:min(720px,calc(100% - 32px));margin:8vh auto}.brand{font-size:15px;color:#d7a56d;letter-spacing:.08em;text-transform:uppercase}
h1{font:48px/1.05 Georgia,serif;margin:18px 0 10px}p{color:#aaa49b}.card{border:1px solid #302e2b;background:#141414;border-radius:16px;padding:22px;margin:16px 0}
.row{display:flex;gap:10px}input,button{font:inherit;border-radius:10px;padding:12px 14px}input{min-width:0;flex:1;background:#090909;color:white;border:1px solid #45413d;text-transform:uppercase;letter-spacing:.15em}
button{border:0;background:#e9d5bb;color:#18120d;font-weight:700;cursor:pointer}button.alt{background:#282624;color:#eee}button:disabled{opacity:.45;cursor:not-allowed}
.status{display:flex;align-items:center;gap:9px}.dot{width:9px;height:9px;border-radius:50%;background:#777}.dot.on{background:#76d18b;box-shadow:0 0 15px #76d18b}
pre{white-space:pre-wrap;max-height:280px;overflow:auto;background:#090909;border-radius:10px;padding:14px;color:#c9c3ba;font:13px/1.5 ui-monospace,monospace}
.small{font-size:13px;color:#777}</style></head><body><main>
<div class="brand">PeerPixel Worker</div><h1>Your graphics card,<br>helping somebody else.</h1>
<p>This page stays on your computer. Pair once, benchmark once, then leave the worker running whenever you want.</p>
<section class="card"><b>1. Pair this machine</b><p id="pairText">Paste the code from peerpixel.cc.</p><div class="row"><input id="code" maxlength="12" placeholder="PAIRING CODE"><button id="pair">Pair</button></div></section>
<section class="card"><b>2. Get it ready</b><p>The first render warms the model; only the second is timed.</p><div class="row"><button id="bench">Download and benchmark</button><button class="alt" id="run">Start worker</button><button class="alt" id="stop">Stop</button></div></section>
<section class="card"><div class="status"><span class="dot" id="dot"></span><b id="state">Checking...</b></div><pre id="log">No commands run yet.</pre><div class="small">Dashboard: 127.0.0.1 only</div></section>
</main><script>
const $=id=>document.getElementById(id); async function call(path,body){const r=await fetch(path,{method:body?'POST':'GET',headers:{'content-type':'application/json','x-peerpixel-token':'__TOKEN__'},body:body&&JSON.stringify(body)});const j=await r.json();if(!r.ok)throw Error(j.error||'Request failed');return j}
async function refresh(){try{const s=await call('/api/state');$('pairText').textContent=s.paired?'Paired as '+s.deviceId+'.':'Paste the code from peerpixel.cc.';$('state').textContent=s.running?'Running: '+s.running:(s.paired?'Ready':'Not paired');$('dot').className='dot '+(s.running?'on':'');$('log').textContent=s.log||'No commands run yet.';$('bench').disabled=$('run').disabled=!!s.running;$('stop').disabled=!s.running}catch(e){$('state').textContent=e.message}}
$('pair').onclick=async()=>{try{await call('/api/pair',{code:$('code').value});$('code').value='';refresh()}catch(e){$('pairText').textContent=e.message}};
$('bench').onclick=async()=>{await call('/api/start',{command:'bench'});refresh()};$('run').onclick=async()=>{await call('/api/start',{command:'run'});refresh()};$('stop').onclick=async()=>{await call('/api/stop',{});refresh()};refresh();setInterval(refresh,1000)
</script></body></html>"""


class CommandRunner:
    def __init__(self):
        self.process = None
        self.command = None
        self.lines = []
        self.lock = threading.RLock()

    def start(self, command):
        with self.lock:
            if self.process and self.process.poll() is None:
                raise ValueError("Another worker command is already running")
            self.lines = []
            self.command = command
            self.process = subprocess.Popen(
                [sys.executable, "-m", "peerpixel", command],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            threading.Thread(target=self._collect, daemon=True).start()
        return self.state()

    def _collect(self):
        for line in self.process.stdout:
            with self.lock:
                self.lines.append(line.rstrip())
                del self.lines[:-200]

    def stop(self):
        with self.lock:
            if self.process and self.process.poll() is None:
                self.process.terminate()
            self.command = None
        return self.state()

    def state(self):
        with self.lock:
            running = self.command if self.process and self.process.poll() is None else None
            return {"running": running, "log": "\n".join(self.lines)}


def pair_machine(code):
    from .render import Renderer

    renderer = Renderer()
    info = {
        "name": socket.gethostname(),
        "platform": f"{platform.system().lower()}-{platform.machine()}",
        "accelerator": renderer.accelerator,
    }
    result = api.pair(code, info)
    config.write(deviceId=result["deviceId"], token=result["token"], api=config.API)
    return result


class DashboardApp:
    def __init__(self, *, pair=pair_machine, state=None, start=None, stop=None):
        self.pair = pair
        self.state = state or current_state
        self.start = start or RUNNER.start
        self.stop = stop or RUNNER.stop

    def handle(self, method, path, body=None):
        body = body or {}
        try:
            if method == "GET" and path == "/api/state":
                return 200, self.state()
            if method == "POST" and path == "/api/pair":
                code = str(body.get("code", "")).strip().upper()
                if not code:
                    return 400, {"error": "Pairing code required"}
                self.pair(code)
                return 200, self.state()
            if method == "POST" and path == "/api/start":
                command = body.get("command")
                if command not in ("bench", "run"):
                    return 400, {"error": "Unknown command"}
                return 200, self.start(command)
            if method == "POST" and path == "/api/stop":
                return 200, self.stop()
            return 404, {"error": "Not found"}
        except (api.ApiError, ValueError) as error:
            return 400, {"error": str(error)}


RUNNER = CommandRunner()


def request_allowed(host, origin, token, expected_token):
    try:
        local_host = urlsplit("//" + host).hostname in ("127.0.0.1", "localhost")
        local_origin = not origin or (
            urlsplit(origin).hostname in ("127.0.0.1", "localhost")
            and urlsplit(origin).netloc == host
        )
        good_token = secrets.compare_digest(token or "", expected_token or "")
        return local_host and local_origin and good_token
    except (TypeError, ValueError):
        return False


def current_state():
    settings = config.read()
    return {
        "paired": bool(settings.get("token")),
        "deviceId": settings.get("deviceId"),
        **RUNNER.state(),
    }


class Handler(BaseHTTPRequestHandler):
    app = DashboardApp()
    token = ""

    def do_GET(self):
        if self.path == "/":
            if urlsplit("//" + self.headers.get("host", "")).hostname not in ("127.0.0.1", "localhost"):
                self.send_error(403)
                return
            data = PAGE.replace("__TOKEN__", self.token).encode()
            self.send_response(200)
            self.send_header("content-type", "text/html; charset=utf-8")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self._api("GET")

    def do_POST(self):
        self._api("POST")

    def _api(self, method):
        if not request_allowed(
            self.headers.get("host", ""), self.headers.get("origin", ""),
            self.headers.get("x-peerpixel-token", ""), self.token,
        ):
            self.send_error(403)
            return
        try:
            length = int(self.headers.get("content-length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            body = {}
        status, payload = self.app.handle(method, self.path, body)
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("cache-control", "no-store")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, _format, *_args):
        pass


def serve(*, open_browser=True, port=8765):
    Handler.token = secrets.token_urlsafe(24)
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"PeerPixel dashboard: {url}")
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        RUNNER.stop()
        server.server_close()
