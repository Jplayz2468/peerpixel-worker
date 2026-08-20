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
from urllib.parse import parse_qs, urlsplit

from . import api, config, dashboard_state, download


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PeerPixel Worker</title><style>
:root{color-scheme:dark;font:16px/1.45 Inter,system-ui,sans-serif;background:#090909;color:#f5f0e8}*{box-sizing:border-box}body{margin:0;min-height:100vh;background:radial-gradient(circle at 12% 0,#342519 0,transparent 32%),#090909}main{width:min(920px,calc(100% - 28px));margin:38px auto 70px}.brand{color:#e1a868;letter-spacing:.12em;text-transform:uppercase;font-size:13px}h1{font:clamp(36px,6vw,64px)/1 Georgia,serif;margin:12px 0}h2{margin:0 0 5px}p{color:#aaa49b}.tabs{display:flex;gap:8px;margin:26px 0}.tabs button{background:#201e1b;color:#aaa}.tabs .active{background:#ead5bb;color:#18120d}.view{display:none}.view.active{display:block}.card{border:1px solid #302d29;background:#141414;border-radius:18px;padding:22px;margin:14px 0}.step{display:grid;grid-template-columns:42px 1fr auto;align-items:center;gap:14px}.num{width:38px;height:38px;border-radius:50%;display:grid;place-items:center;background:#292622;color:#bbb}.done .num{background:#386746;color:#d8ffe1}.row{display:flex;gap:10px;flex-wrap:wrap}input,button{font:inherit;border-radius:11px;padding:12px 15px}input{min-width:180px;flex:1;background:#090909;color:white;border:1px solid #45413d;text-transform:uppercase;letter-spacing:.15em}button{border:0;background:#ead5bb;color:#18120d;font-weight:750;cursor:pointer}button.alt{background:#282624;color:#eee}button:disabled{opacity:.45;cursor:not-allowed}.hero{display:grid;grid-template-columns:1fr 1fr;gap:14px}.status{display:flex;align-items:center;gap:9px}.dot{width:10px;height:10px;border-radius:50%;background:#777}.dot.on{background:#76d18b;box-shadow:0 0 16px #76d18b}.big{font:34px/1.1 Georgia,serif;margin:8px 0}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.metric{background:#0c0c0c;border-radius:12px;padding:15px}.metric span{display:block;color:#888;font-size:12px;text-transform:uppercase}.bar{height:12px;background:#292622;border-radius:20px;overflow:hidden}.fill{height:100%;background:linear-gradient(90deg,#d38d4d,#f3d6a8);width:0;transition:width .25s}.preview{width:100%;aspect-ratio:1;background:#090909;border-radius:14px;object-fit:cover}.muted,.small{font-size:13px;color:#777}pre{white-space:pre-wrap;max-height:220px;overflow:auto;background:#090909;border-radius:10px;padding:14px;color:#c9c3ba;font:13px/1.5 ui-monospace,monospace}@media(max-width:680px){.hero{grid-template-columns:1fr}.metrics{grid-template-columns:1fr}.step{grid-template-columns:42px 1fr}.step button{grid-column:2}}
.fill.working{width:35%!important;animation:work 1.25s ease-in-out infinite}@keyframes work{0%{transform:translateX(-110%)}100%{transform:translateX(300%)}}
</style></head><body><main>
<div class="brand">PeerPixel Worker · stays on this computer</div><h1>Turn spare compute<br>into pixels.</h1><p>Setup happens once. Running is what you use day to day.</p>
<nav class="tabs"><button id="setupTab" class="active">Setup · one time</button><button id="runTab">Running</button></nav>
<div id="setupView" class="view active">
<section id="pairStep" class="card step"><div class="num">1</div><div><h2>Pair this device</h2><p id="pairText">Paste the code from peerpixel.cc.</p><div class="row"><input id="code" maxlength="12" placeholder="PAIRING CODE"><button id="pair">Pair</button></div></div></section>
<section id="downloadStep" class="card step"><div class="num">2</div><div><h2>Download the model</h2><p>About 15 GB. Progress appears below and interrupted downloads resume.</p></div><button id="download">Download</button></section>
<section id="benchStep" class="card step"><div class="num">3</div><div><h2>Warm up and benchmark</h2><p>Loading can take several minutes the first time. That is normal, not a freeze.</p></div><button id="bench">Benchmark</button></section>
<section id="readyStep" class="card step"><div class="num">4</div><div><h2>Ready to contribute</h2><p id="readyText">Finish the steps above.</p></div></section>
</div>
<div id="runView" class="view"><div class="hero"><section class="card"><div class="status"><span class="dot" id="dot"></span><b id="state">Checking…</b></div><div class="big" id="phase">Stopped</div><p id="prompt">Start the worker, then you can leave this page open or close it.</p><div class="row"><button id="run">Start worker</button><button class="alt" id="stop">Stop</button></div></section><section class="card"><img class="preview" id="preview" alt="Latest completed image"><div class="small">Latest completed image</div></section></div>
<section class="card"><div class="row"><b id="progressText">Waiting for work</b><span class="muted" id="elapsed">0s</span></div><div class="bar"><div class="fill" id="fill"></div></div></section>
<section class="metrics"><div class="metric"><span>Images this session</span><b id="images">0</b></div><div class="metric"><span>Earned this session</span><b id="earned">0 px</b></div><div class="metric"><span>Estimated rate</span><b id="rate">Calculating…</b></div></section></div>
<details class="card"><summary>Details and command log</summary><pre id="log">No commands run yet.</pre><div class="small">Protected local dashboard · 127.0.0.1 only</div></details>
</main><script>
const $=id=>document.getElementById(id); async function call(path,body){const r=await fetch(path,{method:body?'POST':'GET',headers:{'content-type':'application/json','x-peerpixel-token':'__TOKEN__'},body:body&&JSON.stringify(body)});const j=await r.json();if(!r.ok)throw Error(j.error||'Request failed');return j}
function tab(which){$('setupView').classList.toggle('active',which==='setup');$('runView').classList.toggle('active',which==='run');$('setupTab').classList.toggle('active',which==='setup');$('runTab').classList.toggle('active',which==='run')} $('setupTab').onclick=()=>tab('setup');$('runTab').onclick=()=>tab('run');
let lastImage=0;async function refresh(){try{const s=await call('/api/state');$('pairText').textContent=s.paired?'Paired as '+s.deviceId+'.':'Paste the code from peerpixel.cc.';[['pairStep',s.paired],['downloadStep',s.modelReady],['benchStep',s.approved],['readyStep',s.ready]].forEach(([id,ok])=>$(id).classList.toggle('done',!!ok));$('readyText').textContent=s.ready?'All set. Open Running and press Start worker.':'Finish the steps above.';const active=!!s.running;$('state').textContent=s.connected?'Connected':active?'Starting…':'Stopped';$('phase').textContent=s.phase==='loading'?'Loading model…':s.phase==='rendering'?'Generating image':s.phase==='online'?'Online · waiting for work':active?'Starting worker':'Stopped';$('dot').className='dot '+(s.connected?'on':'');$('prompt').textContent=s.prompt|| (s.phase==='loading'?'The first model load can take several minutes. It is working.':'Start the worker, then you can leave this page open or close it.');$('log').textContent=s.log||'No commands run yet.';$('download').disabled=$('bench').disabled=$('run').disabled=active;$('stop').disabled=!active;const total=Number(s.steps)||0,step=Number(s.step)||0;$('fill').style.width=(total?Math.min(100,100*step/total):0)+'%';$('progressText').textContent=s.phase==='rendering'?`Step ${step} of ${total}`:'Waiting for work';$('elapsed').textContent=(s.elapsedSeconds||0)+'s';$('images').textContent=s.images||0;$('earned').textContent=(s.earnedPixels||0)+' px';$('rate').textContent=s.pixelsPerHour==null?'Calculating…':Number(s.pixelsPerHour).toFixed(1)+' px/hour';if(s.lastImageAt&&s.lastImageAt!==lastImage){lastImage=s.lastImageAt;$('preview').src='/api/preview?t='+lastImage+'&token=__TOKEN__'}}catch(e){$('state').textContent=e.message}}
$('pair').onclick=async()=>{try{await call('/api/pair',{code:$('code').value});$('code').value='';refresh()}catch(e){$('pairText').textContent=e.message}};
$('download').onclick=async()=>{await call('/api/start',{command:'download'});refresh()};$('bench').onclick=async()=>{await call('/api/start',{command:'bench'});refresh()};$('run').onclick=async()=>{await call('/api/start',{command:'run'});tab('run');refresh()};$('stop').onclick=async()=>{await call('/api/stop',{});refresh()};
async function showWork(){try{const s=await call('/api/state'),working=['loading','benchmarking','downloading'].includes(s.phase);$('fill').classList.toggle('working',working);if(working){$('state').textContent='Working…';$('phase').textContent=s.phase==='downloading'?'Downloading model…':s.phase==='loading'?'Loading model…':'Warming up and benchmarking…';$('progressText').textContent='Working, this is not stuck';$('prompt').textContent='This can take several minutes the first time. It is working.'}}catch{}}refresh();showWork();setInterval(()=>{refresh();showWork()},1000)
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
                if command not in ("download", "bench", "run"):
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
    runtime = dashboard_state.read()
    try:
        model_ready = any((download._repo_dir() / "snapshots").glob("*/model_index.json"))
    except Exception:
        model_ready = False
    return {
        "paired": bool(settings.get("token")),
        "deviceId": settings.get("deviceId"),
        "modelReady": model_ready,
        "benchMs": settings.get("benchMs"),
        "approved": bool(settings.get("approved")),
        "ready": bool(settings.get("token")) and model_ready and bool(settings.get("approved")),
        **runtime,
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
        if self.path.startswith("/api/preview"):
            query_token = parse_qs(urlsplit(self.path).query).get("token", [""])[0]
            if not request_allowed(self.headers.get("host", ""), self.headers.get("origin", ""), query_token, self.token):
                self.send_error(403)
                return
            try:
                data = dashboard_state.preview_path().read_bytes()
            except OSError:
                self.send_error(404)
                return
            self.send_response(200); self.send_header("content-type", "image/jpeg"); self.send_header("cache-control", "no-store"); self.send_header("content-length", str(len(data))); self.end_headers(); self.wfile.write(data)
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
    # 8765 is a common port and a second worker on the same machine would want
    # its own. Falling over with a traceback is the worst possible outcome for
    # the people this page exists for, so try the preferred port, then a few
    # after it, then let the OS pick anything free.
    server = None
    for candidate in [port, port + 1, port + 2, port + 3, 0]:
        try:
            server = ThreadingHTTPServer(("127.0.0.1", candidate), Handler)
            break
        except OSError:
            continue
    if server is None:
        raise SystemExit("could not open a local port for the dashboard")
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
