"""
The window, served to a browser.

The packaged app has no native GUI toolkit in it. Instead it starts this tiny
server on the loopback address and opens the person's own browser at it, which
reuses the one thing the tool already renders beautifully -- the HTML digest --
and keeps the whole front end inspectable text rather than a compiled widget.

Three deliberate safety choices, because "a local web server" is a phrase that
should make anyone handling private recordings nervous:

  - It binds to 127.0.0.1 only. It is not reachable from the network; another
    machine cannot open it.
  - Every action carries a random token minted at launch and embedded in the
    page. A request without it is refused, so a malicious website you happen to
    have open cannot quietly POST audio to your loopback port and drive the app.
  - The Host header must be loopback. That closes the DNS-rebinding trick where a
    hostile page resolves its own domain to 127.0.0.1 to bypass the origin rules.

The server owns no pipeline logic. It is a thin skin over `AppController`, which
is the same engine the CLI and the tests use.
"""

from __future__ import annotations

import json
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .controller import AppController, Brain

# Extensions the picker will accept an upload for. Mirrors the pipeline's own
# ingest lists; anything else is refused before it is written to the inbox.
_ACCEPTED = {
    ".mp3", ".wav", ".m4a", ".m4b", ".flac", ".ogg", ".oga", ".opus", ".aac",
    ".wma", ".amr", ".aiff", ".webm", ".mp4", ".mov", ".txt", ".srt", ".vtt", ".md",
}
_MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024   # 2 GiB per file, a generous ceiling


def _safe_name(raw: str) -> str:
    """A filename that cannot escape the inbox: basename only, no separators."""
    name = Path(raw.replace("\\", "/")).name.strip()
    name = name.replace("\x00", "")
    return name or "recording"


class _Job:
    """The state of one processing run, polled by the page."""

    def __init__(self) -> None:
        self.running = False
        self.lines: list[str] = []
        self.summary: dict | None = None
        self.error: str | None = None
        self._lock = threading.Lock()

    def start(self) -> bool:
        with self._lock:
            if self.running:
                return False
            self.running = True
            self.lines = []
            self.summary = None
            self.error = None
            return True

    def log(self, message: str) -> None:
        with self._lock:
            self.lines.append(message)

    def finish(self, summary: dict | None, error: str | None) -> None:
        with self._lock:
            self.summary = summary
            self.error = error
            self.running = False

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "running": self.running,
                "lines": list(self.lines),
                "summary": self.summary,
                "error": self.error,
            }


class AppServer:
    """Owns the controller, the launch token, and the current job."""

    def __init__(self, controller: AppController | None = None) -> None:
        self.controller = controller or AppController()
        self.token = secrets.token_urlsafe(24)
        self.job = _Job()
        self.controller.ensure_installed()

    # ---- actions the handler calls -------------------------------------
    def preflight(self, brain: Brain) -> list[dict]:
        return [
            {"ok": i.ok, "fatal": i.fatal, "name": i.name, "detail": i.detail}
            for i in self.controller.preflight(brain)
        ]

    def apply_settings(self, passphrase: str | None, groq_key: str | None) -> None:
        if passphrase is not None:
            self.controller.set_passphrase(passphrase)
        if groq_key is not None:
            self.controller.set_groq_key(groq_key)

    def save_upload(self, filename: str, body: bytes) -> Path:
        name = _safe_name(filename)
        if Path(name).suffix.lower() not in _ACCEPTED:
            raise ValueError(f"{name}: not a recording or transcript the app accepts")
        cfg = self.controller.load_config(Brain.CLOUD)
        inbox = cfg.path("inbox")
        inbox.mkdir(parents=True, exist_ok=True)
        dest = inbox / name
        dest.write_bytes(body)
        return dest

    def start_processing(self, brain: Brain) -> bool:
        if not self.job.start():
            return False

        def worker() -> None:
            try:
                summary = self.controller.process(brain, progress=self.job.log)
                self.job.finish(summary, None)
            except Exception as exc:  # noqa: BLE001 - surface it to the page, never crash the server
                self.job.finish(None, str(exc))

        threading.Thread(target=worker, daemon=True).start()
        return True

    def digest_html(self, include_personal: bool) -> str:
        out = self.controller.write_digest(include_personal=include_personal)
        return out.read_text(encoding="utf-8")

    # ---- the http layer -------------------------------------------------
    def make_server(self, host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
        app = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):  # noqa: N802 - silence the default stderr spam
                pass

            # -- guards --
            def _loopback_host(self) -> bool:
                host_header = (self.headers.get("Host") or "").split(":")[0]
                return host_header in ("127.0.0.1", "localhost", "")

            def _authed(self) -> bool:
                token = self.headers.get("X-Token") or parse_qs(
                    urlparse(self.path).query
                ).get("token", [""])[0]
                return secrets.compare_digest(token or "", app.token)

            def _send(self, code: int, body: bytes, ctype: str) -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                # This page never wants to be embedded or cached anywhere.
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Frame-Options", "DENY")
                self.end_headers()
                self.wfile.write(body)

            def _json(self, code: int, obj) -> None:
                self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

            def _guard(self) -> bool:
                if not self._loopback_host():
                    self._send(403, b"forbidden", "text/plain")
                    return False
                if not self._authed():
                    self._json(403, {"error": "bad or missing token"})
                    return False
                return True

            def _brain(self, raw: str | None) -> Brain:
                return Brain.OFFLINE if (raw or "").lower() == "offline" else Brain.CLOUD

            # -- routes --
            def do_GET(self):  # noqa: N802
                route = urlparse(self.path).path
                if route in ("/", "/index.html"):
                    if not self._loopback_host():
                        self._send(403, b"forbidden", "text/plain")
                        return
                    self._send(200, _PAGE.replace("__TOKEN__", app.token).encode("utf-8"),
                               "text/html; charset=utf-8")
                    return
                if not self._guard():
                    return
                if route == "/api/state":
                    brain = self._brain(parse_qs(urlparse(self.path).query).get("brain", [""])[0])
                    self._json(200, {"preflight": app.preflight(brain), "job": app.job.snapshot()})
                elif route == "/api/status":
                    self._json(200, app.job.snapshot())
                elif route == "/api/digest":
                    q = parse_qs(urlparse(self.path).query)
                    personal = q.get("personal", ["0"])[0] == "1"
                    try:
                        self._send(200, app.digest_html(personal).encode("utf-8"),
                                   "text/html; charset=utf-8")
                    except Exception as exc:  # noqa: BLE001
                        self._send(500, f"could not build the digest: {exc}".encode(), "text/plain")
                else:
                    self._json(404, {"error": "no such route"})

            def do_POST(self):  # noqa: N802
                if not self._guard():
                    return
                route = urlparse(self.path).path
                length = int(self.headers.get("Content-Length") or 0)
                if length > _MAX_UPLOAD_BYTES:
                    self._json(413, {"error": "file is too large"})
                    return
                body = self.rfile.read(length) if length else b""

                if route == "/api/settings":
                    try:
                        data = json.loads(body or b"{}")
                    except ValueError:
                        self._json(400, {"error": "bad json"})
                        return
                    app.apply_settings(data.get("passphrase"), data.get("groq_key"))
                    self._json(200, {"preflight": app.preflight(self._brain(data.get("brain")))})
                elif route == "/api/upload":
                    try:
                        dest = app.save_upload(self.headers.get("X-Filename", ""), body)
                    except ValueError as exc:
                        self._json(400, {"error": str(exc)})
                        return
                    self._json(200, {"saved": dest.name})
                elif route == "/api/process":
                    try:
                        data = json.loads(body or b"{}")
                    except ValueError:
                        data = {}
                    started = app.start_processing(self._brain(data.get("brain")))
                    self._json(200 if started else 409,
                               {"started": started, "error": None if started else "already running"})
                else:
                    self._json(404, {"error": "no such route"})

        return ThreadingHTTPServer((host, port), Handler)


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Plaud Bridge</title>
<style>
  :root { color-scheme: light dark; --bg:#fff; --fg:#1c1c1e; --muted:#6b6b70;
    --line:#e3e3e8; --accent:#2b6cb0; --ok:#1a7f37; --bad:#c0392b; --card:#f7f7f9; }
  @media (prefers-color-scheme: dark) { :root { --bg:#161618; --fg:#ececef;
    --muted:#9a9aa2; --line:#2c2c30; --accent:#6aa9e9; --ok:#4ac26b; --bad:#e57373; --card:#1e1e21; } }
  * { box-sizing: border-box; }
  body { margin:0; font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;
    background:var(--bg); color:var(--fg); }
  .wrap { max-width:720px; margin:0 auto; padding:24px 18px 60px; }
  h1 { font-size:1.5rem; margin:.2rem 0 .1rem; }
  p.sub { color:var(--muted); margin:.1rem 0 1.4rem; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:12px;
    padding:16px 18px; margin:14px 0; }
  label { display:block; font-weight:600; margin:.6rem 0 .3rem; }
  input[type=text], input[type=password] { width:100%; padding:9px 11px; font-size:15px;
    border:1px solid var(--line); border-radius:8px; background:var(--bg); color:var(--fg); }
  .row { display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
  .brain { display:flex; gap:8px; }
  .brain button { flex:1; }
  button { font:inherit; font-weight:600; padding:10px 16px; border-radius:9px;
    border:1px solid var(--line); background:var(--bg); color:var(--fg); cursor:pointer; }
  button.primary { background:var(--accent); color:#fff; border-color:transparent; }
  button.sel { outline:2px solid var(--accent); }
  button:disabled { opacity:.5; cursor:not-allowed; }
  .check { display:flex; gap:8px; padding:4px 0; }
  .dot { width:10px; height:10px; border-radius:50%; margin-top:6px; flex:none; }
  .dot.ok{background:var(--ok)} .dot.bad{background:var(--bad)} .dot.warn{background:#d99e00}
  .name{font-weight:600} .detail{color:var(--muted); font-size:.9rem}
  #log { white-space:pre-wrap; font:13px/1.5 ui-monospace,Menlo,Consolas,monospace;
    background:var(--bg); border:1px solid var(--line); border-radius:8px; padding:10px;
    max-height:220px; overflow:auto; margin-top:10px; }
  .files { color:var(--muted); font-size:.9rem; margin-top:8px; }
  small { color:var(--muted); }
</style></head>
<body><div class="wrap">
  <h1>Plaud Bridge</h1>
  <p class="sub">Turn your recordings into a digest. Nothing private leaves this computer.</p>

  <div class="card">
    <label>1 &middot; Your passphrase <small>(encrypts private recordings &mdash; there is no recovery)</small></label>
    <input id="pass" type="password" placeholder="at least 16 characters" autocomplete="off">
    <label>Analysis brain</label>
    <div class="brain">
      <button id="b-offline" onclick="setBrain('offline')">Offline &middot; fully private</button>
      <button id="b-cloud" class="sel" onclick="setBrain('cloud')">Free cloud key</button>
    </div>
    <div id="groqwrap" style="display:none">
      <label>Free Groq key <small>(get one free at groq.com)</small></label>
      <input id="groq" type="password" placeholder="gsk_..." autocomplete="off">
    </div>
    <div class="row" style="margin-top:12px">
      <button onclick="saveSettings()">Save</button>
      <span id="saved" class="detail"></span>
    </div>
  </div>

  <div class="card">
    <label>2 &middot; Readiness</label>
    <div id="checks"><span class="detail">checking...</span></div>
  </div>

  <div class="card">
    <label>3 &middot; Pick your recordings</label>
    <input id="files" type="file" multiple>
    <div class="files" id="filelist"></div>
    <div class="row" style="margin-top:12px">
      <button id="go" class="primary" onclick="run()">Process</button>
      <button id="view" onclick="openDigest()" disabled>Open digest</button>
    </div>
    <div id="log" style="display:none"></div>
  </div>

  <p><small>Family and spousal recordings are always processed offline and encrypted, whatever is chosen above.</small></p>
</div>
<script>
const TOKEN="__TOKEN__"; let BRAIN="cloud"; let POLL=null;
const H={"X-Token":TOKEN,"Content-Type":"application/json"};
function setBrain(b){BRAIN=b;
  document.getElementById("b-offline").classList.toggle("sel",b==="offline");
  document.getElementById("b-cloud").classList.toggle("sel",b==="cloud");
  document.getElementById("groqwrap").style.display=b==="cloud"?"block":"none";
  refresh();}
async function saveSettings(){
  const body={passphrase:document.getElementById("pass").value,
    groq_key:document.getElementById("groq").value, brain:BRAIN};
  await fetch("/api/settings",{method:"POST",headers:H,body:JSON.stringify(body)});
  document.getElementById("saved").textContent="saved"; refresh();}
function renderChecks(items){
  const el=document.getElementById("checks"); el.innerHTML="";
  let blocked=false;
  for(const it of items){
    const cls=it.ok?"ok":(it.fatal?"bad":"warn"); if(it.fatal&&!it.ok) blocked=true;
    const d=document.createElement("div"); d.className="check";
    d.innerHTML=`<span class="dot ${cls}"></span><div><div class="name">${it.name}</div><div class="detail">${it.detail}</div></div>`;
    el.appendChild(d);}
  document.getElementById("go").disabled=blocked;}
async function refresh(){
  const r=await fetch("/api/state?brain="+BRAIN+"&token="+TOKEN);
  const j=await r.json(); renderChecks(j.preflight);}
document.getElementById("files").addEventListener("change",e=>{
  const names=[...e.target.files].map(f=>f.name).join(", ");
  document.getElementById("filelist").textContent=names?("selected: "+names):"";});
async function run(){
  const files=document.getElementById("files").files;
  if(!files.length){alert("Pick at least one recording first.");return;}
  const log=document.getElementById("log"); log.style.display="block"; log.textContent="Uploading...";
  document.getElementById("go").disabled=true; document.getElementById("view").disabled=true;
  for(const f of files){
    await fetch("/api/upload",{method:"POST",
      headers:{"X-Token":TOKEN,"X-Filename":f.name},body:f});}
  await fetch("/api/process",{method:"POST",headers:H,body:JSON.stringify({brain:BRAIN})});
  POLL=setInterval(poll,1000); poll();}
async function poll(){
  const j=await (await fetch("/api/status?token="+TOKEN)).json();
  const log=document.getElementById("log"); log.textContent=j.lines.join("\\n");
  log.scrollTop=log.scrollHeight;
  if(!j.running){clearInterval(POLL);
    document.getElementById("go").disabled=false;
    if(j.error){log.textContent+="\\n\\nSomething went wrong: "+j.error;}
    else{document.getElementById("view").disabled=false;}}}
function openDigest(){window.open("/api/digest?token="+TOKEN,"_blank");}
refresh();
</script>
</body></html>"""
