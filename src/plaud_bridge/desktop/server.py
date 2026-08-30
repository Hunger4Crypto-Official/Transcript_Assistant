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
        self._pending_update = None
        self._httpd = None
        # Phone mode: names beyond loopback the Host check may accept, and the
        # URL a phone on the same network opens. Both empty until a launcher
        # explicitly turns phone mode on -- reachable-from-the-network is never
        # a default for a tool holding recordings.
        self._extra_hosts: set[str] = set()
        self.lan_url = ""
        self.controller.ensure_installed()

    def enable_phone(self, lan_ip: str, port: int) -> str:
        """
        Let this machine's Wi-Fi address reach the app, and say where.

        The token stays mandatory on every request -- phone mode widens WHERE
        the app answers, never WHO it answers. The returned URL carries the
        token, so opening it on the phone is the authentication.
        """
        self._extra_hosts.add(lan_ip)
        self.lan_url = f"http://{lan_ip}:{port}/?token={self.token}"
        return self.lan_url

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

    def digest_html(self, include_personal: bool, days: int = 3650) -> str:
        out = self.controller.write_digest(days=days, include_personal=include_personal)
        return out.read_text(encoding="utf-8")

    def update_check(self) -> dict:
        """
        Whether a newer release exists. Soft in every failure direction: the
        page renders "no banner" for offline, private-repo-without-token, and
        no-releases alike. `checks_disabled` is reported so the page can say
        nothing at all rather than "up to date" it cannot know.
        """
        from . import update

        if update.checks_disabled():
            return {"available": False, "disabled": True}
        info = update.check_for_update()
        if info is None:
            return {"available": False}
        self._pending_update = info
        return {"available": True, "version": info.version,
                "current": info.current, "notes": info.notes}

    def update_apply(self) -> dict:
        """
        Install the update the person clicked on, then schedule our own exit.

        The verified download and the handoff script are update.apply_update's
        job; this only refuses when there is nothing pending (the page must
        check first -- applying an update the server never offered would let a
        stale click install who-knows-what) and, on success, stops the server a
        beat after the response flushes so the updater script can take over.
        """
        from . import update

        info = getattr(self, "_pending_update", None)
        if info is None:
            return {"applied": False, "error": "no update has been offered; check first"}
        try:
            message = update.apply_update(info)
        except update.UpdateError as exc:
            return {"applied": False, "error": str(exc)}
        threading.Timer(1.5, self._shutdown_for_update).start()
        return {"applied": True, "message": message}

    def _shutdown_for_update(self) -> None:  # pragma: no cover - exercised on Windows
        httpd = getattr(self, "_httpd", None)
        if httpd is not None:
            threading.Thread(target=httpd.shutdown, daemon=True).start()

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

            def _host_allowed(self) -> bool:
                """Loopback always; the LAN address only when phone mode is on."""
                if self._loopback_host():
                    return True
                host_header = (self.headers.get("Host") or "").split(":")[0]
                return host_header in app._extra_hosts

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
                if not self._host_allowed():
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
                    if not self._host_allowed():
                        self._send(403, b"forbidden", "text/plain")
                        return
                    # The page embeds the token, so WHO gets the page is the
                    # whole game. On loopback, serving it bare is fine -- only
                    # local processes can connect. Over the LAN it must be
                    # earned: the phone URL carries the token, and a bare GET
                    # from someone else on the Wi-Fi gets nothing.
                    if not self._loopback_host() and not self._authed():
                        self._send(403, b"forbidden", "text/plain")
                        return
                    self._send(200, _PAGE.replace("__TOKEN__", app.token).encode("utf-8"),
                               "text/html; charset=utf-8")
                    return
                if route == "/manifest.webmanifest":
                    # What lets a phone's "Add to Home Screen" install this as
                    # an app. Token-guarded like everything else; the page
                    # declares the link with its token attached.
                    if not self._guard():
                        return
                    icon = (
                        "data:image/svg+xml,"
                        "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E"
                        "%3Crect width='100' height='100' rx='20' fill='%232b6cb0'/%3E"
                        "%3Cg fill='%23fff'%3E%3Crect x='24' y='38' width='9' height='24' rx='4'/%3E"
                        "%3Crect x='38' y='28' width='9' height='44' rx='4'/%3E"
                        "%3Crect x='52' y='42' width='9' height='16' rx='4'/%3E"
                        "%3Crect x='66' y='32' width='9' height='36' rx='4'/%3E%3C/g%3E%3C/svg%3E"
                    )
                    manifest = {
                        "name": "Plaud Bridge",
                        "short_name": "Plaud",
                        "start_url": f"/?token={app.token}",
                        "display": "standalone",
                        "background_color": "#161618",
                        "theme_color": "#2b6cb0",
                        "icons": [{"src": icon, "sizes": "any", "type": "image/svg+xml"}],
                    }
                    self._send(200, json.dumps(manifest).encode("utf-8"),
                               "application/manifest+json")
                    return
                if not self._guard():
                    return
                if route == "/api/state":
                    brain = self._brain(parse_qs(urlparse(self.path).query).get("brain", [""])[0])
                    from .. import __version__

                    self._json(200, {"preflight": app.preflight(brain),
                                     "job": app.job.snapshot(), "version": __version__,
                                     "lan_url": app.lan_url})
                elif route == "/api/status":
                    self._json(200, app.job.snapshot())
                elif route == "/api/digest":
                    q = parse_qs(urlparse(self.path).query)
                    personal = q.get("personal", ["0"])[0] == "1"
                    try:
                        days = max(1, min(int(q.get("days", ["3650"])[0]), 3650))
                    except ValueError:
                        days = 3650
                    try:
                        self._send(200, app.digest_html(personal, days).encode("utf-8"),
                                   "text/html; charset=utf-8")
                    except Exception as exc:  # noqa: BLE001
                        self._send(500, f"could not build the digest: {exc}".encode(), "text/plain")
                elif route == "/api/update/check":
                    try:
                        self._json(200, app.update_check())
                    except Exception:  # noqa: BLE001 - a failed check is "no banner"
                        self._json(200, {"available": False})
                elif route == "/api/recordings":
                    self._json(200, {"recordings": app.controller.recent_recordings()})
                elif route == "/api/search":
                    q = parse_qs(urlparse(self.path).query).get("q", [""])[0]
                    self._json(200, app.controller.search(q))
                elif route == "/api/followups":
                    status = parse_qs(urlparse(self.path).query).get("status", ["open"])[0]
                    self._json(200, app.controller.followups(None if status == "all" else status))
                elif route == "/api/quarantine":
                    self._json(200, {"entries": app.controller.quarantine()})
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
                elif route == "/api/update/apply":
                    result = app.update_apply()
                    self._json(200 if result.get("applied") else 400, result)
                elif route == "/api/ask":
                    try:
                        data = json.loads(body or b"{}")
                    except ValueError:
                        self._json(400, {"error": "bad json"})
                        return
                    question = str(data.get("question") or "").strip()
                    if not question:
                        self._json(400, {"error": "ask something first"})
                        return
                    # Answering can take a while (retrieval + a model call), but
                    # it is a read: safe to run on the request thread, and the
                    # page shows its own "thinking" state meanwhile.
                    self._json(200, app.controller.ask(
                        question, self._brain(data.get("brain")),
                        include_personal=bool(data.get("personal")),
                    ))
                elif route == "/api/followups/mark":
                    try:
                        data = json.loads(body or b"{}")
                    except ValueError:
                        self._json(400, {"error": "bad json"})
                        return
                    result = app.controller.followup_mark(
                        str(data.get("id") or ""), str(data.get("status") or ""))
                    self._json(200 if result["ok"] else 400, result)
                elif route == "/api/quarantine/act":
                    try:
                        data = json.loads(body or b"{}")
                    except ValueError:
                        self._json(400, {"error": "bad json"})
                        return
                    rid = str(data.get("id") or "")
                    action = str(data.get("action") or "")
                    if action == "release":
                        result = app.controller.quarantine_release(rid)
                    elif action == "forget":
                        result = app.controller.quarantine_forget(rid)
                    else:
                        result = {"ok": False, "error": "unknown action"}
                    self._json(200 if result["ok"] else 400, result)
                elif route == "/api/backup":
                    result = app.controller.backup()
                    self._json(200 if result["ok"] else 400, result)
                else:
                    self._json(404, {"error": "no such route"})

        httpd = ThreadingHTTPServer((host, port), Handler)
        # The apply step needs a handle to stop the server once the updater
        # script is spawned; stashing it here keeps make_server's contract
        # (build and return, never serve) unchanged.
        self._httpd = httpd
        return httpd


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Plaud Bridge</title>
<meta name="theme-color" content="#2b6cb0">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="mobile-web-app-capable" content="yes">
<style>
  :root { color-scheme: light dark; --bg:#fff; --fg:#1c1c1e; --muted:#6b6b70;
    --line:#e3e3e8; --accent:#2b6cb0; --ok:#1a7f37; --bad:#c0392b; --warn:#b7791f; --card:#f7f7f9; }
  @media (prefers-color-scheme: dark) { :root { --bg:#161618; --fg:#ececef;
    --muted:#9a9aa2; --line:#2c2c30; --accent:#6aa9e9; --ok:#4ac26b; --bad:#e57373; --warn:#d9a441; --card:#1e1e21; } }
  * { box-sizing: border-box; }
  body { margin:0; font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;
    background:var(--bg); color:var(--fg); }
  .wrap { max-width:860px; margin:0 auto; padding:20px 18px 60px; }
  h1 { font-size:1.4rem; margin:.2rem 0 .1rem; }
  p.sub { color:var(--muted); margin:.1rem 0 1rem; }
  .tabs { display:flex; gap:4px; flex-wrap:wrap; border-bottom:1px solid var(--line);
    margin:0 0 14px; position:sticky; top:0; background:var(--bg); padding-top:6px; z-index:5; }
  .tabs button { border:none; background:none; font-weight:600; color:var(--muted);
    padding:9px 13px; border-radius:9px 9px 0 0; border-bottom:2px solid transparent; }
  .tabs button.sel { color:var(--fg); border-bottom-color:var(--accent); }
  .card { background:var(--card); border:1px solid var(--line); border-radius:12px;
    padding:16px 18px; margin:14px 0; overflow-x:auto; }
  label { display:block; font-weight:600; margin:.6rem 0 .3rem; }
  input[type=text], input[type=password], textarea, select {
    width:100%; padding:9px 11px; font-size:15px; font-family:inherit;
    border:1px solid var(--line); border-radius:8px; background:var(--bg); color:var(--fg); }
  textarea { resize:vertical; min-height:64px; }
  .row { display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
  .brain { display:flex; gap:8px; } .brain button { flex:1; }
  button { font:inherit; font-weight:600; padding:9px 15px; border-radius:9px;
    border:1px solid var(--line); background:var(--bg); color:var(--fg); cursor:pointer; }
  button.primary { background:var(--accent); color:#fff; border-color:transparent; }
  button.small { padding:4px 10px; font-size:.85rem; }
  button.danger { color:var(--bad); }
  button.sel { outline:2px solid var(--accent); }
  button:disabled { opacity:.5; cursor:not-allowed; }
  .check { display:flex; gap:8px; padding:4px 0; }
  .dot { width:10px; height:10px; border-radius:50%; margin-top:6px; flex:none; }
  .dot.ok{background:var(--ok)} .dot.bad{background:var(--bad)} .dot.warn{background:var(--warn)}
  .name{font-weight:600} .detail{color:var(--muted); font-size:.9rem}
  #log { white-space:pre-wrap; font:13px/1.5 ui-monospace,Menlo,Consolas,monospace;
    background:var(--bg); border:1px solid var(--line); border-radius:8px; padding:10px;
    max-height:220px; overflow:auto; margin-top:10px; }
  .files { color:var(--muted); font-size:.9rem; margin-top:8px; }
  small { color:var(--muted); }
  table.list { width:100%; border-collapse:collapse; font-size:.92rem; }
  table.list th { text-align:left; color:var(--muted); font-weight:600;
    border-bottom:1px solid var(--line); padding:6px 8px; }
  table.list td { padding:7px 8px; border-bottom:1px solid var(--line); vertical-align:top; }
  .badge { display:inline-block; font-size:.75rem; font-weight:700; padding:1px 8px;
    border-radius:99px; border:1px solid var(--line); color:var(--muted); }
  .badge.refusal { color:#fff; background:var(--bad); border-color:transparent; }
  .badge.no-announcement { color:#fff; background:var(--warn); border-color:transparent; }
  .badge.personal { color:#fff; background:#9f3a6d; border-color:transparent; }
  .hit { border-left:3px solid var(--accent); padding:6px 10px; margin:8px 0; }
  .hit .meta { color:var(--muted); font-size:.85rem; }
  .cite { border-left:3px solid var(--ok); padding:4px 10px; margin:6px 0; font-size:.92rem; }
  .empty { color:var(--muted); padding:14px 4px; }
  .notice { border-left:3px solid var(--warn); padding:6px 10px; margin:8px 0; color:var(--muted); }
  #askanswer { white-space:pre-wrap; }
  .footer { color:var(--muted); font-size:.85rem; margin-top:20px; }
</style></head>
<body><div class="wrap">
  <h1>Plaud Bridge</h1>
  <p class="sub">Your recordings, digested. Nothing private leaves this computer.</p>

  <div class="card" id="updatebar" style="display:none">
    <div class="row">
      <div style="flex:1">
        <span class="name" id="updatetitle">Update available</span>
        <div class="detail" id="updatenotes"></div>
      </div>
      <button class="primary" id="updatebtn" onclick="applyUpdate()">Update now</button>
    </div>
    <div class="detail" id="updatemsg"></div>
  </div>

  <div class="tabs" id="tabs">
    <button data-tab="home" class="sel" onclick="showTab('home')">Process</button>
    <button data-tab="library" onclick="showTab('library')">Library</button>
    <button data-tab="search" onclick="showTab('search')">Search</button>
    <button data-tab="ask" onclick="showTab('ask')">Ask</button>
    <button data-tab="followups" onclick="showTab('followups')">Follow-ups</button>
    <button data-tab="held" onclick="showTab('held')">Held</button>
    <button data-tab="tools" onclick="showTab('tools')">Tools</button>
  </div>

  <!-- ============ Process ============ -->
  <div id="tab-home">
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

  <!-- ============ Library ============ -->
  <div id="tab-library" hidden>
    <div class="card">
      <div class="row">
        <label style="margin:0">Digest window</label>
        <select id="digestdays" style="width:auto">
          <option value="7">Last 7 days</option>
          <option value="30">Last 30 days</option>
          <option value="90">Last 90 days</option>
          <option value="365">Last year</option>
          <option value="3650" selected>Everything</option>
        </select>
        <label style="margin:0; font-weight:400"><input type="checkbox" id="digestpersonal"> include personal</label>
        <button class="primary" onclick="openDigest()">Open digest</button>
      </div>
      <div class="detail" id="personalwarn" style="display:none">
        A digest with personal sections in it is a document worth not leaving on a shared screen.
      </div>
    </div>
    <div class="card">
      <div class="row" style="justify-content:space-between">
        <label style="margin:0">Recordings</label>
        <button class="small" onclick="loadLibrary()">Refresh</button>
      </div>
      <div id="librarylist" class="empty">loading...</div>
    </div>
  </div>

  <!-- ============ Search ============ -->
  <div id="tab-search" hidden>
    <div class="card">
      <label>Find what was actually said</label>
      <div class="row">
        <input id="searchq" type="text" placeholder="e.g. elimination period" style="flex:1"
               onkeydown="if(event.key==='Enter')doSearch()">
        <button class="primary" onclick="doSearch()">Search</button>
      </div>
      <div id="searchhonesty" class="detail" style="margin-top:8px"></div>
      <div id="searchout"></div>
    </div>
  </div>

  <!-- ============ Ask ============ -->
  <div id="tab-ask" hidden>
    <div class="card">
      <label>Ask your archive a question</label>
      <textarea id="askq" placeholder="what did I promise the Hendersons about the conversion rider?"></textarea>
      <div class="row" style="margin-top:8px">
        <button class="primary" id="askbtn" onclick="doAsk()">Ask</button>
        <label style="margin:0; font-weight:400"><input type="checkbox" id="askpersonal"> include personal</label>
        <span class="detail" id="askstatus"></span>
      </div>
      <div id="asknote" class="notice" style="display:none"></div>
      <div id="askanswer" style="margin-top:10px"></div>
      <div id="askcites"></div>
    </div>
  </div>

  <!-- ============ Follow-ups ============ -->
  <div id="tab-followups" hidden>
    <div class="card">
      <div class="row" style="justify-content:space-between">
        <label style="margin:0">Commitments, oldest first</label>
        <div class="row">
          <select id="fustatus" style="width:auto" onchange="loadFollowups()">
            <option value="open" selected>Open</option>
            <option value="all">All</option>
          </select>
          <button class="small" onclick="loadFollowups()">Refresh</button>
        </div>
      </div>
      <div id="fulist" class="empty">loading...</div>
    </div>
  </div>

  <!-- ============ Held (quarantine) ============ -->
  <div id="tab-held" hidden>
    <div class="card">
      <div class="row" style="justify-content:space-between">
        <label style="margin:0">Held for review</label>
        <button class="small" onclick="loadHeld()">Refresh</button>
      </div>
      <p class="detail">The consent gate held these. Releasing one is you affirming, as a person,
      that consent was actually obtained. Recordings where someone <b>objected</b> cannot be
      released here at all.</p>
      <div id="heldlist" class="empty">loading...</div>
    </div>
  </div>

  <!-- ============ Tools ============ -->
  <div id="tab-tools" hidden>
    <div class="card">
      <label>Back up everything</label>
      <p class="detail">One encrypted file: vault, index, outbox, quarantine, and your tuned
      config. Safe on an external drive or a cloud folder. Restoring needs your passphrase
      &mdash; there is no recovery without it.</p>
      <div class="row">
        <button class="primary" id="backupbtn" onclick="doBackup()">Back up now</button>
        <span class="detail" id="backupout"></span>
      </div>
    </div>
    <div class="card">
      <label>Use it on your phone</label>
      <div id="phoneoff" class="detail">Phone mode is off. Start the app with
        <b>--phone</b> (or set PLAUD_BRIDGE_PHONE=1) and this card shows the address to open
        on a phone connected to the same Wi-Fi. Once open, use your browser's
        &ldquo;Add to Home Screen&rdquo; and it installs like an app.</div>
      <div id="phoneon" style="display:none">
        <div class="detail">Open this on your phone (same Wi-Fi), then &ldquo;Add to Home Screen&rdquo;:</div>
        <div class="row" style="margin-top:6px">
          <input id="phoneurl" type="text" readonly style="flex:1">
          <button class="small" onclick="navigator.clipboard&&navigator.clipboard.writeText($('phoneurl').value)">Copy</button>
        </div>
        <div class="detail" style="margin-top:6px">Home network only: the link carries this
        session's key, and traffic on your Wi-Fi is not encrypted. Never use phone mode on
        public Wi-Fi. The address changes each time the app starts.</div>
      </div>
    </div>
    <div class="card">
      <label>About</label>
      <div class="detail" id="aboutver">version: ...</div>
      <div class="detail">Updates are offered on this page when a new release exists; nothing
      installs without your click. Set PLAUD_BRIDGE_NO_UPDATE_CHECK=1 to disable checking.</div>
    </div>
  </div>

  <div class="footer">Loopback-only &middot; token-guarded &middot; nothing here can send anything on your behalf.</div>
</div>
<script>
const TOKEN="__TOKEN__"; let BRAIN="cloud"; let POLL=null;
const H={"X-Token":TOKEN,"Content-Type":"application/json"};
const $=id=>document.getElementById(id);
function esc(s){const d=document.createElement("div");d.textContent=String(s==null?"":s);return d.innerHTML;}
async function GET(p){return (await fetch(p+(p.includes("?")?"&":"?")+"token="+TOKEN)).json();}
async function POST(p,b){return (await fetch(p,{method:"POST",headers:H,body:JSON.stringify(b||{})})).json();}

/* ---- tabs ---- */
const LOADERS={library:loadLibrary,followups:loadFollowups,held:loadHeld};
const LOADED={};
function showTab(name){
  for(const b of document.querySelectorAll("#tabs button")) b.classList.toggle("sel",b.dataset.tab===name);
  for(const t of ["home","library","search","ask","followups","held","tools"]) $("tab-"+t).hidden=(t!==name);
  if(LOADERS[name]&&!LOADED[name]){LOADED[name]=true;LOADERS[name]();}}

/* ---- settings + readiness ---- */
function setBrain(b){BRAIN=b;
  $("b-offline").classList.toggle("sel",b==="offline");
  $("b-cloud").classList.toggle("sel",b==="cloud");
  $("groqwrap").style.display=b==="cloud"?"block":"none";
  refresh();}
async function saveSettings(){
  await POST("/api/settings",{passphrase:$("pass").value,groq_key:$("groq").value,brain:BRAIN});
  $("saved").textContent="saved"; refresh();}
function renderChecks(items){
  const el=$("checks"); el.innerHTML=""; let blocked=false;
  for(const it of items){
    const cls=it.ok?"ok":(it.fatal?"bad":"warn"); if(it.fatal&&!it.ok) blocked=true;
    const d=document.createElement("div"); d.className="check";
    d.innerHTML=`<span class="dot ${cls}"></span><div><div class="name">${esc(it.name)}</div><div class="detail">${esc(it.detail)}</div></div>`;
    el.appendChild(d);}
  $("go").disabled=blocked;}
async function refresh(){
  const j=await GET("/api/state?brain="+BRAIN);
  renderChecks(j.preflight);
  if(j.version) $("aboutver").textContent="version: "+j.version;
  if(j.lan_url){$("phoneoff").style.display="none";$("phoneon").style.display="block";
    $("phoneurl").value=j.lan_url;}}

/* ---- process ---- */
$("files").addEventListener("change",e=>{
  const names=[...e.target.files].map(f=>f.name).join(", ");
  $("filelist").textContent=names?("selected: "+names):"";});
async function run(){
  const files=$("files").files;
  if(!files.length){alert("Pick at least one recording first.");return;}
  const log=$("log"); log.style.display="block"; log.textContent="Uploading...";
  $("go").disabled=true; $("view").disabled=true;
  for(const f of files){
    await fetch("/api/upload",{method:"POST",headers:{"X-Token":TOKEN,"X-Filename":f.name},body:f});}
  await POST("/api/process",{brain:BRAIN});
  POLL=setInterval(poll,1000); poll();}
async function poll(){
  const j=await GET("/api/status");
  const log=$("log"); log.textContent=j.lines.join("\n"); log.scrollTop=log.scrollHeight;
  if(!j.running){clearInterval(POLL); $("go").disabled=false;
    if(j.error){log.textContent+="\n\nSomething went wrong: "+j.error;}
    else{$("view").disabled=false; LOADED.library=false; LOADED.followups=false; LOADED.held=false;
      if(j.summary&&j.summary.quarantined>0){
        log.textContent+="\n"+j.summary.quarantined+" recording(s) were held for consent review — see the Held tab.";}}}}
function openDigest(){
  const days=$("digestdays")?$("digestdays").value:"3650";
  const personal=$("digestpersonal")&&$("digestpersonal").checked?"1":"0";
  window.open("/api/digest?token="+TOKEN+"&days="+days+"&personal="+personal,"_blank");}
document.addEventListener("change",e=>{
  if(e.target&&e.target.id==="digestpersonal")
    $("personalwarn").style.display=e.target.checked?"block":"none";});

/* ---- library ---- */
async function loadLibrary(){
  const el=$("librarylist"); el.className=""; el.textContent="loading...";
  const j=await GET("/api/recordings");
  if(!j.recordings.length){el.className="empty";el.textContent="Nothing processed yet. Drop a recording in the Process tab.";return;}
  let html='<table class="list"><tr><th>When</th><th>Recording</th><th>Profile</th><th>Min</th><th></th></tr>';
  for(const r of j.recordings){
    html+=`<tr><td>${esc(r.when)}</td><td>${esc(r.name)}</td>
      <td>${esc(r.profile)}${r.personal?' <span class="badge personal">personal</span>':""}</td>
      <td>${esc(r.minutes)}</td>
      <td>${r.stage==="quarantined"?'<span class="badge">held</span>':(r.encrypted?'<span class="badge">encrypted</span>':"")}</td></tr>`;}
  el.innerHTML=html+"</table>";}

/* ---- search ---- */
async function doSearch(){
  const q=$("searchq").value.trim(); if(!q) return;
  $("searchout").innerHTML='<div class="empty">searching...</div>'; $("searchhonesty").textContent="";
  const j=await GET("/api/search?q="+encodeURIComponent(q));
  const honesty=[];
  honesty.push(`searched ${j.scanned} of ${j.total} recording(s)`);
  if(j.unopened) honesty.push(`${j.unopened} could not be opened — this search is not complete`);
  $("searchhonesty").textContent=honesty.join(" · ");
  if(!j.matches.length){$("searchout").innerHTML='<div class="empty">Nothing said that. '+(j.unopened?"(But some recordings could not be searched.)":"")+"</div>";return;}
  let html="";
  for(const m of j.matches){
    html+=`<div class="hit"><div class="meta">${esc(m.when)} · ${esc(m.source)} · ${esc(m.stamp)}${m.personal?' · <span class="badge personal">personal</span>':""}</div>
      <div>${m.speaker?"<b>"+esc(m.speaker)+":</b> ":""}${esc(m.text)}</div></div>`;}
  $("searchout").innerHTML=html;}

/* ---- ask ---- */
async function doAsk(){
  const q=$("askq").value.trim(); if(!q) return;
  $("askbtn").disabled=true; $("askstatus").textContent="thinking...";
  $("askanswer").textContent=""; $("askcites").innerHTML=""; $("asknote").style.display="none";
  try{
    const j=await POST("/api/ask",{question:q,brain:BRAIN,personal:$("askpersonal").checked});
    $("askanswer").textContent=j.text||"(no answer)";
    const bits=[];
    if(j.confidence) bits.push("confidence: "+j.confidence);
    if(j.cost_usd) bits.push("$"+j.cost_usd);
    $("askstatus").textContent=bits.join(" · ");
    if(j.degraded||j.note||j.unanswered){
      const n=$("asknote"); n.style.display="block";
      n.textContent=[j.degraded?"No model was reachable, so these are ranked excerpts rather than an answer.":"",
                     j.unanswered||"", j.note||""].filter(Boolean).join(" ");}
    let html="";
    for(const c of j.citations)
      html+=`<div class="cite">&ldquo;${esc(c.quote)}&rdquo;<div class="detail">${esc(c.source)} @ ${esc(c.stamp)}</div></div>`;
    $("askcites").innerHTML=html;
  }catch(e){$("askstatus").textContent="could not ask: "+e;}
  $("askbtn").disabled=false;}

/* ---- follow-ups ---- */
async function loadFollowups(){
  const el=$("fulist"); el.className=""; el.textContent="loading...";
  const j=await GET("/api/followups?status="+$("fustatus").value);
  if(j.error){el.className="empty";el.textContent=j.error;return;}
  if(!j.items.length){el.className="empty";el.textContent="Nothing outstanding.";return;}
  let html='<table class="list"><tr><th>Commitment</th><th>Age</th><th>Profile</th><th></th></tr>';
  for(const f of j.items){
    const act=f.status==="open"
      ?`<button class="small" onclick="markFu('${esc(f.id)}','done')">Done</button>
        <button class="small" onclick="markFu('${esc(f.id)}','dropped')">Drop</button>`
      :`<span class="badge">${esc(f.status)}</span> <button class="small" onclick="markFu('${esc(f.id)}','open')">Reopen</button>`;
    html+=`<tr><td>${esc(f.text)}${f.due?'<div class="detail">due: '+esc(f.due)+"</div>":""}</td>
      <td>${esc(f.age_days)}d</td><td>${esc(f.profile)}</td><td>${act}</td></tr>`;}
  el.innerHTML=html+"</table>";}
async function markFu(id,status){
  const j=await POST("/api/followups/mark",{id:id,status:status});
  if(!j.ok) alert(j.error);
  loadFollowups();}

/* ---- held / quarantine ---- */
async function loadHeld(){
  const el=$("heldlist"); el.className=""; el.textContent="loading...";
  const j=await GET("/api/quarantine");
  if(!j.entries.length){el.className="empty";el.textContent="Nothing is held. The gate is quiet.";return;}
  let html='<table class="list"><tr><th>Recording</th><th>Why held</th><th></th></tr>';
  for(const e of j.entries){
    const badge=`<span class="badge ${esc(e.klass)}">${esc(e.klass)}</span>`;
    let act="";
    if(e.released) act='<span class="detail">released</span>';
    else if(e.klass==="refusal") act='<span class="detail">not releasable here</span>';
    else if(e.has_media) act=`<button class="small" onclick="heldAct('${esc(e.id)}','release')">Release</button>`;
    act+=` <button class="small danger" onclick="heldForget('${esc(e.id)}')">Forget</button>`;
    html+=`<tr><td>${esc(e.source)}<div class="detail">${esc(e.when)} · ${esc(e.id)}</div></td>
      <td>${badge}<div class="detail">${esc(e.reason)}</div></td><td>${act}</td></tr>`;}
  el.innerHTML=html+"</table>";}
async function heldAct(id,action){
  const j=await POST("/api/quarantine/act",{id:id,action:action});
  if(!j.ok) alert(j.error); else if(j.note) alert(j.note);
  loadHeld();}
async function heldForget(id){
  const typed=prompt("This permanently deletes the recording and every trace of it.\nType FORGET to confirm:");
  if(typed!=="FORGET") return;
  const j=await POST("/api/quarantine/act",{id:id,action:"forget"});
  if(!j.ok) alert(j.error);
  loadHeld();}

/* ---- tools ---- */
async function doBackup(){
  $("backupbtn").disabled=true; $("backupout").textContent="backing up...";
  try{
    const j=await POST("/api/backup",{});
    $("backupout").textContent=j.ok?("saved "+j.path+" ("+j.size_mb+" MB)"):("could not back up: "+j.error);
  }catch(e){$("backupout").textContent="could not back up: "+e;}
  $("backupbtn").disabled=false;}

/* ---- updates ---- */
async function checkUpdate(){
  try{
    const j=await GET("/api/update/check");
    if(!j.available) return;
    $("updatetitle").textContent="Update available: "+j.version+" (you have "+j.current+")";
    $("updatenotes").textContent=j.notes||"";
    $("updatebar").style.display="block";
  }catch(e){/* no banner is the correct rendering of "could not check" */}}
async function applyUpdate(){
  const btn=$("updatebtn"); btn.disabled=true;
  const msg=$("updatemsg"); msg.textContent="Downloading and verifying...";
  try{
    const j=await POST("/api/update/apply",{});
    if(j.applied){msg.textContent=j.message+" You can close this tab; the app reopens itself.";}
    else{msg.textContent="Could not update: "+(j.error||"unknown"); btn.disabled=false;}
  }catch(e){msg.textContent="Could not update: "+e; btn.disabled=false;}}
const MF=document.createElement("link"); MF.rel="manifest";
MF.href="/manifest.webmanifest?token="+TOKEN; document.head.appendChild(MF);
refresh(); checkUpdate();
</script>
</body></html>"""
