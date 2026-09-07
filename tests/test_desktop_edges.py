"""
The desktop layer's edges: every refusal, fallback, and failure answer.

The happy paths live in the other test_desktop_* files. These pin what the
window does when something is wrong -- a body that is not JSON, a vault that
is locked, a preflight with ffmpeg missing, an update endpoint that blows up,
a player that hangs up mid-stream -- because a privacy tool's failure answers
are part of its contract: a clean 500 with the reason, never a dead socket or
a half-deleted recording. Same discipline as the rest: real HTTP on loopback,
the LLM stubbed, nothing leaves the machine.
"""

from __future__ import annotations

import hashlib
import json
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from _fixtures import CLIENT_CALL, StubLLM
from plaud_bridge.desktop import controller as controller_module
from plaud_bridge.desktop import launch, update
from plaud_bridge.desktop.controller import (
    AppController,
    Brain,
    LocalLLMStatus,
    default_base_dir,
    probe_local_llm,
)
from plaud_bridge.desktop.server import _MAX_UPLOAD_BYTES, AppServer
from plaud_bridge.desktop.update import UpdateError, UpdateInfo
from plaud_bridge.storage import VaultError

ROOT = Path(__file__).resolve().parents[1]
PASSPHRASE = "a-long-enough-desktop-passphrase"

# Routes to the sales_trainer profile, the one shipped profile with
# encrypt_at_rest false, so its original stays plaintext in inbox/_processed.
TRAINING_CALL = """\
Sasson: Let's run a role play on objection handling for the discovery script.
Trainee: Okay, give me the toughest close you have.
Sasson: Three questions about your pipeline and appointment set activity first.
Trainee: My activity was ninety dials this week.
"""


class TrainerStub(StubLLM):
    """The shared stub, plus a correct router score for coaching content."""

    def __call__(self, cfg, system, user, local_only=False, max_tokens=None):
        if '"scores"' in user and "role play" in user.lower():
            self.calls.append({"local_only": local_only, "system": system[:80]})
            return {"scores": [
                {"profile_id": "sales_trainer", "score": 0.95,
                 "evidence": ["role play coaching session"]},
            ]}, self._response()
        return super().__call__(cfg, system, user, local_only, max_tokens)


# =========================================================================
# Fixtures and helpers
# =========================================================================
@pytest.fixture
def app(tmp_path, monkeypatch):
    stub = TrainerStub()
    for module in ("plaud_bridge.profiles.router", "plaud_bridge.profiles.extractor"):
        monkeypatch.setattr(f"{module}.complete_json", stub)
    monkeypatch.setenv("PLAUD_BRIDGE_PASSPHRASE", PASSPHRASE)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("PLAUD_BRIDGE_NO_UPDATE_CHECK", raising=False)
    return AppController(base_dir=tmp_path / "home", template_dir=ROOT / "config")


@dataclass
class Live:
    controller: AppController
    app: AppServer
    httpd: ThreadingHTTPServer
    base: str
    token: str


@pytest.fixture
def live(app):
    srv = AppServer(app)
    httpd = srv.make_server("127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        yield Live(app, srv, httpd, base, srv.token)
    finally:
        httpd.shutdown()
        httpd.server_close()


def _req(live: Live, path: str, *, method="GET", body=None, headers=None,
         token=True, host=None):
    h = dict(headers or {})
    if token:
        h["X-Token"] = live.token
    if host:
        h["Host"] = host
    data = None
    if body is not None:
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        h.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(live.base + path, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read(), dict(resp.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), dict(exc.headers)


def _raw(live: Live, request: bytes) -> bytes:
    """One request over a bare socket, for shapes urllib will not send."""
    host, port = live.httpd.server_address
    with socket.create_connection((host, port), timeout=10) as sock:
        sock.sendall(request)
        chunks = []
        while chunk := sock.recv(65536):
            chunks.append(chunk)
    return b"".join(chunks)


def _feed(controller: AppController, name: str, body: str) -> dict:
    picked = controller.base_dir / name
    picked.parent.mkdir(parents=True, exist_ok=True)
    picked.write_text(body, encoding="utf-8")
    controller.add_files([picked])
    return controller.process(Brain.CLOUD)


def _wait_job(live: Live, seconds: float = 60.0) -> dict:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        _status, body, _ = _req(live, "/api/status")
        snap = json.loads(body)
        if not snap["running"]:
            return snap
        time.sleep(0.05)
    raise AssertionError("the processing job never finished")


@contextmanager
def _http_stub(status: int, body: bytes):
    """A loopback server answering every GET the same way."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


# =========================================================================
# Server: the guards, on every method
# =========================================================================
def test_an_api_call_with_a_foreign_host_header_is_forbidden_even_with_the_token(live):
    status, body, _ = _req(live, "/api/state?brain=cloud", host="evil.example.com")
    assert status == 403 and body == b"forbidden"


def test_a_post_without_the_token_is_refused_before_any_body_is_acted_on(live):
    status, body, _ = _req(live, "/api/settings", method="POST",
                           body={"passphrase": "x"}, token=False)
    assert status == 403
    assert json.loads(body)["error"] == "bad or missing token"
    assert live.controller.preflight(Brain.CLOUD)[-1].ok, "the bare POST changed the passphrase"


def test_unknown_routes_answer_404_for_get_and_post(live):
    for method in ("GET", "POST"):
        status, body, _ = _req(live, "/api/nothing-here", method=method,
                               body=b"" if method == "POST" else None)
        assert status == 404, method
        assert json.loads(body)["error"] == "no such route"


# =========================================================================
# Server: bodies that are not JSON
# =========================================================================
@pytest.mark.parametrize("route", [
    "/api/settings", "/api/ask", "/api/followups/mark", "/api/quarantine/act",
])
def test_a_body_that_is_not_json_is_a_400_naming_the_problem(live, route):
    status, body, _ = _req(live, route, method="POST", body=b"{not json")
    assert status == 400
    assert json.loads(body) == {"error": "bad json"}


def test_a_process_request_with_a_broken_body_still_starts_on_the_cloud_brain(live, monkeypatch):
    seen: list[Brain] = []

    def fake_process(brain, progress=None):
        seen.append(brain)
        return {"processed": 0, "quarantined": 0, "failed": 0, "skipped": 0, "cost_usd": 0.0}

    monkeypatch.setattr(live.controller, "process", fake_process)
    status, body, _ = _req(live, "/api/process", method="POST", body=b"{not json")
    assert status == 200 and json.loads(body)["started"] is True
    snap = _wait_job(live)
    assert snap["error"] is None and seen == [Brain.CLOUD]


# =========================================================================
# Server: settings
# =========================================================================
def test_the_settings_route_applies_both_secrets_and_the_cloud_brain_goes_green(live, monkeypatch):
    monkeypatch.delenv("PLAUD_BRIDGE_PASSPHRASE")
    status, body, _ = _req(live, "/api/settings", method="POST", body={
        "passphrase": "a-brand-new-passphrase-typed-in", "groq_key": " gsk_test ",
        "brain": "cloud"})
    assert status == 200
    import os

    assert os.environ["PLAUD_BRIDGE_PASSPHRASE"] == "a-brand-new-passphrase-typed-in"
    assert os.environ["GROQ_API_KEY"] == "gsk_test", "the key was not stripped"
    items = {i["name"]: i for i in json.loads(body)["preflight"]}
    assert items["passphrase"]["ok"] is True
    brain = items["analysis brain (cloud)"]
    assert brain["ok"] is True and "groq" in brain["detail"]


def test_apply_settings_with_none_leaves_each_secret_exactly_as_it_was(app, monkeypatch):
    import os

    monkeypatch.setenv("GROQ_API_KEY", "keep-me")
    srv = AppServer(app)
    srv.apply_settings(None, None)
    assert os.environ["PLAUD_BRIDGE_PASSPHRASE"] == PASSPHRASE
    assert os.environ["GROQ_API_KEY"] == "keep-me"
    srv.apply_settings("only-the-passphrase-changes", None)
    assert os.environ["PLAUD_BRIDGE_PASSPHRASE"] == "only-the-passphrase-changes"
    assert os.environ["GROQ_API_KEY"] == "keep-me"
    srv.apply_settings(None, "")
    assert os.environ["GROQ_API_KEY"] == "", "an explicit empty key must clear it"
    assert os.environ["PLAUD_BRIDGE_PASSPHRASE"] == "only-the-passphrase-changes"


# =========================================================================
# Server: upload ceiling
# =========================================================================
def test_an_upload_declared_over_the_ceiling_is_refused_before_its_body_is_read(live):
    request = (
        f"POST /api/upload HTTP/1.0\r\nHost: 127.0.0.1\r\nX-Token: {live.token}\r\n"
        f"X-Filename: huge.mp3\r\nContent-Length: {_MAX_UPLOAD_BYTES + 1}\r\n\r\n"
    ).encode()
    # No body follows. If the server tried to read the declared length it
    # would hang until the socket timeout instead of answering.
    raw = _raw(live, request)
    head, _, body = raw.partition(b"\r\n\r\n")
    assert head.startswith(b"HTTP/1.0 413")
    assert json.loads(body) == {"error": "file is too large"}
    inbox = live.controller.load_config(Brain.CLOUD).path("inbox")
    assert not (inbox / "huge.mp3").exists()


# =========================================================================
# Server: a processing crash reaches the page, not the server
# =========================================================================
def test_a_processing_crash_is_reported_in_the_job_and_the_server_survives(live, monkeypatch):
    def explode(brain, progress=None):
        progress("Starting...")
        raise RuntimeError("the pipeline fell over")

    monkeypatch.setattr(live.controller, "process", explode)
    status, body, _ = _req(live, "/api/process", method="POST", body={"brain": "cloud"})
    assert status == 200 and json.loads(body)["started"] is True
    snap = _wait_job(live)
    assert snap["error"] == "the pipeline fell over"
    assert snap["summary"] is None and snap["lines"] == ["Starting..."]
    # And the next run is allowed: the crash released the job lock.
    status, body, _ = _req(live, "/api/process", method="POST", body={})
    assert json.loads(body)["started"] is True
    _wait_job(live)


# =========================================================================
# Server: digest, insights, brief windows and failures
# =========================================================================
def test_a_digest_that_cannot_build_answers_500_with_the_reason(live, monkeypatch):
    def broken(include_personal, days=3650):
        raise RuntimeError("index is unreadable")

    monkeypatch.setattr(live.app, "digest_html", broken)
    status, body, headers = _req(live, "/api/digest")
    assert status == 500
    assert body == b"could not build the digest: index is unreadable"
    assert headers["Content-Type"] == "text/plain"


@pytest.mark.parametrize("raw,expected", [
    ("abc", 3650), ("999999", 3650), ("0", 1), ("-4", 1), ("30", 30),
])
def test_the_digest_window_falls_back_and_clamps(live, monkeypatch, raw, expected):
    seen = []

    def recorder(include_personal, days=3650):
        seen.append((include_personal, days))
        return "<html>digest</html>"

    monkeypatch.setattr(live.app, "digest_html", recorder)
    status, body, _ = _req(live, f"/api/digest?days={raw}&personal=1")
    assert status == 200 and body == b"<html>digest</html>"
    assert seen == [(True, expected)]


def test_the_insights_window_falls_back_to_ninety_days(live, monkeypatch):
    seen = []

    def recorder(days=90, include_personal=False):
        seen.append((days, include_personal))
        return {"report": None, "error": ""}

    monkeypatch.setattr(live.controller, "insights", recorder)
    status, _, _ = _req(live, "/api/insights?days=lots")
    assert status == 200 and seen == [(90, False)]


def test_the_brief_window_falls_back_to_a_week_and_a_failure_is_500(live, monkeypatch):
    seen = []

    def recorder(days=7, include_personal=False, brain=Brain.CLOUD):
        seen.append((days, include_personal, brain))
        return "<html>brief</html>"

    monkeypatch.setattr(live.controller, "brief_html", recorder)
    status, body, _ = _req(live, "/api/brief?days=soon&brain=offline&personal=1")
    assert status == 200 and body == b"<html>brief</html>"
    assert seen == [(7, True, Brain.OFFLINE)]

    def broken(days=7, include_personal=False, brain=Brain.CLOUD):
        raise RuntimeError("no archive")

    monkeypatch.setattr(live.controller, "brief_html", broken)
    status, body, _ = _req(live, "/api/brief")
    assert status == 500 and body == b"could not build the brief: no archive"


# =========================================================================
# Server: update check and apply
# =========================================================================
def test_a_crashing_update_check_renders_as_no_banner(live, monkeypatch):
    def explode(**_kw):
        raise RuntimeError("github exploded")

    monkeypatch.setattr(update, "check_for_update", explode)
    status, body, _ = _req(live, "/api/update/check")
    assert status == 200 and json.loads(body) == {"available": False}


def test_no_newer_release_means_no_banner_and_nothing_pending_to_apply(live, monkeypatch):
    monkeypatch.setattr(update, "check_for_update", lambda **_kw: None)
    status, body, _ = _req(live, "/api/update/check")
    assert status == 200 and json.loads(body) == {"available": False}
    status, body, _ = _req(live, "/api/update/apply", method="POST", body={})
    assert status == 400
    result = json.loads(body)
    assert result["applied"] is False and "check first" in result["error"]


def test_a_successful_apply_reports_it_and_schedules_the_servers_own_exit(live, monkeypatch):
    info = UpdateInfo(version="v99.0.0", current="1.0.0", zip_url="http://127.0.0.1:9/z",
                      checksum_url="http://127.0.0.1:9/z.sha256")
    monkeypatch.setattr(update, "check_for_update", lambda **_kw: info)
    applied = []
    monkeypatch.setattr(update, "apply_update", lambda i: applied.append(i.version) or "restarting")
    stopped = threading.Event()
    monkeypatch.setattr(live.app, "_shutdown_for_update", stopped.set)

    status, body, _ = _req(live, "/api/update/check")
    assert json.loads(body)["available"] is True
    status, body, _ = _req(live, "/api/update/apply", method="POST", body={})
    assert status == 200
    assert json.loads(body) == {"applied": True, "message": "restarting"}
    assert applied == ["v99.0.0"]
    assert stopped.wait(10), "the server never scheduled its own shutdown after applying"


def test_shutdown_for_update_stops_the_server_it_holds_and_tolerates_having_none(app):
    srv = AppServer(app)
    srv._httpd = None
    srv._shutdown_for_update()   # nothing to stop; must not raise

    stopped = threading.Event()
    srv._httpd = SimpleNamespace(shutdown=stopped.set)
    srv._shutdown_for_update()
    assert stopped.wait(10), "shutdown was never called on the held server"


# =========================================================================
# Server: the media route's failure answers
# =========================================================================
def test_a_locked_vault_answers_media_with_a_clean_json_500(live, monkeypatch):
    _feed(live.controller, "client-marcus.txt", CLIENT_CALL)
    rid = live.controller.recent_recordings()[0]["id"]
    assert live.controller.recent_recordings()[0]["encrypted"] is True

    monkeypatch.delenv("PLAUD_BRIDGE_PASSPHRASE")
    status, body, headers = _req(live, f"/api/media?id={rid}")
    assert status == 500
    assert headers["Content-Type"] == "application/json"
    error = json.loads(body)["error"]
    assert "PLAUD_BRIDGE_PASSPHRASE" in error and "not set" in error
    # The transcript beside it says the same thing rather than going blank.
    status, body, _ = _req(live, f"/api/transcript?id={rid}")
    assert status == 404 and "passphrase" in json.loads(body)["error"]

    # Unlocking again is enough: the server needed no restart.
    monkeypatch.setenv("PLAUD_BRIDGE_PASSPHRASE", PASSPHRASE)
    status, body, _ = _req(live, f"/api/media?id={rid}")
    assert status == 200 and body == CLIENT_CALL.encode()


def test_a_plaintext_original_without_a_range_declares_its_length_and_offers_ranges(live):
    assert _feed(live.controller, "training.txt", TRAINING_CALL)["processed"] == 1
    row = live.controller.recent_recordings()[0]
    assert row["encrypted"] is False
    status, body, headers = _req(live, f"/api/media?id={row['id']}")
    assert status == 200 and body == TRAINING_CALL.encode()
    assert headers["Content-Length"] == str(len(TRAINING_CALL.encode()))
    assert headers["Accept-Ranges"] == "bytes"


def _fake_stream(chunks, total=None, encrypted=True):
    return {"encrypted": encrypted, "content_type": "audio/mpeg", "iter": iter(chunks),
            "total": total, "partial": False, "start": 0, "stop": None}


def test_a_stream_of_many_chunks_is_delivered_whole_and_in_order(live, monkeypatch):
    chunks = [b"one-", b"two-", b"three"]
    monkeypatch.setattr(live.controller, "media_stream",
                        lambda rid, start=None, end=None: _fake_stream(chunks))
    status, body, headers = _req(live, "/api/media?id=x")
    assert status == 200 and body == b"one-two-three"
    assert "Content-Length" not in headers and headers["Accept-Ranges"] == "none"


def test_tampering_found_mid_stream_stops_the_body_and_never_troubles_the_server(live, monkeypatch):
    """
    What was sent was sent; what follows is nothing. The handler swallows the
    vault's error so the serving thread ends cleanly (no handle_error) and the
    next request is answered as if nothing happened.
    """
    def tampered():
        yield b"first-chunk"
        raise VaultError("chunk 1 failed authentication")

    monkeypatch.setattr(live.controller, "media_stream",
                        lambda rid, start=None, end=None: _fake_stream(tampered()))
    errors = []
    monkeypatch.setattr(live.httpd, "handle_error", lambda req, addr: errors.append(addr))
    status, body, _ = _req(live, "/api/media?id=x")
    assert status == 200 and body == b"first-chunk"
    assert errors == [], "the mid-stream vault error escaped the handler"
    status, _, _ = _req(live, "/api/status")
    assert status == 200


def test_a_player_that_hangs_up_mid_stream_is_not_an_error(live, monkeypatch):
    finished = threading.Event()

    def endless():
        try:
            for _ in range(256):
                yield b"x" * (1 << 20)
        finally:
            finished.set()

    monkeypatch.setattr(live.controller, "media_stream",
                        lambda rid, start=None, end=None: _fake_stream(endless()))
    errors = []
    monkeypatch.setattr(live.httpd, "handle_error", lambda req, addr: errors.append(addr))

    host, port = live.httpd.server_address
    with socket.create_connection((host, port), timeout=10) as sock:
        sock.sendall(f"GET /api/media?id=x&token={live.token} HTTP/1.0\r\n"
                     f"Host: 127.0.0.1\r\n\r\n".encode())
        assert sock.recv(4096).startswith(b"HTTP/1.0 200")
    # The player is gone; the server's writes fail and the stream is dropped.
    assert finished.wait(30), "the server kept streaming to a closed connection"
    assert errors == [], "a hang-up was treated as a server error"
    status, _, _ = _req(live, "/api/status")
    assert status == 200


# =========================================================================
# Server: ask, follow-ups, quarantine, demo over POST
# =========================================================================
def test_ask_refuses_an_empty_question_and_answers_a_real_one_honestly(live):
    for payload in ({}, {"question": "   "}):
        status, body, _ = _req(live, "/api/ask", method="POST", body=payload)
        assert status == 400 and json.loads(body)["error"] == "ask something first"

    _feed(live.controller, "client-marcus.txt", CLIENT_CALL)
    status, body, _ = _req(live, "/api/ask", method="POST",
                           body={"question": "what about the elimination period?"})
    assert status == 200
    answer = json.loads(body)
    assert answer["degraded"] is True, "no model is reachable here; this cannot be narrated"
    assert isinstance(answer["text"], str) and isinstance(answer["citations"], list)


def test_marking_a_follow_up_with_a_bad_status_is_the_engines_refusal(live):
    status, body, _ = _req(live, "/api/followups/mark", method="POST",
                           body={"id": "abc", "status": "maybe"})
    assert status == 400
    assert "unknown status 'maybe'" in json.loads(body)["error"]


def test_quarantine_actions_refuse_unknown_actions_and_unknown_recordings(live):
    status, body, _ = _req(live, "/api/quarantine/act", method="POST",
                           body={"id": "rec_x", "action": "shred"})
    assert status == 400 and json.loads(body)["error"] == "unknown action"
    status, body, _ = _req(live, "/api/quarantine/act", method="POST",
                           body={"id": "rec_x", "action": "release"})
    assert status == 400 and json.loads(body)["error"] == "no such quarantined recording"
    status, body, _ = _req(live, "/api/quarantine/act", method="POST",
                           body={"id": "rec_x", "action": "forget"})
    assert status == 400 and json.loads(body)["error"] == "no such recording"


def test_a_held_folder_with_nothing_to_release_says_so(app):
    cfg = app.load_config(Brain.CLOUD)
    qdir = cfg.path("quarantine") / "rec_empty"
    qdir.mkdir(parents=True)
    (qdir / "WHY.md").write_text("# Held\n\n## Reasons\n- no consent announcement detected\n")
    entries = app.quarantine()
    assert [e["id"] for e in entries] == ["rec_empty"] and entries[0]["has_media"] is False
    result = app.quarantine_release("rec_empty")
    assert result["ok"] is False and "no media" in result["error"]
    assert not any(cfg.path("inbox").glob("*")), "something was released from an empty folder"


def test_loading_the_samples_never_replaces_a_real_recording_and_runs_the_pipeline(live):
    from plaud_bridge.demo import SAMPLES

    # A person's own recording happens to share a sample's filename. Loading
    # the samples must leave it exactly as it is, not replace it with fiction.
    inbox = live.controller.load_config(Brain.CLOUD).path("inbox")
    inbox.mkdir(parents=True, exist_ok=True)
    theirs = inbox / SAMPLES[0].filename
    theirs.write_text(CLIENT_CALL, encoding="utf-8")

    status, body, _ = _req(live, "/api/demo", method="POST", body={})
    assert status == 200
    result = json.loads(body)
    assert result["ok"] is True and result["started"] is True
    assert result["written"] == len(SAMPLES) - 1 and result["skipped"] == 1
    assert SAMPLES[0].filename not in result["names"]
    assert theirs.read_text(encoding="utf-8") == CLIENT_CALL, "a sample overwrote a real file"

    snap = _wait_job(live, seconds=180)
    assert snap["error"] is None, snap["error"]
    assert snap["summary"]["processed"] + snap["summary"]["quarantined"] >= len(SAMPLES)
    names = {r["name"] for r in live.controller.recent_recordings()}
    assert set(result["names"]) <= names, "the samples never reached the library"
    assert SAMPLES[0].filename in names, "the person's own recording was not processed"


def test_a_sample_install_that_fails_is_a_400_and_starts_nothing(live, monkeypatch):
    def broken():
        raise OSError("inbox is read-only")

    monkeypatch.setattr(live.controller, "install_samples", broken)
    status, body, _ = _req(live, "/api/demo", method="POST", body={})
    assert status == 400
    assert json.loads(body) == {"ok": False, "error": "inbox is read-only"}
    assert live.app.job.snapshot()["running"] is False


# =========================================================================
# Controller: preflight branches
# =========================================================================
def test_preflight_names_ffmpeg_as_the_fatal_item_when_it_is_missing(app, monkeypatch):
    from plaud_bridge.audio import prepare

    monkeypatch.setattr(prepare, "shutil", SimpleNamespace(which=lambda _tool: None))
    item = next(i for i in app.preflight(Brain.CLOUD) if i.name == "ffmpeg")
    assert item.fatal and not item.ok
    assert item.detail.startswith("not found. Audio needs it.")
    assert "'ffmpeg' is not on PATH" in item.detail
    assert not app.is_ready(Brain.CLOUD)


@pytest.mark.parametrize("status,ok", [
    (LocalLLMStatus.NOT_RUNNING, False),
    (LocalLLMStatus.MODEL_MISSING, False),
    (LocalLLMStatus.READY, True),
])
def test_the_offline_brain_line_follows_the_probe_and_offers_the_cloud_way_out(
        app, monkeypatch, status, ok):
    probed = []

    def fake_probe(base_url, model, timeout=2.0):
        probed.append((base_url, model))
        return status, "probe says so"

    monkeypatch.setattr(controller_module, "probe_local_llm", fake_probe)
    item = next(i for i in app.preflight(Brain.OFFLINE) if i.name == "analysis brain (offline)")
    assert item.fatal is True and item.ok is ok
    assert probed and probed[0][1] == app.load_config(Brain.OFFLINE).get("llm.local.model")
    if ok:
        assert item.detail == "probe says so"
    else:
        assert item.detail == "probe says so Or switch to the free cloud key."


def test_an_offline_config_missing_its_model_is_fatal_without_probing(app, monkeypatch):
    real = app.load_config

    def blanked(brain):
        cfg = real(brain)
        cfg._d["llm"]["local"]["model"] = ""
        return cfg

    monkeypatch.setattr(app, "load_config", blanked)
    monkeypatch.setattr(controller_module, "probe_local_llm",
                        lambda *a, **k: pytest.fail("probed with nothing to probe"))
    item = next(i for i in app.preflight(Brain.OFFLINE) if i.name == "analysis brain (offline)")
    assert item.fatal and not item.ok
    assert "missing base_url or model" in item.detail


def test_a_local_asr_that_is_installed_turns_that_line_green(app, monkeypatch):
    from plaud_bridge.asr import registry

    local = SimpleNamespace(name="local", is_cloud=False, available=lambda: (True, "ready"))
    cloud = SimpleNamespace(name="groq", is_cloud=True, available=lambda: (True, "ready"))
    monkeypatch.setattr(registry, "build_asr_chain", lambda cfg, glossary: [cloud, local])
    item = next(i for i in app.preflight(Brain.CLOUD) if i.name == "local transcription")
    assert item.ok and item.detail == "ready"
    # A cloud-only chain must not count: private recordings cannot use it.
    monkeypatch.setattr(registry, "build_asr_chain", lambda cfg, glossary: [cloud])
    item = next(i for i in app.preflight(Brain.CLOUD) if i.name == "local transcription")
    assert not item.ok and "needed for private recordings" in item.detail


def test_a_groq_key_makes_the_cloud_brain_reachable_and_the_app_ready(app, monkeypatch):
    item = next(i for i in app.preflight(Brain.CLOUD) if i.name.startswith("analysis brain"))
    assert not item.ok and "Paste a Groq key" in item.detail
    assert app.is_ready(Brain.CLOUD) is False

    app.set_groq_key("  gsk_live  ")
    import os

    assert os.environ["GROQ_API_KEY"] == "gsk_live"
    item = next(i for i in app.preflight(Brain.CLOUD) if i.name == "analysis brain (cloud)")
    assert item.ok and item.detail == "ready: groq"
    assert app.is_ready(Brain.CLOUD) is True


# =========================================================================
# Controller: the probe's odd answers
# =========================================================================
def test_a_server_that_errors_is_still_a_server_so_the_advice_is_pull_not_install():
    with _http_stub(500, b"internal") as base:
        status, message = probe_local_llm(base, "llama3.3:70b")
    assert status is LocalLLMStatus.MODEL_MISSING
    assert "ollama pull llama3.3:70b" in message and "install" not in message


def test_a_model_list_that_is_not_json_reads_as_model_missing():
    with _http_stub(200, b"<html>not an api</html>") as base:
        status, message = probe_local_llm(base, "llama3.3:70b")
    assert status is LocalLLMStatus.MODEL_MISSING
    assert "is not pulled" in message


# =========================================================================
# Controller: where an install lives
# =========================================================================
def test_the_home_override_wins_on_every_platform(monkeypatch, tmp_path):
    monkeypatch.setenv("PLAUD_BRIDGE_HOME", str(tmp_path / "elsewhere"))
    assert default_base_dir() == tmp_path / "elsewhere"


def test_the_default_home_follows_the_platforms_convention(monkeypatch, tmp_path):
    monkeypatch.delenv("PLAUD_BRIDGE_HOME", raising=False)

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local"))
    assert default_base_dir() == tmp_path / "Local" / "PlaudBridge"
    monkeypatch.delenv("LOCALAPPDATA")
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    assert default_base_dir() == tmp_path / "Roaming" / "PlaudBridge"

    monkeypatch.setattr(sys, "platform", "darwin")
    assert default_base_dir() == Path.home() / "Library" / "Application Support" / "PlaudBridge"

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert default_base_dir() == tmp_path / "xdg" / "PlaudBridge"


def test_a_frozen_app_seeds_from_the_bundled_config_when_it_is_there(monkeypatch, tmp_path):
    bundle = tmp_path / "meipass"
    (bundle / "config").mkdir(parents=True)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)
    assert controller_module._bundled_config_template() == bundle / "config"
    # A bundle without its config falls back to the checkout's, not nowhere.
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path / "hollow"), raising=False)
    assert controller_module._bundled_config_template() == ROOT / "config"


def test_an_app_packaged_without_its_config_fails_loudly_on_first_run(tmp_path):
    controller = AppController(base_dir=tmp_path / "home", template_dir=tmp_path / "no-such")
    with pytest.raises(FileNotFoundError, match="packaged without its config"):
        controller.ensure_installed()
    assert not (tmp_path / "home").exists()


def test_add_files_skips_what_is_not_a_file(app, tmp_path):
    folder = tmp_path / "a-folder"
    folder.mkdir()
    assert app.add_files([tmp_path / "missing.mp3", folder]) == []
    inbox = app.load_config(Brain.CLOUD).path("inbox")
    assert not any(inbox.iterdir())


# =========================================================================
# Controller: everything a locked vault refuses, honestly
# =========================================================================
def _lock_after_a_marked_follow_up(app, monkeypatch) -> str:
    """Process a call, mark a follow-up (which writes encrypted state), lock."""
    assert _feed(app, "client-marcus.txt", CLIENT_CALL)["processed"] == 1
    items = app.followups("open")["items"]
    assert items
    assert app.followup_mark(items[0]["id"], "done")["ok"]
    monkeypatch.delenv("PLAUD_BRIDGE_PASSPHRASE")
    return app.recent_recordings()[0]["id"]


def test_the_worklist_and_the_roster_report_a_locked_vault_instead_of_guessing(app, monkeypatch):
    _lock_after_a_marked_follow_up(app, monkeypatch)
    worklist = app.followups("open")
    assert worklist["items"] == [] and "PLAUD_BRIDGE_PASSPHRASE" in worklist["error"]
    marked = app.followup_mark("anything", "done")
    assert marked["ok"] is False and "PLAUD_BRIDGE_PASSPHRASE" in marked["error"]
    roster = app.people()
    assert roster["people"] == [] and "PLAUD_BRIDGE_PASSPHRASE" in roster["error"]


def test_forgetting_under_a_locked_vault_deletes_nothing_rather_than_half(app, monkeypatch):
    rid = _lock_after_a_marked_follow_up(app, monkeypatch)
    result = app.quarantine_forget(rid)
    assert result["ok"] is False and "vault is locked" in result["error"]
    assert [r["id"] for r in app.recent_recordings()] == [rid], "a locked forget half-deleted"


def test_a_backup_under_a_locked_vault_is_refused_with_no_file_written(app, monkeypatch):
    _feed(app, "client-marcus.txt", CLIENT_CALL)
    monkeypatch.delenv("PLAUD_BRIDGE_PASSPHRASE")
    result = app.backup()
    assert result["ok"] is False and result["path"] == ""
    assert "PLAUD_BRIDGE_PASSPHRASE" in result["error"]
    backups = app.base_dir / "backups"
    assert not backups.exists() or not any(backups.iterdir())


def test_an_insights_engine_error_is_an_answer_not_an_exception(app, monkeypatch):
    """A defensive branch: `trend` does not raise today, but the page's contract is
    that if the engine ever does, it sees {report: None, error} and not a 500."""
    from plaud_bridge import insights

    def broken(*_a, **_k):
        raise insights.InsightsError("the index will not open")

    monkeypatch.setattr(insights, "trend", broken)
    assert app.insights() == {"report": None, "error": "the index will not open"}


# =========================================================================
# Launcher
# =========================================================================
class _FakeHTTPD:
    def __init__(self, serve):
        self._serve = serve
        self.calls: list[str] = []

    def serve_forever(self):
        self.calls.append("serve")
        self._serve()

    def shutdown(self):
        self.calls.append("shutdown")

    def server_close(self):
        self.calls.append("close")


def _launch(monkeypatch, argv, *, lan_url="", serve=None, browser=None):
    """Run main() against a fake server; returns (rc, app, httpd, build kwargs, opened)."""
    def interrupted():
        raise KeyboardInterrupt

    httpd = _FakeHTTPD(serve or interrupted)
    app = SimpleNamespace(lan_url=lan_url, token="tok")
    built = []

    def fake_build(base_dir=None, host="127.0.0.1", port=0, phone=False):
        built.append(phone)
        return app, httpd, "http://127.0.0.1:4321/?token=tok"

    opened = []
    monkeypatch.setattr(launch, "build", fake_build)
    monkeypatch.setattr(launch.webbrowser, "open", browser or opened.append)
    rc = launch.main(argv)
    return rc, httpd, built, opened


def test_main_prints_the_url_opens_the_browser_and_cleans_up_on_ctrl_c(monkeypatch, capsys):
    monkeypatch.delenv("PLAUD_BRIDGE_PHONE", raising=False)
    rc, httpd, built, opened = _launch(monkeypatch, [])
    out = capsys.readouterr().out
    assert rc == 0 and built == [False]
    assert "http://127.0.0.1:4321/?token=tok" in out
    assert "Keep this window open" in out and "Stopping." in out
    assert "phone" not in out.lower()
    assert opened == ["http://127.0.0.1:4321/?token=tok"]
    assert httpd.calls == ["serve", "shutdown", "close"]


def test_main_with_no_browser_prints_only_and_a_quiet_exit_still_cleans_up(monkeypatch, capsys):
    monkeypatch.delenv("PLAUD_BRIDGE_PHONE", raising=False)
    rc, httpd, _built, opened = _launch(monkeypatch, ["--no-browser"], serve=lambda: None)
    out = capsys.readouterr().out
    assert rc == 0 and opened == []
    assert "Stopping." not in out
    assert httpd.calls == ["serve", "shutdown", "close"]


def test_main_survives_a_browser_that_cannot_open(monkeypatch, capsys):
    monkeypatch.delenv("PLAUD_BRIDGE_PHONE", raising=False)

    def no_display(_url):
        raise RuntimeError("no display")

    rc, httpd, _built, _opened = _launch(monkeypatch, [], browser=no_display)
    assert rc == 0 and httpd.calls == ["serve", "shutdown", "close"]
    assert "http://127.0.0.1:4321/?token=tok" in capsys.readouterr().out


def test_phone_mode_prints_the_phone_address_and_its_warning(monkeypatch, capsys):
    monkeypatch.delenv("PLAUD_BRIDGE_PHONE", raising=False)
    rc, _httpd, built, _opened = _launch(monkeypatch, ["--phone", "--no-browser"],
                                         lan_url="http://192.168.1.50:4321/?token=tok")
    out = capsys.readouterr().out
    assert rc == 0 and built == [True]
    assert "On your phone (same Wi-Fi)" in out
    assert "http://192.168.1.50:4321/?token=tok" in out
    assert "Home network only" in out


def test_phone_mode_with_no_network_says_so_instead_of_pretending(monkeypatch, capsys):
    monkeypatch.setenv("PLAUD_BRIDGE_PHONE", "1")
    rc, _httpd, built, _opened = _launch(monkeypatch, ["--no-browser"], lan_url="")
    out = capsys.readouterr().out
    assert rc == 0 and built == [True], "PLAUD_BRIDGE_PHONE=1 must turn phone mode on"
    assert "no network address was found" in out and "loopback-only" in out


def test_phone_env_set_to_zero_is_off(monkeypatch, capsys):
    monkeypatch.setenv("PLAUD_BRIDGE_PHONE", "0")
    _rc, _httpd, built, _opened = _launch(monkeypatch, ["--no-browser"])
    assert built == [False]


def test_lan_ip_falls_back_to_the_hostname_and_then_to_loopback(monkeypatch):
    class Dead:
        def __init__(self, *_a):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def connect(self, _addr):
            raise OSError("network is unreachable")

    fake = SimpleNamespace(AF_INET=0, SOCK_DGRAM=0, socket=Dead,
                           gethostname=lambda: "box",
                           gethostbyname=lambda _n: "10.0.0.7")
    monkeypatch.setattr(launch, "socket", fake)
    assert launch._lan_ip() == "10.0.0.7"

    def unresolvable(_n):
        raise OSError("unknown host")

    fake.gethostbyname = unresolvable
    assert launch._lan_ip() == "127.0.0.1"


# =========================================================================
# Updater
# =========================================================================
@contextmanager
def _github(files: dict[str, bytes]):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            name = self.path.rsplit("/", 1)[-1]
            body = files.get(name, b"missing")
            self.send_response(200 if name in files else 404)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def _info(base: str) -> UpdateInfo:
    return UpdateInfo(version="v99.0.0", current="1.0.0",
                      zip_url=f"{base}/dl/{update.ASSET_NAME}",
                      checksum_url=f"{base}/dl/{update.CHECKSUM_ASSET}")


def test_a_malformed_published_checksum_refuses_the_install(tmp_path):
    files = {update.ASSET_NAME: b"bundle", update.CHECKSUM_ASSET: b"deadbeef  bundle"}
    with _github(files) as base:
        with pytest.raises(UpdateError, match="malformed"):
            update.download_and_verify(_info(base), tmp_path)
    assert not (tmp_path / update.ASSET_NAME).exists()


def test_a_download_past_the_size_ceiling_is_cut_off_and_removed(tmp_path, monkeypatch):
    bundle = b"x" * 4096
    files = {update.ASSET_NAME: bundle,
             update.CHECKSUM_ASSET: hashlib.sha256(bundle).hexdigest().encode()}
    monkeypatch.setattr(update, "_MAX_DOWNLOAD_BYTES", 1024)
    with _github(files) as base:
        with pytest.raises(UpdateError, match="size ceiling"):
            update.download_and_verify(_info(base), tmp_path)
    assert not (tmp_path / update.ASSET_NAME).exists(), "an oversized download was left behind"


def test_the_install_dir_is_the_folder_holding_the_executable():
    assert update._install_dir() == Path(sys.executable).resolve().parent


def test_a_frozen_build_that_is_not_windows_refuses_to_self_update(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(UpdateError, match="only wired for the Windows build"):
        update.apply_update(_info("http://127.0.0.1:9"))


def test_the_frozen_windows_apply_stages_verifies_writes_the_script_and_hands_off(
        tmp_path, monkeypatch):
    bundle = b"pretend this is the app zip"
    files = {update.ASSET_NAME: bundle,
             update.CHECKSUM_ASSET: hashlib.sha256(bundle).hexdigest().encode()}
    staging = tmp_path / "staging"
    staging.mkdir()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(update.tempfile, "mkdtemp", lambda prefix="": str(staging))
    spawned = []

    with _github(files) as base:
        message = update.apply_update(_info(base), spawn=spawned.append)

    assert message == "updating to v99.0.0; the app will restart itself"
    assert spawned == [staging / "apply-update.bat"]
    assert (staging / update.ASSET_NAME).read_bytes() == bundle
    script = spawned[0].read_text(encoding="utf-8")
    exe = Path(sys.executable).name
    assert f'ren "%EXE%" "{exe}.old"' in script
    assert str(staging / update.ASSET_NAME) in script
    assert str(Path(sys.executable).resolve().parent) in script


def test_a_frozen_windows_apply_with_a_bad_checksum_spawns_nothing(tmp_path, monkeypatch):
    files = {update.ASSET_NAME: b"tampered bundle",
             update.CHECKSUM_ASSET: hashlib.sha256(b"the real one").hexdigest().encode()}
    staging = tmp_path / "staging"
    staging.mkdir()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(update.tempfile, "mkdtemp", lambda prefix="": str(staging))
    spawned = []
    with _github(files) as base:
        with pytest.raises(UpdateError, match="checksum"):
            update.apply_update(_info(base), spawn=spawned.append)
    assert spawned == []
    assert not (staging / "apply-update.bat").exists()
    assert not (staging / update.ASSET_NAME).exists()
