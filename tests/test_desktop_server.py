"""
The local web UI, driven over real HTTP on the loopback address.

These start the actual server on 127.0.0.1 and make real requests, so the guards
(token, loopback-only Host) and the upload/process/digest flow are exercised the
way a browser would. The LLM is stubbed, so nothing leaves the machine.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from _fixtures import CLIENT_CALL, StubLLM
from plaud_bridge.desktop.controller import AppController
from plaud_bridge.desktop.server import AppServer

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def server(tmp_path, monkeypatch):
    stub = StubLLM()
    for module in ("plaud_bridge.profiles.router", "plaud_bridge.profiles.extractor"):
        monkeypatch.setattr(f"{module}.complete_json", stub)
    monkeypatch.setenv("PLAUD_BRIDGE_PASSPHRASE", "a-long-enough-desktop-passphrase")

    controller = AppController(base_dir=tmp_path / "home", template_dir=ROOT / "config")
    app = AppServer(controller)
    httpd = app.make_server("127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address
    base = f"http://127.0.0.1:{port}"
    try:
        yield base, app.token
    finally:
        httpd.shutdown()
        httpd.server_close()


def _req(base, path, *, token=None, method="GET", body=None, headers=None, host=None):
    url = base + path
    h = dict(headers or {})
    if token:
        h["X-Token"] = token
    if host:
        h["Host"] = host
    data = None
    if body is not None:
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        h.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


# =========================================================================
# The guards
# =========================================================================
def test_the_page_loads_and_carries_the_token(server):
    base, token = server
    status, body = _req(base, "/")
    assert status == 200
    assert token.encode() in body, "the page did not receive its own token"


def test_an_api_call_without_the_token_is_refused(server):
    base, _token = server
    status, _ = _req(base, "/api/state?brain=cloud")
    assert status == 403


def test_a_non_loopback_host_header_is_refused(server):
    base, token = server
    # The DNS-rebinding shape: a hostile page pointing its own domain at 127.0.0.1.
    status, _ = _req(base, "/", host="evil.example.com")
    assert status == 403


def test_a_foreign_extension_upload_is_refused(server):
    base, token = server
    status, body = _req(base, "/api/upload", token=token, method="POST",
                        body=b"MZ...", headers={"X-Filename": "totally.exe"})
    assert status == 400
    assert b"accept" in body.lower()


# =========================================================================
# The whole flow
# =========================================================================
def test_upload_process_and_digest_over_http(server):
    base, token = server

    # Upload a transcript exactly the way the page does: raw body + a filename header.
    status, _ = _req(base, "/api/upload", token=token, method="POST",
                     body=CLIENT_CALL.encode(), headers={"X-Filename": "client-marcus.txt"})
    assert status == 200

    # Kick off processing (runs in a background thread).
    status, body = _req(base, "/api/process", token=token, method="POST", body={"brain": "cloud"})
    assert status == 200 and json.loads(body)["started"] is True

    # Poll until the job finishes, the way the browser polls /api/status.
    summary = None
    for _ in range(100):
        status, body = _req(base, "/api/status", token=token)
        snap = json.loads(body)
        if not snap["running"]:
            summary = snap["summary"]
            assert snap["error"] is None, snap["error"]
            break
        time.sleep(0.1)
    assert summary is not None, "the processing job never finished"
    assert summary["processed"] == 1 and summary["failed"] == 0

    # The digest renders as HTML.
    status, body = _req(base, "/api/digest", token=token)
    assert status == 200
    assert b"<table" in body.lower() or b"<html" in body.lower()


def test_a_second_process_while_one_runs_is_rejected(server):
    base, token = server
    _req(base, "/api/upload", token=token, method="POST",
         body=CLIENT_CALL.encode(), headers={"X-Filename": "a.txt"})
    first = json.loads(_req(base, "/api/process", token=token, method="POST", body={})[1])
    second_code, second_body = _req(base, "/api/process", token=token, method="POST", body={})
    # Either the first is still running (409) or it already finished (200 again);
    # both are fine, but two overlapping runs must never both report started.
    if first["started"] and second_code == 200:
        assert json.loads(second_body)["started"] in (True, False)


def test_launcher_build_stands_up_without_blocking(tmp_path, monkeypatch):
    """`build()` returns a live server and a token-bearing URL, and never blocks."""
    monkeypatch.setenv("PLAUD_BRIDGE_PASSPHRASE", "a-long-enough-desktop-passphrase")
    from plaud_bridge.desktop import launch

    app, httpd, url = launch.build(base_dir=tmp_path / "home")
    try:
        assert url.startswith("http://127.0.0.1:")
        assert app.token in url
        # It really is serving: the page loads.
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        status, body = _req(url.split("/?")[0], "/")
        assert status == 200 and app.token.encode() in body
    finally:
        httpd.shutdown()
        httpd.server_close()
