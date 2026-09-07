"""
The HTTP client, driven over a real socket.

`http_util` was written off as "untestable without network". It is not. A
loopback `ThreadingHTTPServer` on 127.0.0.1:0 can be scripted to return any
status, drop the connection, or stall past the timeout -- every failure the
real endpoints produce, with no packet leaving the machine.

What is pinned: which statuses are retried and which are not, how many attempts
are made and with which backoff schedule, what the error carries when it gives
up, and -- because the same header carries the API key on every call -- that the
key never lands in an error message or a log line.

`ScriptedServer` and the `server` / `no_backoff` fixtures are shared with the
ASR and LLM provider tests, which point their base URLs at the same stub.
"""

from __future__ import annotations

import email.policy
import json
import logging
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from plaud_bridge import http_util
from plaud_bridge.http_util import (
    RETRYABLE_STATUS,
    HttpError,
    _sleep_backoff,
    post_json,
    post_multipart,
)

SECRET = "sk-test-secret-that-must-not-leak-0123456789"


# =========================================================================
# The loopback stub
# =========================================================================
@dataclass
class Exchange:
    """One request as the stub saw it."""

    path: str
    headers: dict[str, str]
    body: bytes

    @property
    def json(self) -> Any:
        return json.loads(self.body.decode("utf-8"))

    def multipart(self) -> dict[str, dict[str, Any]]:
        """Parse the multipart body into {field: {"value"/"bytes", "filename", "type"}}."""
        raw = b"Content-Type: " + self.headers["content-type"].encode() + b"\r\n\r\n" + self.body
        msg = BytesParser(policy=email.policy.default).parsebytes(raw)
        out: dict[str, dict[str, Any]] = {}
        for part in msg.iter_parts():
            name = part.get_param("name", header="content-disposition")
            payload = part.get_payload(decode=True)
            out[str(name)] = {
                "bytes": payload,
                "value": payload.decode("utf-8", "replace"),
                "filename": part.get_filename(),
                "type": part.get_content_type(),
            }
        return out


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's name
        length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(length)
        server: ScriptedServer = self.server.owner  # type: ignore[attr-defined]
        with server.lock:
            server.seen.append(Exchange(
                self.path, {k.lower(): v for k, v in self.headers.items()}, body,
            ))
            step = server.script.popleft() if server.script else ("status", 599, b"script exhausted", "text/plain")

        kind = step[0]
        if kind == "drop":
            # Hang up without a status line. urllib reports RemoteDisconnected,
            # which the client must treat as a network error.
            self.close_connection = True
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            return
        if kind == "stall":
            time.sleep(step[1])
            step = ("status", 200, b"{}", "application/json")

        _, status, payload, ctype = step
        try:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except OSError:
            # The client gave up (timeout test). Nothing to report.
            pass

    def log_message(self, fmt: str, *args: Any) -> None:
        """Silence. The assertions are the report."""


@dataclass
class ScriptedServer:
    """
    A loopback HTTP server whose next responses are queued by the test.

    Steps are consumed in request order regardless of path, so a test that
    drives two providers in sequence scripts their replies in that order and
    asserts on `seen[i].path` to confirm who asked.
    """

    seen: list[Exchange] = field(default_factory=list)
    script: deque = field(default_factory=deque)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.httpd.owner = self  # type: ignore[attr-defined]
        # serve_forever MUST be running before shutdown() is ever called, or
        # shutdown() blocks forever waiting for a loop that never started.
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}"

    def respond(self, status: int, body: bytes | str = b"", ctype: str = "application/json") -> None:
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.script.append(("status", status, body, ctype))

    def respond_json(self, obj: Any, status: int = 200) -> None:
        self.respond(status, json.dumps(obj))

    def drop(self) -> None:
        self.script.append(("drop",))

    def stall(self, seconds: float) -> None:
        self.script.append(("stall", seconds))

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


@pytest.fixture
def server():
    stub = ScriptedServer()
    try:
        yield stub
    finally:
        stub.close()


@pytest.fixture
def no_backoff(monkeypatch):
    """Replace the sleep with a recorder. Returns the list of attempts backed off on."""
    attempts: list[int] = []
    monkeypatch.setattr(http_util, "_sleep_backoff", lambda attempt, **kw: attempts.append(attempt))
    return attempts


def unused_port() -> int:
    """A port nothing is listening on, so a connect is refused rather than hanging."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


# =========================================================================
# HttpError
# =========================================================================
def test_http_error_keeps_status_and_truncates_the_body_at_2000_chars():
    err = HttpError("boom", status=502, body="x" * 5000)
    assert err.status == 502
    assert len(err.body) == 2000
    assert str(err) == "boom"


def test_a_network_failure_with_no_status_counts_as_retryable():
    assert HttpError("no route").retryable is True
    assert HttpError("no route").status is None


@pytest.mark.parametrize("status", sorted(RETRYABLE_STATUS))
def test_every_listed_transient_status_is_retryable(status):
    assert HttpError("x", status=status).retryable is True


@pytest.mark.parametrize("status", [400, 401, 403, 404, 413, 422])
def test_client_errors_are_not_retryable(status):
    """Sending the same bad request four more times cannot make it valid."""
    assert HttpError("x", status=status).retryable is False


# =========================================================================
# Backoff schedule
# =========================================================================
def test_backoff_grows_exponentially_with_jitter_and_stops_at_the_cap(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(http_util.time, "sleep", slept.append)
    # random() == 0.5 puts the jitter multiplier at exactly 1.0.
    monkeypatch.setattr(http_util.random, "random", lambda: 0.5)

    for attempt in range(4):
        _sleep_backoff(attempt, base=2.0, cap=5.0)

    assert slept == [1.0, 2.0, 4.0, 5.0]


def test_backoff_jitter_stays_within_sixty_to_one_forty_percent(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(http_util.time, "sleep", slept.append)

    monkeypatch.setattr(http_util.random, "random", lambda: 0.0)
    _sleep_backoff(1, base=2.0)
    monkeypatch.setattr(http_util.random, "random", lambda: 0.999999)
    _sleep_backoff(1, base=2.0)

    assert slept[0] == pytest.approx(2.0 * 0.6)
    assert slept[1] == pytest.approx(2.0 * 1.4, abs=1e-4)


# =========================================================================
# post_json over the wire
# =========================================================================
def test_post_json_sends_the_payload_as_json_with_the_callers_headers(server):
    server.respond_json({"ok": True, "echo": 1})

    result = post_json(f"{server.url}/v1/thing", {"a": [1, 2], "b": "c"},
                       headers={"Authorization": f"Bearer {SECRET}", "X-Custom": "yes"})

    assert result == {"ok": True, "echo": 1}
    assert len(server.seen) == 1
    seen = server.seen[0]
    assert seen.path == "/v1/thing"
    assert seen.json == {"a": [1, 2], "b": "c"}
    assert seen.headers["content-type"] == "application/json"
    assert seen.headers["authorization"] == f"Bearer {SECRET}"
    assert seen.headers["x-custom"] == "yes"


def test_post_json_retries_transient_failures_and_returns_the_eventual_body(server, no_backoff):
    server.respond(503, "overloaded")
    server.respond(503, "still overloaded")
    server.respond_json({"finally": "yes"})

    assert post_json(server.url, {}, {}, max_retries=4) == {"finally": "yes"}
    assert len(server.seen) == 3
    # Backed off after attempt 0 and attempt 1; the third attempt succeeded.
    assert no_backoff == [0, 1]


def test_post_json_does_not_retry_a_client_error(server, no_backoff):
    server.respond(400, '{"error": "bad request"}')
    server.respond_json({"never": "reached"})

    with pytest.raises(HttpError) as info:
        post_json(server.url, {}, {}, max_retries=4)

    assert info.value.status == 400
    assert info.value.retryable is False
    assert info.value.body == '{"error": "bad request"}'
    assert "HTTP 400" in str(info.value)
    assert len(server.seen) == 1
    assert no_backoff == []


def test_post_json_retries_a_rate_limit_until_the_budget_is_spent(server, no_backoff):
    for _ in range(10):
        server.respond(429, "slow down")

    with pytest.raises(HttpError) as info:
        post_json(server.url, {}, {}, max_retries=2)

    assert info.value.status == 429
    assert info.value.body == "slow down"
    # max_retries=2 means three attempts: the first plus two retries.
    assert len(server.seen) == 3
    # No sleep after the final attempt -- there is nothing to wait for.
    assert no_backoff == [0, 1]


def test_post_json_reports_the_error_body_from_the_final_attempt(server, no_backoff):
    server.respond(503, "first")
    server.respond(503, "second")

    with pytest.raises(HttpError) as info:
        post_json(server.url, {}, {}, max_retries=1)

    assert info.value.body == "second"


def test_a_server_that_hangs_up_without_replying_is_an_http_error_not_a_crash(server, no_backoff):
    """
    Regression. urllib wraps a failure during the send in URLError but lets a
    failure while reading the reply propagate raw, so a server that accepted
    the upload and then hung up escaped `_request` as
    `http.client.RemoteDisconnected`. That skipped the retry loop, skipped the
    provider's `except HttpError`, and skipped the registry's failover to the
    next provider -- the recording crashed instead of falling back to local.
    """
    server.drop()

    with pytest.raises(HttpError) as info:
        post_json(server.url, {}, {}, max_retries=0)

    assert info.value.status is None
    assert info.value.retryable is True
    assert "network error contacting" in str(info.value)
    assert "closed connection" in str(info.value)


def test_an_error_response_whose_body_cannot_be_read_still_reports_its_status(monkeypatch):
    import urllib.error

    class _Severed:
        """A response body whose socket died between the status line and the read."""

        def read(self, *args):
            raise ConnectionResetError("connection reset by peer")

        def close(self):
            # HTTPError wraps its fp in a closer that calls this at GC time;
            # without it the fake raises from __del__ and pytest reports an
            # unraisable exception that is noise, not a finding.
            pass

    def refuse(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 500, "boom", {}, _Severed())

    monkeypatch.setattr(http_util.urllib.request, "urlopen", refuse)

    with pytest.raises(HttpError) as info:
        post_json("http://127.0.0.1:9/x", {}, {}, max_retries=0)

    assert info.value.status == 500
    assert info.value.body == ""


def test_post_json_surfaces_a_dropped_connection_as_a_network_error_and_retries_it(server, no_backoff):
    server.drop()
    server.respond_json({"recovered": True})

    assert post_json(server.url, {}, {}, max_retries=1) == {"recovered": True}
    assert len(server.seen) == 2
    assert no_backoff == [0]


def test_post_json_gives_up_on_a_refused_connection_after_the_retry_budget(no_backoff):
    url = f"http://127.0.0.1:{unused_port()}/nothing"

    with pytest.raises(HttpError) as info:
        post_json(url, {}, {}, max_retries=2)

    assert info.value.status is None
    assert info.value.retryable is True
    assert "network error contacting" in str(info.value)
    assert url in str(info.value)
    assert no_backoff == [0, 1]


def test_post_json_turns_a_stalled_server_into_a_timeout_error(server, no_backoff):
    server.stall(3.0)

    started = time.monotonic()
    with pytest.raises(HttpError) as info:
        post_json(server.url, {}, {}, timeout=1, max_retries=0)

    assert time.monotonic() - started < 2.5
    assert info.value.status is None
    assert "timeout contacting" in str(info.value)
    assert no_backoff == []


def test_post_json_rejects_a_non_json_body_without_retrying(server, no_backoff):
    server.respond(200, "<html>a captive portal</html>", ctype="text/html")
    server.respond_json({"never": "reached"})

    with pytest.raises(HttpError) as info:
        post_json(server.url, {}, {}, max_retries=3)

    assert "non-JSON response" in str(info.value)
    assert info.value.status is None
    assert len(server.seen) == 1
    assert no_backoff == []


def test_post_json_with_a_negative_retry_budget_says_nothing_was_attempted(server):
    with pytest.raises(HttpError) as info:
        post_json(server.url, {}, {}, max_retries=-1)

    assert "no request attempted" in str(info.value)
    assert "max_retries" in str(info.value)
    assert server.seen == []


def test_the_bearer_token_never_appears_in_errors_or_logs(server, no_backoff, caplog):
    """
    The same header carries the API key on every call. A stack trace pasted
    into a bug report, or a log file shipped with the app, must not contain it.
    """
    server.respond(503, "overloaded")
    server.respond(401, '{"error": {"message": "invalid api key"}}')

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(HttpError) as info:
            post_json(server.url, {"prompt": "hi"},
                      headers={"Authorization": f"Bearer {SECRET}"}, max_retries=3)

    assert info.value.status == 401
    assert SECRET not in str(info.value)
    assert SECRET not in info.value.body
    assert SECRET not in repr(info.value.__cause__)
    assert SECRET not in caplog.text


# =========================================================================
# post_multipart over the wire
# =========================================================================
def _audio(tmp_path: Path, name: str = "chunk-000.wav", size: int = 2048) -> Path:
    path = tmp_path / name
    path.write_bytes(bytes(range(256)) * (size // 256))
    return path


def test_post_multipart_encodes_fields_and_the_file_the_way_the_endpoint_expects(server, tmp_path):
    audio = _audio(tmp_path)
    server.respond_json({"text": "hello"})

    result = post_multipart(
        f"{server.url}/audio/transcriptions",
        fields={"model": "whisper-large-v3-turbo", "language": "en", "prompt": "Plaud, Sasson"},
        file_path=audio, file_field="file",
        headers={"Authorization": f"Bearer {SECRET}"},
    )

    assert result == {"text": "hello"}
    seen = server.seen[0]
    assert seen.path == "/audio/transcriptions"
    assert seen.headers["content-type"].startswith("multipart/form-data; boundary=----plaudbridge")
    assert seen.headers["content-length"] == str(len(seen.body))
    assert seen.headers["authorization"] == f"Bearer {SECRET}"

    parts = seen.multipart()
    assert parts["model"]["value"] == "whisper-large-v3-turbo"
    assert parts["language"]["value"] == "en"
    assert parts["prompt"]["value"] == "Plaud, Sasson"
    assert parts["file"]["filename"] == "chunk-000.wav"
    assert parts["file"]["type"] == "audio/x-wav"
    assert parts["file"]["bytes"] == audio.read_bytes()


def test_post_multipart_falls_back_to_octet_stream_for_an_unknown_extension(server, tmp_path):
    audio = _audio(tmp_path, name="chunk.plaudraw")
    server.respond_json({})

    post_multipart(server.url, {}, audio, "file", headers={})

    assert server.seen[0].multipart()["file"]["type"] == "application/octet-stream"


def test_post_multipart_retries_transient_failures_and_resends_the_whole_file(server, tmp_path, no_backoff):
    audio = _audio(tmp_path)
    server.respond(502, "bad gateway")
    server.respond_json({"text": "second time lucky"})

    assert post_multipart(server.url, {"model": "m"}, audio, "file", {}, max_retries=2) == {
        "text": "second time lucky"
    }
    assert len(server.seen) == 2
    assert no_backoff == [0]
    # The retry is a full resend, not a resumed stream.
    assert server.seen[1].multipart()["file"]["bytes"] == audio.read_bytes()


def test_post_multipart_does_not_retry_a_rejected_upload(server, tmp_path, no_backoff):
    server.respond(413, "file too large")

    with pytest.raises(HttpError) as info:
        post_multipart(server.url, {}, _audio(tmp_path), "file", {}, max_retries=4)

    assert info.value.status == 413
    assert info.value.body == "file too large"
    assert len(server.seen) == 1
    assert no_backoff == []


def test_post_multipart_reports_the_last_transient_error_when_the_budget_runs_out(server, tmp_path, no_backoff):
    server.respond(504, "gateway timeout one")
    server.respond(504, "gateway timeout two")
    server.respond(504, "gateway timeout three")

    with pytest.raises(HttpError) as info:
        post_multipart(server.url, {}, _audio(tmp_path), "file", {}, max_retries=1)

    assert info.value.status == 504
    assert info.value.body == "gateway timeout two"
    assert len(server.seen) == 2
    assert no_backoff == [0]


def test_post_multipart_rejects_a_non_json_body_without_retrying(server, tmp_path, no_backoff):
    server.respond(200, "not json", ctype="text/plain")

    with pytest.raises(HttpError) as info:
        post_multipart(server.url, {}, _audio(tmp_path), "file", {}, max_retries=3)

    assert "non-JSON response" in str(info.value)
    assert len(server.seen) == 1
    assert no_backoff == []


def test_post_multipart_with_a_negative_retry_budget_says_nothing_was_attempted(server, tmp_path):
    with pytest.raises(HttpError) as info:
        post_multipart(server.url, {}, _audio(tmp_path), "file", {}, max_retries=-1)

    assert "no request attempted" in str(info.value)
    assert server.seen == []
