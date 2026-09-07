"""
The app's new faces: People, Insights, Brief, and the Moment player.

Same discipline as the other desktop tests: the controller and the HTTP routes
are driven the way the page drives them, with the LLM stubbed so nothing
leaves the machine. What gets pinned is the honesty carried through from the
engines -- a speaker label presented as attribution rather than identity, an
assembled brief labelled as assembled, and above all the media contract:
original audio streams out of the vault decrypted chunk by chunk, and the
plaintext never touches disk on the way.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from _fixtures import CLIENT_CALL, FAMILY_DINNER, StubLLM
from plaud_bridge.desktop import AppController, Brain
from plaud_bridge.desktop.server import AppServer

ROOT = Path(__file__).resolve().parents[1]

# Routes to the sales_trainer profile (keyword floor: "role play", "objection
# handling", "pipeline"), which is the one shipped profile with
# encrypt_at_rest false -- so its original lands in _processed as plaintext,
# which is what exercises the byte-exact Range path of the player.
TRAINING_CALL = """\
Sasson: Let's run a role play on objection handling for the discovery script.
Trainee: Okay, give me the toughest close you have.
Sasson: Three questions about your pipeline and appointment set activity first.
Trainee: My activity was ninety dials this week.
Sasson: Good. Now handle this objection: the premium is too high.
Trainee: I would reframe with the rapport framework before talking price.
"""


class TrainerStub(StubLLM):
    """The shared stub, except the router scores coaching content correctly.

    The base stub only knows two answers -- insurance markers or family -- so
    a role-play transcript would land on `father` and be encrypted, which is
    the opposite of what the plaintext-range tests need to exist.
    """

    def __call__(self, cfg, system, user, local_only=False, max_tokens=None):
        if '"scores"' in user and "role play" in user.lower():
            self.calls.append({"local_only": local_only, "system": system[:80]})
            return {"scores": [
                {"profile_id": "sales_trainer", "score": 0.95,
                 "evidence": ["role play coaching session"]},
            ]}, self._response()
        return super().__call__(cfg, system, user, local_only, max_tokens)


@pytest.fixture
def app(tmp_path, monkeypatch):
    stub = TrainerStub()
    for module in ("plaud_bridge.profiles.router", "plaud_bridge.profiles.extractor"):
        monkeypatch.setattr(f"{module}.complete_json", stub)
    monkeypatch.setenv("PLAUD_BRIDGE_PASSPHRASE", "a-long-enough-desktop-passphrase")
    return AppController(base_dir=tmp_path / "home", template_dir=ROOT / "config"), stub


def _feed(controller: AppController, name: str, body: str) -> dict:
    picked = controller.base_dir / name
    picked.parent.mkdir(parents=True, exist_ok=True)
    picked.write_text(body, encoding="utf-8")
    controller.add_files([picked])
    return controller.process(Brain.CLOUD)


# =========================================================================
# People
# =========================================================================
def test_the_roster_lists_the_counterparty_as_attribution_not_identity(app):
    controller, _ = app
    assert _feed(controller, "client-marcus.txt", CLIENT_CALL)["processed"] == 1

    result = controller.people()
    assert result["error"] == ""
    names = {p["display_name"] for p in result["people"]}
    assert "Marcus" in names
    marcus = next(p for p in result["people"] if p["display_name"] == "Marcus")
    # Nobody enrolled a voiceprint, so the page must not present the label
    # as a verified identity.
    assert marcus["voice_verified"] is False
    assert marcus["conversations"] == 1 and marcus["minutes_heard"] >= 0


def test_personal_recordings_stay_off_the_roster_unless_asked(app):
    controller, _ = app
    _feed(controller, "client-marcus.txt", CLIENT_CALL)
    _feed(controller, "dinner.txt", FAMILY_DINNER)

    default_names = {p["display_name"] for p in controller.people()["people"]}
    assert "Kid" not in default_names, "a personal speaker leaked onto the default roster"
    with_personal = {p["display_name"]
                     for p in controller.people(include_personal=True)["people"]}
    assert "Kid" in with_personal


# =========================================================================
# Insights
# =========================================================================
def test_insights_report_carries_checkable_arithmetic(app):
    controller, _ = app
    _feed(controller, "client-marcus.txt", CLIENT_CALL)

    result = controller.insights(days=90)
    assert result["error"] == ""
    report = result["report"]
    assert report["days"] == 90
    assert len(report["recordings"]) == 1
    rec = report["recordings"][0]
    speakers = {s["speaker"]: s for s in rec["speakers"]}
    assert "Sasson" in speakers and "Marcus" in speakers
    # Shares are fractions of everything spoken; together they cover it.
    total_share = sum(s["share"] for s in rec["speakers"])
    assert 0.99 <= total_share <= 1.01
    assert report["unopened"] == []


def test_insights_on_an_empty_archive_is_an_answer_not_an_error(app):
    controller, _ = app
    result = controller.insights()
    assert result["error"] == ""
    assert result["report"]["recordings"] == []


# =========================================================================
# Brief
# =========================================================================
def test_the_brief_with_no_model_renders_and_says_it_was_assembled(app):
    controller, _ = app
    _feed(controller, "client-marcus.txt", CLIENT_CALL)

    html = controller.brief_html(days=7)
    assert "<html" in html.lower()
    # No brain is reachable in this sandbox, and the reader must be told the
    # difference between a narrated memo and a templated one.
    assert "Assembled, not narrated" in html


# =========================================================================
# The moment player: transcript + media
# =========================================================================
def test_the_transcript_lines_are_shaped_for_the_player(app):
    controller, _ = app
    _feed(controller, "client-marcus.txt", CLIENT_CALL)
    rid = controller.recent_recordings()[0]["id"]

    result = controller.transcript(rid)
    assert result["ok"] is True and result["lines"]
    line = result["lines"][0]
    assert set(line) == {"start", "end", "speaker", "text"}
    assert isinstance(line["start"], float) and line["text"]
    speakers = {ln["speaker"] for ln in result["lines"]}
    assert "Marcus" in speakers


def test_an_unknown_recording_gets_an_honest_refusal_not_an_empty_transcript(app):
    controller, _ = app
    result = controller.transcript("rec_does_not_exist")
    assert result["ok"] is False and result["lines"] == []
    assert "unknown" in result["error"]


def test_an_encrypted_original_streams_back_byte_identical(app):
    """The vault round trip: what went in encrypted comes out exact, in chunks."""
    controller, _ = app
    _feed(controller, "client-marcus.txt", CLIENT_CALL)
    rid = controller.recent_recordings()[0]["id"]

    media = controller.media_stream(rid)
    assert media is not None and media["encrypted"] is True
    assert media["partial"] is False and media["total"] is None
    assert b"".join(media["iter"]) == CLIENT_CALL.encode()


def test_a_plaintext_original_honours_an_exact_byte_range(app):
    controller, _ = app
    summary = _feed(controller, "training-roleplay.txt", TRAINING_CALL)
    assert summary["processed"] == 1
    rows = controller.recent_recordings()
    assert rows[0]["encrypted"] is False, "the trainer profile should be plaintext at rest"
    rid = rows[0]["id"]

    media = controller.media_stream(rid, start=0, end=9)
    assert media is not None and media["partial"] is True
    assert media["total"] == len(TRAINING_CALL.encode())
    assert b"".join(media["iter"]) == TRAINING_CALL.encode()[:10]


def test_a_missing_original_is_none_not_a_crash(app):
    controller, _ = app
    assert controller.media_stream("rec_never_existed") is None


# =========================================================================
# The HTTP routes, driven like a browser
# =========================================================================
@pytest.fixture
def server(app):
    controller, _ = app
    srv = AppServer(controller)
    httpd = srv.make_server("127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        yield controller, base, srv.token
    finally:
        httpd.shutdown()
        httpd.server_close()


def _req(base, path, *, token=None, headers=None):
    h = dict(headers or {})
    if token:
        h["X-Token"] = token
    req = urllib.request.Request(base + path, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, resp.read(), dict(resp.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), dict(exc.headers)


def test_the_page_script_survives_python_string_escaping():
    """
    The page is a Python raw string on purpose: its JS contains "\\n" inside
    string literals, and a cooked string would bake real newlines into them --
    an unterminated-literal SyntaxError that kills the entire script in the
    browser while every server-side test still passes. This pins the escape
    surviving to the browser; the node check below (when node exists) proves
    the whole script parses.
    """
    from plaud_bridge.desktop.server import _PAGE

    assert 'join("\\n")' in _PAGE, (
        "a JS \\n escape was cooked into a real newline -- _PAGE must stay "
        "a raw string")
    assert '\njoin("' not in _PAGE


def test_the_page_script_is_valid_javascript(tmp_path):
    import re
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed; the escape test above still guards")
    from plaud_bridge.desktop.server import _PAGE

    match = re.search(r"<script>(.*)</script>", _PAGE.replace("__TOKEN__", "t"), re.S)
    assert match, "the page lost its script block"
    path = tmp_path / "page.js"
    path.write_text(match.group(1), encoding="utf-8")
    proc = subprocess.run([node, "--check", str(path)], capture_output=True, text=True)
    assert proc.returncode == 0, f"the page's JavaScript does not parse:\n{proc.stderr}"


def test_the_page_carries_the_new_tabs(server):
    _controller, base, token = server
    status, body, _ = _req(base, "/")
    assert status == 200
    for tab in (b">Brief<", b">People<", b">Insights<"):
        assert tab in body, f"the page is missing its {tab} tab"


def test_media_and_the_new_reads_refuse_requests_without_the_token(server):
    _controller, base, _token = server
    for path in ("/api/media?id=x", "/api/transcript?id=x", "/api/people",
                 "/api/insights", "/api/brief"):
        status, _, _ = _req(base, path)
        assert status == 403, f"{path} answered without the token"


def test_the_whole_moment_flow_over_http(server):
    controller, base, token = server
    _feed(controller, "client-marcus.txt", CLIENT_CALL)
    rid = controller.recent_recordings()[0]["id"]

    # The transcript the player syncs to.
    status, body, _ = _req(base, f"/api/transcript?id={rid}", token=token)
    assert status == 200 and json.loads(body)["lines"]

    # The audio element's request: query token, exact decrypted bytes.
    status, body, headers = _req(base, f"/api/media?id={rid}&token={token}")
    assert status == 200
    assert body == CLIENT_CALL.encode()
    # An encrypted stream must tell the player not to try scrubbing.
    assert headers.get("Accept-Ranges") == "none"

    # A Range request against an encrypted original degrades to the whole
    # stream rather than staging plaintext anywhere to satisfy the seek.
    status, body, _ = _req(base, f"/api/media?id={rid}&token={token}",
                           headers={"Range": "bytes=0-9"})
    assert status == 200 and body == CLIENT_CALL.encode()

    status, body, _ = _req(base, "/api/media?id=rec_gone&token=" + token)
    assert status == 404


def test_a_plaintext_range_request_gets_a_true_206_over_http(server):
    controller, base, token = server
    _feed(controller, "training-roleplay.txt", TRAINING_CALL)
    rid = controller.recent_recordings()[0]["id"]

    status, body, headers = _req(base, f"/api/media?id={rid}&token={token}",
                                 headers={"Range": "bytes=3-12"})
    assert status == 206
    assert body == TRAINING_CALL.encode()[3:13]
    total = len(TRAINING_CALL.encode())
    assert headers.get("Content-Range") == f"bytes 3-12/{total}"

    # A range past the end is 416, the answer scrubbing code expects.
    status, _, _ = _req(base, f"/api/media?id={rid}&token={token}",
                        headers={"Range": f"bytes={total + 10}-"})
    assert status == 416


def test_people_insights_and_brief_round_trip_over_http(server):
    controller, base, token = server
    _feed(controller, "client-marcus.txt", CLIENT_CALL)

    status, body, _ = _req(base, "/api/people", token=token)
    assert status == 200
    assert "Marcus" in {p["display_name"] for p in json.loads(body)["people"]}

    status, body, _ = _req(base, "/api/insights?days=90", token=token)
    assert status == 200
    assert json.loads(body)["report"]["recordings"]

    status, body, _ = _req(base, "/api/brief?days=7", token=token)
    assert status == 200
    assert b"Assembled, not narrated" in body
