"""
The app's tabs: library, search, ask, follow-ups, quarantine triage, backup.

Each tab is a thin view over the same engine the CLI uses, and these tests
drive the controller (and the HTTP routes) the way the page does -- with the
LLM stubbed, so nothing leaves the machine and no key is needed. The point
being pinned: the app exposes the engine's honesty (a search says what it
could not open; a refusal cannot be released by a click) rather than a
prettier, softer version of it.
"""

from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path

import pytest

from _fixtures import CLIENT_CALL, StubLLM
from plaud_bridge.desktop import AppController, Brain
from plaud_bridge.desktop.server import AppServer

ROOT = Path(__file__).resolve().parents[1]

# A client call where the other party explicitly refuses. Never releasable
# from the app; the friction is the point.
REFUSED_CALL = """\
Sasson: Hey Marcus, before we get started I record these calls for my notes. Is that okay?
Marcus: Hold on, I really don't want this being recorded.
Sasson: Understood, no problem.
Marcus: So about that term policy through work, the elimination period question.
Sasson: Your income is the asset here, not the house.
"""

# The same kind of call with no consent exchange at all: quarantined as
# "no announcement", releasable after human review.
UNANNOUNCED_CALL = "\n".join(CLIENT_CALL.strip().splitlines()[2:]) + "\n"


@pytest.fixture
def app(tmp_path, monkeypatch):
    stub = StubLLM()
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
# Library
# =========================================================================
def test_the_library_lists_what_was_processed(app):
    controller, _ = app
    assert _feed(controller, "client-marcus.txt", CLIENT_CALL)["processed"] == 1

    rows = controller.recent_recordings()
    assert len(rows) == 1
    row = rows[0]
    assert row["name"] == "client-marcus.txt"
    assert row["profile"] == "insurance_agent" and row["personal"] is False
    assert row["encrypted"] is True and row["stage"] == "complete"
    assert row["minutes"] >= 0


def test_an_empty_archive_is_an_empty_list_not_an_error(app):
    controller, _ = app
    assert controller.recent_recordings() == []
    assert controller.quarantine() == []
    assert controller.followups()["items"] == []


# =========================================================================
# Search
# =========================================================================
def test_search_finds_the_words_and_keeps_the_honesty_fields(app):
    controller, _ = app
    _feed(controller, "client-marcus.txt", CLIENT_CALL)

    result = controller.search("elimination period")
    assert result["complete"] is True and result["unopened"] == 0
    assert result["matches"], "the phrase is in the transcript and search missed it"
    hit = result["matches"][0]
    assert "elimination" in hit["text"].lower()
    assert hit["stamp"] and hit["source"] == "client-marcus.txt"


# =========================================================================
# Ask
# =========================================================================
def test_ask_degrades_honestly_with_no_model_reachable(app):
    """
    The sandbox has no cloud key and no local model, so the answer must be the
    engine's degraded mode -- ranked excerpts that say they are excerpts --
    never an exception and never a fabricated answer.
    """
    controller, _ = app
    _feed(controller, "client-marcus.txt", CLIENT_CALL)

    out = controller.ask("what did I promise about the elimination period quotes?",
                         Brain.CLOUD)
    assert isinstance(out["text"], str)
    assert out["degraded"] is True, "no provider is reachable here; this cannot be a model answer"


# =========================================================================
# Follow-ups
# =========================================================================
def test_followups_can_be_marked_done_from_the_app(app):
    controller, _ = app
    _feed(controller, "client-marcus.txt", CLIENT_CALL)

    open_items = controller.followups("open")["items"]
    assert open_items, "the stubbed analysis carries commitments; none were collected"

    first = open_items[0]
    marked = controller.followup_mark(first["id"], "done")
    assert marked["ok"] is True and marked["status"] == "done"

    remaining = {i["id"] for i in controller.followups("open")["items"]}
    assert first["id"] not in remaining, "a done item is still on the open worklist"
    everything = {i["id"]: i["status"] for i in controller.followups(None)["items"]}
    assert everything[first["id"]] == "done"


# =========================================================================
# Held (quarantine triage)
# =========================================================================
def test_a_refusal_cannot_be_released_by_a_click(app):
    controller, _ = app
    summary = _feed(controller, "client-refused.txt", REFUSED_CALL)
    assert summary["quarantined"] == 1

    entries = controller.quarantine()
    assert len(entries) == 1 and entries[0]["klass"] == "refusal"

    result = controller.quarantine_release(entries[0]["id"])
    assert result["ok"] is False
    assert "objected" in result["error"], "the refusal must say WHY it will not release"


def test_an_unannounced_call_releases_back_to_the_inbox(app):
    controller, _ = app
    summary = _feed(controller, "backlog-call.txt", UNANNOUNCED_CALL)
    assert summary["quarantined"] == 1

    entry = controller.quarantine()[0]
    assert entry["klass"] != "refusal"

    result = controller.quarantine_release(entry["id"])
    assert result["ok"] is True, result["error"]
    inbox = controller.load_config(Brain.CLOUD).path("inbox")
    assert any(inbox.iterdir()), "released media did not land back in the inbox"


def test_forgetting_a_held_recording_removes_it(app):
    controller, _ = app
    _feed(controller, "client-refused.txt", REFUSED_CALL)
    rid = controller.quarantine()[0]["id"]

    result = controller.quarantine_forget(rid)
    assert result["ok"] is True, result["error"]
    assert controller.quarantine() == [], "the forgotten recording is still listed as held"


# =========================================================================
# Backup
# =========================================================================
def test_backup_produces_one_encrypted_file(app):
    controller, _ = app
    _feed(controller, "client-marcus.txt", CLIENT_CALL)

    result = controller.backup()
    assert result["ok"] is True, result["error"]
    blob = Path(result["path"])
    assert blob.exists() and blob.read_bytes()[:4] == b"PBS1", (
        "the backup is not the vault's encrypted stream format"
    )


# =========================================================================
# The routes the tabs actually call
# =========================================================================
def test_the_tab_routes_round_trip_over_http(app):
    controller, _ = app
    _feed(controller, "client-marcus.txt", CLIENT_CALL)

    server = AppServer(controller)
    httpd = server.make_server("127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def get(path):
        req = urllib.request.Request(base + path, headers={"X-Token": server.token})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())

    def post(path, payload):
        req = urllib.request.Request(
            base + path, data=json.dumps(payload).encode(),
            headers={"X-Token": server.token, "Content-Type": "application/json"},
            method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())

    try:
        assert get("/api/recordings")["recordings"][0]["name"] == "client-marcus.txt"
        assert get("/api/search?q=elimination%20period")["matches"]
        items = get("/api/followups?status=open")["items"]
        assert items
        assert post("/api/followups/mark", {"id": items[0]["id"], "status": "done"})["ok"]
        assert get("/api/quarantine") == {"entries": []}
        backup = post("/api/backup", {})
        assert backup["ok"] and Path(backup["path"]).exists()

        # And the page itself carries the tabs the routes exist for.
        req = urllib.request.Request(base + "/", headers={"X-Token": server.token})
        with urllib.request.urlopen(req, timeout=10) as resp:
            page = resp.read().decode()
        for tab in ("Library", "Search", "Ask", "Follow-ups", "Held", "Tools"):
            assert tab in page, f"the {tab} tab is missing from the page"
    finally:
        httpd.shutdown()
        httpd.server_close()
