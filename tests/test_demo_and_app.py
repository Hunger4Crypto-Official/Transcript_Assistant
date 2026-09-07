"""
The two commands that make a fresh install usable: `demo` and `app`.

`demo` furnishes an empty archive with fictional recordings so nobody has to
record their life for a week before they can judge the tool. `app` is the same
local server the packaged Windows build runs, started from a terminal, with a
`--probe` self-check that stands the whole stack up and exits.

What these pin: the samples are labelled as fiction and go through the real
pipeline (no pre-baked database), they never overwrite a real recording, and
the app probe genuinely serves a token-guarded page rather than reporting
success from a variable nobody set.
"""

from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path

import pytest

from _fixtures import CLIENT_CALL, StubLLM, build_sandbox
from plaud_bridge import demo
from plaud_bridge.cli import main
from plaud_bridge.db import Database
from plaud_bridge.desktop import AppController, Brain
from plaud_bridge.desktop.server import AppServer

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    cfg, _stub = build_sandbox(tmp_path, monkeypatch)
    return cfg


def cli(cfg, *argv) -> int:
    return main(["--config", str(cfg.root / "config"), *argv])


# =========================================================================
# demo: honest fiction
# =========================================================================
def test_every_sample_announces_itself_as_fiction(sandbox):
    """
    A sample must be impossible to mistake for a real recording later. The
    banner is a transcript line, so it survives into the stored transcript and
    shows up in `open`, in search, and in the player.
    """
    assert cli(sandbox, "demo") == 0
    for sample in demo.SAMPLES:
        body = (sandbox.path("inbox") / sample.filename).read_text(encoding="utf-8")
        assert body.splitlines()[0].startswith("[SAMPLE]")
        assert "fictional sample data" in body.splitlines()[0]
        assert sample.filename.startswith("sample-")


def test_the_samples_speak_as_the_configured_owner(sandbox):
    """
    The roster marks the owner as the owner and the talk-time numbers are
    about the right person only if the samples use the configured label.
    """
    owner = demo.owner_label(sandbox)
    assert owner == "Sasson"
    assert cli(sandbox, "demo") == 0
    body = (sandbox.path("inbox") / demo.CLIENT_CALL.filename).read_text(encoding="utf-8")
    assert f"{owner}:" in body
    assert "{OWNER}" not in body, "the owner placeholder was left unsubstituted"


def test_demo_never_overwrites_something_already_in_the_inbox(sandbox):
    """The inbox holds real recordings. A demo command is not a reason to lose one."""
    victim = sandbox.path("inbox") / demo.CLIENT_CALL.filename
    victim.parent.mkdir(parents=True, exist_ok=True)
    victim.write_text("a real recording that happens to share the name", encoding="utf-8")

    assert cli(sandbox, "demo") == 0
    assert victim.read_text(encoding="utf-8") == "a real recording that happens to share the name"

    # --force is the deliberate way to replace them.
    assert cli(sandbox, "demo", "--force") == 0
    assert victim.read_text(encoding="utf-8").startswith("[SAMPLE]")


def test_clean_removes_only_the_samples(sandbox):
    mine = sandbox.path("inbox") / "my-real-recording.txt"
    assert cli(sandbox, "demo") == 0
    mine.write_text(CLIENT_CALL, encoding="utf-8")

    assert cli(sandbox, "demo", "--clean") == 0
    assert mine.exists(), "a real recording was removed by a sample cleanup"
    for sample in demo.SAMPLES:
        assert not (sandbox.path("inbox") / sample.filename).exists()


def test_demo_process_populates_the_archive_through_the_real_pipeline(sandbox):
    """
    No pre-baked database: the samples are processed by the same pipeline any
    recording goes through, so what a person explores is real output.
    """
    assert cli(sandbox, "demo", "--process") == 0

    db = Database(sandbox.path("database"))
    try:
        rows = db.query(limit=50)
    finally:
        db.close()
    assert len(rows) >= 3, "the samples did not reach the archive"

    profiles = {r["governing_profile"] for r in rows}
    stages = {r["stage"] for r in rows}
    # The set is chosen to show several states at once, including the gate
    # holding the consent-free call.
    assert "insurance_agent" in profiles
    assert "quarantined" in stages, "the no-consent sample should have been held"


def test_the_furnished_archive_answers_every_read_only_view(sandbox):
    """The point of the samples: no tab is empty afterwards."""
    assert cli(sandbox, "demo", "--process") == 0
    for argv in (("digest",), ("brief",), ("people",), ("insights",),
                 ("followups",), ("status",), ("quarantine",)):
        assert cli(sandbox, *argv) in (0, 2), f"{argv[0]} failed on the demo archive"


def test_cleaning_an_empty_inbox_is_not_an_error(sandbox):
    assert cli(sandbox, "demo", "--clean") == 0


# =========================================================================
# app: the headless probe
# =========================================================================
def test_the_app_probe_serves_a_real_token_guarded_page(sandbox, tmp_path, capsys):
    home = tmp_path / "apphome"
    assert cli(sandbox, "app", "--probe", "--home", str(home)) == 0
    out = capsys.readouterr().out
    assert "app probe: ok" in out
    assert "token guard: on" in out
    assert "phone mode:  off" in out


def test_the_probe_reports_failure_rather_than_claiming_success(sandbox, tmp_path,
                                                               monkeypatch, capsys):
    """
    Mutation-style: if the probe stopped checking the response, this would
    still print "ok". Breaking the page's token makes the check the only thing
    that can tell the difference.
    """
    from plaud_bridge.desktop import server as server_module

    monkeypatch.setattr(server_module, "_PAGE", "<html>a page with no token</html>")
    code = cli(sandbox, "app", "--probe", "--home", str(tmp_path / "apphome2"))
    assert code == 1
    assert "app probe: FAILED" in capsys.readouterr().out


# =========================================================================
# The app's own sample loader
# =========================================================================
@pytest.fixture
def app(tmp_path, monkeypatch):
    stub = StubLLM()
    for module in ("plaud_bridge.profiles.router", "plaud_bridge.profiles.extractor"):
        monkeypatch.setattr(f"{module}.complete_json", stub)
    monkeypatch.setenv("PLAUD_BRIDGE_PASSPHRASE", "a-long-enough-desktop-passphrase")
    return AppController(base_dir=tmp_path / "home", template_dir=ROOT / "config")


def test_the_app_can_install_the_samples(app):
    result = app.install_samples()
    assert result["ok"] is True and result["written"] == len(demo.SAMPLES)
    inbox = app.load_config(Brain.CLOUD).path("inbox")
    for sample in demo.SAMPLES:
        assert (inbox / sample.filename).exists()

    # Running it twice writes nothing new rather than duplicating.
    again = app.install_samples()
    assert again["written"] == 0 and again["skipped"] == len(demo.SAMPLES)


def test_the_demo_route_starts_a_real_run_over_http(app):
    srv = AppServer(app)
    httpd = srv.make_server("127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        req = urllib.request.Request(
            base + "/api/demo", data=b"{}", method="POST",
            headers={"X-Token": srv.token, "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            payload = json.loads(resp.read())
        assert payload["ok"] is True
        assert payload["written"] == len(demo.SAMPLES)
        assert payload["started"] is True
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_the_page_offers_the_samples_and_the_palette():
    """Both are inline-onclick driven, so their functions must be global."""
    from plaud_bridge.desktop.server import _PAGE

    assert "loadSamples()" in _PAGE
    # Declared at top level, not nested inside loadLibrary, or the button's
    # onclick would never resolve it.
    assert "\nasync function loadSamples()" in _PAGE
    for fn in ("function openPalette()", "function runPalette(", "function paletteKey("):
        assert fn in _PAGE
    assert "Ctrl-K" in _PAGE
