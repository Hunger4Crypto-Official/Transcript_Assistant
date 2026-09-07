"""
The engine behind the clickable app.

These drive `AppController` the way a window would -- first run, brain switch,
readiness, and a real recording all the way to a digest -- with the LLM stubbed
so nothing leaves the machine and no key is needed. The point is that the app is
a thin front door: the same `Pipeline`, the same config, the same vault.
"""

from __future__ import annotations

import json
import socket
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from _fixtures import CLIENT_CALL, StubLLM
from plaud_bridge.desktop import AppController, Brain, LocalLLMStatus, probe_local_llm

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def app(tmp_path, monkeypatch):
    """A controller seeded from the shipped config, with the LLM stubbed."""
    stub = StubLLM()
    for module in ("plaud_bridge.profiles.router", "plaud_bridge.profiles.extractor"):
        monkeypatch.setattr(f"{module}.complete_json", stub)
    monkeypatch.setenv("PLAUD_BRIDGE_PASSPHRASE", "a-long-enough-desktop-passphrase")
    controller = AppController(base_dir=tmp_path / "home", template_dir=ROOT / "config")
    return controller, stub


# =========================================================================
# First run
# =========================================================================
def test_first_run_lays_down_a_working_config(app):
    controller, _ = app
    assert not controller.config_dir.exists()
    controller.ensure_installed()
    assert (controller.config_dir / "pipeline.yaml").exists()
    assert (controller.config_dir / "profiles").is_dir()


def test_a_second_run_does_not_clobber_edited_config(app):
    controller, _ = app
    controller.ensure_installed()
    edited = controller.config_dir / "pipeline.yaml"
    edited.write_text(edited.read_text() + "\n# my own tweak\n", encoding="utf-8")
    controller.ensure_installed()
    assert "# my own tweak" in edited.read_text(), "an update overwrote the person's config"


# =========================================================================
# The brain switch
# =========================================================================
def test_offline_brain_permits_no_cloud_llm(app):
    controller, _ = app
    cfg = controller.load_config(Brain.OFFLINE)
    assert cfg.get("llm.providers") == ["local"], "offline let a cloud provider into the chain"


def test_cloud_brain_still_ends_in_a_local_fallback(app):
    controller, _ = app
    cfg = controller.load_config(Brain.CLOUD)
    providers = cfg.get("llm.providers")
    assert "groq" in providers and providers[-1] == "local", (
        "cloud mode should try the free key first and still fall back to local"
    )


# =========================================================================
# Readiness
# =========================================================================
def test_preflight_flags_a_missing_passphrase_as_fatal(app, monkeypatch):
    controller, _ = app
    monkeypatch.delenv("PLAUD_BRIDGE_PASSPHRASE", raising=False)
    passphrase = [i for i in controller.preflight(Brain.CLOUD) if i.name == "passphrase"]
    assert passphrase and passphrase[0].fatal and not passphrase[0].ok


def test_set_passphrase_clears_that_check(app):
    controller, _ = app
    controller.set_passphrase("another-long-enough-passphrase")
    passphrase = [i for i in controller.preflight(Brain.CLOUD) if i.name == "passphrase"]
    assert passphrase and passphrase[0].ok


# =========================================================================
# All the way to a digest
# =========================================================================
def test_a_picked_recording_processes_and_produces_a_digest(app, tmp_path):
    controller, _ = app

    # A person picks a transcript from anywhere on disk (text skips ASR/ffmpeg,
    # so this exercises the whole front-to-back path without audio tooling).
    picked = tmp_path / "client-marcus.txt"
    picked.write_text(CLIENT_CALL, encoding="utf-8")

    landed = controller.add_files([picked])
    assert landed and landed[0].exists(), "the picked file did not reach the inbox"

    summary = controller.process(Brain.CLOUD)
    assert summary["processed"] == 1, summary
    assert summary["failed"] == 0

    digest = controller.write_digest()
    assert digest.exists() and digest.suffix == ".html"
    body = digest.read_text(encoding="utf-8")
    assert "<html" in body.lower() or "<table" in body.lower(), "the digest is not HTML"


def test_progress_lines_are_reported_to_the_caller(app, tmp_path):
    controller, _ = app
    picked = tmp_path / "note.txt"
    picked.write_text(CLIENT_CALL, encoding="utf-8")
    controller.add_files([picked])

    lines: list[str] = []
    controller.process(Brain.CLOUD, progress=lines.append)
    assert any("Starting" in ln for ln in lines) and any("Done" in ln for ln in lines)


# =========================================================================
# The offline brain diagnoses its own absence
# =========================================================================
# Three worlds a person can be in, each needing a different command typed:
# Ollama not running at all, running without the configured model, and truly
# ready. A loopback stub plays the server so no real Ollama is involved.

def _dead_port() -> int:
    """A port nothing listens on: bind, read the number, close."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextmanager
def _models_server(model_ids: list[str]):
    """A loopback OpenAI-compatible /models endpoint listing `model_ids`."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler's spelling
            body = json.dumps(
                {"object": "list", "data": [{"id": m} for m in model_ids]}
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):  # keep pytest output clean
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()
        server.server_close()


def _aim_offline_brain_at(controller: AppController, base_url: str) -> str:
    """Point the installed config's llm.local at a stub; returns the model name."""
    controller.ensure_installed()
    pipeline = controller.config_dir / "pipeline.yaml"
    pipeline.write_text(
        pipeline.read_text(encoding="utf-8").replace("http://localhost:11434/v1", base_url),
        encoding="utf-8",
    )
    return str(controller.load_config(Brain.OFFLINE).get("llm.local.model"))


def _offline_brain_item(controller: AppController):
    items = [i for i in controller.preflight(Brain.OFFLINE)
             if i.name == "analysis brain (offline)"]
    assert items, "the offline preflight lost its analysis-brain line"
    return items[0]


def test_probe_calls_a_dead_port_not_running():
    status, message = probe_local_llm(f"http://127.0.0.1:{_dead_port()}/v1", "llama3.3:70b")
    assert status is LocalLLMStatus.NOT_RUNNING
    assert "ollama.com" in message and "ollama pull llama3.3:70b" in message


def test_probe_spots_a_model_that_was_never_pulled():
    with _models_server(["mistral:7b"]) as base_url:
        status, message = probe_local_llm(base_url, "llama3.3:70b")
    assert status is LocalLLMStatus.MODEL_MISSING
    assert "ollama pull llama3.3:70b" in message


def test_probe_reports_ready_and_names_the_model():
    with _models_server(["llama3.3:70b"]) as base_url:
        status, message = probe_local_llm(base_url, "llama3.3:70b")
    assert status is LocalLLMStatus.READY
    assert "llama3.3:70b" in message


def test_probe_knows_latest_is_ollamas_implicit_tag():
    """`ollama pull llama3` lists as llama3:latest; do not demand a re-pull."""
    with _models_server(["llama3:latest"]) as base_url:
        status, _ = probe_local_llm(base_url, "llama3")
    assert status is LocalLLMStatus.READY
    with _models_server(["llama3.3:70b"]) as base_url:
        status, _ = probe_local_llm(base_url, "llama3.3")
    assert status is LocalLLMStatus.MODEL_MISSING, "a different tag is a different model"


def test_preflight_tells_an_ollamaless_person_what_to_install(app):
    controller, _ = app
    model = _aim_offline_brain_at(controller, f"http://127.0.0.1:{_dead_port()}/v1")
    item = _offline_brain_item(controller)
    assert item.fatal and not item.ok
    assert "install from ollama.com" in item.detail
    assert f"ollama pull {model}" in item.detail


def test_preflight_tells_a_modelless_person_what_to_pull(app):
    controller, _ = app
    with _models_server(["qwen2.5:3b"]) as base_url:
        model = _aim_offline_brain_at(controller, base_url)
        item = _offline_brain_item(controller)
    assert item.fatal and not item.ok
    assert "is not pulled" in item.detail
    assert f"ollama pull {model}" in item.detail


def test_preflight_goes_green_when_the_model_is_really_there(app):
    controller, _ = app
    controller.ensure_installed()
    model = str(controller.load_config(Brain.OFFLINE).get("llm.local.model"))
    with _models_server([model]) as base_url:
        _aim_offline_brain_at(controller, base_url)
        item = _offline_brain_item(controller)
    assert item.ok, item.detail
    assert model in item.detail


def test_offline_brain_enables_the_local_provider_in_memory(app):
    """
    Choosing Offline must be enough: the template ships llm.local disabled so
    the CLI assumes nothing, but the app's switch is the person's decision.
    The on-disk file stays untouched.
    """
    controller, _ = app
    cfg = controller.load_config(Brain.OFFLINE)
    assert cfg.get("llm.local.enabled") is True
    on_disk = (controller.config_dir / "pipeline.yaml").read_text(encoding="utf-8")
    assert "enabled: false" in on_disk, "the offline switch rewrote the person's config"
