"""
The engine behind the clickable app.

These drive `AppController` the way a window would -- first run, brain switch,
readiness, and a real recording all the way to a digest -- with the LLM stubbed
so nothing leaves the machine and no key is needed. The point is that the app is
a thin front door: the same `Pipeline`, the same config, the same vault.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from _fixtures import CLIENT_CALL, StubLLM
from plaud_bridge.desktop import AppController, Brain

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
