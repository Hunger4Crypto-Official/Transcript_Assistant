"""
Offline mode.

The requirement: install on a machine that has never been attached to a network
and have everything work, while the same code still runs with Groq when you do
have a connection.

Offline is an assertion, so these tests are about refusal. A tool that claims to
be offline and then quietly downloads a 3GB model on first use has told you
something false about where your recordings went.
"""

import pytest
import yaml

from _fixtures import build_sandbox
from plaud_bridge.config import Config, ConfigError
from plaud_bridge.runtime import (
    OfflineError,
    cloud_providers_enabled,
    is_offline,
    model_path,
    models_dir,
    require_local,
    resolve_local_model,
)


def _reconfigure(tmp_path, **blocks):
    """Rewrite the sandbox config, merging one level deep, and reload it."""
    path = tmp_path / "config" / "pipeline.yaml"
    raw = yaml.safe_load(path.read_text())
    for block, values in blocks.items():
        raw.setdefault(block, {})
        for key, value in values.items():
            if isinstance(value, dict) and isinstance(raw[block].get(key), dict):
                raw[block][key].update(value)
            else:
                raw[block][key] = value
    path.write_text(yaml.safe_dump(raw))
    return Config.load(tmp_path / "config", root=tmp_path)


def _all_local(tmp_path):
    """Offline on, every cloud provider off. The air-gapped shape."""
    return _reconfigure(
        tmp_path,
        runtime={"offline": True},
        asr={"providers": ["local"], "groq": {"enabled": False}},
        llm={"providers": ["local"], "anthropic": {"enabled": False},
             "groq": {"enabled": False}, "local": {"enabled": True}},
    )


# =========================================================================
# The assertion
# =========================================================================
def test_offline_refuses_to_load_while_a_cloud_provider_is_enabled(tmp_path, monkeypatch):
    """The claim is already false at that point. Startup is where to say so."""
    build_sandbox(tmp_path, monkeypatch)
    with pytest.raises(ConfigError) as excinfo:
        _reconfigure(tmp_path, runtime={"offline": True})

    message = str(excinfo.value)
    assert "runtime.offline" in message
    assert "asr.groq" in message or "llm.anthropic" in message
    assert "cannot be both" in message


def test_offline_loads_once_the_cloud_providers_are_off(tmp_path, monkeypatch):
    build_sandbox(tmp_path, monkeypatch)
    cfg = _all_local(tmp_path)
    assert is_offline(cfg)
    assert cloud_providers_enabled(cfg) == []


def test_the_shipped_config_is_online_and_says_so(tmp_path, monkeypatch):
    """The default has to keep working for someone with a connection."""
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    assert not is_offline(cfg)
    assert cloud_providers_enabled(cfg), "the online default should have cloud providers"


# =========================================================================
# Model resolution
# =========================================================================
def test_a_model_on_disk_resolves_to_its_path(tmp_path, monkeypatch):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    weights = model_path(cfg, "whisper", "large-v3")
    weights.mkdir(parents=True)

    target, local = resolve_local_model(cfg, "large-v3", "whisper")
    assert local
    assert target == str(weights)


def test_a_slashed_model_name_flattens_into_one_directory(tmp_path, monkeypatch):
    """So the models directory copies onto a USB stick without surprises."""
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    weights = model_path(cfg, "diarization", "pyannote__speaker-diarization-3.1")
    weights.mkdir(parents=True)

    target, local = resolve_local_model(cfg, "pyannote/speaker-diarization-3.1", "diarization")
    assert local
    assert target == str(weights)


def test_a_missing_model_passes_through_when_online(tmp_path, monkeypatch):
    """Online, an unresolved name means "download it", which is fine."""
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    target, local = resolve_local_model(cfg, "large-v3", "whisper")
    assert not local
    assert target == "large-v3"
    assert require_local(cfg, "large-v3", "whisper", "speech recognition") == "large-v3"


def test_a_missing_model_offline_refuses_and_says_where_to_get_it(tmp_path, monkeypatch):
    build_sandbox(tmp_path, monkeypatch)
    cfg = _all_local(tmp_path)

    with pytest.raises(OfflineError) as excinfo:
        require_local(cfg, "large-v3", "whisper", "speech recognition")

    message = str(excinfo.value)
    assert str(model_path(cfg, "whisper", "large-v3")) in message, "it did not name the path"
    assert "fetch_models.py" in message, "it did not say how to fix it"
    assert "offline means offline" in message


def test_a_present_model_offline_is_returned_without_complaint(tmp_path, monkeypatch):
    build_sandbox(tmp_path, monkeypatch)
    cfg = _all_local(tmp_path)
    model_path(cfg, "whisper", "large-v3").mkdir(parents=True)

    assert require_local(cfg, "large-v3", "whisper", "speech recognition") == str(
        model_path(cfg, "whisper", "large-v3")
    )


def test_models_dir_is_relative_to_the_project(tmp_path, monkeypatch):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    assert models_dir(cfg) == (tmp_path / "models").resolve()


# =========================================================================
# Diarization degrades rather than failing
# =========================================================================
def test_diarization_needs_no_token_once_the_weights_are_local(tmp_path, monkeypatch):
    """
    A token is only needed to download. Demanding one afterwards would make
    offline speaker separation impossible for no reason.
    """
    from plaud_bridge.diarize.engine import _available

    build_sandbox(tmp_path, monkeypatch)
    cfg = _all_local(tmp_path)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)

    cfg = _reconfigure(tmp_path, diarization={"enabled": True})
    model_path(cfg, "diarization", "pyannote__speaker-diarization-3.1").mkdir(parents=True)
    ok, why = _available(cfg)
    # pyannote.audio is not installed in CI, so the honest answer is that the
    # package is missing -- not that the token is.
    assert "TOKEN" not in why, why
    if ok:
        assert "local" in why


def test_diarization_offline_without_weights_explains_itself(tmp_path, monkeypatch):
    from plaud_bridge.diarize.engine import _available

    build_sandbox(tmp_path, monkeypatch)
    cfg = _all_local(tmp_path)
    # The sandbox turns diarization off; this test is about what it says when it
    # is on and the weights are missing.
    cfg = _reconfigure(tmp_path, diarization={"enabled": True})
    ok, why = _available(cfg)
    assert not ok
    assert "fetch_models.py" in why or "not installed" in why


# =========================================================================
# doctor
# =========================================================================
def test_doctor_offline_flags_providers_that_would_reach_the_network(
    tmp_path, monkeypatch, capsys
):
    from plaud_bridge.cli import build_parser, cmd_doctor

    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    args = build_parser().parse_args(
        ["--config", str(cfg.root / "config"), "doctor", "--offline"]
    )
    cmd_doctor(args)

    out = capsys.readouterr().out
    assert "offline:providers" in out
    assert "would reach the network" in out
    assert "offline:asr" in out
    assert "fetch_models.py" in out


def test_doctor_offline_passes_the_provider_check_when_air_gapped(
    tmp_path, monkeypatch, capsys
):
    build_sandbox(tmp_path, monkeypatch)
    cfg = _all_local(tmp_path)
    model_path(cfg, "whisper", "large-v3").mkdir(parents=True)

    from plaud_bridge.cli import build_parser, cmd_doctor

    args = build_parser().parse_args(
        ["--config", str(cfg.root / "config"), "doctor", "--offline"]
    )
    cmd_doctor(args)

    out = capsys.readouterr().out
    assert "no cloud provider is enabled" in out
    assert "'large-v3' on disk" in out


def test_the_fetch_script_runs_nothing_on_import():
    """It belongs on the networked machine and must not be a runtime dependency."""
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "fetch_models.py"
    assert script.exists()
    spec = importlib.util.spec_from_file_location("fetch_models", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)          # importing must be side-effect free
    assert hasattr(module, "main")
