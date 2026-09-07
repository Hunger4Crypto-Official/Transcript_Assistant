"""
The diarization engine's plumbing: availability, loading, and degradation.

pyannote is not installed here and the network will not hand over its weights,
so every test that reaches the library reaches a stand-in placed in
`sys.modules`. That is the honest way to test this file: the questions are
"what does it say when the library is missing", "what does it pass to
`from_pretrained`", and "what happens when the model blows up mid-call", and
none of those needs a real model to answer.
"""

from __future__ import annotations

import sys
import types

import pytest

from _fixtures import build_sandbox
from plaud_bridge.diarize import engine
from plaud_bridge.diarize.engine import DiarizationError, _available, _load_pipeline, speaker_turns
from plaud_bridge.models import Segment
from plaud_bridge.runtime import model_path

MODEL = "pyannote/speaker-diarization-3.1"


# ---------------------------------------------------------------------------
# stand-ins
# ---------------------------------------------------------------------------
class FakeTurn:
    def __init__(self, start, end):
        self.start, self.end = start, end


class FakeAnnotation:
    def __init__(self, turns):
        self._turns = turns

    def itertracks(self, yield_label=False):
        for start, end, label in self._turns:
            yield FakeTurn(start, end), None, label


class FakePipeline:
    """Records how it was constructed and called, and returns what it is told to."""

    loads: list[tuple[tuple, dict]] = []
    turns: list[tuple[float, float, str]] = []
    fail_with: Exception | None = None

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.devices: list[object] = []

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.loads.append((args, kwargs))
        return cls()

    def to(self, device):
        self.devices.append(device)
        return self

    def __call__(self, path, **kwargs):
        self.calls.append((path, kwargs))
        if self.fail_with is not None:
            raise self.fail_with
        return FakeAnnotation(self.turns)


class FakeDevice:
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        return isinstance(other, FakeDevice) and other.name == self.name

    def __repr__(self):
        return f"device({self.name})"


def install_fake_pyannote(monkeypatch, *, turns=(), fail_with=None):
    pkg = types.ModuleType("pyannote")
    pkg.__path__ = []  # marks it as a package so "pyannote.audio" resolves under it
    audio = types.ModuleType("pyannote.audio")
    FakePipeline.loads = []
    FakePipeline.turns = list(turns)
    FakePipeline.fail_with = fail_with
    audio.Pipeline = FakePipeline
    pkg.audio = audio
    monkeypatch.setitem(sys.modules, "pyannote", pkg)
    monkeypatch.setitem(sys.modules, "pyannote.audio", audio)
    monkeypatch.setattr(engine, "_PIPELINE_CACHE", {})
    return FakePipeline


def remove_pyannote(monkeypatch):
    """A None entry in sys.modules makes `import pyannote.audio` raise ImportError."""
    monkeypatch.setitem(sys.modules, "pyannote", None)
    monkeypatch.setitem(sys.modules, "pyannote.audio", None)


def install_fake_torch(monkeypatch, *, cuda: bool):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: cuda)
    torch.device = FakeDevice
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def enabled_sandbox(tmp_path, monkeypatch, **diarization):
    """The sandbox turns diarization off; these tests are about it being on."""
    block = {"enabled": True}
    block.update(diarization)
    cfg, _ = build_sandbox(tmp_path, monkeypatch, overrides={"diarization": block})
    return cfg


# ---------------------------------------------------------------------------
# _available: every reason it can say no, in the order it checks them
# ---------------------------------------------------------------------------
def test_a_provider_other_than_pyannote_means_no_diarization(tmp_path, monkeypatch):
    cfg = enabled_sandbox(tmp_path, monkeypatch, provider="none")
    assert _available(cfg) == (False, "no diarization provider configured")


def test_a_missing_library_is_reported_as_the_library_not_the_token(tmp_path, monkeypatch):
    cfg = enabled_sandbox(tmp_path, monkeypatch)
    remove_pyannote(monkeypatch)
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_present")

    ok, why = _available(cfg)
    assert not ok
    assert "pyannote.audio is not installed" in why
    assert "TOKEN" not in why


def test_local_weights_need_no_token_at_all(tmp_path, monkeypatch):
    cfg = enabled_sandbox(tmp_path, monkeypatch)
    install_fake_pyannote(monkeypatch)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    model_path(cfg, "diarization", MODEL.replace("/", "__")).mkdir(parents=True)

    assert _available(cfg) == (True, "ready (local weights)")


def test_offline_without_local_weights_names_the_fetch_script(tmp_path, monkeypatch):
    cfg = enabled_sandbox(tmp_path, monkeypatch)
    install_fake_pyannote(monkeypatch)
    monkeypatch.setattr(engine, "is_offline", lambda c: True)
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_present")

    ok, why = _available(cfg)
    assert not ok
    assert "runtime.offline is on" in why
    assert MODEL in why
    assert "fetch_models.py" in why


def test_online_without_weights_or_token_says_which_variable_is_missing(tmp_path, monkeypatch):
    cfg = enabled_sandbox(tmp_path, monkeypatch)
    install_fake_pyannote(monkeypatch)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)

    assert _available(cfg) == (False, "HUGGINGFACE_TOKEN is not set")

    # Whitespace is not a token.
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "   ")
    assert _available(cfg) == (False, "HUGGINGFACE_TOKEN is not set")


def test_the_token_variable_name_comes_from_config(tmp_path, monkeypatch):
    cfg = enabled_sandbox(tmp_path, monkeypatch, pyannote={
        "model": MODEL, "hf_token_env": "MY_HF_TOKEN", "device": "auto",
    })
    install_fake_pyannote(monkeypatch)
    monkeypatch.delenv("MY_HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "the-wrong-variable")

    assert _available(cfg) == (False, "MY_HF_TOKEN is not set")
    monkeypatch.setenv("MY_HF_TOKEN", "hf_right")
    assert _available(cfg) == (True, "ready")


# ---------------------------------------------------------------------------
# _load_pipeline: what reaches from_pretrained, and the device dance
# ---------------------------------------------------------------------------
def test_local_weights_are_loaded_by_path_without_a_token(tmp_path, monkeypatch):
    cfg = enabled_sandbox(tmp_path, monkeypatch)
    fake = install_fake_pyannote(monkeypatch)
    install_fake_torch(monkeypatch, cuda=False)
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_should_not_be_used")
    local = model_path(cfg, "diarization", MODEL.replace("/", "__"))
    local.mkdir(parents=True)

    pipe = _load_pipeline(cfg)

    assert fake.loads == [((str(local),), {})], (
        "local weights were loaded by name or with a token, which means a "
        "download could be attempted for a model already on disk"
    )
    assert pipe.devices == [FakeDevice("cpu")]


def test_remote_weights_are_loaded_by_name_with_the_token(tmp_path, monkeypatch):
    cfg = enabled_sandbox(tmp_path, monkeypatch)
    fake = install_fake_pyannote(monkeypatch)
    install_fake_torch(monkeypatch, cuda=True)
    monkeypatch.setenv("HUGGINGFACE_TOKEN", " hf_secret ")

    pipe = _load_pipeline(cfg)

    assert fake.loads == [((MODEL,), {"use_auth_token": "hf_secret"})]
    assert pipe.devices == [FakeDevice("cuda")], "auto should pick cuda when it is available"


def test_an_explicit_device_is_used_as_written(tmp_path, monkeypatch):
    cfg = enabled_sandbox(tmp_path, monkeypatch, pyannote={
        "model": MODEL, "hf_token_env": "HUGGINGFACE_TOKEN", "device": "cpu",
    })
    install_fake_pyannote(monkeypatch)
    install_fake_torch(monkeypatch, cuda=True)
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_secret")

    assert _load_pipeline(cfg).devices == [FakeDevice("cpu")]


def test_a_missing_token_is_passed_as_none_rather_than_an_empty_string(tmp_path, monkeypatch):
    cfg = enabled_sandbox(tmp_path, monkeypatch)
    fake = install_fake_pyannote(monkeypatch)
    install_fake_torch(monkeypatch, cuda=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)

    _load_pipeline(cfg)
    assert fake.loads == [((MODEL,), {"use_auth_token": None})]


def test_torch_being_absent_does_not_stop_the_pipeline_loading(tmp_path, monkeypatch):
    cfg = enabled_sandbox(tmp_path, monkeypatch)
    install_fake_pyannote(monkeypatch)
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_secret")

    pipe = _load_pipeline(cfg)
    assert isinstance(pipe, FakePipeline)
    assert pipe.devices == [], "nothing should have been moved anywhere without torch"


def test_the_pipeline_is_loaded_once_and_reused(tmp_path, monkeypatch):
    cfg = enabled_sandbox(tmp_path, monkeypatch)
    fake = install_fake_pyannote(monkeypatch)
    install_fake_torch(monkeypatch, cuda=False)
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_secret")

    first = _load_pipeline(cfg)
    second = _load_pipeline(cfg)
    assert first is second
    assert len(fake.loads) == 1, "the model was loaded twice for the same config"


# ---------------------------------------------------------------------------
# speaker_turns: clusters without words
# ---------------------------------------------------------------------------
def test_speaker_turns_raises_with_the_reason_when_diarization_is_unavailable(tmp_path, monkeypatch):
    cfg = enabled_sandbox(tmp_path, monkeypatch)
    remove_pyannote(monkeypatch)

    with pytest.raises(DiarizationError, match="pyannote.audio is not installed"):
        speaker_turns(tmp_path / "clip.wav", cfg)


def test_speaker_turns_returns_wordless_segments_with_the_models_labels(tmp_path, monkeypatch):
    cfg = enabled_sandbox(tmp_path, monkeypatch)
    install_fake_pyannote(monkeypatch, turns=[
        (0.0, 12.5, "SPEAKER_00"), (12.5, 20.0, "SPEAKER_01"), (20.0, 31.0, "SPEAKER_00"),
    ])
    install_fake_torch(monkeypatch, cuda=False)
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_secret")
    clip = tmp_path / "clip.wav"

    out = speaker_turns(clip, cfg)

    assert [(s.start, s.end, s.speaker, s.text) for s in out] == [
        (0.0, 12.5, "SPEAKER_00", ""),
        (12.5, 20.0, "SPEAKER_01", ""),
        (20.0, 31.0, "SPEAKER_00", ""),
    ]
    # The shipped config bounds the speaker count, and those bounds reach the model.
    pipe = engine._PIPELINE_CACHE[MODEL]
    assert pipe.calls == [(str(clip), {"min_speakers": 1, "max_speakers": 6})]


def test_speaker_bounds_of_zero_are_not_passed_to_the_model(tmp_path, monkeypatch):
    cfg = enabled_sandbox(tmp_path, monkeypatch, min_speakers=0, max_speakers=None)
    install_fake_pyannote(monkeypatch, turns=[(0.0, 5.0, "A")])
    install_fake_torch(monkeypatch, cuda=False)
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_secret")

    speaker_turns(tmp_path / "clip.wav", cfg)
    assert engine._PIPELINE_CACHE[MODEL].calls[0][1] == {}


# ---------------------------------------------------------------------------
# diarize: it degrades, it never raises upward
# ---------------------------------------------------------------------------
def _segments(*spans):
    return [Segment(start=s, end=e, text="...") for s, e in spans]


def test_diarization_that_is_unavailable_is_skipped_without_touching_the_model(
    tmp_path, monkeypatch
):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)   # diarization.enabled is False here
    monkeypatch.setattr(engine, "_load_pipeline", lambda c: pytest.fail("must not load"))
    monkeypatch.setattr(engine, "named_speakers", lambda *a, **k: pytest.fail("must not run"))

    segments = _segments((0.0, 10.0), (10.0, 20.0))
    out = engine.diarize(tmp_path / "clip.wav", segments, cfg)
    assert out is segments
    assert [s.speaker for s in out] == ["SPEAKER", "SPEAKER"]


def test_a_model_that_raises_leaves_every_segment_with_the_default_label(tmp_path, monkeypatch):
    cfg = enabled_sandbox(tmp_path, monkeypatch)
    install_fake_pyannote(monkeypatch, fail_with=RuntimeError("CUDA out of memory"))
    install_fake_torch(monkeypatch, cuda=False)
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_secret")
    named = []
    monkeypatch.setattr(engine, "named_speakers", lambda *a, **k: named.append(1) or {})

    segments = _segments((0.0, 10.0), (10.0, 20.0))
    out = engine.diarize(tmp_path / "clip.wav", segments, cfg)

    assert out is segments, "the caller's segments must come back, not a copy"
    assert [s.speaker for s in out] == ["SPEAKER", "SPEAKER"]
    assert named == [], "identification ran on clusters that do not exist"


def test_a_model_that_finds_no_turns_leaves_every_segment_with_the_default_label(
    tmp_path, monkeypatch
):
    cfg = enabled_sandbox(tmp_path, monkeypatch)
    install_fake_pyannote(monkeypatch, turns=[])
    install_fake_torch(monkeypatch, cuda=False)
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_secret")
    monkeypatch.setattr(engine, "named_speakers", lambda *a, **k: pytest.fail("should not run"))

    out = engine.diarize(tmp_path / "clip.wav", _segments((0.0, 10.0), (10.0, 20.0)), cfg)
    assert [s.speaker for s in out] == ["SPEAKER", "SPEAKER"]


def test_each_segment_takes_the_turn_that_overlaps_it_most(tmp_path, monkeypatch):
    cfg = enabled_sandbox(tmp_path, monkeypatch, assume_owner_is_dominant_speaker=False)
    install_fake_pyannote(monkeypatch, turns=[(0.0, 10.0, "SPEAKER_00"), (10.0, 30.0, "SPEAKER_01")])
    install_fake_torch(monkeypatch, cuda=False)
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_secret")
    monkeypatch.setattr(engine, "named_speakers", lambda *a, **k: {})

    # The middle segment straddles the boundary: 4s in the first turn, 6s in the second.
    out = engine.diarize(tmp_path / "clip.wav", _segments((0.0, 5.0), (6.0, 16.0), (20.0, 30.0)), cfg)

    # With the owner rule off and nobody recognised, the model's own labels stand.
    assert [s.speaker for s in out] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_01"]


def test_a_segment_no_turn_overlaps_keeps_the_default_label(tmp_path, monkeypatch):
    cfg = enabled_sandbox(tmp_path, monkeypatch, assume_owner_is_dominant_speaker=False)
    install_fake_pyannote(monkeypatch, turns=[(0.0, 10.0, "SPEAKER_00")])
    install_fake_torch(monkeypatch, cuda=False)
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_secret")
    monkeypatch.setattr(engine, "named_speakers", lambda *a, **k: {})

    out = engine.diarize(tmp_path / "clip.wav", _segments((0.0, 5.0), (40.0, 50.0)), cfg)
    assert [s.speaker for s in out] == ["SPEAKER_00", "SPEAKER"]


def test_fallback_numbering_skips_a_label_that_is_already_taken(tmp_path, monkeypatch):
    """
    The owner label is whatever the config says. If it happens to be "Speaker 1",
    the next unrecognised cluster must not also become "Speaker 1", or two
    different people collapse into one name on the transcript.
    """
    cfg = enabled_sandbox(tmp_path, monkeypatch, owner_label="Speaker 1")
    turns = [(0.0, 60.0, "SPEAKER_00"), (60.0, 80.0, "SPEAKER_01"), (80.0, 90.0, "SPEAKER_02")]
    install_fake_pyannote(monkeypatch, turns=turns)
    install_fake_torch(monkeypatch, cuda=False)
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_secret")
    monkeypatch.setattr(engine, "named_speakers", lambda *a, **k: {})

    out = engine.diarize(tmp_path / "clip.wav", _segments(*[(s, e) for s, e, _ in turns]), cfg)

    assert [s.speaker for s in out] == ["Speaker 1", "Speaker 2", "Speaker 3"]
    assert len({s.speaker for s in out}) == 3
