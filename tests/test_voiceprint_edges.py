"""
Named speakers: the branches test_voiceprint.py leaves alone.

Mostly the embedding wrapper. pyannote, torch and numpy are all absent here, so
each is replaced by a stand-in in `sys.modules` that records what it was asked
and returns what the test decided. That pins the contract this code has with
those libraries -- which arguments reach `from_pretrained`, that a span goes
through `crop`, that a many-row result is averaged -- which is what can break
when someone edits this file, and is testable without a single weight on disk.
"""

from __future__ import annotations

import sys
import types

import pytest

from _fixtures import build_sandbox
from plaud_bridge.diarize import voiceprint
from plaud_bridge.diarize.voiceprint import (
    MIN_CROP_SECONDS,
    ClusterMatch,
    Embedder,
    VoiceprintError,
    VoiceprintStore,
    _decide,
    _spans_by_cluster,
    average,
    identify,
    named_speakers,
    normalise,
)
from plaud_bridge.models import Segment
from plaud_bridge.runtime import model_path
from plaud_bridge.storage import Vault, VaultError

EMBEDDING_MODEL = "pyannote/embedding"


# ---------------------------------------------------------------------------
# helpers and stand-ins
# ---------------------------------------------------------------------------
def vec(*values: float) -> list[float]:
    return normalise(list(values))


def rounded(values) -> list[float]:
    """pytest.approx reaches for whatever `numpy` is in sys.modules; this does not."""
    return [round(float(v), 9) for v in values]


def store_for(cfg) -> VoiceprintStore:
    return VoiceprintStore(Vault(cfg.path("vault")))


def segs(*spans) -> list[Segment]:
    return [Segment(start=s, end=e, text="...", speaker=c) for s, e, c in spans]


class FakeModel:
    loads: list[tuple[tuple, dict]] = []

    def __init__(self):
        self.devices: list[object] = []

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.loads.append((args, kwargs))
        return cls()

    def to(self, device):
        self.devices.append(device)
        return self


class FakeInference:
    """Returns `raw` for a whole file and `raw` for a crop, recording each ask."""

    raw: object = [3.0, 4.0]
    built: list[tuple[object, str]] = []

    def __init__(self, model, window="sliding"):
        self.model = model
        self.window = window
        self.whole_calls: list[str] = []
        self.crop_calls: list[tuple[str, tuple[float, float]]] = []
        FakeInference.built.append((model, window))

    def __call__(self, path):
        self.whole_calls.append(path)
        return self.raw

    def crop(self, path, span):
        self.crop_calls.append((path, (span.start, span.end)))
        return self.raw


class FakePSegment:
    def __init__(self, start, end):
        self.start, self.end = start, end

    @property
    def duration(self):
        return self.end - self.start


class FakeArray:
    """Just enough of numpy's ndarray for `_flatten`: ndim, mean over axes, reshape(-1)."""

    def __init__(self, rows):
        self.rows = rows

    @property
    def ndim(self):
        return 2 if self.rows and isinstance(self.rows[0], list) else 1

    def mean(self, axis):
        assert axis == (0,), f"_flatten asked to average over {axis}, expected the window axis"
        width = len(self.rows[0])
        return FakeArray([sum(r[i] for r in self.rows) / len(self.rows) for i in range(width)])

    def reshape(self, shape):
        assert shape == -1
        return self

    def __iter__(self):
        return iter(self.rows)


def install_fake_numpy(monkeypatch):
    np = types.ModuleType("numpy")

    def asarray(raw, dtype=None):
        return FakeArray([list(map(float, r)) if isinstance(r, (list, tuple)) else float(r)
                          for r in raw])

    np.asarray = asarray
    monkeypatch.setitem(sys.modules, "numpy", np)


def install_fake_pyannote(monkeypatch, *, raw=None):
    pkg = types.ModuleType("pyannote")
    pkg.__path__ = []
    audio = types.ModuleType("pyannote.audio")
    core = types.ModuleType("pyannote.core")
    FakeModel.loads = []
    FakeInference.built = []
    FakeInference.raw = [3.0, 4.0] if raw is None else raw
    audio.Model = FakeModel
    audio.Inference = FakeInference
    core.Segment = FakePSegment
    pkg.audio, pkg.core = audio, core
    monkeypatch.setitem(sys.modules, "pyannote", pkg)
    monkeypatch.setitem(sys.modules, "pyannote.audio", audio)
    monkeypatch.setitem(sys.modules, "pyannote.core", core)
    monkeypatch.setattr(Embedder, "_CACHE", {})


def remove_pyannote(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyannote", None)
    monkeypatch.setitem(sys.modules, "pyannote.audio", None)


class FakeDevice:
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        return isinstance(other, FakeDevice) and other.name == self.name

    def __repr__(self):
        return f"device({self.name})"


def install_fake_torch(monkeypatch, *, cuda: bool):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: cuda)
    torch.device = FakeDevice
    monkeypatch.setitem(sys.modules, "torch", torch)


def online_with_token(monkeypatch):
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_secret")


# ---------------------------------------------------------------------------
# the maths
# ---------------------------------------------------------------------------
def test_averaging_nothing_is_an_error_not_a_zero_vector():
    with pytest.raises(VoiceprintError, match="no vectors to average"):
        average([])


def test_averaging_vectors_of_different_widths_is_refused():
    with pytest.raises(VoiceprintError, match="different sizes"):
        average([vec(1.0, 0.0), vec(1.0, 0.0, 0.0)])


# ---------------------------------------------------------------------------
# the store
# ---------------------------------------------------------------------------
def test_a_store_that_does_not_decrypt_raises_the_vault_error_unchanged(sandbox, monkeypatch):
    """
    A wrong passphrase is a vault problem and has to surface as one, so the
    caller can tell "set the right passphrase" apart from "the file is corrupt".
    """
    cfg, _ = sandbox
    store = store_for(cfg)
    store.enroll("Marcus", vec(1.0, 0.0, 0.0))
    store.save()

    monkeypatch.setenv("PLAUD_BRIDGE_PASSPHRASE", "a-completely-different-passphrase")
    with pytest.raises(VaultError):
        store_for(cfg).load()
    # And it is not swallowed into an empty list by the query path either.
    with pytest.raises(VaultError):
        store_for(cfg).people()


def test_a_name_with_no_letters_or_digits_is_rejected(sandbox):
    cfg, _ = sandbox
    with pytest.raises(VoiceprintError, match="does not reduce to a usable id"):
        store_for(cfg).enroll("???", vec(1.0, 0.0, 0.0))
    assert store_for(cfg).people() == []


# ---------------------------------------------------------------------------
# Embedder.available: every reason it can say no
# ---------------------------------------------------------------------------
def test_identification_disabled_in_config_says_which_key(tmp_path, monkeypatch):
    cfg, _ = build_sandbox(tmp_path, monkeypatch,
                           overrides={"diarization": {"identify": {"enabled": False}}})
    assert Embedder.available(cfg) == (
        False, "disabled in config (diarization.identify.enabled)"
    )


def test_a_missing_library_is_the_reason_before_any_token_is_checked(sandbox, monkeypatch):
    cfg, _ = sandbox
    remove_pyannote(monkeypatch)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)

    ok, why = Embedder.available(cfg)
    assert not ok
    assert "pyannote.audio is not installed" in why
    assert "TOKEN" not in why


def test_local_embedding_weights_are_ready_without_a_token(sandbox, monkeypatch):
    cfg, _ = sandbox
    install_fake_pyannote(monkeypatch)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    model_path(cfg, "diarization", "pyannote__embedding").mkdir(parents=True)

    assert Embedder.available(cfg) == (True, "ready (local weights)")


def test_offline_without_embedding_weights_names_the_fetch_flag(sandbox, monkeypatch):
    cfg, _ = sandbox
    install_fake_pyannote(monkeypatch)
    monkeypatch.setattr(voiceprint, "is_offline", lambda c: True)
    online_with_token(monkeypatch)

    ok, why = Embedder.available(cfg)
    assert not ok
    assert "runtime.offline is on" in why
    assert "fetch_models.py --embedding" in why


def test_online_without_weights_or_token_says_both_are_missing(sandbox, monkeypatch):
    cfg, _ = sandbox
    install_fake_pyannote(monkeypatch)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)

    assert Embedder.available(cfg) == (
        False, "HUGGINGFACE_TOKEN is not set, and the weights are not on disk yet"
    )
    online_with_token(monkeypatch)
    assert Embedder.available(cfg) == (True, "ready")


def test_require_turns_unavailability_into_a_voiceprint_error(sandbox, monkeypatch):
    cfg, _ = sandbox
    remove_pyannote(monkeypatch)
    with pytest.raises(VoiceprintError, match="speaker identification is unavailable: pyannote"):
        Embedder(cfg).require()


# ---------------------------------------------------------------------------
# Embedder: loading the model
# ---------------------------------------------------------------------------
def test_local_embedding_weights_are_loaded_by_path_without_a_token(sandbox, monkeypatch):
    cfg, _ = sandbox
    install_fake_pyannote(monkeypatch)
    install_fake_torch(monkeypatch, cuda=False)
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_should_not_be_used")
    local = model_path(cfg, "diarization", "pyannote__embedding")
    local.mkdir(parents=True)

    inference = Embedder(cfg)._inference()

    assert FakeModel.loads == [((str(local),), {})]
    assert inference.window == "whole", "anything but a whole-window embedding yields one row per window"
    assert inference.model.devices == [FakeDevice("cpu")]


def test_remote_embedding_weights_are_loaded_by_name_with_the_token(sandbox, monkeypatch):
    cfg, _ = sandbox
    install_fake_pyannote(monkeypatch)
    install_fake_torch(monkeypatch, cuda=True)
    monkeypatch.setenv("HUGGINGFACE_TOKEN", " hf_secret ")

    inference = Embedder(cfg)._inference()

    assert FakeModel.loads == [((EMBEDDING_MODEL,), {"use_auth_token": "hf_secret"})]
    assert inference.model.devices == [FakeDevice("cuda")]


def test_the_embedding_model_survives_torch_being_absent_and_is_loaded_once(sandbox, monkeypatch):
    cfg, _ = sandbox
    install_fake_pyannote(monkeypatch)
    monkeypatch.setitem(sys.modules, "torch", None)
    online_with_token(monkeypatch)

    first = Embedder(cfg)._inference()
    second = Embedder(cfg)._inference()

    assert first is second
    assert len(FakeModel.loads) == 1
    assert first.model.devices == []


# ---------------------------------------------------------------------------
# Embedder: turning audio into a vector
# ---------------------------------------------------------------------------
def test_a_whole_file_is_embedded_as_a_unit_vector(sandbox, monkeypatch):
    cfg, _ = sandbox
    install_fake_pyannote(monkeypatch, raw=[3.0, 4.0])
    install_fake_numpy(monkeypatch)
    install_fake_torch(monkeypatch, cuda=False)
    online_with_token(monkeypatch)
    clip = cfg.path("inbox") / "marcus.wav"

    out = Embedder(cfg).embed(clip)

    assert rounded(out) == [0.6, 0.8]
    inference = Embedder._CACHE[EMBEDDING_MODEL]
    assert inference.whole_calls == [str(clip)]
    assert inference.crop_calls == []


def test_a_span_is_embedded_through_crop_with_the_exact_bounds(sandbox, monkeypatch):
    cfg, _ = sandbox
    install_fake_pyannote(monkeypatch, raw=[0.0, 5.0])
    install_fake_numpy(monkeypatch)
    install_fake_torch(monkeypatch, cuda=False)
    online_with_token(monkeypatch)
    clip = cfg.path("inbox") / "call.wav"

    out = Embedder(cfg).embed(clip, 12.5, 20.0)

    assert rounded(out) == [0.0, 1.0]
    inference = Embedder._CACHE[EMBEDDING_MODEL]
    assert inference.crop_calls == [(str(clip), (12.5, 20.0))]
    assert inference.whole_calls == []


def test_a_span_shorter_than_the_crop_floor_is_refused_before_the_model_runs(sandbox, monkeypatch):
    cfg, _ = sandbox
    install_fake_pyannote(monkeypatch)
    install_fake_numpy(monkeypatch)
    install_fake_torch(monkeypatch, cuda=False)
    online_with_token(monkeypatch)

    with pytest.raises(VoiceprintError, match="too short to embed"):
        Embedder(cfg).embed(cfg.path("inbox") / "call.wav", 10.0, 10.0 + MIN_CROP_SECONDS / 2)
    assert Embedder._CACHE[EMBEDDING_MODEL].crop_calls == []


def test_a_many_row_embedding_is_averaged_over_its_windows(sandbox, monkeypatch):
    """A sliding-window model returns one row per window; the summary is their mean."""
    cfg, _ = sandbox
    install_fake_pyannote(monkeypatch, raw=[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    install_fake_numpy(monkeypatch)
    install_fake_torch(monkeypatch, cuda=False)
    online_with_token(monkeypatch)

    out = Embedder(cfg).embed(cfg.path("inbox") / "call.wav")

    # Column means are (2/3, 1/3); normalised that is (2, 1) / sqrt(5).
    assert rounded(out) == rounded([2 / 5 ** 0.5, 1 / 5 ** 0.5])


def test_a_model_that_returns_nothing_is_an_error_not_a_silent_vector(sandbox, monkeypatch):
    cfg, _ = sandbox
    install_fake_pyannote(monkeypatch, raw=[])
    install_fake_numpy(monkeypatch)
    install_fake_torch(monkeypatch, cuda=False)
    online_with_token(monkeypatch)

    with pytest.raises(VoiceprintError, match="returned nothing"):
        Embedder(cfg).embed(cfg.path("inbox") / "call.wav")


# ---------------------------------------------------------------------------
# identification edges
# ---------------------------------------------------------------------------
def test_best_is_the_top_score_or_nothing():
    assert ClusterMatch("A", 10.0, [("Marcus", 0.9), ("Dana", 0.2)]).best == ("Marcus", 0.9)
    assert ClusterMatch("A", 10.0, []).best is None


def test_zero_length_segments_contribute_no_spans():
    spans = _spans_by_cluster(segs((5.0, 5.0, "A"), (7.0, 6.0, "A"), (0.0, 3.0, "B")))
    assert spans == {"B": [(0.0, 3.0)]}


def test_nobody_enrolled_means_the_model_is_never_touched(sandbox, monkeypatch):
    cfg, _ = sandbox
    monkeypatch.setattr(Embedder, "require", lambda self: pytest.fail("the model was loaded"))
    assert identify(cfg.path("inbox") / "x.wav", segs((0.0, 30.0, "S")), cfg, store_for(cfg)) == []


def test_enough_speech_in_spans_all_too_short_to_embed_stays_unnamed(sandbox, monkeypatch):
    """Five 0.7s interjections total 3.5s, clearing min_speech but not the crop floor."""
    cfg, _ = sandbox
    store = store_for(cfg)
    store.enroll("Marcus", vec(1.0, 0.0, 0.0))
    monkeypatch.setattr(Embedder, "require", lambda self: None)
    monkeypatch.setattr(Embedder, "embed", lambda *a, **k: pytest.fail("embedded a too-short span"))

    spans = [(float(i * 2), float(i * 2) + 0.7, "S") for i in range(5)]
    matches = identify(cfg.path("inbox") / "x.wav", segs(*spans), cfg, store)

    assert len(matches) == 1
    assert matches[0].matched is None
    assert matches[0].scores == []
    assert matches[0].seconds == pytest.approx(3.5)
    assert matches[0].reason == "no span was long enough to embed"


class FlakyEmbedder:
    """Raises or returns per span, so a test can say which crops go wrong."""

    def __init__(self, by_span):
        self.by_span = by_span
        self.calls = []

    def embed(self, audio, start=None, end=None):
        self.calls.append((start, end))
        result = self.by_span[(start, end)]
        if isinstance(result, Exception):
            raise result
        return result


def test_a_span_the_model_cannot_embed_is_skipped_and_the_rest_still_count(sandbox, monkeypatch):
    cfg, _ = sandbox
    store = store_for(cfg)
    store.enroll("Marcus", vec(1.0, 0.0, 0.0))
    flaky = FlakyEmbedder({
        (0.0, 20.0): VoiceprintError("silence"),
        (30.0, 40.0): RuntimeError("decoder choked"),
        (50.0, 58.0): vec(0.99, 0.1, 0.0),
    })
    monkeypatch.setattr(Embedder, "embed", lambda self, a, s=None, e=None: flaky.embed(a, s, e))
    monkeypatch.setattr(Embedder, "require", lambda self: None)

    matches = identify(
        cfg.path("inbox") / "x.wav",
        segs((0.0, 20.0, "S"), (30.0, 40.0, "S"), (50.0, 58.0, "S")),
        cfg, store,
    )

    assert sorted(flaky.calls) == [(0.0, 20.0), (30.0, 40.0), (50.0, 58.0)]
    assert matches[0].matched == "Marcus"


def test_when_every_span_fails_to_embed_the_cluster_stays_unnamed(sandbox, monkeypatch):
    cfg, _ = sandbox
    store = store_for(cfg)
    store.enroll("Marcus", vec(1.0, 0.0, 0.0))
    flaky = FlakyEmbedder({
        (0.0, 20.0): VoiceprintError("silence"),
        (30.0, 40.0): RuntimeError("decoder choked"),
    })
    monkeypatch.setattr(Embedder, "embed", lambda self, a, s=None, e=None: flaky.embed(a, s, e))
    monkeypatch.setattr(Embedder, "require", lambda self: None)

    matches = identify(cfg.path("inbox") / "x.wav", segs((0.0, 20.0, "S"), (30.0, 40.0, "S")),
                       cfg, store)

    assert matches[0].matched is None
    assert matches[0].scores == [], "a score table was produced from no embedding at all"
    assert matches[0].reason


def test_a_cluster_with_no_score_table_is_told_nobody_was_enrolled():
    match = ClusterMatch("A", 30.0, [])
    _decide([match], threshold=0.55, margin=0.08)
    assert match.matched is None
    assert match.reason == "nobody is enrolled to compare against"


# ---------------------------------------------------------------------------
# the pipeline entry point, all the way through
# ---------------------------------------------------------------------------
def test_named_speakers_maps_only_the_confident_clusters(sandbox, monkeypatch):
    cfg, _ = sandbox
    store = store_for(cfg)
    store.enroll("Marcus", vec(1.0, 0.0, 0.0))
    store.enroll("Dana", vec(0.0, 1.0, 0.0))
    store.save()

    by_span = {
        (0.0, 30.0): vec(0.99, 0.14, 0.0),   # Marcus, comfortably
        (30.0, 60.0): vec(0.0, 0.0, 1.0),    # nobody enrolled sounds like this
    }
    monkeypatch.setattr(Embedder, "available", staticmethod(lambda c: (True, "ready")))
    monkeypatch.setattr(Embedder, "embed", lambda self, a, s=None, e=None: by_span[(s, e)])

    named = named_speakers(
        cfg.path("inbox") / "x.wav",
        segs((0.0, 30.0, "SPEAKER_00"), (30.0, 60.0, "SPEAKER_01")),
        cfg,
    )

    assert named == {"SPEAKER_00": "Marcus"}
    assert "SPEAKER_01" not in named, "an unrecognised voice was given a name"
