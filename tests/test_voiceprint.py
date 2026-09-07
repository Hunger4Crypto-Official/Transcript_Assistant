"""
Named speakers: the store, the maths, the guards, and the engine wiring.

The embedding model itself is never loaded here. Every test substitutes vectors
it chose deliberately, because the questions worth asking are "does a 0.9 match
become a name" and "does a 0.9 tie between two brothers stay silent", and both
are answered better by arithmetic than by shipping a 90MB model into CI.
"""

from __future__ import annotations

import json

import pytest

from _fixtures import build_sandbox
from plaud_bridge.diarize import engine
from plaud_bridge.diarize.voiceprint import (
    ClusterMatch,
    Embedder,
    VoiceprintError,
    VoiceprintStore,
    _decide,
    average,
    cosine,
    identify,
    named_speakers,
    normalise,
    slugify,
)
from plaud_bridge.models import Segment
from plaud_bridge.storage import Vault, VaultError


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def vec(*values: float) -> list[float]:
    """A unit vector pointing where the test wants it to point."""
    return normalise(list(values))


def store_for(cfg) -> VoiceprintStore:
    return VoiceprintStore(Vault(cfg.path("vault")))


def segs(*spans) -> list[Segment]:
    """(start, end, cluster) triples into segments."""
    return [Segment(start=s, end=e, text="...", speaker=c) for s, e, c in spans]


class FakeEmbedder:
    """
    Stands in for the model.

    Returns whatever vector the test mapped to the span's cluster, so a test can
    say "cluster A sounds like this" without any audio existing.
    """

    def __init__(self, by_span: dict[tuple[float, float], list[float]]):
        self.by_span = by_span
        self.calls: list[tuple[float, float]] = []

    def embed(self, audio, start=None, end=None):
        self.calls.append((start, end))
        try:
            return self.by_span[(start, end)]
        except KeyError as exc:  # pragma: no cover - a wiring mistake, not a code path
            raise AssertionError(f"no fake embedding for span {start}-{end}") from exc


# ---------------------------------------------------------------------------
# the maths
# ---------------------------------------------------------------------------
def test_slugify_makes_a_stable_id():
    assert slugify("Marcus O'Neill") == "marcus-o-neill"
    assert slugify("  MARCUS  ") == "marcus"
    assert slugify("Marcus") == slugify("marcus")


def test_normalise_rejects_silence():
    with pytest.raises(VoiceprintError, match="zero magnitude"):
        normalise([0.0, 0.0, 0.0])


def test_cosine_is_one_for_identical_and_zero_for_orthogonal():
    a = vec(1.0, 0.0, 0.0)
    assert cosine(a, a) == pytest.approx(1.0)
    assert cosine(a, vec(0.0, 1.0, 0.0)) == pytest.approx(0.0)


def test_cosine_refuses_mismatched_sizes_and_says_why():
    with pytest.raises(VoiceprintError, match="identify.model changed"):
        cosine(vec(1.0, 0.0), vec(1.0, 0.0, 0.0))


def test_average_returns_a_unit_vector_between_its_inputs():
    result = average([vec(1.0, 0.0), vec(0.0, 1.0)])
    assert sum(v * v for v in result) == pytest.approx(1.0)
    assert result[0] == pytest.approx(result[1])


# ---------------------------------------------------------------------------
# the store
# ---------------------------------------------------------------------------
def test_enrollment_round_trips_through_the_vault(sandbox):
    cfg, _ = sandbox
    store = store_for(cfg)
    store.enroll("Marcus", vec(1.0, 0.0, 0.0), source="marcus.wav", seconds=12.0)
    store.save()

    reloaded = store_for(cfg)
    people = reloaded.people()
    assert [p.name for p in people] == ["Marcus"]
    assert people[0].id == "marcus"
    assert people[0].seconds == pytest.approx(12.0)
    assert people[0].samples[0].source == "marcus.wav"


def test_the_store_on_disk_is_ciphertext(sandbox):
    cfg, _ = sandbox
    store = store_for(cfg)
    store.enroll("Marcus", vec(1.0, 0.0, 0.0))
    path = store.save()

    raw = path.read_bytes()
    assert b"Marcus" not in raw
    assert b"marcus" not in raw
    # And it is not merely obfuscated: a wrong passphrase must fail closed.
    with pytest.raises(VaultError):
        Vault(cfg.path("vault"), passphrase_env="NOT_THE_PASSPHRASE_ENV").decrypt_bytes(
            raw, b"voiceprints"
        )


def test_saving_without_a_passphrase_refuses_rather_than_writing_plaintext(sandbox, monkeypatch):
    cfg, _ = sandbox
    store = store_for(cfg)
    store.enroll("Marcus", vec(1.0, 0.0, 0.0))
    monkeypatch.delenv("PLAUD_BRIDGE_PASSPHRASE", raising=False)

    with pytest.raises(VoiceprintError, match="never written in plaintext"):
        store.save()
    assert not store.path.exists()


def test_a_second_clip_adds_a_sample_and_replace_starts_over(sandbox):
    cfg, _ = sandbox
    store = store_for(cfg)
    store.enroll("Marcus", vec(1.0, 0.0, 0.0), seconds=10.0)
    store.enroll("Marcus", vec(0.9, 0.1, 0.0), seconds=8.0)
    assert len(store.find("Marcus").samples) == 2
    assert store.find("Marcus").seconds == pytest.approx(18.0)

    store.enroll("Marcus", vec(0.0, 1.0, 0.0), seconds=5.0, replace=True)
    assert len(store.find("Marcus").samples) == 1
    assert store.find("Marcus").seconds == pytest.approx(5.0)


def test_re_enrolling_updates_the_spelling_without_creating_a_second_person(sandbox):
    cfg, _ = sandbox
    store = store_for(cfg)
    store.enroll("marcus", vec(1.0, 0.0, 0.0))
    store.enroll("Marcus", vec(0.9, 0.1, 0.0))
    # Same slug, so one person, and every future transcript uses the newer spelling.
    assert [p.name for p in store.people()] == ["Marcus"]
    assert len(store.find("marcus").samples) == 2

    # A genuinely different name is a genuinely different person, though.
    store.enroll("Marcus Reed", vec(0.0, 0.0, 1.0))
    assert {p.id for p in store.people()} == {"marcus", "marcus-reed"}


def test_forget_removes_a_person_permanently(sandbox):
    cfg, _ = sandbox
    store = store_for(cfg)
    store.enroll("Marcus", vec(1.0, 0.0, 0.0))
    store.enroll("Dana", vec(0.0, 1.0, 0.0))
    store.save()

    store = store_for(cfg)
    assert store.forget("marcus").name == "Marcus"
    store.save()

    reloaded = store_for(cfg)
    assert [p.name for p in reloaded.people()] == ["Dana"]
    assert reloaded.forget("nobody") is None


def test_a_store_from_a_newer_build_is_refused_not_overwritten(sandbox):
    cfg, _ = sandbox
    vault = Vault(cfg.path("vault"))
    vault.write("voiceprints", json.dumps({"version": 99, "people": []}), "voiceprints")

    with pytest.raises(VoiceprintError, match="version 99"):
        VoiceprintStore(vault).load()


def test_a_corrupt_store_says_so_rather_than_starting_empty(sandbox):
    cfg, _ = sandbox
    vault = Vault(cfg.path("vault"))
    vault.write("voiceprints", "this is not json", "voiceprints")

    with pytest.raises(VoiceprintError, match="unreadable"):
        VoiceprintStore(vault).load()


def test_an_unnamed_person_is_rejected(sandbox):
    cfg, _ = sandbox
    with pytest.raises(VoiceprintError, match="needs a name"):
        store_for(cfg).enroll("   ", vec(1.0, 0.0, 0.0))


# ---------------------------------------------------------------------------
# identification
# ---------------------------------------------------------------------------
def test_a_confident_match_gets_the_name(sandbox, monkeypatch):
    cfg, _ = sandbox
    store = store_for(cfg)
    store.enroll("Marcus", vec(1.0, 0.0, 0.0))
    store.enroll("Dana", vec(0.0, 1.0, 0.0))

    fake = FakeEmbedder({(0.0, 30.0): vec(0.99, 0.14, 0.0)})
    monkeypatch.setattr(Embedder, "embed", lambda self, a, s=None, e=None: fake.embed(a, s, e))
    monkeypatch.setattr(Embedder, "require", lambda self: None)

    matches = identify(cfg.path("inbox") / "x.wav", segs((0.0, 30.0, "SPEAKER_00")), cfg, store)
    assert [(m.cluster, m.matched) for m in matches] == [("SPEAKER_00", "Marcus")]


def test_a_stranger_stays_unnamed(sandbox, monkeypatch):
    cfg, _ = sandbox
    store = store_for(cfg)
    store.enroll("Marcus", vec(1.0, 0.0, 0.0))

    # Orthogonal to the only enrolled person: nobody in the room is Marcus.
    fake = FakeEmbedder({(0.0, 30.0): vec(0.0, 1.0, 0.0)})
    monkeypatch.setattr(Embedder, "embed", lambda self, a, s=None, e=None: fake.embed(a, s, e))
    monkeypatch.setattr(Embedder, "require", lambda self: None)

    matches = identify(cfg.path("inbox") / "x.wav", segs((0.0, 30.0, "SPEAKER_00")), cfg, store)
    assert matches[0].matched is None
    assert "below the" in matches[0].reason


def test_two_people_who_sound_alike_are_left_unnamed(sandbox, monkeypatch):
    """
    The margin guard. Relatives sound alike; a coin flip between them puts a
    wrong name on a transcript, which is worse than no name at all.
    """
    cfg, _ = sandbox
    store = store_for(cfg)
    store.enroll("Marcus", vec(1.0, 0.02, 0.0))
    store.enroll("Marcus's brother", vec(1.0, 0.0, 0.02))

    fake = FakeEmbedder({(0.0, 30.0): vec(1.0, 0.01, 0.01)})
    monkeypatch.setattr(Embedder, "embed", lambda self, a, s=None, e=None: fake.embed(a, s, e))
    monkeypatch.setattr(Embedder, "require", lambda self: None)

    matches = identify(cfg.path("inbox") / "x.wav", segs((0.0, 30.0, "SPEAKER_00")), cfg, store)
    assert matches[0].matched is None
    assert "margin" in matches[0].reason


def test_a_cluster_with_almost_no_speech_is_not_guessed_at(sandbox, monkeypatch):
    cfg, _ = sandbox
    store = store_for(cfg)
    store.enroll("Marcus", vec(1.0, 0.0, 0.0))
    monkeypatch.setattr(Embedder, "require", lambda self: None)

    matches = identify(cfg.path("inbox") / "x.wav", segs((0.0, 1.0, "SPEAKER_00")), cfg, store)
    assert matches[0].matched is None
    assert "1.0s of speech" in matches[0].reason


def test_one_person_cannot_be_two_speakers_in_the_same_room():
    """The higher score keeps the name; the other cluster is told why it lost."""
    strong = ClusterMatch("A", 60.0, [("Marcus", 0.91), ("Dana", 0.20)])
    weak = ClusterMatch("B", 40.0, [("Marcus", 0.72), ("Dana", 0.10)])
    _decide([strong, weak], threshold=0.55, margin=0.08)

    assert strong.matched == "Marcus"
    assert weak.matched is None
    assert "already matched cluster A" in weak.reason


def test_the_score_table_survives_even_when_nothing_matches(sandbox, monkeypatch):
    """`speakers identify` is how a person tunes the threshold, so show the misses."""
    cfg, _ = sandbox
    store = store_for(cfg)
    store.enroll("Marcus", vec(1.0, 0.0, 0.0))
    store.enroll("Dana", vec(0.0, 1.0, 0.0))

    fake = FakeEmbedder({(0.0, 30.0): vec(0.4, 0.4, 0.82)})
    monkeypatch.setattr(Embedder, "embed", lambda self, a, s=None, e=None: fake.embed(a, s, e))
    monkeypatch.setattr(Embedder, "require", lambda self: None)

    matches = identify(cfg.path("inbox") / "x.wav", segs((0.0, 30.0, "SPEAKER_00")), cfg, store)
    assert matches[0].matched is None
    assert {name for name, _ in matches[0].scores} == {"Marcus", "Dana"}


def test_the_longest_spans_are_the_ones_embedded(tmp_path, monkeypatch):
    cfg, _ = build_sandbox(tmp_path, monkeypatch,
                           overrides={"diarization": {"identify": {"max_crops": 2}}})
    store = store_for(cfg)
    store.enroll("Marcus", vec(1.0, 0.0, 0.0))

    fake = FakeEmbedder({
        (0.0, 20.0): vec(1.0, 0.0, 0.0),
        (30.0, 40.0): vec(1.0, 0.0, 0.0),
    })
    monkeypatch.setattr(Embedder, "embed", lambda self, a, s=None, e=None: fake.embed(a, s, e))
    monkeypatch.setattr(Embedder, "require", lambda self: None)

    identify(
        cfg.path("inbox") / "x.wav",
        segs((0.0, 20.0, "S"), (30.0, 40.0, "S"), (50.0, 50.5, "S")),
        cfg,
        store,
    )
    # The half-second span is below the crop floor and the crop cap is two, so
    # only the twenty and the ten second spans should have been embedded.
    assert sorted(fake.calls) == [(0.0, 20.0), (30.0, 40.0)]


# ---------------------------------------------------------------------------
# the pipeline entry point
# ---------------------------------------------------------------------------
def test_nobody_enrolled_means_no_work_and_no_names(sandbox):
    cfg, _ = sandbox
    assert named_speakers(cfg.path("inbox") / "x.wav", segs((0.0, 30.0, "S")), cfg) == {}


def test_an_empty_store_is_not_an_error(sandbox):
    cfg, _ = sandbox
    store = store_for(cfg)
    store.enroll("Marcus", vec(1.0, 0.0, 0.0))
    store.forget("Marcus")
    store.save()
    assert named_speakers(cfg.path("inbox") / "x.wav", segs((0.0, 30.0, "S")), cfg) == {}


def test_no_passphrase_skips_identification_instead_of_failing(sandbox, monkeypatch):
    cfg, _ = sandbox
    store = store_for(cfg)
    store.enroll("Marcus", vec(1.0, 0.0, 0.0))
    store.save()
    monkeypatch.delenv("PLAUD_BRIDGE_PASSPHRASE", raising=False)

    assert named_speakers(cfg.path("inbox") / "x.wav", segs((0.0, 30.0, "S")), cfg) == {}


def test_identification_can_be_turned_off(tmp_path, monkeypatch):
    cfg, _ = build_sandbox(tmp_path, monkeypatch,
                           overrides={"diarization": {"identify": {"enabled": False}}})
    store = store_for(cfg)
    store.enroll("Marcus", vec(1.0, 0.0, 0.0))
    store.save()

    assert named_speakers(cfg.path("inbox") / "x.wav", segs((0.0, 30.0, "S")), cfg) == {}


def test_a_missing_model_skips_identification_rather_than_failing_the_transcript(sandbox,
                                                                                monkeypatch):
    cfg, _ = sandbox
    store = store_for(cfg)
    store.enroll("Marcus", vec(1.0, 0.0, 0.0))
    store.save()
    monkeypatch.setattr(Embedder, "available", staticmethod(lambda c: (False, "no weights")))

    assert named_speakers(cfg.path("inbox") / "x.wav", segs((0.0, 30.0, "S")), cfg) == {}


# ---------------------------------------------------------------------------
# engine wiring
# ---------------------------------------------------------------------------
class FakeAnnotation:
    def __init__(self, turns):
        self._turns = turns

    def itertracks(self, yield_label=False):
        for start, end, label in self._turns:
            yield type("T", (), {"start": start, "end": end})(), None, label


def run_diarize(cfg, monkeypatch, turns, names):
    monkeypatch.setattr(engine, "_available", lambda c: (True, "ready"))
    monkeypatch.setattr(engine, "_load_pipeline", lambda c: (lambda path, **kw: FakeAnnotation(turns)))
    monkeypatch.setattr(engine, "named_speakers", lambda a, s, c: names)
    segments = [Segment(start=s, end=e, text="...") for s, e, _ in turns]
    return engine.diarize(cfg.path("inbox") / "x.wav", segments, cfg)


def test_a_recognised_cluster_gets_its_name_in_the_transcript(sandbox, monkeypatch):
    cfg, _ = sandbox
    turns = [(0.0, 60.0, "SPEAKER_00"), (60.0, 80.0, "SPEAKER_01")]
    out = run_diarize(cfg, monkeypatch, turns, {"SPEAKER_01": "Marcus"})

    # The dominant cluster is the wearer; the enrolled one keeps its name.
    assert out[0].speaker == cfg.get("diarization.owner_label")
    assert out[1].speaker == "Marcus"


def test_an_enrolled_wearer_is_not_renamed_by_the_owner_rule(sandbox, monkeypatch):
    """
    Enrolling yourself must win over the dominant-speaker heuristic, otherwise
    the owner label would overwrite the one label the model was sure about.
    """
    cfg, _ = sandbox
    turns = [(0.0, 60.0, "SPEAKER_00"), (60.0, 80.0, "SPEAKER_01")]
    out = run_diarize(cfg, monkeypatch, turns, {"SPEAKER_00": "Sasson"})

    assert out[0].speaker == "Sasson"
    assert out[1].speaker == "Speaker 1"


def test_unrecognised_clusters_do_not_collide_with_a_name(sandbox, monkeypatch):
    cfg, _ = sandbox
    turns = [(0.0, 60.0, "S0"), (60.0, 80.0, "S1"), (80.0, 90.0, "S2")]
    out = run_diarize(cfg, monkeypatch, turns, {"S1": "Marcus"})

    labels = [s.speaker for s in out]
    assert labels[1] == "Marcus"
    assert len(set(labels)) == 3


def test_identification_blowing_up_does_not_lose_the_transcript(sandbox, monkeypatch):
    cfg, _ = sandbox
    turns = [(0.0, 60.0, "S0"), (60.0, 80.0, "S1")]

    def explode(*_args, **_kwargs):
        raise RuntimeError("model file is truncated")

    monkeypatch.setattr(engine, "_available", lambda c: (True, "ready"))
    monkeypatch.setattr(engine, "_load_pipeline", lambda c: (lambda p, **kw: FakeAnnotation(turns)))
    monkeypatch.setattr(engine, "named_speakers", explode)

    segments = [Segment(start=s, end=e, text="...") for s, e, _ in turns]
    out = engine.diarize(cfg.path("inbox") / "x.wav", segments, cfg)

    assert [s.speaker for s in out] == [cfg.get("diarization.owner_label"), "Speaker 1"]
