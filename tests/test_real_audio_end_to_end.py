"""
A real audio file, all the way through, with only the recogniser stubbed.

This is the closest this suite can get to the question that matters to
somebody holding an mp3: *if I drop this in, does the whole thing work?* Every
stage runs for real -- ingest picks up a genuine mp3, ffmpeg decodes and
normalises it, the chunker cuts real WAV, the router and extractor run, the
compliance gate decides, the vault encrypts, and a digest comes out the other
end. The one substitution is the speech recogniser, which needs model weights
this suite will not download.

The substitution is made honest by making the stub *verify what it is handed*:
it opens each chunk with Python's own `wave` module and asserts it is real,
non-empty, correctly-formatted PCM at the configured rate before it returns
anything. So "the recogniser was stubbed" does not quietly mean "the audio
path was skipped" -- if normalisation produced a broken or empty WAV, this
test fails inside the stub rather than passing on faith.

It skips when ffmpeg is absent, and says so.
"""

from __future__ import annotations

import shutil
import wave
from pathlib import Path

import pytest

from _fixtures import StubLLM, build_sandbox
from plaud_bridge.asr import registry
from plaud_bridge.asr.base import ASRProvider, ASRResult
from plaud_bridge.db import Database
from plaud_bridge.models import Segment
from plaud_bridge.pipeline import Pipeline
from test_real_audio import make_audio

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe are not installed, so the real-audio path is unverified here",
)

# What the fake recogniser "hears": a coaching session, which routes to a
# profile that does not require consent. That matters here -- see
# `test_real_audio_cannot_confirm_consent_without_diarization` for why a
# consent-requiring profile could not reach the end of this path at all.
HEARD = [
    ("Sasson", "Let's run a role play on objection handling for your discovery script."),
    ("Priya", "Ready. Give me the hardest objection you get."),
    ("Sasson", "The premium is too high, I need to think about it."),
    ("Priya", "I would ask what specifically feels high before talking about price."),
    ("Sasson", "Good. That is the rapport framework rather than a script."),
    ("Priya", "My pipeline activity was ninety dials and four appointments set."),
    ("Sasson", "The dials are fine. Closing is where I would spend next month."),
    ("Priya", "Can you review a recording of my next discovery call?"),
    ("Sasson", "Send it Monday and I will mark it up line by line."),
]

# A consented client call, used only by the consent test below.
HEARD_CLIENT = [
    ("Sasson", "Morning Dana, before we start I record these calls for my notes. Is that okay?"),
    ("Dana", "Yes, that's fine with me."),
    ("Sasson", "Tell me about the term policy and your disability coverage."),
    ("Dana", "Just a small term policy through work. The elimination period question worries me."),
]


class VerifyingStubASR(ASRProvider):
    """
    A recogniser that checks its input is real audio before inventing words.

    This is what keeps the stub from hiding an audio bug: it is handed the
    actual normalised chunk the pipeline produced, and it refuses anything
    that is not decodable 16-bit mono PCM at the configured rate.
    """

    name = "verifying_stub"
    is_cloud = False        # so the local-only compliance path is exercised

    seen: list[Path] = []
    script: list[tuple[str, str]] = HEARD

    def available(self):
        return True, "ready"

    def transcribe_file(self, path: Path, offset: float = 0.0,
                        language: str | None = None) -> ASRResult:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            rate = handle.getframerate()
            width = handle.getsampwidth()
            frames = handle.getnframes()
        assert channels == 1, f"the recogniser was handed {channels}-channel audio"
        assert width == 2, "the recogniser was handed audio that is not 16-bit PCM"
        assert rate == int(self.cfg.get("audio.sample_rate", 16000))
        assert frames > 0, "the recogniser was handed a WAV with no audio in it"
        VerifyingStubASR.seen.append(path)

        segments = []
        for i, (speaker, text) in enumerate(VerifyingStubASR.script):
            start = offset + i * 3.0
            segments.append(Segment(start=start, end=start + 2.8,
                                    speaker=speaker, text=text, confidence=0.95))
        return ASRResult(segments=segments, language="en",
                         provider=self.name, model="stub-v1")


class CoachingRouterStub(StubLLM):
    """Routes the coaching script to the profile that does not require consent."""

    def __call__(self, cfg, system, user, local_only=False, max_tokens=None):
        if '"scores"' in user and "role play" in user.lower():
            self.calls.append({"local_only": local_only, "system": system[:80]})
            return {"scores": [
                {"profile_id": "sales_trainer", "score": 0.95,
                 "evidence": ["role play coaching session"]},
            ]}, self._response()
        return super().__call__(cfg, system, user, local_only, max_tokens)


@pytest.fixture
def audio_sandbox(tmp_path, monkeypatch):
    cfg, _stub = build_sandbox(tmp_path, monkeypatch, stub=CoachingRouterStub())
    VerifyingStubASR.seen = []
    VerifyingStubASR.script = HEARD
    monkeypatch.setitem(registry._PROVIDERS, "verifying_stub", VerifyingStubASR)
    cfg._d.setdefault("asr", {})["providers"] = ["verifying_stub"]
    return cfg


def test_a_real_mp3_goes_all_the_way_through_the_pipeline(audio_sandbox, tmp_path):
    cfg = audio_sandbox
    make_audio(cfg.path("inbox") / "coaching-call.mp3", 12.0)

    pipe = Pipeline(cfg)
    try:
        stats = pipe.run()
    finally:
        pipe.close()

    assert stats.failed == 0, "a real mp3 failed to process"
    assert stats.processed == 1, f"expected one recording, got {stats.processed}"
    assert VerifyingStubASR.seen, "the recogniser was never handed any audio"

    db = Database(cfg.path("database"))
    try:
        rows = db.query(limit=10)
    finally:
        db.close()
    assert len(rows) == 1
    row = rows[0]
    assert row["source_name"] == "coaching-call.mp3"
    assert row["stage"] == "complete"
    assert row["governing_profile"] == "sales_trainer"
    # The duration came from ffprobe on real audio, not from a guess.
    assert 10.0 < float(row["duration_seconds"]) < 14.0


def test_real_audio_cannot_confirm_consent_without_diarization(tmp_path, monkeypatch):
    """
    The most important thing this file has to say, and it is a design truth
    rather than a bug.

    With diarization off -- the default, because it needs a HuggingFace token
    and an accepted model licence -- every segment of a recording is labelled
    `SPEAKER`. The consent detector therefore cannot tell who announced the
    recording or who agreed, so it refuses to certify consent and the gate
    holds the recording. A consented client call still lands in quarantine.

    That is the correct behaviour: the alternative is a tool that certifies
    consent it cannot actually see. But it means anyone processing real audio
    into a consent-requiring profile will meet the Held tab on day one, which
    is why it is pinned here and called out in the README rather than left to
    be discovered.
    """
    cfg, _stub = build_sandbox(tmp_path, monkeypatch)
    VerifyingStubASR.seen = []
    VerifyingStubASR.script = HEARD_CLIENT
    monkeypatch.setitem(registry._PROVIDERS, "verifying_stub", VerifyingStubASR)
    cfg._d.setdefault("asr", {})["providers"] = ["verifying_stub"]
    assert cfg.get("diarization.enabled") is False

    make_audio(cfg.path("inbox") / "client-call.mp3", 10.0)
    pipe = Pipeline(cfg)
    try:
        stats = pipe.run()
    finally:
        pipe.close()

    assert stats.quarantined == 1 and stats.processed == 0
    why = next((cfg.path("quarantine")).glob("*/WHY.md"))
    reasons = why.read_text(encoding="utf-8")
    assert "only one speaker detected" in reasons
    assert "release" in reasons, "the held recording must say how to release it"


def test_the_words_from_real_audio_come_back_out_of_the_vault(audio_sandbox):
    """The round trip a person actually cares about: put audio in, read it back."""
    from plaud_bridge.archive import Archive

    cfg = audio_sandbox
    make_audio(cfg.path("inbox") / "coaching-call.mp3", 12.0)
    pipe = Pipeline(cfg)
    try:
        pipe.run()
    finally:
        pipe.close()

    db = Database(cfg.path("database"))
    try:
        archive = Archive(cfg, db)
        row = db.query(limit=1)[0]
        segments = archive.segments(row)
    finally:
        db.close()

    assert segments, "no transcript came back for a real recording"
    said = " ".join(str(s.get("text", "")) for s in segments)
    assert "objection handling" in said or "appointments set" in said


def test_the_original_audio_is_recoverable_and_still_plays(audio_sandbox, tmp_path):
    """
    Recovery is the promise that makes encryption acceptable. What comes back
    has to be the original bytes -- and still be decodable audio, which is the
    part a byte comparison alone would not tell a person.
    """
    import subprocess

    from plaud_bridge.media import locate_original, stream_plaintext

    cfg = audio_sandbox
    source = make_audio(cfg.path("inbox") / "coaching-call.mp3", 8.0)
    original_bytes = source.read_bytes()

    pipe = Pipeline(cfg)
    try:
        pipe.run()
    finally:
        pipe.close()

    db = Database(cfg.path("database"))
    try:
        rid = db.query(limit=1)[0]["id"]
        info = locate_original(cfg, db, rid)
        assert info is not None, "the original audio was not kept"
        recovered = b"".join(stream_plaintext(cfg, info))
    finally:
        db.close()

    assert recovered == original_bytes, "the recovered audio is not the original"

    # And it is still audio: ffprobe reads a duration out of the recovered bytes.
    out = tmp_path / "recovered.mp3"
    out.write_bytes(recovered)
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(out)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, "the recovered file will not open as audio"
    assert 7.0 < float(proc.stdout.strip()) < 9.5


def test_a_long_real_recording_is_chunked_and_stitched_back_into_one_transcript(
        audio_sandbox):
    """
    Multi-chunk is where a real recording differs most from a text fixture:
    the recogniser is called several times and the pieces must come back as
    one transcript, in order, with no chunk silently dropped.
    """
    cfg = audio_sandbox
    cfg._d.setdefault("audio", {})["chunk_seconds"] = 30
    cfg._d["audio"]["chunk_overlap_seconds"] = 3
    make_audio(cfg.path("inbox") / "long-coaching-call.mp3", 75.0)

    pipe = Pipeline(cfg)
    try:
        stats = pipe.run()
    finally:
        pipe.close()

    assert stats.processed == 1 and stats.failed == 0
    assert len(VerifyingStubASR.seen) > 1, (
        "a 75s recording should have been cut into more than one chunk")

    from plaud_bridge.archive import Archive

    db = Database(cfg.path("database"))
    try:
        archive = Archive(cfg, db)
        segments = archive.segments(db.query(limit=1)[0])
    finally:
        db.close()

    assert segments
    starts = [float(s.get("start", 0)) for s in segments]
    assert starts == sorted(starts), "the stitched transcript is out of order"
