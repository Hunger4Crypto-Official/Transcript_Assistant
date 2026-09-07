"""
The audio path, against audio that actually exists.

Every other test in this suite feeds the pipeline TEXT transcripts, because
that is what runs anywhere with no ffmpeg, no model weights and no network.
That is the right default, and it leaves one honest gap: the code that decodes,
normalises, chunks and probes real media had never met a real media file. A
loudness filter that silently produces an empty WAV, a chunk window that drops
the last two seconds, a duration probe that returns a string -- none of those
are visible to a test that never decodes anything.

So these generate real audio with ffmpeg (a tone, no fixture binary in the
repository) and run the real `AudioPreparer` over it. They skip when ffmpeg is
not installed, which is honest rather than convenient: the skip line says the
audio path went unverified on this machine, instead of a green tick that
means nothing.

What is deliberately NOT here: transcription. That needs model weights this
suite will not download. The seam is exactly where it should be -- everything
up to the recogniser's door is proven here, and the recogniser has its own
tests with a stub behind it.
"""

from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

import pytest

from _fixtures import build_sandbox
from plaud_bridge.audio.prepare import AudioError, AudioPreparer, probe_duration

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe are not installed, so the real-audio path is unverified here",
)


def make_audio(path: Path, seconds: float, *, freq: int = 220,
               rate: int = 44100, channels: int = 2) -> Path:
    """
    Write a real, decodable audio file with ffmpeg.

    Stereo at 44.1kHz on purpose: the preparer's job is to convert down to
    mono at the ASR's sample rate, and a fixture already in the target format
    would prove nothing about the conversion.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", f"sine=frequency={freq}:duration={seconds}",
         "-ac", str(channels), "-ar", str(rate), str(path)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"could not build the test audio: {proc.stderr[:300]}"
    assert path.exists() and path.stat().st_size > 0
    return path


@pytest.fixture
def preparer(tmp_path, monkeypatch):
    cfg, _stub = build_sandbox(tmp_path, monkeypatch)
    return AudioPreparer(cfg), cfg


# =========================================================================
# Normalisation: the conversion the recogniser depends on
# =========================================================================
def test_a_real_mp3_normalises_to_mono_pcm_at_the_configured_rate(preparer, tmp_path):
    prep, _cfg = preparer
    src = make_audio(tmp_path / "in" / "call.mp3", 6.0)

    dest, duration = prep.normalise(src, tmp_path / "work")

    assert dest.exists() and dest.suffix == ".wav"
    assert 5.0 < duration < 7.5, f"duration came back as {duration}"

    # The actual PCM, read by a decoder that is not ffmpeg: if the filter chain
    # produced something malformed, `wave` is where it shows.
    with wave.open(str(dest), "rb") as handle:
        assert handle.getnchannels() == prep.channels == 1
        assert handle.getframerate() == prep.sample_rate
        assert handle.getsampwidth() == 2, "expected 16-bit PCM"
        frames = handle.getnframes()
    assert frames > 0, "normalisation produced a WAV with no audio in it"
    # Frames and the probed duration have to agree, or something is lying.
    assert abs(frames / prep.sample_rate - duration) < 0.5


@pytest.mark.parametrize("suffix", [".mp3", ".m4a", ".wav", ".flac", ".ogg"])
def test_every_container_the_inbox_accepts_really_decodes(preparer, tmp_path, suffix):
    """
    The accepted-extension list is a promise. This is the only test that
    checks the promise against a decoder rather than against itself.
    """
    prep, _cfg = preparer
    src = make_audio(tmp_path / "in" / f"clip{suffix}", 3.0)
    dest, duration = prep.normalise(src, tmp_path / f"work{suffix.replace('.', '')}")
    assert dest.exists() and duration > 1.0


def test_a_recording_over_the_duration_budget_is_refused_before_decoding(preparer,
                                                                        tmp_path):
    """
    The budget exists so a corrupt or hostile file cannot fill the disk with
    scratch WAV. Refusing must happen on the probe, not after the write.
    """
    prep, _cfg = preparer
    prep.max_duration = 2.0
    src = make_audio(tmp_path / "in" / "long.mp3", 6.0)

    work = tmp_path / "work"
    with pytest.raises(AudioError) as exc:
        prep.normalise(src, work)
    assert "budget" in str(exc.value)
    # Nothing decoded: the point of refusing early.
    assert not list(work.glob("*.norm.wav")), "it decoded before refusing"


def test_a_file_that_is_not_audio_fails_loudly(preparer, tmp_path):
    prep, _cfg = preparer
    fake = tmp_path / "in" / "not-audio.mp3"
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_bytes(b"this is not an mp3, it is a sentence")

    with pytest.raises(AudioError):
        prep.normalise(fake, tmp_path / "work")


# =========================================================================
# Chunking: the seam where audio is most likely to be quietly lost
# =========================================================================
def test_a_short_recording_stays_one_chunk(preparer, tmp_path):
    prep, _cfg = preparer
    src = make_audio(tmp_path / "in" / "short.wav", 4.0, rate=16000, channels=1)

    chunks = prep.chunk(src, tmp_path / "work")
    assert len(chunks) == 1
    assert chunks[0].index == 0 and chunks[0].start == 0.0
    assert chunks[0].path.exists()


def test_a_long_recording_chunks_with_overlap_and_loses_no_audio(preparer, tmp_path):
    """
    The property that matters: every second of the source appears in some
    chunk. A stride bug drops the tail, which in production looks like a
    conversation whose last minute was never transcribed -- and nothing in the
    output says so.
    """
    prep, _cfg = preparer
    prep.chunk_seconds = 40.0
    prep.overlap = 5.0
    seconds = 100.0
    src = make_audio(tmp_path / "in" / "long.wav", seconds, rate=16000, channels=1)

    chunks = prep.chunk(src, tmp_path / "work", duration=seconds)
    assert len(chunks) > 1, "a 100s recording should not be one 40s chunk"

    for chunk in chunks:
        assert chunk.path.exists() and chunk.path.stat().st_size > 0
        real = probe_duration(chunk.path, prep.ffprobe)
        assert abs(real - chunk.duration) < 0.75, (
            f"chunk {chunk.index} claims {chunk.duration:.1f}s but holds {real:.1f}s")

    # Coverage: walk the chunks in order and confirm they tile the recording.
    covered_to = 0.0
    for chunk in sorted(chunks, key=lambda c: c.start):
        assert chunk.start <= covered_to + 0.01, (
            f"a gap: nothing covers {covered_to:.1f}s-{chunk.start:.1f}s")
        covered_to = max(covered_to, chunk.start + chunk.duration)
    assert covered_to >= seconds - 1.0, (
        f"the last {seconds - covered_to:.1f}s of the recording is in no chunk")

    # And the overlap is real, so a word split across a boundary is heard whole.
    assert chunks[1].start < chunks[0].start + chunks[0].duration


def test_the_probe_reads_a_real_duration_as_a_number(preparer, tmp_path):
    prep, _cfg = preparer
    src = make_audio(tmp_path / "in" / "probe.mp3", 5.0)
    duration = probe_duration(src, prep.ffprobe)
    assert isinstance(duration, float) and 4.5 < duration < 5.6
