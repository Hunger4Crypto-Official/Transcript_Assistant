"""
Audio preparation edges: ffmpeg's absence and failures become actionable
errors, and the chunk arithmetic never produces a sliver or a negative stride.

ffmpeg is installed where these run, so its absence, a hang, and a failed
decode are simulated at the subprocess boundary and the message a person
would read is pinned. The stitcher's own guards -- nothing to stitch, a
segment that normalises to nothing, an exact same-speaker repeat -- and the
ASR provider's glossary prompt are pinned alongside as the next stage of the
same audio path.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from plaud_bridge.asr.base import ASRProvider, ASRResult
from plaud_bridge.asr.stitch import _similar, stitch
from plaud_bridge.audio.prepare import AudioChunk, AudioError, AudioPreparer, _run
from plaud_bridge.config import Glossary
from plaud_bridge.models import Segment


class _Cfg:
    """Dotted-key settings for the preparer, without a whole sandbox."""

    def __init__(self, **values):
        self.values = values

    def get(self, key, default=None):
        return self.values.get(key, default)


class _Proc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def _probe(seconds) -> _Proc:
    return _Proc(stdout=f'{{"format": {{"duration": "{seconds}"}}}}')


# =========================================================================
# The subprocess boundary
# =========================================================================
def test_a_missing_binary_is_an_install_hint_not_a_traceback(monkeypatch):
    def gone(*_a, **_k):
        raise FileNotFoundError(2, "No such file", "ffprobe-nope")

    monkeypatch.setattr(subprocess, "run", gone)
    with pytest.raises(AudioError, match="ffprobe-nope not found on PATH. Install ffmpeg."):
        _run(["ffprobe-nope", "-v", "error"])


def test_a_hung_binary_is_reported_with_its_timeout(monkeypatch):
    def hangs(cmd, *_a, **kw):
        raise subprocess.TimeoutExpired(cmd, kw.get("timeout"))

    monkeypatch.setattr(subprocess, "run", hangs)
    with pytest.raises(AudioError, match="ffmpeg timed out after 7s"):
        _run(["ffmpeg", "-i", "x"], timeout=7)


def test_check_tools_names_the_missing_tool_and_how_to_install_it(monkeypatch):
    prep = AudioPreparer(_Cfg(**{"audio.ffmpeg_binary": "ffmpeg", "audio.ffprobe_binary": "ffprobe-nope"}))
    monkeypatch.setattr("shutil.which", lambda tool: None if tool == "ffprobe-nope" else "/usr/bin/ffmpeg")
    with pytest.raises(AudioError) as excinfo:
        prep.check_tools()
    message = str(excinfo.value)
    assert message.startswith("'ffprobe-nope' is not on PATH.")
    assert "brew install ffmpeg" in message and "apt install ffmpeg" in message


def test_chunk_size_is_zero_for_a_chunk_whose_file_is_gone(tmp_path):
    present = tmp_path / "a.wav"
    present.write_bytes(b"\0" * (512 * 1024))
    assert AudioChunk(0, present, 0.0, 10.0, 0.0).size_mb == pytest.approx(0.5)
    assert AudioChunk(1, tmp_path / "missing.wav", 10.0, 10.0, 0.0).size_mb == 0.0


# =========================================================================
# normalise: a container with no readable duration
# =========================================================================
def test_a_source_with_no_readable_duration_is_capped_and_refused_after_decoding(
        tmp_path, monkeypatch):
    """
    When the source container reports no duration the decode runs under the
    -t ceiling, and the post-decode probe is the only place the budget can be
    enforced. The message says why the number was not known up front.
    """
    prep = AudioPreparer(_Cfg(**{"audio.max_duration_seconds": 3600}))
    monkeypatch.setattr(AudioPreparer, "check_tools", lambda self: None)
    calls: list[list[str]] = []

    def fake_run(cmd, *_a, **_k):
        calls.append(cmd)
        if cmd[0] == "ffprobe":
            target = Path(cmd[-1])
            if target.suffix == ".wav":
                return _probe(3660)      # the decode ran to the -t ceiling
            return _Proc(returncode=1, stderr="no duration in container")
        Path(cmd[-1]).write_bytes(b"RIFF")
        return _Proc()

    monkeypatch.setattr("plaud_bridge.audio.prepare._run", fake_run)
    with pytest.raises(AudioError) as excinfo:
        prep.normalise(tmp_path / "odd.m4a", tmp_path / "work")

    message = str(excinfo.value)
    assert message.startswith("odd.m4a decodes to more than the audio.max_duration_seconds budget of 1.0 hours")
    assert "Its container reports no duration" in message
    decode = next(c for c in calls if c[0] == "ffmpeg")
    assert decode[decode.index("-t") + 1] == "3660.000", "the decode was not capped a minute past budget"


# =========================================================================
# chunk arithmetic
# =========================================================================
def test_an_overlap_at_least_the_window_is_refused_before_any_ffmpeg_call(tmp_path, monkeypatch):
    prep = AudioPreparer(_Cfg(**{"audio.chunk_seconds": 30, "audio.chunk_overlap_seconds": 30}))
    monkeypatch.setattr(AudioPreparer, "check_tools", lambda self: None)
    calls = []
    monkeypatch.setattr("plaud_bridge.audio.prepare._run", lambda cmd, *a, **k: calls.append(cmd))
    with pytest.raises(AudioError, match="chunk_overlap_seconds must be smaller than the chunk window"):
        prep.chunk(tmp_path / "long.wav", tmp_path / "work", duration=100.0)
    assert not calls


def test_a_trailing_sliver_under_half_a_second_is_not_cut(tmp_path, monkeypatch):
    """
    60.3s in 30s windows with no overlap would arithmetically be three chunks;
    the third would hold 0.3s of audio, which no recogniser can use, so it is
    not cut and the two real chunks cover 60s.
    """
    prep = AudioPreparer(_Cfg(**{"audio.chunk_seconds": 30, "audio.chunk_overlap_seconds": 0}))
    monkeypatch.setattr(AudioPreparer, "check_tools", lambda self: None)

    def fake_run(cmd, *_a, **_k):
        Path(cmd[-1]).write_bytes(b"RIFF")
        return _Proc()

    monkeypatch.setattr("plaud_bridge.audio.prepare._run", fake_run)
    src = tmp_path / "long.wav"
    src.write_bytes(b"RIFF")
    chunks = prep.chunk(src, tmp_path / "work", duration=60.3)

    assert [(c.index, c.start, c.duration, c.overlap_lead) for c in chunks] == [
        (0, 0.0, 30.0, 0.0), (1, 30.0, 30.0, 0.0),
    ]
    assert all(c.path.exists() for c in chunks)
    assert not (tmp_path / "work" / "long.chunks" / "long.002.wav").exists()


def test_a_failed_chunk_names_its_index_and_ffmpeg_s_complaint(tmp_path, monkeypatch):
    prep = AudioPreparer(_Cfg(**{"audio.chunk_seconds": 30, "audio.chunk_overlap_seconds": 5}))
    monkeypatch.setattr(AudioPreparer, "check_tools", lambda self: None)

    def fake_run(cmd, *_a, **_k):
        if "-ss" in cmd and cmd[cmd.index("-ss") + 1] == "25.000":
            return _Proc(returncode=1, stderr="Invalid data found when processing input")
        Path(cmd[-1]).write_bytes(b"RIFF")
        return _Proc()

    monkeypatch.setattr("plaud_bridge.audio.prepare._run", fake_run)
    src = tmp_path / "long.wav"
    src.write_bytes(b"RIFF")
    with pytest.raises(AudioError, match="ffmpeg chunk 1 failed: Invalid data found"):
        prep.chunk(src, tmp_path / "work", duration=100.0)


# =========================================================================
# Stitching
# =========================================================================
def test_nothing_to_stitch_is_an_empty_timeline():
    assert stitch([], []) == []


def test_a_segment_that_normalises_to_nothing_resembles_nothing():
    assert _similar("...", "...") == 0.0
    assert _similar("", "hello") == 0.0
    assert _similar("hello!", "Hello") == 1.0


def test_a_blank_segment_that_survives_the_overlap_pass_is_dropped_at_the_end():
    """A single chunk skips the overlap pass entirely, so the final sweep is what removes it."""
    out = stitch([[Segment(0.0, 1.0, "   ", "A"), Segment(1.0, 2.0, "real words", "A")]], [0.0])
    assert [s.text for s in out] == ["real words"]


def test_an_exact_same_speaker_repeat_within_a_second_and_a_half_is_merged():
    """Two copies of one segment collapse into one span; a different speaker's copy stays."""
    segments = [
        Segment(10.0, 12.0, "Right, that makes sense.", "Sasson"),
        Segment(11.0, 13.0, "right that makes sense", "Sasson"),   # exact repeat, same speaker
        Segment(12.5, 14.0, "Right, that makes sense.", "Marcus"),  # a real exchange
        Segment(20.0, 22.0, "Right, that makes sense.", "Sasson"),  # too far apart to be a dup
    ]
    out = stitch([segments], [0.0], [0.0])
    assert [(s.speaker, s.start, s.end) for s in out] == [
        ("Sasson", 10.0, 13.0), ("Marcus", 12.5, 14.0), ("Sasson", 20.0, 22.0),
    ]


# =========================================================================
# The ASR contract
# =========================================================================
class _Backend(ASRProvider):
    name = "test"
    is_cloud = False

    def available(self):
        return True, "ready"

    def transcribe_file(self, path, offset=0.0, language=None):
        return ASRResult(provider=self.name)


def test_the_asr_prompt_is_the_glossary_bias_terms_or_nothing():
    glossary = Glossary(asr_bias_terms=["IUL", "elimination period"], proper_nouns=["Marcus"])
    assert _Backend(_Cfg(), glossary).prompt() == "IUL, elimination period, Marcus"
    assert _Backend(_Cfg()).prompt() == ""
