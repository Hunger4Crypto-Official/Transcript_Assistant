"""
Transcript import, chunk stitching, and log redaction.

These are the quiet failures: nothing raises, the run reports success, and the
transcript is simply missing words. Each test below corresponds to a specific
way content used to disappear between the file on disk and the stored artifact.
"""

import logging

import pytest

from plaud_bridge.asr.stitch import stitch
from plaud_bridge.audio.prepare import AudioError, probe_duration
from plaud_bridge.logging_setup import RedactingFilter
from plaud_bridge.models import Segment
from plaud_bridge.pipeline import _parse_text_transcript


def _text(segments):
    return " ".join(s.text for s in segments)


# =========================================================================
# SRT
# =========================================================================
def test_srt_keeps_numeric_caption_lines():
    """
    Only the first line of a cue is the index. Filtering every numeric line
    deletes policy numbers, dollar figures, and years from the transcript.
    """
    srt = (
        "1\n00:00:01,000 --> 00:00:04,000\nHe paid\n1035\ninto the exchange\n\n"
        "2\n00:00:05,000 --> 00:00:07,000\n2024\n"
    )
    segments = _parse_text_transcript(srt, ".srt")
    assert "1035" in _text(segments)
    assert any(s.text.strip() == "2024" for s in segments), (
        "a cue whose only content was a number produced no segment at all"
    )


def test_srt_real_timestamps_survive():
    srt = "1\n00:01:02,500 --> 00:01:05,000\nMarcus: I'll email you tonight.\n"
    segments = _parse_text_transcript(srt, ".srt")
    assert len(segments) == 1
    assert segments[0].start == pytest.approx(62.5)
    assert segments[0].end == pytest.approx(65.0)


# =========================================================================
# Speaker labels
# =========================================================================
@pytest.mark.parametrize("line,keeps", [
    ("Well: I told him no.", "Well"),
    ("Note: he agreed to the quote.", "Note"),
    ("See https://example.com/x for details", "example.com"),
])
def test_a_sentence_opener_is_not_a_speaker(line, keeps):
    segments = _parse_text_transcript(line + "\n", ".txt")
    assert keeps in _text(segments), f"'{keeps}' was eaten as a speaker label"
    assert segments[0].speaker == "SPEAKER"


def test_a_repeated_label_is_recognised_as_a_speaker():
    body = "Sasson: How was practice?\nKid: Good.\nSasson: Nice.\n"
    segments = _parse_text_transcript(body, ".txt")
    assert [s.speaker for s in segments] == ["Sasson", "Kid", "Sasson"]
    assert "How was practice?" in _text(segments)


# =========================================================================
# Synthetic timeline
# =========================================================================
def test_plain_text_timeline_never_goes_backwards():
    body = (
        "[00:00] Sasson: one two three four five six seven eight nine ten eleven twelve\n"
        "[00:02] Marcus: yes\n"
        "[00:01] Sasson: and another thing entirely\n"
    )
    segments = _parse_text_transcript(body, ".txt")
    for earlier, later in zip(segments, segments[1:], strict=False):
        assert later.start >= earlier.start
        assert later.start >= earlier.end - 1e-6, (
            "a segment starts before the previous one ends; the rendered "
            "transcript and the consent window are both out of order"
        )


def test_markdown_headings_do_not_empty_the_transcript():
    body = "# Meeting notes\n\nSasson: The elimination period matters.\nMarcus: Understood.\n"
    segments = _parse_text_transcript(body, ".md")
    assert segments, "a markdown transcript produced nothing"
    assert "elimination period" in _text(segments)


# =========================================================================
# Stitching
# =========================================================================
def test_stitch_uses_the_chunk_start_not_the_first_segment():
    """
    The chunk begins at 592s but VAD trimmed its silence, so the first segment
    lands at 597. The duplicate window is 592-600 either way; anchoring on the
    segment pushes it to 605 and deletes real speech after the boundary.
    """
    first = [Segment(580.0, 585.0, "right, that makes sense")]
    second = [
        Segment(597.0, 600.0, "so the elimination period applies"),
        Segment(601.0, 604.0, "so the elimination period is ninety days"),
        Segment(606.0, 608.0, "understood"),
    ]
    out = stitch([first, second], [0.0, 8.0], [0.0, 592.0])
    texts = [s.text for s in out]
    assert "so the elimination period is ninety days" in texts, (
        "speech past the overlap window was deleted as a duplicate"
    )


def test_stitch_does_not_collapse_non_latin_segments():
    segments = [
        Segment(10.0, 10.7, "你好，我是王先生"),
        Segment(10.9, 11.6, "很高兴认识你"),
        Segment(12.0, 13.0, "我们开始吧"),
    ]
    out = stitch([segments], [0.0], [0.0])
    assert len(out) == 3, "non-Latin segments normalised to empty and were deduped away"


def test_stitch_keeps_two_speakers_saying_the_same_word():
    segments = [
        Segment(10.0, 10.5, "Yeah.", speaker="Sasson"),
        Segment(11.0, 11.5, "Yeah.", speaker="Marcus"),
    ]
    out = stitch([segments], [0.0], [0.0])
    assert len(out) == 2, "two people agreeing became one segment"


def test_stitch_still_removes_a_real_overlap_duplicate():
    first = [Segment(0.0, 6.0, "the own occupation definition matters here")]
    second = [
        Segment(4.0, 6.0, "the own-occupation definition matters here."),
        Segment(9.0, 11.0, "next point"),
    ]
    out = stitch([first, second], [0.0, 4.0], [0.0, 4.0])
    assert len(out) == 2
    assert out[1].text == "next point"


# =========================================================================
# Logging
# =========================================================================
def _record(msg, exc_info=None):
    return logging.LogRecord(
        "plaud_bridge.test", logging.INFO, __file__, 1, msg, (), exc_info
    )


PATTERNS = {
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "email": r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b",
}


def test_redact_content_actually_redacts():
    record = _record("client said: my ssn is 123-45-6789, email marcus@example.com")
    RedactingFilter(redact_content=True, patterns=PATTERNS).filter(record)
    body = record.getMessage()
    assert "123-45-6789" not in body
    assert "marcus@example.com" not in body
    assert "[redacted-ssn]" in body


def test_tracebacks_are_redacted_too():
    """Provider errors fold the raw response body into the exception."""
    try:
        raise ValueError("rejected transcript 'my ssn is 123-45-6789' token gsk_LIVEKEY12345678")
    except ValueError:
        import sys

        record = _record("unexpected failure on %s", exc_info=sys.exc_info())

    RedactingFilter(redact_content=True, patterns=PATTERNS).filter(record)
    rendered = logging.Formatter("%(message)s").format(record)
    assert "123-45-6789" not in rendered
    assert "gsk_LIVEKEY12345678" not in rendered
    assert "[redacted-key]" in rendered


def test_api_keys_are_stripped_even_with_content_redaction_off():
    record = _record("Authorization: Bearer sk-abcdefghijklmnop")
    RedactingFilter(redact_content=False).filter(record)
    assert "abcdefghijklmnop" not in record.getMessage()


# =========================================================================
# ffprobe
# =========================================================================
def test_a_null_duration_is_an_actionable_error(monkeypatch, tmp_path):
    class _Proc:
        returncode = 0
        stdout = '{"format": {"duration": null}}'
        stderr = ""

    monkeypatch.setattr("plaud_bridge.audio.prepare._run", lambda *a, **k: _Proc())
    with pytest.raises(AudioError) as excinfo:
        probe_duration(tmp_path / "clip.m4a")
    assert "duration" in str(excinfo.value)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "-30", "0"])
def test_a_nonsense_duration_is_refused(monkeypatch, tmp_path, value):
    """
    float() will happily return NaN, infinity, or a negative from a crafted or
    corrupt container, and any of those turns the chunk arithmetic into a
    non-terminating loop or a nonsense chunk count. It is rejected at the probe.
    """
    class _Proc:
        returncode = 0
        stdout = '{"format": {"duration": "' + value + '"}}'
        stderr = ""

    monkeypatch.setattr("plaud_bridge.audio.prepare._run", lambda *a, **k: _Proc())
    with pytest.raises(AudioError):
        probe_duration(tmp_path / "clip.m4a")


def test_a_recording_over_the_duration_budget_is_refused_before_decoding(monkeypatch, tmp_path):
    """
    normalise decodes the whole input to uncompressed PCM before anything else
    looks at it, so an over-budget file must be refused before ffmpeg runs, not
    after gigabytes of scratch are on disk. When the source duration is readable
    and over budget, no decode command is issued at all.
    """
    from _fixtures import build_sandbox
    from plaud_bridge.audio.prepare import AudioPreparer

    cfg, _ = build_sandbox(tmp_path, monkeypatch,
                           overrides={"audio": {"max_duration_seconds": 3600}})
    prep = AudioPreparer(cfg)
    monkeypatch.setattr(AudioPreparer, "check_tools", lambda self: None)

    calls: list[list[str]] = []

    def fake_run(cmd, *a, **k):
        calls.append(cmd)
        class _Proc:
            returncode = 0
            # Anything that runs is an ffprobe; report a 3-hour file, over the
            # 1-hour budget. If a decode (ffmpeg) is ever issued, the -c:a arg
            # gives it away and the assertion below catches it.
            stdout = '{"format": {"duration": "10800"}}'
            stderr = ""
        return _Proc()

    monkeypatch.setattr("plaud_bridge.audio.prepare._run", fake_run)

    with pytest.raises(AudioError, match="budget"):
        prep.normalise(tmp_path / "marathon.m4a", tmp_path / "work")

    assert not any("pcm_s16le" in c for c in calls), (
        "an over-budget file was decoded instead of refused before the decode"
    )
