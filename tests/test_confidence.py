"""
Whether the transcript is worth believing.

Speech recognition does not decline. Given music, a restaurant, or a device in a
pocket it returns fluent English that nobody said, and everything downstream
here treats the transcript as fact -- the router files it, the extractor pulls
promises out of it, memory carries those promises into next month's prompt, and
the worklist puts them in front of you as things you owe somebody.

These tests pin the one behaviour that stops an invented sentence becoming a
commitment you believe you made: the recogniser's own confidence is read, and a
transcript it was guessing at says so before anybody reads the summary.
"""

from __future__ import annotations

import pytest

from _fixtures import CLIENT_CALL, build_sandbox, drop
from plaud_bridge.asr.confidence import (
    OK,
    SUSPECT,
    UNKNOWN,
    UNRELIABLE,
    Assessment,
    assess,
    prompt_warning,
)
from plaud_bridge.models import Segment
from plaud_bridge.pipeline import Pipeline


def speech(count: int, confidence: float, seconds: float = 5.0,
           no_speech: float | None = 0.1, start: float = 0.0) -> list[Segment]:
    """A run of segments the recogniser scored however the test needs."""
    out = []
    at = start
    for index in range(count):
        out.append(Segment(start=at, end=at + seconds, text=f"line {index}",
                           confidence=confidence, no_speech=no_speech))
        at += seconds
    return out


# ---------------------------------------------------------------------------
# the judgement
# ---------------------------------------------------------------------------
def test_clean_speech_is_believed(sandbox):
    cfg, _ = sandbox
    result = assess(speech(10, confidence=-0.3), cfg)
    assert result.verdict == OK
    assert result.believable
    assert result.line() == ""
    assert prompt_warning(result) == ""


def test_a_transcript_the_model_was_guessing_at_says_so(sandbox):
    cfg, _ = sandbox
    result = assess(speech(10, confidence=-2.5), cfg)
    assert result.verdict == UNRELIABLE
    assert not result.believable
    assert "not trustworthy" in result.line()
    assert "invents fluent sentences" in result.line()


def test_a_partly_shaky_transcript_is_flagged_without_being_condemned(sandbox):
    cfg, _ = sandbox
    result = assess(speech(6, confidence=-0.3) + speech(4, confidence=-2.5, start=30.0), cfg)
    assert result.verdict == SUSPECT
    assert "shaky" in result.line()
    assert result.low_share == pytest.approx(0.4)


def test_confident_text_over_silence_is_the_signature_that_matters(sandbox):
    """
    The hallucination case: the model is sure of its words and equally sure
    there was no speech there. Scoring only the log probability misses it
    entirely, which is why no_speech is read too.
    """
    cfg, _ = sandbox
    result = assess(speech(10, confidence=-0.2, no_speech=0.95), cfg)
    assert result.verdict == UNRELIABLE
    assert result.silent_share == pytest.approx(1.0)


def test_a_long_invention_does_not_hide_behind_short_real_replies(sandbox):
    """
    Weighted by duration, not by count. Four minutes of invented music is one
    segment; twenty honest "mm-hm"s are twenty. Counting them equally would let
    the thing that matters lose the vote.
    """
    cfg, _ = sandbox
    segments = speech(20, confidence=-0.2, seconds=1.0)
    segments += speech(1, confidence=-3.0, seconds=240.0, start=20.0)
    result = assess(segments, cfg)
    assert result.verdict == UNRELIABLE
    assert result.low_share > 0.9


def test_too_little_speech_to_judge_says_that_rather_than_guessing(sandbox):
    cfg, _ = sandbox
    result = assess(speech(2, confidence=-2.9, seconds=3.0), cfg)
    assert result.verdict == UNKNOWN
    assert "too little to judge" in result.reason
    assert result.believable, "unknown is not an accusation"


def test_imported_text_is_unknown_rather_than_clean(sandbox):
    """
    A .txt import has no scores. Reporting it as good would be claiming a check
    that never happened.
    """
    cfg, _ = sandbox
    unscored = [Segment(start=0.0, end=60.0, text="pasted", confidence=None)]
    result = assess(unscored, cfg)
    assert result.verdict == UNKNOWN
    assert "no confidence scores" in result.reason


def test_an_empty_transcript_is_not_an_error(sandbox):
    cfg, _ = sandbox
    assert assess([], cfg).verdict == UNKNOWN


def test_the_thresholds_are_config_not_code(tmp_path, monkeypatch):
    cfg, _ = build_sandbox(
        tmp_path, monkeypatch,
        overrides={"asr": {"confidence": {"min_avg_logprob": -0.1,
                                          "unreliable_share": 0.5,
                                          "min_seconds": 1.0}}},
    )
    # Comfortably clean under the shipped floor, condemned under this one.
    assert assess(speech(10, confidence=-0.3), cfg).verdict == UNRELIABLE


def test_the_worst_passages_are_named_so_they_can_be_checked(sandbox):
    cfg, _ = sandbox
    segments = speech(8, confidence=-0.2)
    segments += [Segment(start=100.0, end=110.0, text="the invented sentence",
                         confidence=-4.0, no_speech=0.9)]
    result = assess(segments, cfg)
    assert any("the invented sentence" in w for w in result.worst)
    assert any("01:40" in w for w in result.worst)


def test_the_report_survives_a_round_trip(sandbox):
    cfg, _ = sandbox
    result = assess(speech(10, confidence=-2.5), cfg)
    again = Assessment.from_dict(result.to_dict())
    assert again.verdict == result.verdict
    assert again.low_share == pytest.approx(result.low_share)
    assert again.line() == result.line()


# ---------------------------------------------------------------------------
# what the model is told
# ---------------------------------------------------------------------------
def test_an_unreliable_transcript_tells_the_extractor_to_prefer_nothing(sandbox):
    cfg, _ = sandbox
    warning = prompt_warning(assess(speech(10, confidence=-2.5), cfg))
    assert "Prefer empty fields" in warning
    assert "Do not repair" in warning


def test_the_warning_is_an_instruction_not_a_footnote(sandbox):
    """
    Placed last, next to "return the JSON object now". A caveat that arrives
    before the transcript reads as background, gets noted, and is then extracted
    from confidently anyway.
    """
    cfg, stub = sandbox
    from plaud_bridge.models import Transcript
    from plaud_bridge.profiles import extractor

    seen: dict[str, str] = {}

    def capture(cfg_, system, user, local_only=False, max_tokens=None):
        seen["user"] = user
        return stub(cfg_, system, user, local_only, max_tokens)

    monkeypatch_target = extractor.complete_json
    extractor.complete_json = capture
    try:
        transcript = Transcript(segments=speech(10, confidence=-2.5))
        extractor.extract(transcript, cfg.profile("insurance_agent"), cfg,
                          warning="TRANSCRIPT RELIABILITY: made up.")
    finally:
        extractor.complete_json = monkeypatch_target

    body = seen["user"]
    assert "TRANSCRIPT RELIABILITY" in body
    # After the transcript block, not before it: a caveat ahead of the transcript
    # reads as background. The transcript is fenced, so "after" means after the
    # closing marker.
    assert body.index("TRANSCRIPT RELIABILITY") > body.index("<<<END TRANSCRIPT>>>")
    assert body.rstrip().endswith("Return the JSON object now.")


# ---------------------------------------------------------------------------
# through the pipeline and out the other side
# ---------------------------------------------------------------------------
def test_a_processed_recording_carries_its_verdict(sandbox, monkeypatch):
    cfg, _ = sandbox
    drop(cfg, "client-marcus.txt", CLIENT_CALL)

    # Imported text has no scores of its own, so score them on the way past to
    # exercise the path a real recording takes.
    from plaud_bridge import pipeline as pipeline_module

    real = pipeline_module.assess

    def scored(segments, cfg_):
        for seg in segments:
            seg.confidence, seg.no_speech = -3.0, 0.9
        return real(segments, cfg_)

    monkeypatch.setattr(pipeline_module, "assess", scored)

    pipe = Pipeline(cfg)
    try:
        assert pipe.run().processed == 1
        import json

        payload = json.loads(pipe.db.query()[0]["payload_json"])
        report = Assessment.from_dict(payload["transcript"]["confidence_report"])
        assert report.verdict == UNRELIABLE

        entries = pipe.db.audit_log(action="transcript_confidence")
        assert entries, "an untrustworthy transcript left no trace in the audit log"
    finally:
        pipe.close()


def test_the_digest_says_it_before_the_summary_not_after(sandbox, monkeypatch):
    """
    Everything below that line was extracted from the transcript. Learning the
    transcript was invented in a footnote, after believing the summary, is the
    wrong order.
    """
    cfg, _ = sandbox
    drop(cfg, "client-marcus.txt", CLIENT_CALL)

    from plaud_bridge import pipeline as pipeline_module

    real = pipeline_module.assess

    def scored(segments, cfg_):
        for seg in segments:
            seg.confidence, seg.no_speech = -3.0, 0.9
        return real(segments, cfg_)

    monkeypatch.setattr(pipeline_module, "assess", scored)

    pipe = Pipeline(cfg)
    try:
        pipe.run()
    finally:
        pipe.close()

    from plaud_bridge.db import Database
    from plaud_bridge.digest import DigestBuilder, DigestOptions

    db = Database(cfg.path("database"))
    try:
        text = DigestBuilder(cfg, db).render_markdown(DigestOptions(days=30))
        assert "not trustworthy" in text
        heading = text.index("### client-marcus.txt")
        caveat = text.index("not trustworthy")
        assert caveat > heading
        # The analysis fields come after the caveat, not before it.
        for marker in ("Next Action", "next action", "Open Questions", "open question"):
            if marker in text:
                assert text.index(marker) > caveat
                break
    finally:
        db.close()
