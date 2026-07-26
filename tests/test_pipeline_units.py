"""Stitching, corrections, redaction, consent, transcript import."""

from pathlib import Path

from plaud_bridge.asr.stitch import stitch
from plaud_bridge.compliance.consent import detect_consent
from plaud_bridge.compliance.redact import redact_text
from plaud_bridge.config import Config
from plaud_bridge.correct import apply_corrections
from plaud_bridge.models import Segment, Transcript
from plaud_bridge.pipeline import _parse_text_transcript
from plaud_bridge.profiles.router import _keyword_prescore

ROOT = Path(__file__).resolve().parents[1]
CFG = Config.load(ROOT / "config")


# ---- stitching ----------------------------------------------------------
def test_stitch_removes_overlap_duplicates():
    a = [Segment(0, 5, "the client asked about premiums"),
         Segment(5, 10, "and the elimination period")]
    b = [Segment(8, 12, "and the elimination period"),
         Segment(12, 16, "then we discussed riders")]
    out = stitch([a, b], [0.0, 4.0])
    assert [s.text for s in out] == [
        "the client asked about premiums",
        "and the elimination period",
        "then we discussed riders",
    ]


def test_stitch_tolerates_asr_variation_in_overlap():
    a = [Segment(0, 6, "he wants the own occupation rider")]
    b = [Segment(4, 8, "he wants the own-occupation rider."),
         Segment(8, 12, "and a waiver of premium")]
    out = stitch([a, b], [0.0, 4.0])
    assert len(out) == 2


def test_stitch_keeps_genuine_repetition_outside_overlap():
    a = [Segment(0, 4, "yes exactly")]
    b = [Segment(60, 64, "yes exactly")]
    out = stitch([a, b], [0.0, 0.0])
    assert len(out) == 2


def test_stitch_single_chunk_passthrough():
    a = [Segment(0, 4, "only one chunk here")]
    assert len(stitch([a], [0.0])) == 1


# ---- glossary -----------------------------------------------------------
def test_glossary_corrections_fire():
    segs = [Segment(0, 4, "We discussed the elimination. Period and the I U L policy."),
            Segment(4, 8, "He wants a ten thirty five exchange and a para med exam.")]
    out, report = apply_corrections(segs, CFG.glossary)
    assert "elimination period" in out[0].text
    assert "IUL" in out[0].text
    assert "1035 exchange" in out[1].text
    assert "paramed" in out[1].text
    assert report.total >= 4


def test_glossary_leaves_clean_text_alone():
    segs = [Segment(0, 4, "Nothing here needs correcting at all.")]
    out, report = apply_corrections(segs, CFG.glossary)
    assert report.total == 0
    assert out[0].text == "Nothing here needs correcting at all."


# ---- redaction ----------------------------------------------------------
def test_redaction_catches_common_pii():
    text = "SSN 123-45-6789, phone (702) 555-1234, email a.b@x.com, policy AB-1234567"
    out, report = redact_text(text, CFG.get("compliance.redact_patterns"))
    for token in ("123-45-6789", "555-1234", "a.b@x.com", "AB-1234567"):
        assert token not in out
    assert report.total >= 4


def test_redaction_catches_spoken_digit_strings():
    text = "his number is five five five one two three four five six seven"
    out, _ = redact_text(text, CFG.get("compliance.redact_patterns"))
    assert "SPOKEN_DIGITS_REDACTED" in out


def test_redaction_can_be_disabled():
    text = "SSN 123-45-6789"
    out, report = redact_text(text, CFG.get("compliance.redact_patterns"), enabled=False)
    assert out == text and report.total == 0


# ---- consent ------------------------------------------------------------
def _tr(pairs):
    return Transcript(segments=[Segment(i * 3, (i + 1) * 3, t, s) for i, (t, s) in enumerate(pairs)])


def test_consent_detected_when_announced_and_agreed():
    r = detect_consent(_tr([
        ("Before we start, I record these calls for my notes, is that okay?", "Sasson"),
        ("Yeah that is fine, no problem.", "Client"),
    ]))
    assert r.complete


def test_consent_incomplete_without_agreement():
    r = detect_consent(_tr([
        ("Just so you know, this call is being recorded.", "Sasson"),
        ("So anyway about my mortgage.", "Client"),
    ]))
    assert r.announced and not r.agreed and not r.complete


def test_consent_not_satisfied_by_self_agreement():
    """You cannot consent on the other party's behalf."""
    r = detect_consent(_tr([
        ("I am recording this.", "Sasson"),
        ("Yes absolutely.", "Sasson"),
    ]))
    assert not r.complete


def test_consent_absent_entirely():
    r = detect_consent(_tr([("So tell me about your coverage.", "Sasson"),
                            ("Sure thing.", "Client")]))
    assert not r.announced and not r.complete


def test_consent_window_is_respected():
    tr = Transcript(segments=[
        Segment(0, 3, "Hello there.", "Sasson"),
        Segment(600, 603, "By the way I am recording this.", "Sasson"),
        Segment(603, 606, "Sure, fine.", "Client"),
    ])
    assert not detect_consent(tr, window_seconds=90).announced


# ---- routing prescore ---------------------------------------------------
def test_prescore_separates_work_from_home():
    work = ("So the client wants term life with a conversion rider, and we talked "
            "about the elimination period on the disability policy. The premium is "
            "four hundred a month and the death benefit is one million.")
    scores = {p.profile_id: p.score for p in _keyword_prescore(work, CFG.routable_profiles())}
    assert scores["insurance_agent"] > 0.5
    assert scores["husband"] == 0.0
    assert scores["father"] == 0.0

    home = ("Hey babe, can you pick up the kids after practice? I will grab "
            "groceries and we can do date night Saturday.")
    scores = {p.profile_id: p.score for p in _keyword_prescore(home, CFG.routable_profiles())}
    assert scores["husband"] > 0.5
    assert scores["insurance_agent"] == 0.0


def test_negative_keywords_penalise():
    text = ("This is a role play. Pretend the client asks about the policy premium "
            "and the beneficiary on the term life coverage.")
    scores = {p.profile_id: p.score for p in _keyword_prescore(text, CFG.routable_profiles())}
    assert scores["sales_trainer"] > 0
    # "role play" is a negative keyword for insurance_agent and drags it down.
    assert scores["insurance_agent"] < 0.55


# ---- transcript import --------------------------------------------------
def test_srt_import_preserves_timestamps():
    srt = ("1\n00:00:01,000 --> 00:00:04,500\nSasson: Before we start.\n\n"
           "2\n00:00:04,500 --> 00:00:06,000\nClient: Sure.\n")
    segs = _parse_text_transcript(srt, ".srt")
    assert len(segs) == 2
    assert segs[0].start == 1.0 and segs[0].speaker == "Sasson"
    assert segs[1].end == 6.0


def test_plain_text_import_synthesises_timeline():
    txt = "Sasson: So tell me about your coverage.\nClient: I have a term policy."
    segs = _parse_text_transcript(txt, ".txt")
    assert len(segs) == 2
    assert segs[0].start == 0.0
    assert segs[1].start > segs[0].start
    assert segs[1].speaker == "Client"
