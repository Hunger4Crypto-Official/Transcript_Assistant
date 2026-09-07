"""
The insights arithmetic, pinned to hand-computed numbers.

Every metric here is supposed to be checkable by a person with the transcript
and a calculator, so the tests are exactly that: hand-built segment lists
whose talk shares, monologues, and interruptions are known before the code
runs. If one of these fails, the arithmetic changed -- and arithmetic that
changes under refactoring is the failure mode this feature promised not to
have.
"""

from __future__ import annotations

import json

import pytest

from _fixtures import CLIENT_CALL, FAMILY_DINNER, build_sandbox, drop
from plaud_bridge.archive import Archive
from plaud_bridge.cli import main
from plaud_bridge.db import Database
from plaud_bridge.insights import (
    InsightsError,
    measure,
    metrics_dict,
    recording_metrics,
    render_recording,
    render_trend,
    trend,
)


def seg(start: float, end: float, text: str, speaker: str) -> dict:
    return {"start": start, "end": end, "text": text, "speaker": speaker}


def by_name(metrics, speaker: str):
    return next(s for s in metrics.speakers if s.speaker == speaker)


# =========================================================================
# Per-recording arithmetic
# =========================================================================
def test_talk_share_is_seconds_over_total_spoken_seconds():
    m = measure([
        seg(0, 40, "one long stretch of talking", "Sasson"),
        seg(40, 50, "a shorter reply", "Marcus"),
        seg(50, 70, "more from the producer", "Sasson"),
        seg(70, 90, "and a longer reply", "Marcus"),
    ])
    # Sasson 60s of 90s spoken; Marcus the remaining 30s.
    assert m.spoken_seconds == pytest.approx(90.0)
    sasson, marcus = by_name(m, "Sasson"), by_name(m, "Marcus")
    assert sasson.share == pytest.approx(60.0 / 90.0)
    assert marcus.share == pytest.approx(30.0 / 90.0)
    assert sasson.minutes == pytest.approx(1.0)


def test_words_per_minute_is_words_over_spoken_minutes():
    m = measure([seg(0, 60, "word " * 30, "Sasson")])
    assert by_name(m, "Sasson").words == 30
    assert by_name(m, "Sasson").words_per_minute == pytest.approx(30.0)


def test_question_rate_counts_marks_and_interrogative_openers():
    m = measure([
        seg(0, 5, "Is that okay with you?", "Sasson"),      # ends in ?
        seg(5, 10, "What does that run", "Sasson"),          # opener, mark lost by ASR
        seg(10, 15, "I will send two options over.", "Sasson"),
        seg(15, 20, "The mortgage is the concern.", "Sasson"),
    ])
    s = by_name(m, "Sasson")
    assert s.questions == 2
    assert s.question_rate == pytest.approx(0.5)


def test_longest_monologue_is_the_longest_consecutive_same_speaker_run():
    m = measure([
        seg(0, 10, "part one", "Sasson"),
        seg(10, 20, "part two", "Sasson"),
        seg(20, 30, "part three", "Sasson"),   # a 3-segment, 30 second run
        seg(30, 35, "a reply", "Marcus"),
        seg(35, 40, "back again", "Sasson"),
    ])
    assert by_name(m, "Sasson").longest_monologue_seconds == pytest.approx(30.0)
    assert by_name(m, "Marcus").longest_monologue_seconds == pytest.approx(5.0)


def test_a_long_silence_breaks_a_monologue_run():
    m = measure([
        seg(0, 10, "before the pause", "Sasson"),
        seg(20, 30, "after the pause", "Sasson"),   # 10s gap > the 3s threshold
    ])
    # Not one 30-second monologue: nobody was talking for ten of those seconds.
    assert by_name(m, "Sasson").longest_monologue_seconds == pytest.approx(10.0)


def test_interruptions_approx_counts_overlap_with_the_previous_speaker():
    m = measure([
        seg(0, 5, "still making my point here", "Sasson"),
        seg(4, 6, "but wait", "Marcus"),            # starts before Sasson ended
        seg(6, 8, "go on", "Sasson"),               # starts exactly at the end: not one
    ])
    assert by_name(m, "Marcus").interruptions_approx == 1
    assert by_name(m, "Sasson").interruptions_approx == 0


def test_overlap_with_your_own_previous_segment_is_not_an_interruption():
    m = measure([
        seg(0, 5, "first thought", "Sasson"),
        seg(4, 6, "second thought", "Sasson"),
    ])
    assert by_name(m, "Sasson").interruptions_approx == 0


def test_silence_share_counts_only_gaps_over_the_threshold():
    m = measure([
        seg(0, 10, "before", "Sasson"),
        seg(30, 40, "after", "Marcus"),      # a 20 second hole in a 40 second clock
    ])
    assert m.silence_seconds == pytest.approx(20.0)
    assert m.silence_share == pytest.approx(0.5)

    rhythm = measure([
        seg(0, 10, "before", "Sasson"),
        seg(11, 20, "after", "Marcus"),      # a 1 second breath is not silence
    ])
    assert rhythm.silence_seconds == 0.0


def test_empty_and_one_speaker_inputs_produce_numbers_not_nan():
    empty = measure([])
    assert empty.spoken_seconds == 0.0
    assert empty.silence_share == 0.0
    assert empty.speakers == []
    assert json.dumps(metrics_dict(empty))     # zeros, and serialisable zeros

    memo = measure([seg(0, 30, "note to self about the meeting", "Sasson")],
                   owner_label="Sasson")
    only = by_name(memo, "Sasson")
    assert only.share == pytest.approx(1.0)
    assert only.interruptions_approx == 0
    assert only.words_per_minute > 0
    assert memo.owner == "Sasson"


def test_metrics_dict_is_json_serialisable_and_carries_the_charted_fields():
    m = measure([
        seg(0, 40, "talking", "Sasson"),
        seg(40, 60, "replying?", "Marcus"),
    ], owner_label="Sasson")
    payload = json.loads(json.dumps(metrics_dict(m)))
    speaker = payload["speakers"][0]
    for key in ("share", "words_per_minute", "question_rate",
                "longest_monologue_seconds", "interruptions_approx", "is_owner"):
        assert key in speaker
    assert payload["speakers"][0]["is_owner"] is True    # ordered by talk time
    assert "silence_share" in payload and "spoken_seconds" in payload


# =========================================================================
# Trend, over a real (stubbed) archive
# =========================================================================
@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    drop(cfg, "client.txt", CLIENT_CALL)
    drop(cfg, "dinner.txt", FAMILY_DINNER)
    assert main(["--config", str(tmp_path / "config"), "run"]) == 0
    return cfg


def opened(cfg):
    db = Database(cfg.path("database"))
    return db, Archive(cfg, db)


def test_trend_excludes_personal_profiles_unless_asked(sandbox):
    db, archive = opened(sandbox)
    try:
        report = trend(sandbox, db, archive, days=90)
        assert report.excluded_personal == 1
        assert all(m.profile_id != "father" for m in report.recordings)

        everything = trend(sandbox, db, archive, days=90, include_personal=True)
        assert everything.excluded_personal == 0
        assert len(everything.recordings) == len(report.recordings) + 1
    finally:
        db.close()


def test_trend_focuses_on_the_owner_when_identifiable(sandbox):
    db, archive = opened(sandbox)
    try:
        report = trend(sandbox, db, archive, days=90)
        # The shipped config names the owner Sasson, and the client call has
        # Marcus in it too, so the owner's share must be real and partial.
        assert report.owner_label == "Sasson"
        assert report.owner_recordings >= 1
        assert report.overall.focus == "owner"
        assert 0.0 < report.overall.share < 1.0
        assert report.overall.words_per_minute > 0
        # And the rendering says whose numbers these are.
        assert "You (Sasson)" in render_trend(report)
    finally:
        db.close()


def test_trend_reports_unopenable_recordings_rather_than_guessing(sandbox, monkeypatch):
    monkeypatch.delenv("PLAUD_BRIDGE_PASSPHRASE", raising=False)
    db, archive = opened(sandbox)
    try:
        report = trend(sandbox, db, archive, days=90, include_personal=True)
        assert report.unopened, "a locked vault produced no unopened report"
        # Every recording is either measured or reported, never silently lost.
        assert len(report.recordings) + len(report.unopened) == 2
        assert "could not be opened" in render_trend(report)
    finally:
        db.close()


def test_recording_metrics_raises_for_a_recording_that_does_not_exist(sandbox):
    db, archive = opened(sandbox)
    try:
        with pytest.raises(InsightsError) as excinfo:
            recording_metrics(sandbox, db, archive, "rec_does_not_exist")
        assert not excinfo.value.unopened
    finally:
        db.close()


# =========================================================================
# The CLI route
# =========================================================================
def cli(cfg, *argv) -> int:
    return main(["--config", str(cfg.root / "config"), *argv])


def test_cli_summary_renders_the_owner_numbers(sandbox, capsys):
    assert cli(sandbox, "insights") == 0
    out = capsys.readouterr().out
    assert "Insights" in out
    assert "Talk share" in out


def test_cli_one_recording_breaks_down_per_speaker(sandbox, capsys):
    db = Database(sandbox.path("database"))
    try:
        rid = next(r["id"] for r in db.query(limit=10)
                   if r["governing_profile"] == "insurance_agent")
    finally:
        db.close()
    assert cli(sandbox, "insights", "--recording", rid) == 0
    out = capsys.readouterr().out
    assert "Sasson (you)" in out
    assert "Marcus" in out
    assert "Interruptions" in out


def test_cli_unknown_recording_exits_one(sandbox):
    assert cli(sandbox, "insights", "--recording", "rec_does_not_exist") == 1


def test_cli_empty_window_is_a_note_and_exit_zero(tmp_path, monkeypatch, capsys):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    assert cli(cfg, "insights") == 0
    assert "Nothing to measure" in capsys.readouterr().out


def test_render_recording_shows_known_arithmetic():
    m = measure([
        seg(0, 60, "word " * 30, "Sasson"),
        seg(60, 90, "a reply from the other side", "Marcus"),
    ], owner_label="Sasson")
    m.source_name = "call.txt"
    body = render_recording(m)
    assert "call.txt" in body
    assert "67%" in body      # 60s of 90s spoken
    assert "30 wpm" in body
