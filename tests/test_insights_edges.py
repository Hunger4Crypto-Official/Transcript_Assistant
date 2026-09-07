"""
Insights edges: the honest limits the numbers come with.

A locked recording is an error that says which kind; a window with no
identifiable owner reports everyone's numbers and says so; a month-on-month
delta is only computed when both months exist; the empty renderings still
tell the reader what was excluded or unopened. The episode cutter's own
guards -- not enough vocabulary to call a change, an opening fragment folded
forward -- are pinned alongside because they are the same kind of arithmetic
over the same segments.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from _fixtures import CLIENT_CALL, build_sandbox, drop
from plaud_bridge.archive import Archive
from plaud_bridge.cli import main
from plaud_bridge.db import Database
from plaud_bridge.episodes import _cut, _overlap, _speaker_change
from plaud_bridge.insights import (
    InsightsError,
    TrendReport,
    _aggregate,
    measure,
    recording_metrics,
    render_recording,
    render_trend,
    trend,
)
from plaud_bridge.models import ProfileAnalysis, Recording, RouteMatch, Segment, Transcript


def seg(start, end, text, speaker):
    return {"start": start, "end": end, "text": text, "speaker": speaker}


def add_row(db, recording_id, *, days_ago, lines, profile_id="insurance_agent"):
    """A plaintext recording with a known timeline, straight into the index."""
    rec = Recording(
        id=recording_id, source_name=f"{recording_id}.txt",
        source_path=f"/inbox/{recording_id}.txt", content_hash=f"hash-{recording_id}",
        kind="text", recorded_at=datetime.now(timezone.utc) - timedelta(days=days_ago),
    )
    rec.transcript = Transcript(segments=[
        Segment(i * 10.0, i * 10.0 + 8.0, text, who) for i, (who, text) in enumerate(lines)
    ])
    rec.routes = [RouteMatch(profile_id=profile_id, confidence=0.9)]
    rec.compliance.governing_profile = profile_id
    rec.compliance.encrypt_at_rest = False
    rec.analyses = [ProfileAnalysis(profile_id=profile_id)]
    db.upsert(rec)
    return rec


TALK = [("Sasson", "walk me through what you have in place today"),
        ("Marcus", "a term policy through work and not much else"),
        ("Sasson", "and does your wife have anything separate")]


# =========================================================================
# A recording that will not open
# =========================================================================
def test_a_locked_recording_is_an_unopened_error_not_a_missing_one(tmp_path, monkeypatch):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    drop(cfg, "client.txt", CLIENT_CALL)
    assert main(["--config", str(tmp_path / "config"), "run"]) == 0
    db = Database(cfg.path("database"))
    try:
        rid = db.query(limit=1)[0]["id"]
        monkeypatch.delenv("PLAUD_BRIDGE_PASSPHRASE", raising=False)
        with pytest.raises(InsightsError) as excinfo:
            recording_metrics(cfg, db, Archive(cfg, db), rid)
        assert excinfo.value.unopened is True
        assert f"{rid} exists but its content could not be opened" in str(excinfo.value)
        assert "PLAUD_BRIDGE_PASSPHRASE" in str(excinfo.value)
    finally:
        db.close()


# =========================================================================
# No identifiable owner: everyone's numbers, labelled as such
# =========================================================================
def test_without_an_owner_the_aggregate_is_everyone_and_says_so():
    a = measure([seg(0, 60, "word " * 30, "A"), seg(60, 90, "reply?", "B")])
    b = measure([seg(0, 30, "just me talking here", "C")])
    assert a.owner_metrics is None and b.owner_metrics is None

    agg = _aggregate([a, b])
    assert agg.focus == "everyone"
    assert agg.recordings == 2
    assert agg.seconds == pytest.approx(120.0)
    assert agg.share == pytest.approx(1.0), "everyone's share of everyone's speech is all of it"
    assert agg.words == 30 + 1 + 4
    assert agg.segment_count == 3 and agg.questions == 1
    assert agg.question_rate == pytest.approx(1 / 3)
    assert agg.words_per_minute == pytest.approx(35 / 2.0)
    assert agg.longest_monologue_seconds == pytest.approx(60.0)
    assert agg.interruptions_approx == 0

    empty = _aggregate([])
    assert empty.recordings == 0 and empty.share == 0.0 and empty.focus == "owner"


def test_a_trend_with_no_owner_label_reports_everyone(tmp_path, monkeypatch):
    cfg, _ = build_sandbox(tmp_path, monkeypatch,
                           overrides={"diarization": {"owner_label": ""}})
    db = Database(cfg.path("database"))
    try:
        add_row(db, "rec_a", days_ago=2, lines=TALK)
        report = trend(cfg, db, Archive(cfg, db), days=90)
    finally:
        db.close()

    assert report.owner_label == "" and report.owner_recordings == 0
    assert report.overall.focus == "everyone"
    out = render_trend(report)
    assert "## Everyone (the owner's voice was not identifiable in this window)" in out
    assert "owner identified in" not in out


# =========================================================================
# Deltas need both months
# =========================================================================
def test_a_month_on_month_delta_appears_only_when_both_windows_hold_recordings(
        tmp_path, monkeypatch):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    db = Database(cfg.path("database"))
    try:
        archive = Archive(cfg, db)
        add_row(db, "rec_now", days_ago=5, lines=TALK)
        one_month = trend(cfg, db, archive, days=90)
        assert one_month.deltas == {}, "one month of data is not a comparison"
        assert "vs the 30 before" not in render_trend(one_month)

        # A quieter, question-free Sasson last month.
        add_row(db, "rec_then", days_ago=40, lines=[
            ("Sasson", "one two three four five six"),
            ("Marcus", "a much longer reply that runs and runs and runs on and on"),
            ("Marcus", "and keeps going with more words in it than the first"),
        ])
        report = trend(cfg, db, archive, days=90)
    finally:
        db.close()

    assert report.current.recordings == 1 and report.prior.recordings == 1
    assert set(report.deltas) == {"talk_share", "words_per_minute", "question_rate",
                                  "longest_monologue_seconds"}
    assert report.deltas["talk_share"] == pytest.approx(
        report.current.share - report.prior.share)
    assert report.deltas["talk_share"] > 0, "this month Sasson spoke a larger share"
    out = render_trend(report)
    assert "## Last 30 days vs the 30 before" in out
    assert "| Talk share |" in out and "| Pace (wpm) |" in out


# =========================================================================
# Renderings of the empty and the incomplete
# =========================================================================
def test_a_recording_with_no_speech_renders_a_note_not_a_table():
    m = measure([])
    m.recording_id = "rec_silent"
    out = render_recording(m)
    assert "Nothing to measure: this recording has no spoken segments." in out
    assert "| Speaker |" not in out
    assert "`rec_silent`" in out


def test_an_empty_window_still_says_what_was_excluded_and_unopened():
    report = TrendReport(days=30, excluded_personal=2, unopened=["rec_x  dinner.txt"])
    out = render_trend(report)
    assert "Nothing to measure in this window." in out
    assert "2 personal recording(s) were excluded; --include-personal counts them." in out
    assert "1 recording(s) could not be opened. Set PLAUD_BRIDGE_PASSPHRASE" in out
    plain = render_trend(TrendReport(days=30))
    assert "personal recording(s) were excluded" not in plain
    assert "recording(s) could not be opened" not in plain


def test_unopened_recordings_are_listed_beside_the_numbers_they_are_missing_from(
        tmp_path, monkeypatch):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    drop(cfg, "client.txt", CLIENT_CALL)
    assert main(["--config", str(tmp_path / "config"), "run"]) == 0
    db = Database(cfg.path("database"))
    try:
        add_row(db, "rec_plain", days_ago=1, lines=TALK)
        locked = db.query(limit=10)
        locked_id = next(r["id"] for r in locked if r["source_name"] == "client.txt")
        monkeypatch.delenv("PLAUD_BRIDGE_PASSPHRASE", raising=False)
        report = trend(cfg, db, Archive(cfg, db), days=90)
    finally:
        db.close()

    assert [m.recording_id for m in report.recordings] == ["rec_plain"]
    assert report.unopened == [f"{locked_id}  client.txt"]
    out = render_trend(report)
    assert "**1 recording(s) could not be opened and are not counted:**" in out
    assert f"- `{locked_id}  client.txt`" in out
    assert "these numbers describe less than the whole archive" in out


# =========================================================================
# Episodes: the guards in the cutter
# =========================================================================
def test_not_enough_vocabulary_is_not_a_subject_change():
    assert _overlap(set(), {"policy"}) == 1.0
    assert _overlap({"policy"}, set()) == 1.0
    assert _overlap({"policy", "premium"}, {"premium", "rider"}) == pytest.approx(1 / 3)


def test_no_speakers_on_one_side_is_not_a_speaker_change():
    seg_a = Segment(0, 1, "x", "A")
    assert _speaker_change([], [seg_a]) == 0.0
    assert _speaker_change([seg_a], []) == 0.0
    assert _speaker_change([seg_a], [Segment(1, 2, "y", "B")]) == 1.0


def test_an_opening_fragment_folds_forward_into_the_next_episode():
    """
    A ten-second remark before the first real boundary has nothing behind it
    to absorb it, so it joins the episode that follows and that episode takes
    the opening's reason -- unless the boundary was a silence, which is
    conclusive and keeps the fragment separate.
    """
    segments = [Segment(0, 5, "quick word", "A"), Segment(5, 10, "before we start", "A")]
    segments += [Segment(10 + i * 30, 40 + i * 30, f"the real meeting part {i}", "B")
                 for i in range(6)]

    weak = _cut(segments, [(2, "the people talking changed", False)],
                min_seconds=60, max_seconds=1800)
    assert len(weak) == 1
    assert weak[0].reason == "start of recording"
    assert weak[0].segments[0].text == "quick word" and len(weak[0].segments) == 8
    assert weak[0].index == 0

    strong = _cut(segments, [(2, "50s silence", True)], min_seconds=60, max_seconds=1800)
    assert [e.reason for e in strong] == ["start of recording", "50s silence"]
    assert len(strong[0].segments) == 2
