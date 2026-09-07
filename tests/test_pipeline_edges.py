"""
The pipeline at the edges.

Each of these pins what one stage does when its ordinary assumption breaks: the
original cannot be encrypted, the archive folder will not take a move, the
memory ledger will not update, a route names a profile that no longer exists, a
day cuts into more episodes than the cap, the file vanished between discovery
and reading. The rule they share is the one in `process_file`: never lose the
queue to one bad file, and never quietly leave something in the clear.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from _fixtures import CLIENT_CALL, build_sandbox, drop
from plaud_bridge.episodes import Episode
from plaud_bridge.models import (
    ComplianceVerdict,
    Recording,
    RouteMatch,
    Segment,
    Stage,
    Transcript,
)
from plaud_bridge.pipeline import Pipeline, PipelineError, _parse_text_transcript
from plaud_bridge.storage import VaultError

NO_ANNOUNCEMENT = CLIENT_CALL.split("\n", 2)[2]


@pytest.fixture
def pipe(tmp_path, monkeypatch):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    p = Pipeline(cfg)
    try:
        yield p
    finally:
        p.close()


def _rec(rid="rec_hand", *, profile="sales_trainer", encrypt=False, text="spoken words") -> Recording:
    rec = Recording(id=rid, source_name=f"{rid}.txt", source_path=f"/inbox/{rid}.txt",
                    content_hash=f"hash-{rid}", kind="text",
                    recorded_at=datetime(2026, 1, 2, tzinfo=timezone.utc))
    rec.transcript = Transcript(segments=[Segment(0.0, 2.0, text, "Sasson")])
    rec.routes = [RouteMatch(profile_id=profile, confidence=0.9)]
    rec.compliance = ComplianceVerdict(governing_profile=profile, encrypt_at_rest=encrypt)
    return rec


# =========================================================================
# Audio: the glossary fires and is audited
# =========================================================================
def test_glossary_corrections_on_a_transcribed_recording_are_audited(pipe, monkeypatch, tmp_path):
    """
    Everything real about audio is stubbed -- ffmpeg, the recogniser, the
    diarizer -- so this pins only the stage's own work: corrections applied to
    what came back, and an audit row saying that they were.
    """
    heard = Transcript(segments=[
        Segment(0.0, 4.0, "We discussed the elimination. Period and the I U L policy.", "S"),
    ], asr_provider="stub", asr_model="stub")
    monkeypatch.setattr(pipe.audio, "normalise", lambda path, work: (path, 4.0))
    monkeypatch.setattr(pipe.audio, "chunk", lambda normalised, work, duration: [normalised])
    monkeypatch.setattr("plaud_bridge.pipeline.transcribe",
                        lambda chunks, cfg, glossary, local_only=False: heard)
    monkeypatch.setattr("plaud_bridge.pipeline.diarize",
                        lambda normalised, segments, cfg: segments)

    source = pipe.cfg.path("inbox") / "REC0042.mp3"
    source.write_bytes(b"not really audio")
    rec = Recording(id="rec_audio", source_name=source.name, kind="audio")
    pipe._transcribe_audio(rec, source, pipe.cfg.path("work") / rec.id)

    assert rec.duration_seconds == 4.0
    assert "elimination period" in rec.transcript.segments[0].text
    assert "IUL" in rec.transcript.segments[0].text
    actions = [r["action"] for r in pipe.db.audit_log(recording_id="rec_audio", limit=10)]
    assert "glossary" in actions
    assert "asr_locality" in actions
    locality = next(r for r in pipe.db.audit_log(recording_id="rec_audio") if r["action"] == "asr_locality")
    assert "local_only=True" in locality["detail"], "an unnamed file was allowed to reach cloud ASR"


# =========================================================================
# Text import
# =========================================================================
def test_a_text_file_that_vanished_before_it_was_read_is_a_pipeline_error(pipe, tmp_path):
    rec = Recording(id="rec_gone", source_name="gone.txt", kind="text")
    with pytest.raises(PipelineError, match="cannot read gone.txt"):
        pipe._load_text(rec, tmp_path / "gone.txt")
    assert rec.transcript is None


def test_a_line_holding_a_url_is_not_read_as_a_speaker_label():
    segs = _parse_text_transcript("Sasson: see https://example.com/quote for the numbers", ".txt")
    assert len(segs) == 1
    assert segs[0].speaker == "SPEAKER"
    assert segs[0].text == "Sasson: see https://example.com/quote for the numbers"


def test_vtt_blocks_without_a_timestamp_or_without_text_are_skipped():
    vtt = (
        "WEBVTT\n\n"
        "NOTE this exporter leaves notes\n\n"
        "just-an-identifier\nwith no timing line at all\n\n"
        "2\n00:00:01.000 --> 00:00:02.000\n\n"
        "3\n00:00:02.000 --> 00:00:04.000\n<v Marcus>That is fine.\n"
    )
    segs = _parse_text_transcript(vtt, ".vtt")
    assert [(s.speaker, s.text, s.start) for s in segs] == [("Marcus", "That is fine.", 2.0)]


def test_srt_blocks_without_a_timestamp_are_skipped():
    srt = (
        "a stray line that is not a cue\n\n"
        "1\n00:00:01,000 --> 00:00:04,500\nSasson: Before we start.\n"
    )
    segs = _parse_text_transcript(srt, ".srt")
    assert len(segs) == 1 and segs[0].start == 1.0


# =========================================================================
# Routing and the gate
# =========================================================================
def test_more_episodes_than_the_cap_are_folded_into_the_last_one(tmp_path, monkeypatch):
    cfg, _ = build_sandbox(tmp_path, monkeypatch, overrides={"episodes": {"max_per_recording": 2}})
    pipe = Pipeline(cfg)
    try:
        segments = [Segment(i * 10.0, i * 10.0 + 5.0, f"episode {i} about the term policy", "S")
                    for i in range(4)]
        rec = _rec(profile="insurance_agent")
        rec.transcript = Transcript(segments=segments)
        monkeypatch.setattr("plaud_bridge.pipeline.segment_episodes",
                            lambda transcript, cfg: [Episode(index=i, segments=[s])
                                                     for i, s in enumerate(segments)])

        pipe._route(rec)

        assert len(rec.episodes) == 2
        assert [len(e.segments) for e in rec.episodes] == [1, 3]
        assert rec.episodes[1].segments == segments[1:], "the tail was not folded into the last"
        assert rec.stage == Stage.ROUTED
        assert rec.routes, "routing produced nothing"
        route_row = pipe.db.audit_log(recording_id=rec.id, action="route")[0]
        assert route_row["detail"].startswith("2 episode(s)")
    finally:
        pipe.close()


def test_a_flagged_consent_gap_is_processed_and_recorded_rather_than_quarantined(
    tmp_path, monkeypatch
):
    cfg, _ = build_sandbox(tmp_path, monkeypatch,
                           overrides={"compliance": {"on_missing_consent": "flag"}})
    drop(cfg, "unannounced.txt", NO_ANNOUNCEMENT)
    pipe = Pipeline(cfg)
    try:
        stats = pipe.run()
        assert stats.processed == 1 and stats.quarantined == 0
        row = pipe.db.query()[0]
        assert row["stage"] == "complete"
        assert row["consent_status"] == "not_detected"
        assert not (cfg.path("quarantine") / row["id"]).exists()
        compliance = pipe.db.audit_log(recording_id=row["id"], action="compliance")[0]
        assert "allow=True consent=not_detected" in compliance["detail"]
    finally:
        pipe.close()


def test_a_route_to_a_profile_that_no_longer_exists_is_not_analysed(pipe):
    rec = _rec(profile="insurance_agent")
    rec.routes = [RouteMatch(profile_id="retired_profile", confidence=0.95),
                  RouteMatch(profile_id="insurance_agent", confidence=0.9)]
    rec.episodes = []           # no per-profile portion, so the whole transcript is used

    pipe._analyse(rec)
    assert [a.profile_id for a in rec.analyses] == ["insurance_agent"]
    assert rec.analyses[0].fields, "the analysis came back empty"
    assert rec.stage == Stage.ANALYZED


# =========================================================================
# Memory
# =========================================================================
def test_memory_can_be_switched_off(tmp_path, monkeypatch):
    cfg, _ = build_sandbox(tmp_path, monkeypatch, overrides={"memory": {"enabled": False}})
    drop(cfg, "client-marcus.txt", CLIENT_CALL)
    pipe = Pipeline(cfg)
    try:
        assert pipe.run().processed == 1
        assert not list(pipe.memory.dir.glob("*.enc")) if pipe.memory.dir.exists() else True
        assert pipe.memory.ledger("insurance_agent").to_dict().get("entries", []) == []
    finally:
        pipe.close()


def test_a_memory_failure_does_not_fail_the_recording(pipe, monkeypatch):
    def explode(*_a, **_k):
        raise RuntimeError("ledger is unwritable")

    monkeypatch.setattr(pipe.memory, "update_from_record", explode)
    drop(pipe.cfg, "client-marcus.txt", CLIENT_CALL)

    stats = pipe.run()
    assert stats.processed == 1 and stats.failed == 0
    row = pipe.db.query()[0]
    assert row["stage"] == "complete"
    assert all(Path(a["path"]).exists() for a in pipe.db.all_artifacts())


# =========================================================================
# Reprocessing
# =========================================================================
def test_reprocessing_purges_the_archived_original_but_keeps_the_file_in_hand(pipe):
    rec = _rec()
    out = pipe.cfg.path("outbox") / f"{rec.id}.transcript.md"
    out.write_text("old transcript")
    rec.artifact_paths["transcript"] = str(out)
    pipe.db.upsert(rec)
    pipe.db.record_artifact(rec.id, "transcript", str(out), False, None)

    processed = pipe.cfg.path("inbox") / "_processed"
    processed.mkdir(parents=True)
    in_hand = processed / f"{rec.id}_first.txt"
    stale = processed / f"{rec.id}_second.txt"
    theirs = processed / "rec_other_call.txt"
    for p in (in_hand, stale, theirs):
        p.write_text("original")

    pipe._purge_prior_artifacts(rec.id, keep=in_hand)

    assert in_hand.exists(), "the file being reprocessed was deleted from under the run"
    assert theirs.exists()
    assert not stale.exists() and not out.exists()
    assert pipe.db.all_artifacts() == []
    purge = pipe.db.audit_log(recording_id=rec.id, action="reprocess_purge")[0]
    assert "removed 2 prior artifact file(s)" in purge["detail"]


def test_a_prior_artifact_that_will_not_unlink_is_skipped_and_the_rest_still_go(pipe):
    rec = _rec()
    stubborn = pipe.cfg.path("vault") / "stubborn.enc"
    stubborn.mkdir()
    plain = pipe.cfg.path("outbox") / f"{rec.id}.analysis.json"
    plain.write_text("{}")
    rec.artifact_paths = {"transcript": str(stubborn), "analysis": str(plain)}
    pipe.db.upsert(rec)
    pipe.db.record_artifact(rec.id, "transcript", str(stubborn), True, None)
    pipe.db.record_artifact(rec.id, "analysis", str(plain), False, None)

    pipe._purge_prior_artifacts(rec.id)

    assert stubborn.exists()
    assert not plain.exists()
    assert pipe.db.all_artifacts() == [], "the index still points at purged artifacts"


# =========================================================================
# Quarantine and the archive step
# =========================================================================
def test_quarantine_still_writes_its_reasons_when_the_source_cannot_be_copied(pipe, tmp_path):
    rec = _rec("rec_q", profile="insurance_agent")
    rec.compliance.allow = False
    rec.compliance.reasons = ["QUARANTINED. nobody agreed on tape"]

    pipe._quarantine(rec, tmp_path / "vanished.txt")

    qdir = pipe.cfg.path("quarantine") / "rec_q"
    why = (qdir / "WHY.md").read_text(encoding="utf-8")
    assert "nobody agreed on tape" in why
    assert "run.py release rec_q" in why
    assert not (qdir / "vanished.txt").exists()
    assert rec.stage == Stage.QUARANTINED
    assert pipe.db.audit_log(recording_id="rec_q", action="quarantine")


def test_an_original_that_cannot_be_encrypted_is_left_in_the_inbox_and_said_so(
    pipe, monkeypatch
):
    """
    The one thing this must never do is fall back to archiving the original in
    the clear. It stays where it was, the run still completes, and the audit
    trail names the file that was left unencrypted.
    """
    def refuse(*_a, **_k):
        raise VaultError("disk full while encrypting")

    monkeypatch.setattr(pipe.vault, "write_stream", refuse)
    source = drop(pipe.cfg, "client-marcus.txt", CLIENT_CALL)

    stats = pipe.run()
    assert stats.processed == 1 and stats.failed == 0
    assert source.exists(), "the original was moved or deleted after encryption failed"
    assert not (pipe.cfg.path("inbox") / "_processed").exists(), "it was archived in the clear"

    row = pipe.db.query()[0]
    kinds = {a["kind"] for a in pipe.db.all_artifacts() if a["recording_id"] == row["id"]}
    assert "source" not in kinds and "audio" not in kinds
    assert "source" not in pipe.db.load(row["id"])["artifact_paths"]
    left = pipe.db.audit_log(recording_id=row["id"], action="archive_unencrypted")
    assert left and "client-marcus.txt" in left[0]["detail"]
    assert "left in the inbox in plaintext" in left[0]["detail"]


def test_an_original_that_cannot_be_moved_is_left_where_it_is(pipe, monkeypatch):
    def refuse(*_a, **_k):
        raise OSError("read-only file system")

    monkeypatch.setattr("plaud_bridge.pipeline.shutil.move", refuse)
    rec = _rec()                                        # sales_trainer: not encrypted
    pipe.db.upsert(rec)
    source = drop(pipe.cfg, f"{rec.id}.txt", "spoken words")

    pipe._archive(rec, source)

    assert source.exists()
    assert "source" not in rec.artifact_paths
    assert pipe.db.all_artifacts() == []
    assert not any((pipe.cfg.path("inbox") / "_processed").glob("*"))


def test_an_unencrypted_original_is_archived_under_its_recording_id_and_indexed(pipe):
    rec = _rec()
    pipe.db.upsert(rec)
    source = drop(pipe.cfg, "recording.txt", "spoken words")

    pipe._archive(rec, source)

    dest = pipe.cfg.path("inbox") / "_processed" / f"{rec.id}_recording.txt"
    assert dest.exists() and not source.exists()
    assert rec.artifact_paths["source"] == str(dest)
    rows = pipe.db.all_artifacts()
    assert [(r["kind"], r["encrypted"]) for r in rows] == [("source", 0)]
    assert rows[0]["expires_at"], "an archived original must expire on the raw-audio clock"
