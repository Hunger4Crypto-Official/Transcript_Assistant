"""
Digest edges: "we could not open this" and "there was nothing in it" are
different statements, and the digest has to make the right one.

A withheld analysis whose vault file is gone, will not decrypt, or is not
JSON is reported in the entry by cause; a flagged or errored analysis is
listed under Needs You with its reason and its body is a note rather than a
summary; an analysis that extracted nothing says so. The chart helpers'
degenerate inputs -- a zero-width bar, a zero axis, an undated entry -- are
pinned to draw nothing misleading. The index and media edges live here too:
a database from a newer build refuses to open, the `until` filter cuts where
it says, and a range request that makes no sense is refused rather than
served empty.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from _fixtures import build_sandbox
from plaud_bridge.db import Database
from plaud_bridge.digest import DigestBuilder, DigestOptions
from plaud_bridge.digest.builder import DigestSection, _fmt_quote
from plaud_bridge.digest.charts import (
    _activity_chart,
    _hbar,
    _nice,
    _vcol,
    charts_html,
    inject_charts,
)
from plaud_bridge.media import MediaInfo, locate_original, read_range
from plaud_bridge.models import (
    ProfileAnalysis,
    Recording,
    RouteMatch,
    Segment,
    Transcript,
)
from plaud_bridge.storage import Vault
from plaud_bridge.voice import Voice


def add_row(db, vault, recording_id, *, profile_id="insurance_agent", fields=None,
            attention=False, error="", encrypt=False, artifact_paths=None,
            days_ago: float = 0.05) -> Recording:
    rec = Recording(
        id=recording_id, source_name=f"{recording_id}.txt",
        source_path=f"/inbox/{recording_id}.txt", content_hash=f"hash-{recording_id}",
        kind="text", recorded_at=datetime.now(timezone.utc) - timedelta(days=days_ago),
        duration_seconds=90.0,
    )
    rec.transcript = Transcript(segments=[Segment(0, 90, "talking about the policy", "Sasson")])
    rec.routes = [RouteMatch(profile_id=profile_id, confidence=0.8)]
    rec.compliance.governing_profile = profile_id
    rec.compliance.encrypt_at_rest = encrypt
    rec.analyses = [ProfileAnalysis(profile_id=profile_id, fields=fields or {},
                                    requires_human_attention=attention, error=error)]
    if artifact_paths:
        rec.artifact_paths.update(artifact_paths)
    if encrypt:
        rec.artifact_paths["analysis"] = str(
            vault.write(f"{recording_id}.analysis.json", rec.to_json(), recording_id)
        )
    db.upsert(rec)
    return rec


@pytest.fixture
def bench(tmp_path, monkeypatch):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    db = Database(cfg.path("database"))
    try:
        yield cfg, db, Vault(cfg.path("vault"))
    finally:
        db.close()


def _entry(cfg, db, rid, **opts) -> dict:
    sections = DigestBuilder(cfg, db)._collect(DigestOptions(days=7, **opts))
    return next(e for s in sections for e in s.entries if e["id"] == rid)


# =========================================================================
# Withheld analyses: the cause is the message
# =========================================================================
def test_a_withheld_analysis_whose_vault_file_is_gone_is_reported_as_missing(bench, caplog):
    cfg, db, vault = bench
    rec = add_row(db, vault, "rec_gone", encrypt=True, fields={"next_action": "hidden"})
    Path(rec.artifact_paths["analysis"]).unlink()

    with caplog.at_level(logging.WARNING, logger="plaud_bridge.digest"):
        entry = _entry(cfg, db, "rec_gone")
    assert entry["error"] == "the encrypted analysis file is missing from disk"
    assert entry["fields"] == {}, "the withheld fields must not be invented"
    assert any("missing from disk for rec_gone" in r.getMessage() for r in caplog.records)

    out = DigestBuilder(cfg, db).render_markdown(DigestOptions(days=7))
    assert "> Analysis error: the encrypted analysis file is missing from disk" in out
    assert "hidden" not in out


def test_a_withheld_analysis_that_will_not_decrypt_says_so(bench, monkeypatch):
    cfg, db, vault = bench
    add_row(db, vault, "rec_locked", encrypt=True, fields={"next_action": "hidden"})
    monkeypatch.setenv("PLAUD_BRIDGE_PASSPHRASE", "a-different-passphrase-entirely")
    entry = _entry(cfg, db, "rec_locked")
    assert entry["error"].startswith("could not decrypt the analysis (")
    assert "decryption failed" in entry["error"]
    assert "hidden" not in DigestBuilder(cfg, db).render_markdown(DigestOptions(days=7))


def test_a_withheld_analysis_that_is_not_json_says_it_could_not_be_read(bench):
    cfg, db, vault = bench
    rec = add_row(db, vault, "rec_garbled", encrypt=True)
    # Same name and id, so the vault opens it fine; the contents are the problem.
    vault.write("rec_garbled.analysis.json", "this is not json", "rec_garbled")
    assert Path(rec.artifact_paths["analysis"]).exists()
    entry = _entry(cfg, db, "rec_garbled")
    assert entry["error"].startswith("could not read the analysis (")


def test_a_decrypted_analysis_missing_this_profile_is_reported(bench):
    cfg, db, vault = bench
    rec = add_row(db, vault, "rec_other", encrypt=True)
    # Rewrite the vault copy with an analysis for a different profile only.
    rec.analyses = [ProfileAnalysis(profile_id="sales_trainer")]
    vault.write("rec_other.analysis.json", rec.to_json(), "rec_other")
    entry = _entry(cfg, db, "rec_other")
    assert entry["error"] == "the decrypted analysis had no entry for this profile"


def test_an_analysis_path_that_is_not_a_vault_file_is_treated_as_missing(bench, tmp_path):
    cfg, db, vault = bench
    plain = tmp_path / "rec_plainfile.analysis.json"
    plain.write_text("{}")
    add_row(db, vault, "rec_plainfile", encrypt=True, artifact_paths={"analysis": str(plain)})
    # `encrypt=True` rewrote artifact_paths["analysis"]; put the plain one back.
    payload = db.load("rec_plainfile")
    payload["artifact_paths"]["analysis"] = str(plain)
    restored, why = DigestBuilder(cfg, db)._open_withheld_analysis(
        "rec_plainfile", payload, "insurance_agent")
    assert restored is None
    assert why == "the encrypted analysis file is missing from disk"


# =========================================================================
# Flagged, errored, and empty analyses in the rendering
# =========================================================================
def test_flagged_and_errored_entries_are_listed_under_needs_you_with_their_reason(bench):
    cfg, db, vault = bench
    add_row(db, vault, "rec_flag", attention=True)
    add_row(db, vault, "rec_err", error="provider timed out")
    out = DigestBuilder(cfg, db).render_markdown(DigestOptions(days=7))

    assert "## Needs You" in out
    assert "- **Production** :: `rec_flag.txt` flagged for human review" in out
    assert "- **Production** :: `rec_err.txt` error: provider timed out" in out
    # And each body is a note, not a summary.
    assert "was flagged for human attention and was not summarised" in out
    assert "> Analysis error: provider timed out" in out


def test_an_analysis_that_extracted_nothing_says_so_rather_than_rendering_blank(bench):
    cfg, db, vault = bench
    add_row(db, vault, "rec_empty", fields={"next_action": "", "objections": [],
                                            "statements_needing_review": []})
    out = DigestBuilder(cfg, db).render_markdown(DigestOptions(days=7))
    assert "### rec_empty.txt" in out
    assert "_Nothing extracted for the highlighted fields._" in out
    assert "**Next:**" not in out


def test_an_unknown_profile_in_the_wanted_list_is_skipped_with_a_warning(bench, caplog):
    cfg, db, vault = bench
    with caplog.at_level(logging.WARNING, logger="plaud_bridge.digest"):
        sections = DigestBuilder(cfg, db)._collect(DigestOptions(profile_id="ghost"))
    assert sections == []
    assert any("unknown profile 'ghost', skipping" in r.getMessage() for r in caplog.records)


def test_a_dict_with_only_nested_values_is_shown_as_json_rather_than_dropped():
    out = _fmt_quote({"nested": {"a": 1}, "more": [1, 2]})
    assert out == '{"nested": {"a": 1}, "more": [1, 2]}'
    huge = _fmt_quote({"blob": {"k": "x" * 1000}})
    assert len(huge) == 300, "the last resort is truncated so it cannot blow out the digest"


# =========================================================================
# Chart helpers on degenerate input
# =========================================================================
def test_a_zero_width_bar_and_a_zero_height_column_draw_nothing():
    assert _hbar(0, 0, 0.4, 18, "pb-c0") == ""
    assert _hbar(0, 0, 10, 18, "pb-c0").startswith('<path class="pb-c0"')
    assert _vcol(0, 0, 24, 0.3, "pb-c0", rounded=True) == ""
    assert _vcol(0, 0, 0, 30, "pb-c0", rounded=True) == ""
    assert _vcol(0, 0, 24, 30, "pb-c0", rounded=False).startswith('<rect class="pb-c0"')


def test_a_zero_or_negative_axis_maximum_becomes_one():
    assert _nice(0) == 1.0 and _nice(-5) == 1.0
    assert _nice(0.7) == 1.0 and _nice(13) == 20.0 and _nice(50) == 50.0


def test_an_undated_entry_is_left_out_of_the_activity_chart_but_not_the_bars():
    section = DigestSection("insurance_agent", "Production", 1, entries=[
        {"id": "rec_dated", "when": "2026-09-05 10:00", "minutes": 12.0, "cost": 0.0},
        {"id": "rec_undated", "when": "", "minutes": 30.0, "cost": 0.0},
    ])
    now = datetime(2026, 9, 6, 12, 0, tzinfo=timezone.utc)
    opts = DigestOptions(days=7)
    activity = _activity_chart([section], opts, now, "Minutes per day")
    assert "12 minutes in total" in activity, "only the dated entry can be placed on a day"
    page = charts_html([section], opts, Voice(), now=now)
    assert "42 min · 2 rec" in page, "the bar chart counts both"


def test_injection_with_neither_a_table_nor_a_body_appends_the_fragment():
    assert inject_charts("<p>bare</p>", "<x/>") == "<p>bare</p><x/>"


# =========================================================================
# The index
# =========================================================================
def test_a_database_from_a_newer_build_refuses_to_open(tmp_path):
    path = tmp_path / "bridge.db"
    Database(path).close()
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE meta SET value='99' WHERE key='schema_version'")
    with pytest.raises(RuntimeError, match="schema v99 is newer than this code"):
        Database(path)


def test_the_until_filter_cuts_at_the_timestamp_it_is_given(bench):
    cfg, db, vault = bench
    add_row(db, vault, "rec_old", days_ago=10)
    add_row(db, vault, "rec_new")
    cutoff = datetime.now(timezone.utc) - timedelta(days=5)
    assert [r["id"] for r in db.query(until=cutoff)] == ["rec_old"]
    assert {r["id"] for r in db.query()} == {"rec_old", "rec_new"}


def test_trimming_the_audit_trail_refuses_an_open_ended_cutoff(bench):
    cfg, db, vault = bench
    db.audit("test", "kept")
    assert db.delete_audit_before(None) == 0
    assert len(db.audit_log(action="test")) == 1
    assert db.delete_audit_before(datetime.now(timezone.utc) + timedelta(seconds=1)) == 1


# =========================================================================
# Media
# =========================================================================
def test_an_original_whose_recorded_path_is_gone_is_reported_as_not_kept(bench, tmp_path):
    cfg, db, vault = bench
    add_row(db, vault, "rec_lost", artifact_paths={"audio": str(tmp_path / "vanished.mp3")})
    db.record_artifact("rec_lost", "audio", str(tmp_path / "vanished.mp3"), False, None)
    assert locate_original(cfg, db, "rec_lost") is None


def test_a_range_that_ends_before_it_starts_is_refused(tmp_path):
    path = tmp_path / "clip.wav"
    path.write_bytes(b"x" * 100)
    info = MediaInfo("rec_1", path, encrypted=False, content_type="audio/wav", size_bytes=100)
    with pytest.raises(ValueError, match="range 50-10 is empty"):
        read_range(info, 50, 10, cfg=None)
    with pytest.raises(ValueError, match="outside the file"):
        read_range(info, 100, None, cfg=None)


def test_a_file_that_shrinks_mid_stream_ends_the_body_rather_than_spinning(tmp_path):
    path = tmp_path / "clip.wav"
    path.write_bytes(b"y" * 10)
    # The recorded size is what the index knew; the file on disk is shorter.
    info = MediaInfo("rec_1", path, encrypted=False, content_type="audio/wav", size_bytes=1000)
    body, total = read_range(info, 0, None, cfg=None)
    assert total == 1000
    assert b"".join(body) == b"y" * 10
