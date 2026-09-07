"""
Retention: what the sweep says, and what it does on each branch.

The two big promises -- dry run by default, never delete outside the data
directory -- are pinned in test_audit_regressions.py. These cover the rest:
the rendered plan says what will go and what is already gone, the per-kind
expiry windows, a disabled sweeper that plans nothing, and a file that will not
unlink staying in the index rather than being audited as deleted.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta, timezone
from pathlib import Path

from _fixtures import build_sandbox
from plaud_bridge.compliance.retention import RetentionItem, RetentionPlan, RetentionSweeper
from plaud_bridge.db import Database
from plaud_bridge.models import Recording


def _item(rid: str, path: Path, exists: bool, kind: str = "transcript") -> RetentionItem:
    return RetentionItem(
        recording_id=rid, kind=kind, path=str(path), expires_at="2026-01-02T00:00:00+00:00",
        exists=exists, size_bytes=1_048_576 if exists else 0,
    )


def _indexed(db, rid: str) -> Recording:
    rec = Recording(id=rid, source_name=f"{rid}.txt", source_path=f"/inbox/{rid}.txt",
                    content_hash=f"hash-{rid}", kind="text")
    db.upsert(rec)
    return rec


# =========================================================================
# What the plan says
# =========================================================================
def test_an_empty_plan_says_nothing_expired():
    assert RetentionPlan().render() == "Retention sweep: nothing has expired."


def test_a_dry_run_lists_each_artifact_marks_the_missing_ones_and_says_how_to_delete(tmp_path):
    present = tmp_path / "present.enc"
    gone = tmp_path / "gone.enc"
    plan = RetentionPlan(items=[_item("rec_a", present, True), _item("rec_b", gone, False)])

    out = plan.render()
    assert "DRY RUN, nothing deleted" in out
    assert "2 artifact(s), 1.0 MB" in out
    assert "rec_a  expired 2026-01-02" in out
    assert str(present) in out and str(gone) in out
    assert "Re-run with --execute" in out

    lines = out.splitlines()
    gone_line = next(ln for ln in lines if "rec_b" in ln)
    present_line = next(ln for ln in lines if "rec_a" in ln)
    assert "[already gone]" in gone_line
    assert "[already gone]" not in present_line


def test_a_live_plan_is_labelled_live_and_does_not_offer_execute(tmp_path):
    plan = RetentionPlan(items=[_item("rec_a", tmp_path / "x.enc", True)], dry_run=False)
    out = plan.render()
    assert "(LIVE)" in out
    assert "--execute" not in out


def test_the_plan_names_the_audit_cutoff_when_rows_will_go(tmp_path):
    cutoff = datetime(2024, 3, 1, tzinfo=timezone.utc)
    with_cutoff = RetentionPlan(audit_rows=3, audit_cutoff=cutoff).render()
    assert "audit trail  3 row(s) older than 2024-03-01" in with_cutoff
    assert not RetentionPlan(audit_rows=3, audit_cutoff=cutoff).empty

    without = RetentionPlan(audit_rows=3, audit_cutoff=None).render()
    assert "audit trail  3 row(s)" in without
    assert "older than" not in without


# =========================================================================
# Expiry windows per kind
# =========================================================================
def test_audit_artifacts_expire_on_the_profile_audit_window(tmp_path, monkeypatch):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    db = Database(cfg.path("database"))
    try:
        sweeper = RetentionSweeper(cfg, db)
        profile = cfg.profile("insurance_agent")
        created = datetime(2026, 1, 1, tzinfo=timezone.utc)
        assert sweeper.expires_at("audit", profile, created) == (
            created + timedelta(days=profile.audit_log_days)
        )
    finally:
        db.close()


def test_an_unknown_kind_falls_back_to_the_pipeline_default_window(tmp_path, monkeypatch):
    cfg, _ = build_sandbox(tmp_path, monkeypatch,
                           overrides={"retention": {"default_transcript_days": 7}})
    db = Database(cfg.path("database"))
    try:
        sweeper = RetentionSweeper(cfg, db)
        created = datetime(2026, 1, 1, tzinfo=timezone.utc)
        assert sweeper.expires_at("digest", cfg.profile("father"), created) == (
            created + timedelta(days=7)
        )
    finally:
        db.close()


def test_a_zero_day_window_means_the_artifact_never_expires(tmp_path, monkeypatch):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    db = Database(cfg.path("database"))
    try:
        forever = dataclasses.replace(cfg.profile("insurance_agent"), audit_log_days=0)
        assert RetentionSweeper(cfg, db).expires_at("audit", forever) is None
    finally:
        db.close()


# =========================================================================
# Disabled means disabled
# =========================================================================
def test_a_disabled_sweeper_plans_nothing_even_when_artifacts_have_expired(
    tmp_path, monkeypatch
):
    cfg, _ = build_sandbox(tmp_path, monkeypatch, overrides={"retention": {"enabled": False}})
    db = Database(cfg.path("database"))
    try:
        _indexed(db, "rec_old")
        artifact = cfg.path("vault") / "rec_old.transcript.md.enc"
        artifact.write_bytes(b"PBV1 pretend")
        db.record_artifact("rec_old", "transcript", str(artifact), True,
                           datetime.now(timezone.utc) - timedelta(days=1))
        assert db.expired_artifacts(), "the fixture artifact is not actually expired"

        sweeper = RetentionSweeper(cfg, db)
        assert sweeper.enabled is False
        plan = sweeper.plan(dry_run=False)
        assert plan.empty
        assert plan.items == [] and plan.audit_rows == 0
        assert sweeper.execute(plan) == 0
        assert artifact.exists()
        assert db.all_artifacts(), "a disabled sweep dropped the index row"
    finally:
        db.close()


# =========================================================================
# Each branch of execute
# =========================================================================
def test_a_file_that_will_not_unlink_stays_in_the_index_and_is_not_audited_as_deleted(
    tmp_path, monkeypatch
):
    """
    A directory where the index expected a file is one way unlink can fail. The
    sweep must not drop the row or write a `retention_delete` entry for a thing
    that is still on disk -- that would be an audit trail describing a deletion
    that never happened.
    """
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    db = Database(cfg.path("database"))
    try:
        _indexed(db, "rec_stuck")
        stuck = cfg.path("vault") / "rec_stuck.transcript.md.enc"
        stuck.mkdir(parents=True)
        db.record_artifact("rec_stuck", "transcript", str(stuck), True,
                           datetime.now(timezone.utc) - timedelta(days=1))

        sweeper = RetentionSweeper(cfg, db)
        removed = sweeper.execute(sweeper.plan(dry_run=False))

        assert removed == 0
        assert stuck.exists()
        assert [a["recording_id"] for a in db.all_artifacts()] == ["rec_stuck"], (
            "the index row was dropped for a file that is still on disk"
        )
        assert not db.audit_log(action="retention_delete", limit=10), (
            "the audit trail claims a deletion that did not happen"
        )
    finally:
        db.close()


def test_an_artifact_already_gone_from_disk_still_has_its_index_row_cleared(
    tmp_path, monkeypatch
):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    db = Database(cfg.path("database"))
    try:
        _indexed(db, "rec_gone")
        missing = cfg.path("vault") / "rec_gone.transcript.md.enc"
        db.record_artifact("rec_gone", "transcript", str(missing), True,
                           datetime.now(timezone.utc) - timedelta(days=1))

        sweeper = RetentionSweeper(cfg, db)
        plan = sweeper.plan(dry_run=False)
        assert plan.items[0].exists is False
        assert sweeper.execute(plan) == 0

        assert db.all_artifacts() == []
        trail = db.audit_log(action="retention_delete", recording_id="rec_gone", limit=10)
        assert trail and "kind=transcript" in trail[0]["detail"]
    finally:
        db.close()
