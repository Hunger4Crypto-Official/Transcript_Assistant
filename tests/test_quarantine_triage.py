"""
The quarantine triage surface: `run.py quarantine`.

A backlog run quarantines in bulk, correctly, and these tests pin down the
promises that keep the bulk tools from becoming a rubber stamp: the listing
tells a refusal apart from a missing announcement, --release-all never touches
a refusal and says so, --forget-all routes through Archive.forget so every
trace goes, and neither bulk verb does anything without the typed phrase or an
explicit --yes. If one of these fails, the failure is a consent-policy
regression, not a formatting bug.
"""

from __future__ import annotations

import pytest

from _fixtures import CLIENT_CALL, build_sandbox, drop
from plaud_bridge.archive import Archive
from plaud_bridge.cli import main
from plaud_bridge.db import Database

# The other party explicitly objects inside the consent window. Same shape as
# the REFUSAL fixture in test_privacy_guarantees.py: the work keywords route it
# to insurance_agent, whose require_consent sends it through the gate.
REFUSAL = """\
Sasson: Hey Marcus, before we get started I record these calls for my notes. Is that okay?
Marcus: Hold on, I really don't want this being recorded.
Sasson: Understood, no problem.
Marcus: So about that term policy through work, the elimination period question.
Sasson: Your income is the asset here, not the house. Disability matters.
"""

# The consent exchange stripped from the client call: nobody asked on tape, so
# the gate quarantines for a missing announcement rather than for an objection.
NO_ANNOUNCEMENT = CLIENT_CALL.split("\n", 2)[2]


@pytest.fixture
def held(tmp_path, monkeypatch):
    """A sandbox where the backlog run has quarantined one of each class."""
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    drop(cfg, "client-refused.txt", REFUSAL)
    drop(cfg, "client-unannounced.txt", NO_ANNOUNCEMENT)
    assert main(["--config", str(tmp_path / "config"), "run"]) == 0

    db = Database(cfg.path("database"))
    try:
        rows = db.query(stage="quarantined", limit=10)
    finally:
        db.close()
    ids = {r["source_name"]: r["id"] for r in rows}
    assert set(ids) == {"client-refused.txt", "client-unannounced.txt"}, (
        "the fixtures did not both quarantine; these tests have nothing to triage"
    )
    return cfg, ids["client-refused.txt"], ids["client-unannounced.txt"]


def cli(cfg, *argv) -> int:
    return main(["--config", str(cfg.root / "config"), *argv])


def released_ids(cfg) -> set[str]:
    db = Database(cfg.path("database"))
    try:
        return {r["recording_id"] for r in db.audit_log(action="quarantine_release", limit=50)}
    finally:
        db.close()


def loaded(cfg, recording_id) -> dict | None:
    db = Database(cfg.path("database"))
    try:
        return db.load(recording_id)
    finally:
        db.close()


# =========================================================================
# The listing
# =========================================================================
def test_the_listing_tells_a_refusal_apart_from_a_missing_announcement(held, capsys):
    cfg, refused, unannounced = held
    assert cli(cfg, "quarantine") == 0
    out = capsys.readouterr().out

    assert refused in out and unannounced in out
    # The two rows must carry distinguishable reasons, not one shared shrug.
    assert "objected to being recorded" in out
    assert "REFUSED" in out
    # Refusals lead the listing and are labelled as out of bounds for bulk
    # release, so a skim cannot mistake them for one more row to wave through.
    assert out.index(refused) < out.index(unannounced)
    assert "Never included in --release-all" in out


def test_the_run_summary_points_at_the_triage_command(tmp_path, monkeypatch, capsys):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    drop(cfg, "client-unannounced.txt", NO_ANNOUNCEMENT)
    assert main(["--config", str(tmp_path / "config"), "run"]) == 0
    assert "run.py quarantine" in capsys.readouterr().out


# =========================================================================
# Bulk release
# =========================================================================
def test_release_all_excludes_the_refusal_and_says_so(held, capsys):
    cfg, refused, unannounced = held
    assert cli(cfg, "quarantine", "--release-all", "--yes") == 0
    out = capsys.readouterr().out

    # The exclusion is printed, naming the refusal, not silently applied.
    assert "EXCLUDED" in out
    assert refused in out
    assert "released 1 recording(s)" in out

    # Every release is audited through the existing release path; the refusal
    # never acquired one.
    assert released_ids(cfg) == {unannounced}
    # The released media is back in the inbox; the refusal's stays in quarantine.
    assert (cfg.path("inbox") / "client-unannounced.txt").exists()
    assert (cfg.path("quarantine") / refused / "client-refused.txt").exists()


def test_release_all_does_not_release_the_same_recording_twice(held, capsys):
    cfg, _refused, unannounced = held
    assert cli(cfg, "quarantine", "--release-all", "--yes") == 0
    assert cli(cfg, "quarantine", "--release-all", "--yes") == 0
    assert "nothing eligible" in capsys.readouterr().out

    db = Database(cfg.path("database"))
    try:
        releases = [r for r in db.audit_log(action="quarantine_release", limit=50)
                    if r["recording_id"] == unannounced]
    finally:
        db.close()
    assert len(releases) == 1, "a second --release-all re-audited an already released recording"


# =========================================================================
# Bulk forget
# =========================================================================
def test_forget_all_removes_the_recordings_and_their_traces(held):
    cfg, refused, unannounced = held
    assert cli(cfg, "quarantine", "--forget-all", "--yes") == 0

    for rid in (refused, unannounced):
        assert loaded(cfg, rid) is None, f"{rid} survived --forget-all in the index"
        assert not (cfg.path("quarantine") / rid).exists(), (
            f"{rid} still has a quarantine folder after --forget-all"
        )

    # Archive.plan_forget is the authoritative "what would forget still delete"
    # answer, and after a bulk forget it must have nothing left to name.
    db = Database(cfg.path("database"))
    try:
        archive = Archive(cfg, db)
        for rid in (refused, unannounced):
            assert archive.plan_forget(rid) == []
    finally:
        db.close()


# =========================================================================
# Confirmation discipline
# =========================================================================
def _no_stdin(monkeypatch):
    def refuse(*_args, **_kw):
        raise EOFError("EOF when reading a line")
    monkeypatch.setattr("builtins.input", refuse)


def test_release_all_declines_when_there_is_no_terminal(held, monkeypatch):
    cfg, _refused, _unannounced = held
    _no_stdin(monkeypatch)
    assert cli(cfg, "quarantine", "--release-all") == 1
    assert released_ids(cfg) == set(), "a release happened with nobody there to confirm it"


def test_release_all_declines_on_the_wrong_phrase(held, monkeypatch):
    cfg, _refused, _unannounced = held
    # "RELEASE" is the single-recording phrase; the bulk phrase is deliberately
    # different so muscle memory from `release <id>` cannot confirm a bulk one.
    monkeypatch.setattr("builtins.input", lambda *_: "RELEASE")
    assert cli(cfg, "quarantine", "--release-all") == 1
    assert released_ids(cfg) == set()


def test_release_all_proceeds_on_the_typed_phrase(held, monkeypatch):
    cfg, _refused, unannounced = held
    monkeypatch.setattr("builtins.input", lambda *_: "RELEASE ALL")
    assert cli(cfg, "quarantine", "--release-all") == 0
    assert released_ids(cfg) == {unannounced}


def test_forget_all_declines_when_there_is_no_terminal(held, monkeypatch):
    cfg, refused, unannounced = held
    _no_stdin(monkeypatch)
    assert cli(cfg, "quarantine", "--forget-all") == 1
    for rid in (refused, unannounced):
        assert loaded(cfg, rid) is not None, "forget-all deleted without a confirmation"
        assert (cfg.path("quarantine") / rid).is_dir()
