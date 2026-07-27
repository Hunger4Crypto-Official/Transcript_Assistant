"""
Reading the archive back: search, verify, forget, export.

A transcript archive you cannot search is a filing cabinet you cannot open, and
an encrypted archive you have never tried to decrypt is one you may already have
lost. These test the paths that read stored content back out, including the
awkward ones — encrypted, corrupted, and locked.
"""


import pytest

from _fixtures import CLIENT_CALL, FAMILY_DINNER, build_sandbox, drop
from plaud_bridge.archive import Archive
from plaud_bridge.cli import build_parser, cmd_export, cmd_forget, cmd_search, cmd_verify
from plaud_bridge.db import Database
from plaud_bridge.pipeline import Pipeline


def _processed(tmp_path, monkeypatch, files=(("client-marcus.txt", CLIENT_CALL),)):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    for name, body in files:
        drop(cfg, name, body)
    pipe = Pipeline(cfg)
    try:
        pipe.run()
    finally:
        pipe.close()
    return cfg


def _args(cfg, *argv):
    return build_parser().parse_args(["--config", str(cfg.root / "config"), *argv])


# =========================================================================
# Content search
# =========================================================================
def test_content_search_finds_words_inside_an_encrypted_transcript(tmp_path, monkeypatch):
    """insurance_agent encrypts at rest, so this only works by decrypting."""
    cfg = _processed(tmp_path, monkeypatch)
    db = Database(cfg.path("database"))
    try:
        result = Archive(cfg, db).search_content("elimination period")
        assert result.complete
        assert result.matches, "the phrase is in the transcript and search did not find it"
        assert "elimination period" in result.matches[0].text.lower()
        assert result.matches[0].stamp
    finally:
        db.close()


def test_content_search_is_case_insensitive_and_reports_the_speaker(tmp_path, monkeypatch):
    cfg = _processed(tmp_path, monkeypatch)
    db = Database(cfg.path("database"))
    try:
        result = Archive(cfg, db).search_content("ELIMINATION PERIOD")
        assert result.matches
        assert result.matches[0].speaker == "Sasson"
    finally:
        db.close()


def test_content_search_says_what_it_could_not_open(tmp_path, monkeypatch):
    """
    Under-reporting is the dangerous failure. If a file will not decrypt, the
    user must not conclude the phrase was never said.
    """
    cfg = _processed(tmp_path, monkeypatch)
    monkeypatch.setenv("PLAUD_BRIDGE_PASSPHRASE", "a-completely-different-passphrase")

    db = Database(cfg.path("database"))
    try:
        result = Archive(cfg, db).search_content("elimination period")
        assert not result.matches
        assert result.unopened, "search silently returned zero hits for a file it could not read"
        assert not result.complete
    finally:
        db.close()


def test_content_search_marks_personal_recordings(tmp_path, monkeypatch):
    cfg = _processed(tmp_path, monkeypatch,
                     files=(("dinner-with-kid.txt", FAMILY_DINNER),))
    db = Database(cfg.path("database"))
    try:
        result = Archive(cfg, db).search_content("permission slip")
        assert result.matches
        assert result.matches[0].personal is True
    finally:
        db.close()


def test_search_command_exits_nonzero_when_something_could_not_be_read(
    tmp_path, monkeypatch, capsys
):
    cfg = _processed(tmp_path, monkeypatch)
    monkeypatch.setenv("PLAUD_BRIDGE_PASSPHRASE", "a-completely-different-passphrase")
    assert cmd_search(_args(cfg, "search", "elimination", "--content")) == 2
    assert "could not be opened" in capsys.readouterr().out


# =========================================================================
# Verify
# =========================================================================
def test_verify_passes_on_a_healthy_archive(tmp_path, monkeypatch):
    cfg = _processed(tmp_path, monkeypatch)
    db = Database(cfg.path("database"))
    try:
        report = Archive(cfg, db).verify()
        assert report.checked
        assert report.healthy
        assert not report.problems
    finally:
        db.close()


def test_verify_detects_a_silently_corrupted_artifact(tmp_path, monkeypatch):
    cfg = _processed(tmp_path, monkeypatch)
    db = Database(cfg.path("database"))
    try:
        from pathlib import Path

        target = Path(db.all_artifacts()[0]["path"])
        blob = bytearray(target.read_bytes())
        blob[-3] ^= 0xFF
        target.write_bytes(bytes(blob))

        report = Archive(cfg, db).verify()
        assert not report.healthy
        assert any(p.status == "undecryptable" for p in report.problems)
    finally:
        db.close()


def test_verify_detects_a_missing_artifact(tmp_path, monkeypatch):
    cfg = _processed(tmp_path, monkeypatch)
    db = Database(cfg.path("database"))
    try:
        from pathlib import Path

        Path(db.all_artifacts()[0]["path"]).unlink()
        report = Archive(cfg, db).verify()
        assert any(p.status == "missing" for p in report.problems)
    finally:
        db.close()


def test_verify_does_not_claim_health_it_could_not_confirm(tmp_path, monkeypatch, capsys):
    """A locked vault means unchecked, not fine, and the count must say so."""
    cfg = _processed(tmp_path, monkeypatch)
    monkeypatch.delenv("PLAUD_BRIDGE_PASSPHRASE", raising=False)

    assert cmd_verify(_args(cfg, "verify")) == 1
    out = capsys.readouterr().out
    assert "NOT checked" in out
    assert "0 artifact(s) indexed" not in out, (
        "reported nothing indexed when artifacts exist but were not opened"
    )


def test_verify_reports_orphans_without_deleting_them(tmp_path, monkeypatch):
    cfg = _processed(tmp_path, monkeypatch)
    stray = cfg.path("vault") / "left-behind.enc"
    stray.write_bytes(b"not indexed")

    db = Database(cfg.path("database"))
    try:
        report = Archive(cfg, db).verify()
        assert stray in report.orphans
        assert stray.exists(), "verify deleted something; it only reports"
    finally:
        db.close()


# =========================================================================
# Forget
# =========================================================================
def test_forget_removes_every_trace_except_the_audit_entry(tmp_path, monkeypatch, capsys):
    cfg = _processed(tmp_path, monkeypatch)
    db = Database(cfg.path("database"))
    try:
        rec_id = db.query()[0]["id"]
        planned = Archive(cfg, db).plan_forget(rec_id)
        assert planned, "nothing was found to delete"
    finally:
        db.close()

    assert cmd_forget(_args(cfg, "forget", rec_id, "--yes")) == 0

    for path in planned:
        assert not path.exists(), f"{path} survived forget"

    db = Database(cfg.path("database"))
    try:
        assert db.load(rec_id) is None
        assert db.query() == []
        # The audit trail must remember that a deletion happened.
        trail = db.audit_log(recording_id=rec_id)
        assert any(r["action"] == "forget" for r in trail)
        assert all(r["actor"] == "human" for r in trail if r["action"].startswith("forget"))
    finally:
        db.close()


def test_forget_leaves_other_recordings_alone(tmp_path, monkeypatch):
    cfg = _processed(tmp_path, monkeypatch, files=(
        ("client-marcus.txt", CLIENT_CALL),
        ("dinner-with-kid.txt", FAMILY_DINNER),
    ))
    db = Database(cfg.path("database"))
    try:
        rows = db.query()
        assert len(rows) == 2
        doomed, spared = rows[0]["id"], rows[1]["id"]
        spared_files = Archive(cfg, db).plan_forget(spared)
    finally:
        db.close()

    assert cmd_forget(_args(cfg, "forget", doomed, "--yes")) == 0

    db = Database(cfg.path("database"))
    try:
        assert db.load(spared) is not None
        assert all(p.exists() for p in spared_files)
    finally:
        db.close()


def test_forget_on_an_unknown_id_is_an_error(tmp_path, monkeypatch, capsys):
    cfg = _processed(tmp_path, monkeypatch)
    assert cmd_forget(_args(cfg, "forget", "rec_doesnotexist", "--yes")) == 1
    assert "no recording" in capsys.readouterr().out


# =========================================================================
# Export
# =========================================================================
def test_export_refuses_a_personal_profile_by_default(tmp_path, monkeypatch, capsys):
    cfg = _processed(tmp_path, monkeypatch, files=(("dinner-with-kid.txt", FAMILY_DINNER),))
    assert cmd_export(_args(cfg, "export", "--profile", "father")) == 1
    assert "excluded from exports" in capsys.readouterr().out


def test_export_omits_personal_recordings_from_a_combined_export(tmp_path, monkeypatch, capsys):
    cfg = _processed(tmp_path, monkeypatch, files=(
        ("client-marcus.txt", CLIENT_CALL),
        ("dinner-with-kid.txt", FAMILY_DINNER),
    ))
    assert cmd_export(_args(cfg, "export", "--days", "30")) == 0
    out = capsys.readouterr().out
    assert "permission slip" not in out
    assert "personal recording(s) omitted" in out


def test_export_never_includes_a_suppressed_field(tmp_path, monkeypatch, capsys):
    """
    An export leaves the machine. It is the last place to make an exception for
    a field the profile said never renders.
    """
    cfg = _processed(tmp_path, monkeypatch)
    assert cmd_export(_args(cfg, "export", "--days", "30")) == 0
    out = capsys.readouterr().out
    assert "four hundred thousand" not in out, "a suppressed field reached an export"
    assert "Send two quote options" in out, "the export contained nothing useful"


def test_export_redacts_transcripts_and_says_it_did(tmp_path, monkeypatch, capsys):
    cfg = _processed(tmp_path, monkeypatch)
    assert cmd_export(_args(cfg, "export", "--days", "30", "--transcripts")) == 0
    out = capsys.readouterr().out
    assert "elimination period" in out, "the transcript body is missing"
    assert "Redaction is pattern matching" in out
    assert "not a guarantee" in out, "the export overstated what redaction does"


def test_export_html_is_self_contained(tmp_path, monkeypatch, capsys):
    cfg = _processed(tmp_path, monkeypatch)
    assert cmd_export(_args(cfg, "export", "--days", "30", "--format", "html")) == 0
    out = capsys.readouterr().out
    assert out.lstrip().startswith("<!doctype html>")
    assert "<script" not in out


# =========================================================================
# Watch
# =========================================================================
def test_watch_processes_the_inbox_and_can_run_once(tmp_path, monkeypatch, capsys):
    from plaud_bridge.cli import cmd_watch

    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    drop(cfg, "client-marcus.txt", CLIENT_CALL)

    assert cmd_watch(_args(cfg, "watch", "--once")) == 0

    db = Database(cfg.path("database"))
    try:
        assert len(db.query()) == 1
    finally:
        db.close()


@pytest.mark.parametrize("command", ["verify", "export", "watch"])
def test_the_new_commands_are_reachable_from_the_parser(command):
    args = build_parser().parse_args([command] if command != "watch" else [command, "--once"])
    assert args.func is not None
