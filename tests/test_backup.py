"""
Backup and restore: the vault's answer to a dead disk.

Two promises are pinned down here, and they pull in opposite directions. The
backup must bring EVERYTHING back -- vault, index, sidecar state, outbox,
quarantine, config -- proven by wiping the data directories and restoring onto
the bare floor. And the backup file itself must give NOTHING away: it is meant
for an external drive or a cloud folder, so a phrase spoken on an encrypted
recording must not be findable in its bytes, and a tampered or wrongly-keyed
file must fail with the vault's honest error while restoring nothing at all.
"""

from __future__ import annotations

import shutil

from _fixtures import CLIENT_CALL, FAMILY_DINNER, build_sandbox, drop
from plaud_bridge.archive import Archive
from plaud_bridge.cli import main
from plaud_bridge.db import Database
from plaud_bridge.storage.vault import MAGIC_STREAM


def cli(cfg, *argv) -> int:
    return main(["--config", str(cfg.root / "config"), *argv])


def _processed(tmp_path, monkeypatch, files):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    for name, body in files:
        drop(cfg, name, body)
    assert cli(cfg, "run") == 0
    return cfg


def _recording_ids(cfg) -> set[str]:
    db = Database(cfg.path("database"))
    try:
        return {r["id"] for r in db.query(limit=50)}
    finally:
        db.close()


def _wipe_data(cfg) -> None:
    """
    The dead-disk scenario, minus the parts a backup never claimed to hold:
    inbox and work are transient, and the logs directory stays because the
    logging handler in this process holds its file open.
    """
    for name in ("vault", "outbox", "quarantine"):
        shutil.rmtree(cfg.path(name), ignore_errors=True)
    db = cfg.path("database")
    for path in list(db.parent.iterdir()):
        if path.is_file():
            path.unlink()
    assert not db.exists()


# =========================================================================
# The round trip
# =========================================================================
def test_round_trip_survives_a_wiped_data_directory(tmp_path, monkeypatch):
    cfg = _processed(tmp_path, monkeypatch,
                     (("client-marcus.txt", CLIENT_CALL), ("dinner.txt", FAMILY_DINNER)))
    ids_before = _recording_ids(cfg)
    assert ids_before, "the pipeline processed nothing; the round trip would prove nothing"

    out = tmp_path / "backup.pbb"
    assert cli(cfg, "backup", "--out", str(out)) == 0
    assert out.is_file() and out.stat().st_size > 0

    _wipe_data(cfg)
    assert not cfg.path("vault").exists()

    # Onto the bare floor, so no --force is needed and none is given.
    assert cli(cfg, "restore", str(out)) == 0

    db = Database(cfg.path("database"))
    try:
        archive = Archive(cfg, db)

        # Every indexed artifact exists AND decrypts -- verify() opens each
        # one, so a healthy report is the proof the encrypted copies came back
        # as themselves and not as bytes that merely exist.
        report = archive.verify()
        assert report.healthy, report.render()

        # The words themselves are reachable again, through the vault.
        result = archive.search_content("elimination period")
        assert result.complete
        assert result.matches, "a phrase said on tape was lost in the round trip"

        assert _recording_ids(cfg) == ids_before

        # Both halves of the story are on the audit trail: the backup entry
        # rode along inside the backup, the restore entry was written after.
        assert db.audit_log(action="backup", actor="human", limit=10)
        assert db.audit_log(action="restore", actor="human", limit=10)
    finally:
        db.close()


# =========================================================================
# What the file gives away
# =========================================================================
def test_the_backup_file_holds_no_plaintext_of_encrypted_content(tmp_path, monkeypatch):
    """
    FAMILY_DINNER routes to the father profile, which is code-enforced
    local-only and encrypted at rest. A backup destined for someone else's
    cloud must not demote that: the words must not appear in the file's bytes.
    """
    cfg = _processed(tmp_path, monkeypatch, (("dinner.txt", FAMILY_DINNER),))
    out = tmp_path / "backup.pbb"
    assert cli(cfg, "backup", "--out", str(out)) == 0

    blob = out.read_bytes()
    assert blob.startswith(MAGIC_STREAM), "the bundle is not in the vault's stream format"

    for phrase in (b"permission slip", b"pizza after the game", b"starting on Saturday"):
        assert phrase in FAMILY_DINNER.encode(), "the fixture changed; pick a phrase it contains"
        assert phrase not in blob, f"plaintext {phrase!r} leaked into the backup file"


# =========================================================================
# Fail closed
# =========================================================================
def test_backup_refuses_without_a_passphrase(tmp_path, monkeypatch, capsys):
    cfg = _processed(tmp_path, monkeypatch, (("client-marcus.txt", CLIENT_CALL),))
    out = tmp_path / "backup.pbb"

    monkeypatch.delenv("PLAUD_BRIDGE_PASSPHRASE", raising=False)
    assert cli(cfg, "backup", "--out", str(out)) == 1
    assert not out.exists(), "a refusal must not leave a file behind"
    text = capsys.readouterr()
    assert "PLAUD_BRIDGE_PASSPHRASE" in (text.out + text.err)

    # A passphrase too short to trust is the same refusal, not a weaker vault.
    monkeypatch.setenv("PLAUD_BRIDGE_PASSPHRASE", "short")
    assert cli(cfg, "backup", "--out", str(out)) == 1
    assert not out.exists()


def test_restore_refuses_existing_data_without_force_and_replaces_with_it(
        tmp_path, monkeypatch, capsys):
    cfg = _processed(tmp_path, monkeypatch,
                     (("client-marcus.txt", CLIENT_CALL), ("dinner.txt", FAMILY_DINNER)))
    out = tmp_path / "backup.pbb"
    assert cli(cfg, "backup", "--out", str(out)) == 0

    vault_before = {p: p.stat().st_mtime_ns
                    for p in cfg.path("vault").rglob("*") if p.is_file()}
    ids_before = _recording_ids(cfg)
    capsys.readouterr()

    # Refusal: exit 1, names --force, and touched nothing.
    assert cli(cfg, "restore", str(out)) == 1
    text = capsys.readouterr()
    assert "--force" in (text.out + text.err)
    after = {p: p.stat().st_mtime_ns for p in cfg.path("vault").rglob("*") if p.is_file()}
    assert after == vault_before, "a refused restore modified the vault anyway"
    assert _recording_ids(cfg) == ids_before

    # --force goes through and says exactly what it replaced.
    assert cli(cfg, "restore", str(out), "--force") == 0
    text = capsys.readouterr().out
    assert "replaced" in text
    assert str(cfg.path("vault")) in text
    assert str(cfg.path("database")) in text
    assert _recording_ids(cfg) == ids_before
    assert cli(cfg, "verify") == 0


def test_a_tampered_backup_fails_loudly_and_restores_nothing(tmp_path, monkeypatch, capsys):
    cfg = _processed(tmp_path, monkeypatch, (("client-marcus.txt", CLIENT_CALL),))
    out = tmp_path / "backup.pbb"
    assert cli(cfg, "backup", "--out", str(out)) == 0

    _wipe_data(cfg)

    # One flipped bit in the ciphertext body. The AEAD tag has to catch it.
    blob = bytearray(out.read_bytes())
    blob[len(blob) // 2] ^= 0xFF
    out.write_bytes(bytes(blob))

    capsys.readouterr()
    assert cli(cfg, "restore", str(out)) == 1
    text = capsys.readouterr()
    assert "modified" in (text.out + text.err), "the vault's honest error did not surface"

    # Nothing was written anywhere: no index, no vault, no partial archive
    # pretending to be a whole one.
    assert not cfg.path("database").exists()
    assert not any(cfg.path("vault").rglob("*")) if cfg.path("vault").exists() else True
    assert not cfg.path("outbox").exists() or not any(cfg.path("outbox").rglob("*"))


def test_restore_without_force_leaves_the_config_directory_alone(tmp_path, monkeypatch):
    """
    The config named the destinations, so a bare restore keeps it: a tuned
    profile must never be silently undone by restoring data next to it.
    """
    cfg = _processed(tmp_path, monkeypatch, (("client-marcus.txt", CLIENT_CALL),))
    out = tmp_path / "backup.pbb"
    assert cli(cfg, "backup", "--out", str(out)) == 0

    marker = cfg.root / "config" / "profiles" / "father.yaml"
    tuned = marker.read_text(encoding="utf-8") + "\n# tuned after the backup was taken\n"
    marker.write_text(tuned, encoding="utf-8")

    _wipe_data(cfg)
    assert cli(cfg, "restore", str(out)) == 0
    assert marker.read_text(encoding="utf-8") == tuned, "restore rolled back a config edit"

    # And with --force the backup's copy wins, which is the stated trade.
    assert cli(cfg, "restore", str(out), "--force") == 0
    assert marker.read_text(encoding="utf-8") != tuned
