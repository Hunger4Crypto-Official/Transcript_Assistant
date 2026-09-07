"""
Backup and restore at the edges.

test_backup.py proves the round trip and the two headline refusals. These pin
the rest of restore's "change NOTHING" promise against bundles that decrypt
but are wrong in some other way -- a member that points outside the archive, a
symlink, a missing or lying manifest, a format from the future -- and the one
piece the round-trip test never exercised: the sidecar state file beside the
index.
"""

from __future__ import annotations

import io
import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from _fixtures import CLIENT_CALL, build_sandbox, drop
from plaud_bridge.backup import (
    BACKUP_AAD,
    FORMAT,
    FORMAT_VERSION,
    MANIFEST_NAME,
    BackupError,
    _state_files,
    create_backup,
    default_backup_path,
    restore_backup,
)
from plaud_bridge.db import Database
from plaud_bridge.followups import FollowUp, _load_state, set_status, stable_id
from plaud_bridge.pipeline import Pipeline
from plaud_bridge.storage import Vault


# =========================================================================
# Crafting a bundle by hand
# =========================================================================
def _crafted(tmp_path, manifest, files: dict[str, bytes] | None = None, extra=None) -> Path:
    """
    An encrypted .pbb whose tar holds exactly what the test says. `manifest`
    is a dict (written as JSON), a raw string (written verbatim), or None.
    `extra(tar)` may add members that `tar.add` would normalise away.
    """
    src = tmp_path / "craft-src"
    src.mkdir(exist_ok=True)
    tar_path = tmp_path / "craft.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        if manifest is not None:
            body = manifest if isinstance(manifest, str) else json.dumps(manifest)
            m = src / MANIFEST_NAME
            m.write_text(body, encoding="utf-8")
            tar.add(m, arcname=MANIFEST_NAME)
        for arcname, data in (files or {}).items():
            p = src / arcname
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
            tar.add(p, arcname=arcname)
        if extra is not None:
            extra(tar)

    out = tmp_path / "crafted.pbb"
    staged = Vault(tmp_path / "envelope").write_stream("bundle", tar_path, recording_id=BACKUP_AAD)
    shutil.move(str(staged), str(out))
    return out


def _good_manifest(pieces: dict[str, int]) -> dict:
    return {
        "format": FORMAT, "format_version": FORMAT_VERSION,
        "created_at": "2026-01-01T00:00:00+00:00", "tool_version": "test",
        "pieces": {name: {"files": n} for name, n in pieces.items()},
    }


def _nothing_restored(cfg) -> None:
    assert not cfg.path("database").exists(), "an index appeared"
    for name in ("vault", "outbox", "quarantine"):
        root = cfg.path(name)
        assert not root.exists() or not any(p.is_file() for p in root.rglob("*")), (
            f"{name} was written to by a refused restore"
        )


@pytest.fixture
def bare(tmp_path, monkeypatch):
    """A sandbox that has never run: empty data directories, no index."""
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    assert not cfg.path("database").exists()
    return cfg


def _restore(cfg, source, force=False):
    return restore_backup(cfg, cfg.root / "config", source, force=force)


# =========================================================================
# Backup-side edges
# =========================================================================
def test_state_files_is_empty_when_the_index_directory_does_not_exist(tmp_path):
    assert _state_files(tmp_path / "never" / "bridge.db") == []


def test_the_default_backup_path_is_dated_and_lands_in_the_home_directory():
    when = datetime(2026, 3, 4, 5, 6, 7, tzinfo=timezone.utc)
    path = default_backup_path(when)
    assert path.name == "plaud-backup-20260304-050607.pbb"
    assert path.parent == Path.home()
    assert default_backup_path().suffix == ".pbb"


def test_restoring_a_file_that_does_not_exist_is_refused_before_anything_is_read(
    tmp_path, bare
):
    with pytest.raises(BackupError, match="no such file"):
        _restore(bare, tmp_path / "never-written.pbb")
    _nothing_restored(bare)


def test_backing_up_an_install_that_never_ran_is_refused_as_nothing_to_back_up(
    tmp_path, bare
):
    """
    A backup of nothing is worse than no backup: it looks like a backup. With
    empty data directories and an empty config directory there is no piece to
    put in the manifest, and no file is left at the output path.
    """
    empty_config = tmp_path / "empty-config"
    empty_config.mkdir()
    out = tmp_path / "nothing.pbb"
    with pytest.raises(BackupError, match="nothing to back up"):
        create_backup(bare, empty_config, out)
    assert not out.exists()


# =========================================================================
# Members that are not allowed to exist
# =========================================================================
def test_a_member_that_points_outside_the_archive_is_refused_whole(tmp_path, bare):
    def escape(tar):
        info = tarfile.TarInfo("../escape.txt")
        info.size = 4
        tar.addfile(info, io.BytesIO(b"oops"))

    bundle = _crafted(tmp_path, _good_manifest({"vault": 1}),
                      {"vault/x.enc": b"PBV1"}, extra=escape)
    with pytest.raises(BackupError, match="points outside the archive"):
        _restore(bare, bundle)
    _nothing_restored(bare)
    assert not (tmp_path / "escape.txt").exists()


def test_an_absolute_member_path_is_refused(tmp_path, bare):
    def absolute(tar):
        info = tarfile.TarInfo("/etc/evil")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))

    bundle = _crafted(tmp_path, _good_manifest({}), extra=absolute)
    with pytest.raises(BackupError, match="'/etc/evil'.*points outside"):
        _restore(bare, bundle)
    _nothing_restored(bare)


def test_a_symlink_inside_the_archive_is_refused(tmp_path, bare):
    def link(tar):
        info = tarfile.TarInfo("vault/link.enc")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tar.addfile(info)

    bundle = _crafted(tmp_path, _good_manifest({"vault": 1}), extra=link)
    with pytest.raises(BackupError, match="not a regular file or directory"):
        _restore(bare, bundle)
    _nothing_restored(bare)


# =========================================================================
# The manifest is checked, not trusted
# =========================================================================
def test_a_bundle_that_decrypts_but_holds_no_manifest_is_not_a_backup(tmp_path, bare):
    bundle = _crafted(tmp_path, None, {"vault/x.enc": b"PBV1"})
    with pytest.raises(BackupError, match="holds no manifest"):
        _restore(bare, bundle)
    _nothing_restored(bare)


def test_a_manifest_that_is_not_json_is_refused(tmp_path, bare):
    bundle = _crafted(tmp_path, "{not json", {"vault/x.enc": b"PBV1"})
    with pytest.raises(BackupError, match="manifest is not valid JSON"):
        _restore(bare, bundle)
    _nothing_restored(bare)


def test_a_manifest_from_some_other_tool_is_refused(tmp_path, bare):
    manifest = _good_manifest({})
    manifest["format"] = "somebody-elses-backup"
    bundle = _crafted(tmp_path, manifest)
    with pytest.raises(BackupError, match="manifest is not a plaud-bridge backup's"):
        _restore(bare, bundle)
    _nothing_restored(bare)


def test_a_backup_written_by_a_newer_tool_is_refused_with_an_upgrade_hint(tmp_path, bare):
    manifest = _good_manifest({})
    manifest["format_version"] = FORMAT_VERSION + 1
    bundle = _crafted(tmp_path, manifest)
    with pytest.raises(BackupError, match="newer plaud-bridge") as exc:
        _restore(bare, bundle)
    assert "Upgrade the tool" in str(exc.value)
    assert f"v{FORMAT_VERSION + 1}" in str(exc.value)
    _nothing_restored(bare)


def test_a_manifest_naming_a_piece_this_build_does_not_know_is_refused(tmp_path, bare):
    bundle = _crafted(tmp_path, _good_manifest({"inbox": 1}), {"inbox/raw.mp3": b"RIFF"})
    with pytest.raises(BackupError, match=r"does not know: \['inbox'\]"):
        _restore(bare, bundle)
    _nothing_restored(bare)


def test_a_manifest_that_disagrees_with_the_archive_restores_nothing(tmp_path, bare):
    """
    Two files promised, one present. Restoring the one that is there would make
    a partial archive look like the whole one, so the whole thing is refused.
    """
    bundle = _crafted(tmp_path, _good_manifest({"vault": 2}), {"vault/only.enc": b"PBV1"})
    with pytest.raises(BackupError, match="says 'vault' holds 2 file\\(s\\) but the archive holds 1"):
        _restore(bare, bundle)
    _nothing_restored(bare)


# =========================================================================
# The sidecar state file beside the index
# =========================================================================
def _with_state(tmp_path, monkeypatch):
    """A processed sandbox whose followups.state holds one done status."""
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    drop(cfg, "client-marcus.txt", CLIENT_CALL)
    pipe = Pipeline(cfg)
    try:
        pipe.run()
        rec_id = pipe.db.query()[0]["id"]
    finally:
        pipe.close()

    vault = Vault(cfg.path("vault"))
    fu = FollowUp(id=stable_id("send two quote options", "insurance_agent"),
                  text="send two quote options", profile_id="insurance_agent",
                  recording_id=rec_id, first_seen="2026-01-01")
    set_status(cfg, vault, fu.id, "done", items=[fu])
    state = cfg.path("database").parent / "followups.state"
    assert state.exists()
    return cfg, vault, state, fu.id


def _wipe_except(cfg, keep: Path | None) -> None:
    for name in ("vault", "outbox", "quarantine"):
        shutil.rmtree(cfg.path(name), ignore_errors=True)
    for path in list(cfg.path("database").parent.iterdir()):
        if path.is_file() and path != keep:
            path.unlink()


def test_the_sidecar_state_file_is_backed_up_and_comes_back_intact(tmp_path, monkeypatch):
    cfg, vault, state, fid = _with_state(tmp_path, monkeypatch)
    before = state.read_bytes()

    out = tmp_path / "backup.pbb"
    report = create_backup(cfg, cfg.root / "config", out)
    assert report.counts.get("state") == 1
    assert "state" not in report.skipped

    _wipe_except(cfg, keep=None)
    assert not state.exists()

    restored = _restore(cfg, out)
    assert restored.restored.get("state") == 1
    assert restored.replaced == []
    assert state.read_bytes() == before
    assert _load_state(cfg, vault)[fid]["status"] == "done", (
        "the done status did not survive the round trip"
    )


def test_restore_refuses_to_overwrite_a_sidecar_state_file_without_force(
    tmp_path, monkeypatch
):
    cfg, _vault, state, _fid = _with_state(tmp_path, monkeypatch)
    out = tmp_path / "backup.pbb"
    create_backup(cfg, cfg.root / "config", out)

    # Everything gone but the state file: the one thing left is the one thing
    # the restore would overwrite.
    _wipe_except(cfg, keep=state)
    edited = b"PBV1 edited after the backup was taken"
    state.write_bytes(edited)

    with pytest.raises(BackupError, match="would overwrite data already in place") as exc:
        _restore(cfg, out)
    assert f"state -> {state}" in str(exc.value)
    assert state.read_bytes() == edited, "a refused restore replaced the state file anyway"
    assert not cfg.path("database").exists(), "a refused restore brought the index back"


def test_force_replaces_the_sidecar_state_file_and_says_so(tmp_path, monkeypatch):
    cfg, vault, state, fid = _with_state(tmp_path, monkeypatch)
    before = state.read_bytes()
    out = tmp_path / "backup.pbb"
    create_backup(cfg, cfg.root / "config", out)

    _wipe_except(cfg, keep=state)
    state.write_bytes(b"PBV1 edited after the backup was taken")

    report = _restore(cfg, out, force=True)
    # Two things were in place to be replaced: the state file, and the config
    # directory that --force always takes the backup's copy of.
    assert sorted(report.replaced) == sorted([str(state), str(cfg.root / "config")])
    assert report.restored.get("state") == 1
    assert state.read_bytes() == before
    assert _load_state(cfg, vault)[fid]["status"] == "done"

    db = Database(cfg.path("database"))
    try:
        trail = db.audit_log(action="restore", actor="human", limit=5)
        assert trail and "replaced 2 existing path(s)" in trail[0]["detail"]
    finally:
        db.close()
