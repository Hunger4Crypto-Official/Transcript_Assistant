"""
One encrypted file that brings the archive back from a dead disk.

The vault is deliberately unforgiving: the key is never written down, so losing
the disk means losing everything on it. That is the correct behaviour for a
vault and the wrong ending for years of recordings, and nothing else in this
tool copies anything anywhere. This module is the answer, shaped by one rule:
a backup must be safe to sit on an external drive or in a cloud folder that
this tool does not control.

That rule decides the format. Everything worth keeping -- the vault, the SQLite
index, the sidecar state beside it, the outbox, the quarantine, and the config
directory with its hand-tuned profiles -- goes into one tar, and the tar is
encrypted with the vault's own streaming cipher before it touches the output
path. The vault files inside are already ciphertext, but the index and the
outbox hold work-profile content in the clear, so an unencrypted bundle would
quietly demote everything in it to the sensitivity of whatever folder it lands
in. There is no plaintext fallback: no passphrase, no backup, said out loud --
the same fail-closed stance as the vault itself.

What is deliberately left out: data/inbox (transient input the pipeline
consumes), data/work (scratch), and the logs. None of them can be brought back
by a restore in any meaningful sense, and the inbox may hold raw audio the
governing profile would never allow in an unencrypted-at-rest copy -- which the
backup, once decrypted on some other machine, effectively is.

Restore is the mirror image, with the failure mode inverted: a wrong passphrase
or a tampered file must produce the vault's honest error and change NOTHING.
So the file is decrypted, unpacked, and checked against its manifest entirely
in a temp directory, and only a bundle that survived all of that is moved into
place. The half-restored archive this ordering prevents is worse than no
restore at all, because it looks like a whole one.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import tarfile
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .db import Database
from .logging_setup import get
from .storage import Vault

log = get("backup")

# Every backup is bound to this id through the stream cipher's additional data,
# the way an artifact is bound to its recording id. A vault file renamed to
# .pbb, or a backup renamed to look like a recording, fails to open rather than
# quietly decrypting as the wrong kind of thing.
BACKUP_AAD = "backup"

FORMAT = "plaud-bridge-backup"
FORMAT_VERSION = 1
MANIFEST_NAME = "manifest.json"

# SQLite's shadow files. Never copied raw: the snapshot below folds the WAL
# into the database, and a stale -wal restored beside a different .db is how
# SQLite gets handed someone else's uncommitted pages.
_DB_SHADOWS = ("-wal", "-shm", "-journal")

# The directory pieces whose destination is simply cfg.path(<name>).
_DIR_PIECES = ("vault", "outbox", "quarantine")

_KNOWN_PIECES = frozenset(_DIR_PIECES) | {"config", "index", "state"}


class BackupError(RuntimeError):
    """Raised when a backup or restore cannot proceed. Always actionable."""


@dataclass
class BackupReport:
    path: Path
    size_bytes: int
    counts: dict[str, int] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)


@dataclass
class RestoreReport:
    restored: dict[str, int] = field(default_factory=dict)
    replaced: list[str] = field(default_factory=list)
    config_skipped: bool = False


def default_backup_path(now: datetime | None = None) -> Path:
    now = now or datetime.now(timezone.utc)
    return Path.home() / f"plaud-backup-{now:%Y%m%d-%H%M%S}.pbb"


def _state_files(db_path: Path) -> list[Path]:
    """
    The sidecar files that live beside the index -- followups.state today,
    whatever else earns a place there tomorrow. Enumerated rather than named so
    a new sidecar is backed up the day it appears, not the day someone loses it.
    """
    if not db_path.parent.is_dir():
        return []
    shadows = {db_path.name} | {db_path.name + s for s in _DB_SHADOWS}
    return [
        p for p in sorted(db_path.parent.iterdir())
        if p.is_file() and p.name not in shadows
    ]


def _files_under(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return [p for p in sorted(root.rglob("*")) if p.is_file()]


def _require_ready(vault: Vault, doing: str) -> None:
    ok, why = vault.ready()
    if not ok:
        raise BackupError(
            f"cannot {doing}: {why}\n"
            "A backup is one encrypted file and there is no plaintext fallback: "
            "the index and outbox inside it hold conversations in the clear, and "
            "a backup that quietly skipped the encryption would hand them to "
            "whatever drive or cloud folder it lands on."
        )


# =========================================================================
# Backup
# =========================================================================
def create_backup(cfg, config_dir: Path, out: Path,
                  created_at: datetime | None = None) -> BackupReport:
    """
    Collect everything worth keeping into one vault-encrypted file at `out`.

    The audit entry is written BEFORE the index is snapshotted, so the backup
    carries the record of its own creation -- restore it years later and the
    trail still says when this copy was taken.
    """
    out = Path(out)
    db_path = cfg.path("database")

    staging = Path(tempfile.mkdtemp(prefix="plaud-backup-"))
    try:
        _require_ready(Vault(staging / "envelope"), "back up")

        if db_path.exists():
            db = Database(db_path)
            try:
                db.audit("backup", f"writing encrypted backup {out.name}", actor="human")
            finally:
                db.close()

        # A consistent snapshot through SQLite's own backup API, not a file
        # copy. Copying a live WAL-mode database can capture the .db mid-write
        # with its journal half a step ahead, which restores as corruption.
        snapshot: Path | None = None
        if db_path.exists():
            snapshot = staging / db_path.name
            src, dst = sqlite3.connect(str(db_path)), sqlite3.connect(str(snapshot))
            try:
                src.backup(dst)
            finally:
                src.close()
                dst.close()

        # (piece name, source file, name inside the tar). data/inbox and
        # data/work are deliberately absent; see the module docstring.
        members: list[tuple[str, Path, str]] = []
        for name in _DIR_PIECES:
            root = cfg.path(name)
            members += [(name, p, f"{name}/{p.relative_to(root)}") for p in _files_under(root)]
        root = Path(config_dir)
        members += [("config", p, f"config/{p.relative_to(root)}") for p in _files_under(root)]
        if snapshot is not None:
            members.append(("index", snapshot, f"index/{snapshot.name}"))
        members += [("state", p, f"state/{p.name}") for p in _state_files(db_path)]

        counts: dict[str, int] = {}
        for name, _path, _arc in members:
            counts[name] = counts.get(name, 0) + 1
        skipped = [n for n in (*_DIR_PIECES, "config", "index", "state") if n not in counts]

        if not counts:
            raise BackupError(
                "there is nothing to back up: no vault, no index, no config. "
                "Run the pipeline at least once first."
            )

        manifest = {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "created_at": (created_at or datetime.now(timezone.utc)).isoformat(),
            "tool_version": __version__,
            "pieces": {name: {"files": n} for name, n in counts.items()},
        }
        manifest_path = staging / MANIFEST_NAME
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        tar_path = staging / "bundle.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(manifest_path, arcname=MANIFEST_NAME)
            for _name, path, arcname in members:
                tar.add(path, arcname=arcname)

        # The vault's own streaming format, chunk by chunk, so a backup of a
        # large archive never needs the whole tar in memory. write_stream also
        # inherits the 0600 mode and the truncation-refusing chunk framing.
        encrypted = Vault(staging / "envelope").write_stream(
            "bundle", tar_path, recording_id=BACKUP_AAD
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(encrypted), str(out))
        size = out.stat().st_size
    finally:
        # Every path out of this function, success or not, removes the staging
        # directory -- it briefly held the index and outbox in the clear.
        shutil.rmtree(staging, ignore_errors=True)

    log.info("backup: wrote %s (%d bytes, %s)", out, size,
             ", ".join(f"{k}={v}" for k, v in counts.items()))
    return BackupReport(path=out, size_bytes=size, counts=counts, skipped=skipped)


# =========================================================================
# Restore
# =========================================================================
def _unpack(vault: Vault, source: Path, staging: Path) -> Path:
    """Decrypt and extract into `staging`, trusting nothing about the file."""
    tar_path = vault.read_stream(source, staging / "bundle.tar.gz", BACKUP_AAD)

    unpacked = staging / "unpacked"
    with tarfile.open(tar_path, "r:gz") as tar:
        members = []
        for member in tar.getmembers():
            # The cipher already authenticated the bytes, but a backup made by
            # a different version of this tool is still an input, and inputs
            # do not get to name paths outside their own root.
            if member.name.startswith("/") or ".." in Path(member.name).parts:
                raise BackupError(
                    f"refusing to unpack '{member.name}': it points outside the archive"
                )
            if not (member.isfile() or member.isdir()):
                raise BackupError(
                    f"refusing to unpack '{member.name}': not a regular file or directory"
                )
            members.append(member)
        try:
            tar.extractall(unpacked, members=members, filter="data")
        except TypeError:  # pragma: no cover - Python without extraction filters
            tar.extractall(unpacked, members=members)
    return unpacked


def _read_manifest(unpacked: Path) -> dict:
    """
    The manifest is the backup's own claim about what it holds, checked file by
    file. A bundle that decrypts but disagrees with itself is refused whole:
    restoring the pieces that happen to be present is how a partial archive
    gets mistaken for the entire one.
    """
    path = unpacked / MANIFEST_NAME
    if not path.is_file():
        raise BackupError(
            "the file decrypted but holds no manifest, so it is not a plaud-bridge backup."
        )
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise BackupError(f"the backup's manifest is not valid JSON: {exc}") from exc

    if manifest.get("format") != FORMAT:
        raise BackupError("the file decrypted but its manifest is not a plaud-bridge backup's.")
    if int(manifest.get("format_version", 0)) > FORMAT_VERSION:
        raise BackupError(
            f"this backup was written by a newer plaud-bridge "
            f"(format v{manifest.get('format_version')}, this build reads up to "
            f"v{FORMAT_VERSION}). Upgrade the tool, then restore."
        )

    pieces = manifest.get("pieces") or {}
    unknown = sorted(set(pieces) - _KNOWN_PIECES)
    if unknown:
        raise BackupError(f"the backup names pieces this build does not know: {unknown}")
    for name, meta in pieces.items():
        actual = len(_files_under(unpacked / name))
        expected = int(meta.get("files", -1))
        if actual != expected:
            raise BackupError(
                f"the manifest says '{name}' holds {expected} file(s) but the "
                f"archive holds {actual}. Refusing to restore an archive that "
                "disagrees with its own manifest."
            )
    return manifest


def restore_backup(cfg, config_dir: Path, source: Path, force: bool = False) -> RestoreReport:
    """
    Bring a backup's contents back to their configured paths.

    Everything is decrypted and verified in a temp directory first; the real
    data directories are only touched after the whole bundle has proven itself.
    Existing data refuses to be overwritten without `force`, and the config
    directory -- the one piece guaranteed to exist, since it is what named the
    destinations -- is only ever replaced under `force`.
    """
    source = Path(source)
    if not source.is_file():
        raise BackupError(f"no such file: {source}")

    db_dest = cfg.path("database")
    report = RestoreReport()

    staging = Path(tempfile.mkdtemp(prefix="plaud-restore-"))
    try:
        vault = Vault(staging / "envelope")
        _require_ready(vault, "restore")
        # VaultError propagates untouched from here: "the passphrase is wrong
        # or the file has been modified" is the honest answer, and nothing has
        # been written anywhere it could half-restore.
        unpacked = _unpack(vault, source, staging)
        manifest = _read_manifest(unpacked)
        pieces: dict[str, dict] = manifest["pieces"]

        # What would be overwritten. The config directory is excluded from the
        # refusal -- it always exists, or this function could not have been
        # reached -- and handled below instead.
        occupied: list[str] = []
        for name in pieces:
            if name in _DIR_PIECES and _files_under(cfg.path(name)):
                occupied.append(f"{name} -> {cfg.path(name)}")
            elif name == "index" and db_dest.exists():
                occupied.append(f"index -> {db_dest}")
            elif name == "state":
                for staged in _files_under(unpacked / "state"):
                    if (db_dest.parent / staged.name).exists():
                        occupied.append(f"state -> {db_dest.parent / staged.name}")
        if occupied and not force:
            raise BackupError(
                "restoring would overwrite data already in place:\n  "
                + "\n  ".join(occupied)
                + "\nNothing was changed. Pass --force to replace all of it with "
                "the backup's copy -- the copies on disk now will be gone."
            )

        for name in pieces:
            src_root = unpacked / name
            if name in _DIR_PIECES:
                dest = cfg.path(name)
                if _files_under(dest):
                    report.replaced.append(str(dest))
                if dest.exists():
                    shutil.rmtree(dest)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src_root), str(dest))
            elif name == "config":
                # The live config named every destination above, so without
                # --force it stays exactly as it is.
                if not force:
                    report.config_skipped = True
                    continue
                dest = Path(config_dir)
                if dest.exists():
                    report.replaced.append(str(dest))
                    shutil.rmtree(dest)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src_root), str(dest))
            elif name == "index":
                db_file = next(iter(_files_under(src_root)))
                if db_dest.exists():
                    report.replaced.append(str(db_dest))
                    db_dest.unlink()
                # Stale shadow files beside a replaced database are someone
                # else's uncommitted pages; they go with it.
                for shadow in _DB_SHADOWS:
                    Path(str(db_dest) + shadow).unlink(missing_ok=True)
                db_dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(db_file), str(db_dest))
            elif name == "state":
                db_dest.parent.mkdir(parents=True, exist_ok=True)
                for staged in _files_under(src_root):
                    target = db_dest.parent / staged.name
                    if target.exists():
                        report.replaced.append(str(target))
                        target.unlink()
                    shutil.move(str(staged), str(target))
            report.restored[name] = int(pieces[name].get("files", 0))
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    # Audited after the pieces are in place, because the database being written
    # to may itself be the one that was just restored.
    if db_dest.exists():
        db = Database(db_dest)
        try:
            db.audit(
                "restore",
                f"restored {', '.join(sorted(report.restored))} from {source.name}"
                + (f"; replaced {len(report.replaced)} existing path(s)"
                   if report.replaced else ""),
                actor="human",
            )
        finally:
            db.close()

    log.info("restore: %s -> %s", source, ", ".join(sorted(report.restored)))
    return report
