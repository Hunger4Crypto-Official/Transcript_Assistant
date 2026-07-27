"""
Reading the archive back.

Everything that needs stored content -- searching it, checking it still opens,
exporting it, deleting it -- goes through here, so decryption happens in one
place rather than four. A recording's words live in one of two places depending
on its governing profile: in the SQLite payload when it is not encrypted, and
only in the vault when it is.

Nothing in this module writes content. `forget` deletes, and that is the single
destructive path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .logging_setup import get
from .models import format_stamp
from .storage import Vault, VaultError

log = get("archive")


@dataclass
class Match:
    recording_id: str
    source_name: str
    when: str
    profile_id: str
    personal: bool
    stamp: str
    speaker: str
    text: str


@dataclass
class SearchResult:
    """
    What a search found, and what it did not look at.

    `truncated` and `unopened` exist so a caller can never present a partial
    search as a complete one.
    """

    matches: list[Match] = field(default_factory=list)
    unopened: list[str] = field(default_factory=list)
    scanned: int = 0
    total: int = 0
    truncated: bool = False

    @property
    def complete(self) -> bool:
        return not self.truncated and not self.unopened


@dataclass
class ArtifactState:
    recording_id: str
    kind: str
    path: str
    encrypted: bool
    status: str      # ok | missing | unreadable | undecryptable | unchecked
    detail: str = ""

    @property
    def healthy(self) -> bool:
        return self.status == "ok"


@dataclass
class VerifyReport:
    checked: list[ArtifactState] = field(default_factory=list)
    orphans: list[Path] = field(default_factory=list)
    unreachable: bool = False       # the vault could not be opened at all

    @property
    def problems(self) -> list[ArtifactState]:
        """Things that are actually wrong. Unchecked is not the same as broken."""
        return [a for a in self.checked if not a.healthy and a.status != "unchecked"]

    @property
    def unchecked(self) -> list[ArtifactState]:
        return [a for a in self.checked if a.status == "unchecked"]

    def render(self) -> str:
        lines: list[str] = []
        ok = len([a for a in self.checked if a.healthy])
        lines.append(f"{len(self.checked)} artifact(s) indexed, {ok} verified")

        if self.unreachable:
            lines.append("")
            lines.append(
                f"  {len(self.unchecked)} encrypted artifact(s) were NOT checked: the "
                "vault could not be opened."
            )
            lines.append(
                "  Set PLAUD_BRIDGE_PASSPHRASE and run this again. An unverified "
                "encrypted archive is not a verified one, and this command exists "
                "precisely to stop you assuming otherwise."
            )

        for state in self.problems:
            lines.append(f"  [{state.status:14s}] {state.kind:10s} {state.recording_id}")
            lines.append(f"                   {state.path}")
            if state.detail:
                lines.append(f"                   {state.detail}")

        if self.orphans:
            lines.append("")
            lines.append(f"{len(self.orphans)} file(s) on disk that the index does not know about:")
            for path in self.orphans[:40]:
                lines.append(f"  {path}")
            if len(self.orphans) > 40:
                lines.append(f"  ... and {len(self.orphans) - 40} more")
            lines.append(
                "  These are not deleted automatically. They may be from a database "
                "that was rebuilt, or from a run that failed partway."
            )

        if not self.problems and not self.orphans and not self.unreachable:
            lines.append("")
            lines.append("Everything the index points at exists and opens.")
        return "\n".join(lines)

    @property
    def healthy(self) -> bool:
        return not self.problems and not self.unreachable


def owned_roots(cfg) -> list[Path]:
    """The directories this tool is allowed to delete inside."""
    roots: list[Path] = []
    for name in ("vault", "outbox", "inbox", "quarantine", "work"):
        try:
            roots.append(cfg.path(name).resolve())
        except Exception:  # noqa: BLE001 - a missing path is simply not a root
            continue
    return roots


def is_owned(path: Path, roots: list[Path]) -> bool:
    """
    True when `path` lives inside one of our own directories.

    Deletion targets come from the index, and the index is a file: restored from
    a backup, hand-edited, or written by a version whose paths pointed somewhere
    else. Unlinking whatever it names is how a tool deletes something that was
    never its business. Everything destructive checks this first.
    """
    try:
        resolved = path.resolve()
    except OSError:
        return False
    return any(resolved == root or root in resolved.parents for root in roots)


class Archive:
    def __init__(self, cfg, db, vault: Vault | None = None):
        self.cfg = cfg
        self.db = db
        self.vault = vault or Vault(cfg.path("vault"))
        self.roots = owned_roots(cfg)

    # =====================================================================
    # Reading
    # =====================================================================
    def _payload(self, row: dict[str, Any]) -> dict[str, Any]:
        try:
            return json.loads(row["payload_json"])
        except (ValueError, KeyError):
            return {}

    def full_record(self, row: dict[str, Any]) -> dict[str, Any] | None:
        """
        The complete recording, with content, decrypting if it has to.

        Returns None when the content exists but cannot be opened, so callers
        can tell "there is nothing here" apart from "I cannot read what is here".
        """
        payload = self._payload(row)
        transcript = payload.get("transcript") or {}
        analyses = payload.get("analyses") or []

        withheld = bool(transcript.get("segments_withheld")) or any(
            a.get("fields_withheld") for a in analyses
        )
        if not withheld:
            return payload

        path = str(payload.get("artifact_paths", {}).get("analysis", ""))
        if not path or not Path(path).exists():
            return None
        try:
            return json.loads(self.vault.read_text(Path(path), row["id"]))
        except (VaultError, ValueError, OSError) as exc:
            log.warning("archive: could not open %s (%s)", row["id"], exc)
            return None

    def segments(self, row: dict[str, Any]) -> list[dict[str, Any]] | None:
        record = self.full_record(row)
        if record is None:
            return None
        return (record.get("transcript") or {}).get("segments") or []

    # =====================================================================
    # Search
    # =====================================================================
    def search_content(self, query: str, profile_id: str | None = None,
                       since_days: int | None = None, scan_limit: int = 0,
                       context: int = 0) -> SearchResult:
        """
        Find a phrase in what was actually said.

        `scan_limit` bounds how many RECORDINGS are opened, not how many hits
        come back, and 0 means all of them. That distinction is the whole point:
        this used to take the CLI's `--limit` -- which a user reads as "how many
        results to show" -- and hand it to the row query, so an archive of 600
        recordings had its oldest 550 silently excluded. A search that answers
        "that was never said" because it did not look is worse than no search.

        The result carries what was scanned, what was skipped, and what would
        not open, so the caller can say so out loud.
        """
        needle = query.lower().strip()
        result = SearchResult()
        if not needle:
            return result

        personal = {
            p.id for p in self.cfg.profiles.values() if p.exclude_from_combined_export
        }

        total = self.db.count_recordings(profile_id=profile_id, since_days=since_days)
        rows = self.db.query(
            profile_id=profile_id, since_days=since_days,
            limit=scan_limit if scan_limit > 0 else total or 1,
        )
        result.scanned = len(rows)
        result.total = total
        result.truncated = total > len(rows)

        matches = result.matches
        unopened = result.unopened

        for row in rows:
            segments = self.segments(row)
            if segments is None:
                unopened.append(f"{row['id']}  {row['source_name']}")
                continue

            when = (row["recorded_at"] or row["ingested_at"] or "")[:16].replace("T", " ")
            for index, segment in enumerate(segments):
                text = str(segment.get("text", ""))
                if needle not in text.lower():
                    continue
                if context:
                    window = segments[max(0, index - context): index + context + 1]
                    text = " ".join(str(s.get("text", "")) for s in window)
                matches.append(Match(
                    recording_id=row["id"],
                    source_name=row["source_name"],
                    when=when,
                    profile_id=row["governing_profile"] or "",
                    personal=(row["governing_profile"] or "") in personal,
                    stamp=format_stamp(float(segment.get("start", 0.0))),
                    speaker=str(segment.get("speaker", "")),
                    text=text.strip(),
                ))
        return result

    # =====================================================================
    # Verify
    # =====================================================================
    def verify(self) -> VerifyReport:
        report = VerifyReport()
        vault_ok, _ = self.vault.ready()
        indexed: set[Path] = set()

        for row in self.db.all_artifacts():
            path = Path(row["path"])
            indexed.add(path.resolve())
            encrypted = bool(row["encrypted"])
            state = ArtifactState(
                recording_id=row["recording_id"], kind=row["kind"],
                path=str(path), encrypted=encrypted, status="ok",
            )

            if not path.exists():
                state.status = "missing"
                state.detail = "the index points at a file that is not on disk"
            elif encrypted or path.suffix == ".enc":
                if not vault_ok:
                    # Say so rather than silently omitting it. Reporting "0
                    # artifacts indexed" when there are twelve is how someone
                    # concludes an archive is fine when nothing was looked at.
                    report.unreachable = True
                    state.status = "unchecked"
                    state.detail = "the vault is locked; this file was not opened"
                    report.checked.append(state)
                    continue
                try:
                    if Vault.is_streamed(path):
                        # Decrypt and discard. Reading a day of audio into memory
                        # to prove it decrypts would defeat the point of having
                        # streamed it, and writing it out would defeat the point
                        # of having encrypted it.
                        self.vault.verify_stream(path, row["recording_id"])
                    else:
                        self.vault.read(path, row["recording_id"])
                except VaultError as exc:
                    state.status = "undecryptable"
                    state.detail = str(exc)
                except OSError as exc:
                    state.status = "unreadable"
                    state.detail = str(exc)
            else:
                try:
                    path.read_bytes()
                except OSError as exc:
                    state.status = "unreadable"
                    state.detail = str(exc)

            report.checked.append(state)

        for root in (self.cfg.path("vault"), self.cfg.path("outbox")):
            if not root.is_dir():
                continue
            for path in sorted(root.rglob("*")):
                if path.is_file() and path.resolve() not in indexed:
                    report.orphans.append(path)

        return report

    # =====================================================================
    # Forget
    # =====================================================================
    def plan_forget(self, recording_id: str) -> list[Path]:
        """Every file that `forget` would remove. Read-only."""
        return self._plan_forget(recording_id)[0]

    def _plan_forget(self, recording_id: str) -> tuple[list[Path], list[Path]]:
        """(targets, refused) — refused are paths outside our own directories."""
        targets: list[Path] = []

        payload = self.db.load(recording_id) or {}
        for value in payload.get("artifact_paths", {}).values():
            targets.append(Path(str(value)))

        for artifact in self.db.all_artifacts():
            if artifact["recording_id"] == recording_id:
                targets.append(Path(artifact["path"]))

        quarantine = self.cfg.path("quarantine") / recording_id
        if quarantine.is_dir():
            targets.extend(p for p in sorted(quarantine.rglob("*")) if p.is_file())

        # Archived originals carry the recording id as a filename prefix.
        archive = self.cfg.path("inbox") / "_processed"
        if archive.is_dir():
            targets.extend(sorted(archive.glob(f"{recording_id}_*")))

        seen: set[Path] = set()
        unique: list[Path] = []
        refused: list[Path] = []
        for path in targets:
            try:
                resolved = path.resolve()
            except OSError:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            if not path.exists():
                continue
            if not is_owned(path, self.roots):
                log.error(
                    "refusing to delete %s: it is outside every configured data "
                    "directory. The index should not be pointing there.", path,
                )
                refused.append(path)
                continue
            unique.append(path)
        return unique, refused

    def forget(self, recording_id: str) -> tuple[int, list[str]]:
        """
        Delete a recording and everything belonging to it.

        The audit entry is written BEFORE anything is removed, so a crash
        halfway through still leaves evidence that a deletion was attempted.
        """
        targets, refused = self._plan_forget(recording_id)
        self.db.audit(
            "forget",
            f"deleting {len(targets)} file(s) and the index entry"
            + (f"; refused {len(refused)} outside the data directory" if refused else ""),
            recording_id, actor="human",
        )

        removed = 0
        failures: list[str] = [
            f"{path}: outside every configured data directory, left alone" for path in refused
        ]
        for path in targets:
            try:
                path.unlink()
                removed += 1
            except OSError as exc:
                failures.append(f"{path}: {exc}")

        quarantine = self.cfg.path("quarantine") / recording_id
        if quarantine.is_dir():
            try:
                for leftover in sorted(quarantine.rglob("*"), reverse=True):
                    if leftover.is_dir():
                        leftover.rmdir()
                quarantine.rmdir()
            except OSError as exc:
                failures.append(f"{quarantine}: {exc}")

        self.db.delete_recording(recording_id)
        self.db.audit("forget_complete", f"removed {removed} file(s)", recording_id, actor="human")
        return removed, failures
