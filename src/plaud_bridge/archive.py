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


class Archive:
    def __init__(self, cfg, db, vault: Vault | None = None):
        self.cfg = cfg
        self.db = db
        self.vault = vault or Vault(cfg.path("vault"))

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
                       since_days: int | None = None, limit: int = 200,
                       context: int = 0) -> tuple[list[Match], list[str]]:
        """
        Find a phrase in what was actually said.

        Returns (matches, unopened). `unopened` lists recordings whose content
        is encrypted and could not be decrypted, because silently returning
        fewer results than exist is the wrong answer for a search over your own
        archive -- you would conclude the phrase was never said.
        """
        needle = query.lower().strip()
        matches: list[Match] = []
        unopened: list[str] = []
        if not needle:
            return matches, unopened

        personal = {
            p.id for p in self.cfg.profiles.values() if p.exclude_from_combined_export
        }

        for row in self.db.query(profile_id=profile_id, since_days=since_days, limit=limit):
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
                    stamp=_stamp(float(segment.get("start", 0.0))),
                    speaker=str(segment.get("speaker", "")),
                    text=text.strip(),
                ))
        return matches, unopened

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
        targets: list[Path] = []

        rows = self.db.query(limit=100000)
        row = next((r for r in rows if r["id"] == recording_id), None)
        if row is not None:
            for value in self._payload(row).get("artifact_paths", {}).values():
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
        for path in targets:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if path.exists():
                unique.append(path)
        return unique

    def forget(self, recording_id: str) -> tuple[int, list[str]]:
        """
        Delete a recording and everything belonging to it.

        The audit entry is written BEFORE anything is removed, so a crash
        halfway through still leaves evidence that a deletion was attempted.
        """
        targets = self.plan_forget(recording_id)
        self.db.audit(
            "forget",
            f"deleting {len(targets)} file(s) and the index entry",
            recording_id, actor="human",
        )

        removed = 0
        failures: list[str] = []
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


def _stamp(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"
