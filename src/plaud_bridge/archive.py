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

from .diarize.voiceprint import STORE_RELATIVE as VOICEPRINT_STORE
from .logging_setup import get
from .models import Stage, format_stamp
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
    # Quarantined recordings are reported apart from unreadable ones. They hold
    # no searchable content by design -- the gate stopped them before anything
    # was written -- so telling somebody to check their passphrase, which is
    # what `unopened` advises, sends them after a problem they do not have.
    quarantined: list[str] = field(default_factory=list)
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
    # Files that live in the vault or the outbox deliberately and are not
    # artifacts of any recording: voiceprints, saved answers, drafts. They are
    # counted and named rather than hidden, because a `verify` that quietly
    # skips whole directories is one you cannot use to find a real orphan.
    known: list[tuple[Path, str]] = field(default_factory=list)
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

        if self.known:
            lines.append("")
            by_label: dict[str, int] = {}
            for _path, label in self.known:
                by_label[label] = by_label.get(label, 0) + 1
            summary = ", ".join(f"{count} {label}" for label, count in sorted(by_label.items()))
            lines.append(f"{len(self.known)} file(s) that belong here but are not "
                         f"artifacts: {summary}.")

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


def _labelled(path: Path, expected: list[tuple[Path, str]]) -> str:
    """The label for a deliberate non-artifact, or "" when this is a real orphan."""
    resolved = path.resolve()
    for candidate, label in expected:
        candidate = candidate.resolve() if candidate.exists() else candidate
        if resolved == candidate or candidate in resolved.parents:
            return label
    return ""


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

        A quarantined recording is the third case and reads as the first. The
        gate stops it before anything is persisted, so the index knows its name
        and nothing else -- there is no artifact, and there never was one. It
        used to come back as None, which every caller printed as "could not be
        opened, set PLAUD_BRIDGE_PASSPHRASE": advice that cannot work, about a
        problem that does not exist, and an exit code of 2 on every search,
        export, and answer for as long as the recording sat in quarantine.
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
            if str(row.get("stage") or "") == Stage.QUARANTINED.value:
                return payload
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
            if not segments and str(row.get("stage") or "") == Stage.QUARANTINED.value:
                result.quarantined.append(f"{row['id']}  {row['source_name']}")
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

        expected = self._non_artifacts()
        for root in (self.cfg.path("vault"), self.cfg.path("outbox")):
            if not root.is_dir():
                continue
            for path in sorted(root.rglob("*")):
                if not path.is_file() or path.resolve() in indexed:
                    continue
                label = _labelled(path, expected)
                if label:
                    report.known.append((path, label))
                else:
                    report.orphans.append(path)

        return report

    def _non_artifacts(self) -> list[tuple[Path, str]]:
        """
        Where the features that are not the pipeline are allowed to write.

        Three things live in the vault or the outbox without belonging to any
        recording, and every one of them would otherwise be reported as a file
        the index does not know about, next to advice about rebuilt databases
        and half-finished runs that could not apply to it. Being told your
        voiceprints are debris, every time you check the archive is healthy, is
        how a person learns to skim the one command whose job is to be read.

        Listing them here rather than skipping their directories is deliberate:
        an unexpected file inside data/vault/ask/ is still an orphan.
        """
        vault, outbox = self.cfg.path("vault"), self.cfg.path("outbox")
        return [
            (vault / f"{VOICEPRINT_STORE}.enc", "enrolled voiceprints"),
            (vault / "ask", "saved answers"),
            (outbox / "drafts", "follow-up drafts"),
        ]

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

        # Drafts are plaintext markdown destined to leave the machine, and they
        # carry the recording's commitments plus its id -- in the filename when
        # a draft was built for one recording, and in a "traced to" footer for a
        # combined one. Either way `forget` has to reach them.
        targets.extend(self._drafts_referencing(recording_id))

        # Saved answers quote the recording verbatim. They live encrypted in
        # vault/ask; the ones that cite this recording go too.
        targets.extend(self._answers_referencing(recording_id))

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

    def _drafts_referencing(self, recording_id: str) -> list[Path]:
        drafts = self.cfg.path("outbox") / "drafts"
        if not drafts.is_dir():
            return []
        hits: list[Path] = []
        for path in sorted(drafts.iterdir()):
            if not path.is_file():
                continue
            if recording_id in path.name:
                hits.append(path)
                continue
            try:
                if recording_id in path.read_text(encoding="utf-8", errors="ignore"):
                    hits.append(path)
            except OSError:
                continue
        return hits

    def _answers_referencing(self, recording_id: str) -> list[Path]:
        ask_dir = self.cfg.path("vault") / "ask"
        if not ask_dir.is_dir():
            return []
        hits: list[Path] = []
        for path in sorted(ask_dir.glob("*.enc")):
            try:
                # save_answer writes these with an empty AAD (no recording id
                # is bound, because an answer can cite several); read them the
                # same way or every decrypt fails and no answer is ever found.
                payload = json.loads(self.vault.read_text(path, ""))
            except (VaultError, ValueError, OSError):
                # Cannot read it (locked/corrupt). The locked-vault refusal in
                # `forget` stops us reaching here with the vault down; a corrupt
                # answer is left for `verify` to surface rather than guessed at.
                continue
            cited = set(payload.get("recordings_used") or [])
            cited.update(c.get("recording_id") for c in payload.get("citations") or [])
            if recording_id in cited:
                hits.append(path)
        return hits

    def _derived_stores_present(self, recording_id: str) -> bool:
        """
        Whether any ENCRYPTED store that could hold this recording lives on disk.

        These are the stores `forget` cannot purge without the passphrase: the
        memory ledgers, the saved answers, and the follow-up state file. Their
        mere presence is what makes a locked-vault forget dangerous -- it would
        delete the plaintext half and leave the encrypted half behind.
        """
        ask = self.cfg.path("vault") / "ask"
        if ask.is_dir() and any(ask.glob("*.enc")):
            return True
        try:
            from .memory import MemoryStore
            mem_dir = MemoryStore(self.cfg).dir
        except Exception:  # noqa: BLE001 - memory is optional; absence is not presence
            mem_dir = None
        if mem_dir and Path(mem_dir).is_dir() and any(Path(mem_dir).glob("*.enc")):
            return True
        try:
            from .followups import state_path
            from .storage.vault import MAGIC as _VAULT_MAGIC
            state = state_path(self.cfg)
            # Only an ENCRYPTED state file is a store the passphrase is needed
            # for. The plaintext fallback (no vault at all) can be read and
            # purged without one, so it must not force a refusal -- doing so
            # would make forget impossible for anyone running passphrase-free.
            if state.exists() and state.read_bytes()[:len(_VAULT_MAGIC)] == _VAULT_MAGIC:
                return True
        except Exception:  # noqa: BLE001
            pass
        return False

    def forget(self, recording_id: str) -> tuple[int, list[str]]:
        """
        Delete a recording and everything belonging to it.

        Fail-safe about the vault. `forget` has to reach encrypted derived
        stores -- the memory ledgers, saved answers, and follow-up state -- and
        it cannot open any of them without the passphrase. So if the vault is
        locked and any of those stores exist, it refuses the WHOLE operation and
        removes nothing, rather than deleting the plaintext half (files, index
        row) and stranding the recording's words in an encrypted store it can no
        longer even name. A red-team pass produced exactly that half-deleted
        state. Everything the operation touches is recoverable; a half-delete is
        not.

        The audit entry is written BEFORE anything is removed, so a crash
        halfway through still leaves evidence that a deletion was attempted.
        """
        vault_ok, why = self.vault.ready()
        if not vault_ok and self._derived_stores_present(recording_id):
            self.db.audit(
                "forget_refused",
                f"vault locked ({why}); refused to half-delete", recording_id, actor="human",
            )
            return 0, [
                "the vault is locked and this recording may have content in the "
                "memory ledgers, saved answers, or follow-up state. Refusing to "
                "delete anything rather than leave half of it behind. Set "
                f"PLAUD_BRIDGE_PASSPHRASE and run forget again. ({why})"
            ]

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

        # Encrypted derived stores. The vault is ready (guarded above), so these
        # can be purged now rather than left to leak the words the files carried.
        try:
            from .followups import forget_recording as forget_followups
            forget_followups(self.cfg, self.vault, recording_id)
        except Exception as exc:  # noqa: BLE001 - a convenience store, not the record
            failures.append(f"follow-up state: {exc}")
        try:
            from .memory import MemoryStore
            store = MemoryStore(self.cfg, self.vault)
            store.forget_recording(recording_id)
            failures.extend(store.problems)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"memory ledgers: {exc}")

        self.db.delete_recording(recording_id)
        self.db.audit("forget_complete", f"removed {removed} file(s)", recording_id, actor="human")
        return removed, failures
