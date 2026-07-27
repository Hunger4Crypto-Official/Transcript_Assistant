"""
Carrying what you already know into the next recording.

Every recording is analysed as though it were the first one ever made, and that
is the difference between a very good transcriber and an assistant. The tool
cannot tell you that Marcus already turned down that rider in March, that the
rep you are coaching has been told about his discovery questions three times
now, or that you promised a seven-year-old the batting cages two Saturdays ago
and then a work week happened.

So: a small ledger per profile. Who keeps coming up, what was promised and what
closed it, which subjects recur, and the handful of facts worth carrying. Plus a
short briefing that goes into the next extraction prompt so the next analysis is
made in context instead of from nothing.

Three things shape the design.

  - It is built from artifacts that already exist. Nothing in here calls a
    model. A ledger is a deterministic function of the stored analyses, which is
    what makes `rebuild()` meaningful: delete the file, replay the archive, and
    the same ledger comes back. If that ever stops being true, the ledger has
    started holding something nothing else does, and it is no longer a
    convenience -- it is a second copy of your archive with its own failure
    modes.
  - It is per profile, and that separation is a guarantee rather than a
    convention. What the Husband profile knows must never appear in an Insurance
    Agent prompt. Each ledger is its own file, bound by the cipher's additional
    data to the profile id it belongs to, so a ledger opened under the wrong
    name does not decrypt at all.
  - It is as sensitive as the transcripts it came from. Arguably worse: it is
    the distilled version, the part someone would actually sit and read. It goes
    in the vault or it does not get written. There is no plaintext fallback and
    there is not going to be one.

Entries decay. Something not seen for a while stops being surfaced, because a
briefing full of things that stopped being true is worse than a short one. Decay
is not deletion -- the entry stays in the ledger with its provenance intact, it
just stops being carried forward. Deletion is what `forget` does, and
`forget_recording` here makes sure a deleted recording leaves nothing of itself
behind in memory either. A ledger that still remembers a conversation you
deleted would quietly defeat the one command that promises it is gone.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .compliance.redact import redact_text
from .logging_setup import get
from .storage import Vault, VaultError

log = get("memory")

LEDGER_VERSION = 1

PERSON = "person"
COMMITMENT = "commitment"
TOPIC = "topic"
FACT = "fact"
KINDS = (COMMITMENT, PERSON, TOPIC, FACT)

# How each extracted field is filed. Keyed by the field keys the shipped
# profiles actually declare, because guessing from a field name alone puts
# things in the wrong drawer often enough to be annoying. `memory.field_kinds`
# in pipeline.yaml overrides any of these, and "ignore" drops a field entirely.
#
# Two deliberate omissions. `statements_needing_review` is a per-recording
# compliance artifact that `review` already surfaces, and carrying it forward
# into a prompt would encourage a model to look for the same phrasing again.
# `requires_human_attention` is a flag, not a memory.
DEFAULT_FIELD_KINDS: dict[str, str] = {
    # insurance_agent
    "participants": PERSON,
    "meeting_type": TOPIC,
    "stated_needs": FACT,
    "objections": TOPIC,
    "commitments_by_client": COMMITMENT,
    "commitments_by_producer": COMMITMENT,
    "open_questions": TOPIC,
    # sales_trainer
    "session_type": TOPIC,
    "principles": FACT,
    "objection_patterns": TOPIC,
    "language_that_worked": FACT,
    "language_to_fix": TOPIC,
    "content_hooks": TOPIC,
    "skill_gaps": TOPIC,
    # father
    "worth_remembering": FACT,
    "promises_i_made": COMMITMENT,
    "logistics": TOPIC,
    "asks_from_them": FACT,
    "milestones": FACT,
    # husband
    "commitments_i_made": COMMITMENT,
    "shared_plans": COMMITMENT,
    "she_asked_for": FACT,
    "household_logistics": TOPIC,
    # unfiled
    "topic": TOPIC,
    "suggested_keywords": TOPIC,
    "action_items": COMMITMENT,
    # every profile
    "next_action": COMMITMENT,
}

# Fallback for a profile someone scaffolded themselves. Substring match, first
# hit wins, and a field that matches nothing is left out rather than guessed at.
_KIND_HINTS: tuple[tuple[str, str], ...] = (
    ("participant", PERSON),
    ("attendee", PERSON),
    ("people", PERSON),
    ("commit", COMMITMENT),
    ("promise", COMMITMENT),
    ("next_action", COMMITMENT),
    ("action", COMMITMENT),
    ("plan", COMMITMENT),
    ("objection", TOPIC),
    ("question", TOPIC),
    ("logistic", TOPIC),
    ("skill", TOPIC),
    ("topic", TOPIC),
    ("type", TOPIC),
    ("milestone", FACT),
    ("remember", FACT),
    ("need", FACT),
    ("principle", FACT),
    ("asked", FACT),
    ("asks", FACT),
)

# Days after which an entry of each kind stops being surfaced. A promise
# outlives a passing subject; something you learned about someone outlives both.
DEFAULT_DECAY_DAYS: dict[str, int] = {
    COMMITMENT: 180,
    PERSON: 120,
    TOPIC: 90,
    FACT: 365,
}
DEFAULT_DECAY_FALLBACK = 90

# Fields that close an open commitment when they mention one. No shipped profile
# declares any of these yet, which is the point: a commitment is only closed by
# something that says it was done. Deciding a promise was kept because its words
# came up again would be inventing, and a stale open commitment is a far cheaper
# mistake than a false closed one.
DEFAULT_CLOSURE_FIELDS: tuple[str, ...] = (
    "completed", "closed_items", "delivered", "done", "resolved",
)

_QUOTE_KEYS = ("text", "quote", "statement", "content", "what")
_META_KEYS = ("timestamp", "time", "speaker", "who")
_SECTION_TITLES = {
    COMMITMENT: "Open commitments",
    PERSON: "People who keep coming up",
    TOPIC: "Recurring subjects",
    FACT: "Worth carrying",
}

_WORD = re.compile(r"[a-z0-9']+")
_SPEAKER_LABEL = re.compile(r"^speaker[_ ]?\d*$", re.IGNORECASE)
_NOT_CONTENT = frozenset({"", "none", "n/a", "na", "null", "unknown", "nothing", "other"})


class MemoryLedgerError(RuntimeError):
    """Raised only when a ledger would be written somewhere it does not belong."""


# =========================================================================
# Small helpers
# =========================================================================
def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(when: datetime) -> str:
    return when.astimezone(timezone.utc).isoformat()


def _parse_when(raw: Any) -> datetime | None:
    """Parse a stored timestamp, tolerating the shapes SQLite hands back."""
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _key_for(text: str, limit: int = 120) -> str:
    """
    The identity of an entry.

    Lowercased words joined by single spaces, truncated. Deliberately crude: it
    merges "Send two quote options by Thursday" with the same sentence written
    with different punctuation, and it does not pretend to merge two different
    wordings of the same promise. Over-merging loses a commitment; under-merging
    shows it twice, and only one of those is a real problem.
    """
    return " ".join(_tokens(text))[:limit].strip()


def _overlap(a: str, b: str) -> float:
    """Share of the smaller phrase's words that the larger one also has."""
    left, right = set(_tokens(a)), set(_tokens(b))
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def _phrase(item: Any) -> str:
    """
    Render one extracted item as a single line, verbatim where possible.

    The extraction schemas promise quote-shaped dicts for some fields and let a
    model return {"what": ..., "when": ...} for others. Both have to survive
    into the ledger with the speaker's own words intact, because a ledger that
    paraphrases is a ledger that invents.
    """
    if isinstance(item, bool):
        return ""
    if not isinstance(item, dict):
        return str(item).strip()

    for key in _QUOTE_KEYS:
        value = item.get(key)
        if isinstance(value, (str, int, float)) and str(value).strip():
            body = str(value).strip()
            extras = [
                f"{k}: {v}"
                for k, v in item.items()
                if k not in _QUOTE_KEYS + _META_KEYS
                and not isinstance(v, (dict, list, bool))
                and str(v).strip()
            ]
            return f"{body} ({'; '.join(extras)})" if extras else body

    pairs = [
        f"{k}: {v}"
        for k, v in item.items()
        if not isinstance(v, (dict, list, bool)) and str(v).strip()
    ]
    return "; ".join(pairs)


def _phrases(value: Any, limit: int) -> list[str]:
    """Every rememberable line in one extracted field value."""
    if value is None or isinstance(value, bool):
        return []
    items = value if isinstance(value, list) else [value]
    out: list[str] = []
    for item in items[:limit]:
        text = _phrase(item).strip()
        if text and text.lower() not in _NOT_CONTENT:
            out.append(text)
    return out


# =========================================================================
# The ledger
# =========================================================================
@dataclass
class Sighting:
    """One occurrence of one entry, in one recording. This is the provenance."""

    recording_id: str
    when: str            # when the conversation happened, not when it was processed
    field_key: str       # which extracted field it came out of
    text: str            # what was actually said or written, verbatim
    source_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "recording_id": self.recording_id,
            "when": self.when,
            "field_key": self.field_key,
            "text": self.text,
            "source_name": self.source_name,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Sighting:
        return cls(
            recording_id=str(d.get("recording_id", "")),
            when=str(d.get("when", "")),
            field_key=str(d.get("field_key", "")),
            text=str(d.get("text", "")),
            source_name=str(d.get("source_name", "")),
        )

    @property
    def order(self) -> tuple[str, str, str, str]:
        return (self.when, self.recording_id, self.field_key, self.text)


@dataclass
class Entry:
    """
    One thing the ledger knows, and every recording that put it there.

    There is no free-standing count or timestamp on an entry: mentions, first
    seen, and last seen are all read off the sightings. That is what makes
    `forget_recording` exact rather than approximate -- drop the sightings that
    came from a recording and every derived number is already correct.
    """

    kind: str
    key: str
    sightings: list[Sighting] = field(default_factory=list)
    closed_by: str = ""
    closed_at: str = ""
    closed_note: str = ""

    @property
    def text(self) -> str:
        """The most recent wording. Older phrasings stay on their sightings."""
        return self.sightings[-1].text if self.sightings else ""

    @property
    def mentions(self) -> int:
        return len(self.sightings)

    @property
    def recording_ids(self) -> list[str]:
        seen: list[str] = []
        for sighting in self.sightings:
            if sighting.recording_id not in seen:
                seen.append(sighting.recording_id)
        return seen

    @property
    def first_seen(self) -> str:
        return self.sightings[0].when if self.sightings else ""

    @property
    def last_seen(self) -> str:
        return self.sightings[-1].when if self.sightings else ""

    @property
    def open(self) -> bool:
        return self.kind == COMMITMENT and not self.closed_by

    def sort(self) -> None:
        self.sightings.sort(key=lambda s: s.order)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "key": self.key,
            "sightings": [s.to_dict() for s in self.sightings],
            "closed_by": self.closed_by,
            "closed_at": self.closed_at,
            "closed_note": self.closed_note,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Entry:
        return cls(
            kind=str(d.get("kind", "")),
            key=str(d.get("key", "")),
            sightings=[Sighting.from_dict(s) for s in d.get("sightings", [])],
            closed_by=str(d.get("closed_by", "")),
            closed_at=str(d.get("closed_at", "")),
            closed_note=str(d.get("closed_note", "")),
        )


@dataclass
class Ledger:
    """
    Everything one profile has learned, and which recordings taught it.

    `seen` maps a recording id to a signature of the analyses it contributed.
    That is what makes updating idempotent: the same recording processed twice
    changes nothing, and a recording re-analysed with different results has its
    old contribution removed before the new one is applied.
    """

    profile_id: str
    entries: list[Entry] = field(default_factory=list)
    seen: dict[str, str] = field(default_factory=dict)
    updated_at: str = ""
    version: int = LEDGER_VERSION

    def entry(self, kind: str, key: str) -> Entry | None:
        return next((e for e in self.entries if e.kind == kind and e.key == key), None)

    def record(self, kind: str, key: str, sighting: Sighting) -> Entry:
        existing = self.entry(kind, key)
        if existing is None:
            existing = Entry(kind=kind, key=key)
            self.entries.append(existing)
        existing.sightings.append(sighting)
        existing.sort()
        return existing

    def purge(self, recording_id: str) -> bool:
        """
        Remove every trace of one recording. Returns whether anything changed.

        Reopening a commitment this recording had closed matters as much as
        dropping the sightings: leaving it closed would mean a deleted
        conversation still deciding what you are told about an outstanding
        promise.
        """
        changed = self.seen.pop(recording_id, None) is not None
        kept: list[Entry] = []
        for entry in self.entries:
            before = len(entry.sightings)
            entry.sightings = [s for s in entry.sightings if s.recording_id != recording_id]
            if len(entry.sightings) != before:
                changed = True
            if entry.closed_by == recording_id:
                entry.closed_by = entry.closed_at = entry.closed_note = ""
                changed = True
            if entry.sightings:
                kept.append(entry)
            else:
                changed = True
        self.entries = kept
        return changed

    def sort(self) -> None:
        for entry in self.entries:
            entry.sort()
        self.entries.sort(key=lambda e: (KINDS.index(e.kind) if e.kind in KINDS else 9, e.key))

    @property
    def recordings(self) -> list[str]:
        return sorted(self.seen)

    def to_dict(self) -> dict[str, Any]:
        self.sort()
        return {
            "version": self.version,
            "profile_id": self.profile_id,
            "updated_at": self.updated_at,
            "seen": dict(sorted(self.seen.items())),
            "entries": [e.to_dict() for e in self.entries],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], profile_id: str) -> Ledger:
        ledger = cls(
            profile_id=str(d.get("profile_id") or profile_id),
            entries=[Entry.from_dict(e) for e in d.get("entries", [])],
            seen={str(k): str(v) for k, v in (d.get("seen") or {}).items()},
            updated_at=str(d.get("updated_at", "")),
            version=int(d.get("version", LEDGER_VERSION)),
        )
        ledger.sort()
        return ledger

    def fingerprint(self) -> str:
        """
        A hash of the content, ignoring when it was written.

        `rebuild` produces the same knowledge at a different moment, so the
        moment cannot be part of what "the same ledger" means.
        """
        body = self.to_dict()
        body.pop("updated_at", None)
        return hashlib.sha256(
            json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:32]


@dataclass
class RebuildReport:
    """
    What a rebuild managed to replay, and what it could not open.

    The second half is the important one. A rebuild that silently skipped the
    twelve encrypted recordings it could not decrypt would hand back a confident
    ledger of everything else, and nothing about it would look wrong.
    """

    replayed: int = 0
    unopened: list[str] = field(default_factory=list)
    written: list[str] = field(default_factory=list)
    ledgers: dict[str, Ledger] = field(default_factory=dict)
    saved: bool = False

    @property
    def complete(self) -> bool:
        return not self.unopened

    def render(self) -> str:
        lines = [f"replayed {self.replayed} recording(s) into {len(self.ledgers)} ledger(s)"]
        if self.unopened:
            lines.append("")
            lines.append(f"{len(self.unopened)} recording(s) could not be opened:")
            lines += [f"  {item}" for item in self.unopened[:20]]
            if len(self.unopened) > 20:
                lines.append(f"  ... and {len(self.unopened) - 20} more")
        if not self.saved:
            lines.append("")
            lines.append(
                "Nothing was written. A ledger rebuilt from part of the archive "
                "would look exactly like a complete one, so it is not allowed to "
                "replace the ledger you already have. Fix the passphrase, or pass "
                "force to accept a partial rebuild."
            )
        else:
            lines.append(f"wrote {len(self.written)} ledger(s)")
        return "\n".join(lines)


# =========================================================================
# Config access
# =========================================================================
def _memory_dir(cfg) -> Path:
    """
    Where the ledgers live.

    Not inside the vault directory, despite being encrypted with the same key.
    `verify` treats every file under the vault that the artifact index does not
    know about as an orphan, and five ledger files reported as orphans forever
    would train you to skim past that list -- which is the one list you should
    not be skimming past.
    """
    raw = str(cfg.get("memory.dir", "./data/memory"))
    path = Path(raw)
    return path if path.is_absolute() else (Path(cfg.root) / path).resolve()


def _decay_days(cfg, kind: str) -> int:
    """
    How long an entry of this kind keeps being surfaced. 0 means forever.

    `memory.decay_days` may be a mapping per kind or a single number, because
    both are things a person reasonably writes in a YAML file and only one of
    them raising a TypeError six frames away would be indefensible.
    """
    configured = cfg.get("memory.decay_days")
    if isinstance(configured, (int, float)) and not isinstance(configured, bool):
        return int(configured)
    if isinstance(configured, dict):
        for candidate in (kind, "default"):
            if candidate in configured:
                try:
                    return int(configured[candidate])
                except (TypeError, ValueError):
                    log.warning(
                        "memory.decay_days.%s is not a number; using the built-in default",
                        candidate,
                    )
                    break
    return DEFAULT_DECAY_DAYS.get(kind, DEFAULT_DECAY_FALLBACK)


def _kind_for(cfg, field_key: str) -> str:
    """Which drawer this field's contents go in, or "" for none."""
    if field_key in {str(k) for k in (cfg.get("memory.ignore_fields") or [])}:
        return ""

    overrides = cfg.get("memory.field_kinds") or {}
    raw = overrides.get(field_key, DEFAULT_FIELD_KINDS.get(field_key))
    if raw is None:
        lowered = field_key.lower()
        raw = next((kind for hint, kind in _KIND_HINTS if hint in lowered), "")
    raw = str(raw or "").strip().lower()

    if raw in ("", "ignore", "skip", "none"):
        return ""
    if raw not in KINDS:
        log.warning(
            "memory.field_kinds maps '%s' to '%s', which is not one of %s. Ignoring the field.",
            field_key, raw, ", ".join(KINDS),
        )
        return ""
    return raw


def _closure_fields(cfg) -> set[str]:
    configured = cfg.get("memory.closure_fields")
    if configured is None:
        return set(DEFAULT_CLOSURE_FIELDS)
    return {str(k) for k in configured}


def _stale(cfg, entry: Entry, now: datetime) -> bool:
    days = _decay_days(cfg, entry.kind)
    if days <= 0:
        return False
    last = _parse_when(entry.last_seen)
    if last is None:
        return True
    return now - last > timedelta(days=days)


def _score(cfg, entry: Entry, now: datetime) -> float:
    """
    Rank for the briefing: how often, weighted by how recently.

    Mentions alone would put a subject from four months ago above the client you
    saw on Tuesday. Recency alone would put a single passing remark above the
    thing that has come up in every meeting since March. The floor of 0.25 keeps
    an old-but-still-live entry from collapsing to nothing.
    """
    days = _decay_days(cfg, entry.kind)
    last = _parse_when(entry.last_seen)
    if last is None:
        return 0.0
    age_days = max(0.0, (now - last).total_seconds() / 86400.0)
    freshness = 1.0 if days <= 0 else max(0.0, 1.0 - age_days / days)
    score = entry.mentions * (0.25 + 0.75 * freshness)
    if entry.open:
        # An unkept promise is the whole reason someone would read this.
        score *= float(cfg.get("memory.commitment_boost", 2.0))
    return score


# =========================================================================
# The store
# =========================================================================
class MemoryStore:
    """
    Loads, updates, and persists one ledger per profile.

    Every write goes through the vault. When the vault is locked this object
    keeps working in memory and refuses to persist, and says so in `problems`
    rather than in a log line nobody reads. What it never does is write the
    ledger in the clear because encryption was inconvenient at that moment.
    """

    def __init__(self, cfg, vault: Vault | None = None):
        self.cfg = cfg
        self.vault = vault or Vault(cfg.path("vault"))
        self.dir = _memory_dir(cfg)
        self.problems: list[str] = []
        self._cache: dict[str, Ledger] = {}

    # ---- plumbing -------------------------------------------------------
    def _problem(self, message: str) -> None:
        if message not in self.problems:
            self.problems.append(message)
        log.warning("%s", message)

    @staticmethod
    def _aad(profile_id: str) -> bytes:
        """
        Binds a ledger file to its profile.

        This is the isolation guarantee at rest. Copy the Husband ledger over the
        Insurance Agent one and it does not decrypt, so the failure is a refusal
        rather than a briefing about your marriage appearing in a client prompt.
        """
        return f"memory:{profile_id}".encode()

    def path_for(self, profile_id: str) -> Path:
        return self.dir / f"{profile_id}.ledger.json.enc"

    def ready(self) -> tuple[bool, str]:
        return self.vault.ready()

    # ---- load / save ----------------------------------------------------
    def load(self, profile_id: str) -> Ledger:
        """Read one ledger from disk. A missing or unreadable file reads empty."""
        path = self.path_for(profile_id)
        if not path.exists():
            return Ledger(profile_id=profile_id)

        ok, why = self.ready()
        if not ok:
            self._problem(
                f"the {profile_id} ledger is on disk but cannot be opened: {why} "
                "Nothing will be carried forward for this profile until it is set."
            )
            return Ledger(profile_id=profile_id)

        try:
            raw = self.vault.decrypt_bytes(path.read_bytes(), self._aad(profile_id))
            data = json.loads(raw.decode("utf-8"))
        except (VaultError, ValueError, OSError) as exc:
            self._problem(
                f"could not read the {profile_id} ledger ({exc}). It will be treated "
                f"as empty. Rebuild it from the archive with: run.py memory --rebuild"
            )
            return Ledger(profile_id=profile_id)

        stored = str(data.get("profile_id") or profile_id)
        if stored != profile_id:
            # Belt and braces behind the AAD. If this ever fires, something has
            # moved files around by hand and the safe answer is to know nothing.
            self._problem(
                f"the ledger file for '{profile_id}' says it belongs to '{stored}'. "
                "Refusing to use it; one profile's memory does not go into another's."
            )
            return Ledger(profile_id=profile_id)
        return Ledger.from_dict(data, profile_id)

    def ledger(self, profile_id: str) -> Ledger:
        """The ledger for one profile, loaded once and then kept in memory."""
        if profile_id not in self._cache:
            self._cache[profile_id] = self.load(profile_id)
        return self._cache[profile_id]

    def save(self, profile_id: str | None = None) -> list[str]:
        """
        Persist one loaded ledger, or all of them. Returns what was written.

        An empty return with a locked vault is not a silent failure: `problems`
        carries the sentence explaining it, and the caller is expected to print
        it. Writing plaintext instead is not on the table -- this file is a
        distilled record of private conversations.
        """
        if not self.cfg.get("memory.enabled", True):
            return []

        targets = [profile_id] if profile_id else sorted(self._cache)
        ok, why = self.ready()
        if not ok:
            self._problem(
                f"not writing {len(targets)} ledger(s): {why} The ledger distils "
                "private conversations and is as sensitive as the transcripts it "
                "came from, so it is not written unencrypted instead. Everything "
                "learned this run is still in the archive and a later "
                "`run.py memory --rebuild` will recover it."
            )
            return []

        written: list[str] = []
        for pid in targets:
            ledger = self._cache.get(pid)
            if ledger is None:
                continue
            if ledger.profile_id != pid:
                raise MemoryLedgerError(
                    f"refusing to write the '{ledger.profile_id}' ledger to the "
                    f"'{pid}' file. Profiles do not share memory."
                )
            ledger.sort()
            ledger.updated_at = _iso(_now())
            payload = json.dumps(ledger.to_dict(), ensure_ascii=False, sort_keys=True)
            try:
                blob = self.vault.encrypt_bytes(payload.encode("utf-8"), self._aad(pid))
            except VaultError as exc:
                self._problem(f"could not encrypt the {pid} ledger ({exc}); it was not written.")
                continue

            dest = self.path_for(pid)
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".tmp")
            try:
                tmp.write_bytes(blob)
                os.replace(tmp, dest)
                os.chmod(dest, 0o600)
            except OSError as exc:
                tmp.unlink(missing_ok=True)
                self._problem(f"could not write the {pid} ledger to {dest} ({exc}).")
                continue
            written.append(pid)
        return written

    # ---- updating -------------------------------------------------------
    def update_from_record(self, record: Any, *, now: datetime | None = None,
                           save: bool = True) -> list[str]:
        """
        Fold one analysed recording into the ledgers of the profiles it matched.

        Takes either a `Recording` or the payload dict the index stores, because
        the live pipeline has the former and `rebuild` has the latter, and they
        have to produce identical results or rebuild means nothing.

        Idempotent. The same recording applied twice is a no-op; a recording
        whose analyses have changed has its old contribution removed first.
        Returns the profile ids whose ledger actually changed.
        """
        if not self.cfg.get("memory.enabled", True):
            return []

        payload = record.to_dict() if hasattr(record, "to_dict") else dict(record or {})
        recording_id = str(payload.get("id") or "").strip()
        if not recording_id:
            self._problem("a record with no id was handed to memory; it was not filed.")
            return []

        analyses = [a for a in (payload.get("analyses") or []) if isinstance(a, dict)]
        if not analyses:
            return []

        # An analysis whose fields the index deliberately withholds is not an
        # analysis with nothing in it. Filing it would mark the recording as
        # seen, and the real contents would then never be picked up.
        withheld = [a for a in analyses if a.get("fields_withheld")]
        if withheld:
            self._problem(
                f"{recording_id} was handed to memory with its analysis fields "
                "withheld, which is what the index holds for an encrypted "
                "recording. Nothing was filed. Read it through Archive.full_record "
                "so the vault copy is used."
            )
            return []

        when = (
            _parse_when(payload.get("recorded_at"))
            or _parse_when(payload.get("ingested_at"))
            or (now or _now())
        )
        stamp = _iso(when)
        source_name = str(payload.get("source_name", ""))
        signature = _signature(analyses)

        touched: list[str] = []
        for analysis in analyses:
            profile_id = str(analysis.get("profile_id") or "").strip()
            profile = self.cfg.profiles.get(profile_id)
            if profile is None:
                log.debug("memory: no profile '%s' is configured; skipping", profile_id)
                continue

            # The family profiles are told to stop and flag rather than summarise
            # anything concerning, and the extractor discards the rest of the
            # analysis when they do. Carrying that forward into a future prompt is
            # precisely what the flag exists to prevent.
            if analysis.get("requires_human_attention") or (
                analysis.get("fields") or {}
            ).get("requires_human_attention"):
                log.info(
                    "memory: %s is flagged for human attention; not filing it under %s",
                    recording_id, profile_id,
                )
                continue

            ledger = self.ledger(profile_id)
            if ledger.seen.get(recording_id) == signature:
                continue
            if recording_id in ledger.seen:
                # Re-analysed, with different results. Remove what the old pass
                # contributed rather than adding a second copy alongside it.
                ledger.purge(recording_id)

            self._apply(ledger, profile, analysis, recording_id, stamp, source_name)
            ledger.seen[recording_id] = signature
            ledger.sort()
            if profile_id not in touched:
                touched.append(profile_id)

        if touched and save:
            self.save()
        return touched

    def _apply(self, ledger: Ledger, profile, analysis: dict[str, Any],
               recording_id: str, stamp: str, source_name: str) -> None:
        if ledger.profile_id != profile.id:
            raise MemoryLedgerError(
                f"refusing to file a '{profile.id}' analysis into the "
                f"'{ledger.profile_id}' ledger. Profiles do not share memory."
            )

        fields = analysis.get("fields") or {}
        per_field = int(self.cfg.get("memory.max_items_per_field", 12))
        max_chars = int(self.cfg.get("memory.max_entry_chars", 200))
        closure_fields = _closure_fields(self.cfg)
        owner = str(self.cfg.get("diarization.owner_label", "")).strip().lower()

        closures: list[str] = []
        for field_key, value in sorted(fields.items()):
            if field_key in closure_fields:
                closures += _phrases(value, per_field)
                continue

            spec = profile.field_by_key(field_key)
            if spec is None:
                # A field the profile no longer declares carries no policy we can
                # read, and a field whose sensitivity is unknown gets treated as
                # sensitive. Old analyses from a since-edited schema are skipped
                # rather than filed on a guess.
                log.debug("memory: %s no longer declares '%s'; skipping", profile.id, field_key)
                continue
            if spec.sensitive or field_key in profile.suppress_fields:
                # Health and financial disclosures stay in the vault artifact.
                # The ledger is read back into a prompt; these do not go there.
                continue

            kind = _kind_for(self.cfg, field_key)
            if not kind:
                continue

            for text in _phrases(value, per_field):
                if kind == PERSON:
                    if _SPEAKER_LABEL.match(text) or text.strip().lower() == owner:
                        # A diarization placeholder is not a person, and you are
                        # not news to yourself.
                        continue
                body = text[:max_chars].strip()
                key = _key_for(body if kind != PERSON else body.lower())
                if not key:
                    continue
                ledger.record(kind, key, Sighting(
                    recording_id=recording_id, when=stamp, field_key=field_key,
                    text=body, source_name=source_name,
                ))

        for note in closures:
            self._close_matching(ledger, note, recording_id, stamp)

    def _close_matching(self, ledger: Ledger, note: str, recording_id: str, stamp: str) -> None:
        threshold = float(self.cfg.get("memory.closure_overlap", 0.75))
        for entry in ledger.entries:
            if not entry.open:
                continue
            if _overlap(entry.text, note) < threshold:
                continue
            entry.closed_by = recording_id
            entry.closed_at = stamp
            entry.closed_note = note[: int(self.cfg.get("memory.max_entry_chars", 200))]

    def close_commitment(self, profile_id: str, key: str, recording_id: str,
                         note: str = "", *, save: bool = True) -> bool:
        """Close one commitment by hand. Returns whether anything matched."""
        entry = self.ledger(profile_id).entry(COMMITMENT, key)
        if entry is None or not entry.open:
            return False
        entry.closed_by = recording_id
        entry.closed_at = _iso(_now())
        entry.closed_note = note
        if save:
            self.save(profile_id)
        return True

    # ---- rebuilding -----------------------------------------------------
    def rebuild(self, db, archive=None, *, force: bool = False) -> RebuildReport:
        """
        Throw every ledger away and replay the archive.

        This is the check that the ledger is derived rather than authoritative.
        If a rebuild produces a different ledger from the one that was
        maintained incrementally, the incremental path has a bug, and the
        rebuild is the answer to believe.

        A rebuild that could not open everything does not overwrite what you
        already have unless you force it, because a ledger missing the twelve
        recordings that would not decrypt looks exactly like a complete one.
        """
        report = RebuildReport()

        total = db.count_recordings()
        rows = db.query(limit=max(total, 1))
        # The archive hands them back newest first; memory has to be built in
        # the order the conversations happened or closure and "last seen" come
        # out wrong.
        rows.sort(key=lambda r: (str(r.get("recorded_at") or r.get("ingested_at") or ""),
                                 str(r.get("id") or "")))

        self._cache = {pid: Ledger(profile_id=pid) for pid in self.cfg.profiles}

        for row in rows:
            payload: dict[str, Any] | None
            if archive is not None:
                payload = archive.full_record(row)
            else:
                try:
                    payload = json.loads(row["payload_json"])
                except (KeyError, TypeError, ValueError):
                    payload = None
            if payload is None:
                report.unopened.append(f"{row.get('id', '?')}  {row.get('source_name', '')}")
                continue
            payload.setdefault("id", row.get("id"))
            self.update_from_record(payload, save=False)
            report.replayed += 1

        report.ledgers = dict(self._cache)
        if report.complete or force:
            report.written = self.save()
            report.saved = bool(report.written) or not self._cache
        return report

    # ---- forgetting -----------------------------------------------------
    def _known_profiles(self) -> list[str]:
        """Every profile with a ledger: configured, cached, or sitting on disk."""
        ids = set(self.cfg.profiles) | set(self._cache)
        if self.dir.is_dir():
            for path in self.dir.glob("*.ledger.json.enc"):
                ids.add(path.name[: -len(".ledger.json.enc")])
        return sorted(ids)

    def forget_recording(self, recording_id: str) -> list[str]:
        """
        Remove everything one recording taught, from every profile.

        `forget` promises a recording is gone. A ledger still carrying its
        promises and the names of the people in it, and feeding them into the
        next prompt, would make that promise false in the most visible way
        possible.
        """
        known = self._known_profiles()
        ok, why = self.ready()
        if not ok and any(self.path_for(pid).exists() for pid in known):
            # Locked, so the ledgers on disk cannot even be read, let alone
            # edited. Saying nothing here would let `forget` report a clean
            # deletion over files that still hold the recording's contents.
            self._problem(
                f"{recording_id} could not be removed from the memory ledgers: {why} "
                "The ledgers on disk still hold what this recording taught. Set the "
                f"passphrase and run: run.py memory --forget {recording_id}"
            )
            return []

        changed: list[str] = []
        for profile_id in known:
            if self.ledger(profile_id).purge(recording_id):
                changed.append(profile_id)

        for profile_id in changed:
            if not self.save(profile_id):
                # Refusing to write is right; leaving the caller thinking the
                # deletion completed is not.
                self._problem(
                    f"the {profile_id} ledger still remembers {recording_id} on disk. "
                    "The in-memory copy was cleared but nothing could be written. "
                    "Set PLAUD_BRIDGE_PASSPHRASE and run: run.py memory --forget "
                    f"{recording_id}"
                )
        return changed

    def forget_profile(self, profile_id: str) -> bool:
        """Delete one profile's ledger outright. Returns whether a file was removed."""
        self._cache.pop(profile_id, None)
        path = self.path_for(profile_id)
        if not path.exists():
            return False
        try:
            path.unlink()
        except OSError as exc:
            self._problem(f"could not delete {path} ({exc}).")
            return False
        log.info("memory: deleted the %s ledger", profile_id)
        return True


def _signature(analyses: list[dict[str, Any]]) -> str:
    """A short hash of what an analysis contributed, for idempotency."""
    body = [
        {
            "profile_id": a.get("profile_id", ""),
            "fields": a.get("fields") or {},
            "requires_human_attention": bool(a.get("requires_human_attention")),
        }
        for a in analyses
    ]
    body.sort(key=lambda a: str(a["profile_id"]))
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()[:16]


# =========================================================================
# Rendering
# =========================================================================
def carry_forward_brief(cfg, profile_id: str, store: MemoryStore | None = None, *,
                        budget: int | None = None, now: datetime | None = None) -> str:
    """
    The "what you already know" briefing, for injection into an extraction prompt.

    Plain text, bounded by a character budget, ranked so the things most likely
    to matter survive the cut. Returns an empty string when there is nothing to
    say, when the profile is unknown, or when the vault is locked -- an empty
    brief adds nothing to a prompt, which is the correct behaviour for "I could
    not read my own notes".

    Redacted on the way out even though it is going to a local model, because
    this text is the one thing here written specifically to be handed to a
    model, and which model that is depends on config nobody re-reads.
    """
    if not cfg.get("memory.enabled", True):
        return ""
    profile = cfg.profiles.get(profile_id)
    if profile is None:
        log.warning("memory: no profile '%s' is configured; no brief", profile_id)
        return ""

    store = store or MemoryStore(cfg)
    ledger = store.ledger(profile_id)
    if not ledger.entries:
        return ""

    limit = int(budget if budget is not None else cfg.get("memory.brief_budget_chars", 1500))
    if limit <= 0:
        return ""

    now = now or _now()
    live = [
        e for e in ledger.entries
        if not _stale(cfg, e, now)
        and (e.open or e.kind != COMMITMENT)   # a closed promise is history, not context
        and e.kind in _SECTION_TITLES
    ]
    if not live:
        return ""
    live.sort(key=lambda e: (-_score(cfg, e, now), -_ordinal(e.last_seen), e.key))

    patterns = cfg.get("compliance.redact_patterns") or {}

    def safe(text: str) -> str:
        return redact_text(text, patterns, enabled=profile.redact_before_llm)[0]

    header = [
        safe(
            f"WHAT YOU ALREADY KNOW ({profile.name}, carried forward from "
            f"{len(ledger.seen)} earlier recording(s))"
        ),
        # Kept short on purpose: it is spent out of the same budget as the
        # content, and a preamble long enough to crowd out the notes it
        # introduces would be its own kind of failure.
        "Background from earlier recordings; it is not part of the transcript below. "
        "Do not quote it or copy it into your output.",
    ]

    # Grouped for readability, but chosen in global rank order, so a long list of
    # commitments cannot crowd out the person you saw yesterday.
    chosen: dict[str, list[str]] = {}
    used = len("\n".join(header))
    for entry in live:
        title = _SECTION_TITLES[entry.kind]
        line = safe("- " + _brief_line(entry))
        cost = len(line) + 1 + (0 if title in chosen else len(title) + 2)
        if used + cost > limit:
            # Keep filling with whatever still fits rather than stopping at the
            # first line too long: the tail is cheap and the budget is there to
            # be used.
            continue
        chosen.setdefault(title, []).append(line)
        used += cost

    if not chosen:
        return ""

    out = list(header)
    for kind in KINDS:
        title = _SECTION_TITLES[kind]
        if title in chosen:
            out += ["", f"{title}:"] + chosen[title]

    text = "\n".join(out)
    while len(text) > limit and len(out) > len(header):
        # Belt and braces on the arithmetic above. Overrunning the budget is a
        # broken prompt; a slightly shorter brief is not.
        out.pop()
        text = "\n".join(out)
    return text


def _ordinal(when: str) -> float:
    parsed = _parse_when(when)
    return parsed.timestamp() if parsed else 0.0


def _brief_line(entry: Entry) -> str:
    day = (entry.last_seen or "")[:10]
    if entry.kind == PERSON:
        return f"{entry.text} (last {day}, {entry.mentions} mention(s))"
    if entry.kind == COMMITMENT:
        return f"{entry.text} (open since {(entry.first_seen or '')[:10]})"
    suffix = f", {entry.mentions} mention(s)" if entry.mentions > 1 else ""
    return f"{entry.text} ({day}{suffix})"


def render_ledger(ledger: Ledger, *, cfg=None, now: datetime | None = None,
                  sightings: int = 3) -> str:
    """
    The whole ledger, for a person reading it rather than a model.

    Shows stale entries marked as stale rather than hiding them: "this stopped
    being surfaced three weeks ago" is a useful thing to be able to see, and it
    is the only way to tell decay apart from an entry that was never recorded.
    """
    if not ledger.entries:
        return (
            f"{ledger.profile_id}: nothing recorded yet. Memory is built as recordings "
            "are analysed; run something through first, or rebuild from the archive "
            "with: run.py memory --rebuild"
        )

    now = now or _now()
    lines = [
        f"{ledger.profile_id}: {len(ledger.entries)} entr(ies) from "
        f"{len(ledger.seen)} recording(s)"
        + (f", updated {ledger.updated_at[:16].replace('T', ' ')}" if ledger.updated_at else ""),
        f"fingerprint {ledger.fingerprint()}",
    ]

    for kind in KINDS:
        entries = [e for e in ledger.entries if e.kind == kind]
        if not entries:
            continue
        entries.sort(key=lambda e: (-_ordinal(e.last_seen), e.key))
        lines += ["", f"{_SECTION_TITLES[kind]} ({len(entries)}):"]
        for entry in entries:
            state = "stale " if cfg is not None and _stale(cfg, entry, now) else ""
            if entry.kind == COMMITMENT:
                state = "closed" if entry.closed_by else (state or "open  ")
            lines.append(f"  [{state.strip() or 'live'}] {entry.text}")
            for sighting in entry.sightings[-sightings:]:
                name = f" {sighting.source_name}" if sighting.source_name else ""
                lines.append(
                    f"        {sighting.when[:10]}  {sighting.recording_id}"
                    f"  ({sighting.field_key}){name}"
                )
            if len(entry.sightings) > sightings:
                lines.append(f"        ... and {len(entry.sightings) - sightings} earlier")
            if entry.closed_by:
                lines.append(
                    f"        closed by {entry.closed_by} on {entry.closed_at[:10]}"
                    + (f": {entry.closed_note}" if entry.closed_note else "")
                )
    return "\n".join(lines)
