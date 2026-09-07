"""
Commitments, tracked until they are actually done.

Plaud sells a feature called AutoFlow: it summarises a meeting and emails the
summary out for you. Half of that is genuinely useful. You told a client you
would have two quotes to him by Thursday, and by Wednesday you have had four
more conversations and the promise is gone. The other half is a third-party
cloud service mailing a summary of a private conversation to somebody, which is
the precise thing the rest of this tool exists to avoid.

So this is the useful half on its own. It reads the commitments already
extracted per profile, collapses the same promise mentioned across several
recordings into one item, ages it so the eleven-day-old one sorts above the
one from this morning, and remembers when you mark it done. When you want to
send something, it writes a DRAFT into the outbox for you to read, edit, and
send yourself, from your own mail client, under your own name.

**It never sends anything and it never talks to a mail server.** That is a
design decision, not a missing feature. A tool that can send on your behalf is
a tool that can send the wrong summary to the wrong client while you are
driving, and the only reliable way to be certain that never happens is to not
build the capability at all.

Three rules inherited from the rest of the system, because a follow-up is made
of the same words the recording was:

  - **The strictest profile involved governs the whole operation.** One item
    from a locked profile in the set and the entire draft is phrased locally or
    not at all. Not per-item. Whole draft. Same reasoning as ADR-002.
  - **Redaction happens before a model sees anything, and before a draft is
    written to disk.** A draft is by definition a document meant to leave this
    machine, so it gets the treatment an export gets rather than the treatment
    a digest gets.
  - **Nothing is invented.** Every follow-up carries the id of a recording that
    is really in the index, and a draft is built only out of follow-ups that
    were collected from real analyses. There is no path here that writes a
    sentence nobody said.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from .compliance.redact import redact_text
from .digest import fmt_value, to_html
from .llm import complete_json
from .llm.base import LLMError
from .logging_setup import get
from .storage import Vault, VaultError
from .storage.vault import MAGIC as _VAULT_MAGIC

log = get("followups")

STATUSES = ("open", "done", "dropped")

# Which extraction fields hold something a person owes somebody. The profile
# schemas are config, and they do not agree on a name: the same idea is
# `promises_i_made` for a father, `commitments_by_producer` for a producer, and
# `action_items` in the unfiled catch-all. Rather than hardcode that list and
# have it rot the first time somebody adds a profile, the field is matched on
# what it is called. Override with followups.field_tokens, or name the fields
# outright per profile with followups.fields.
DEFAULT_FIELD_TOKENS = (
    "commitment", "promise", "action", "follow_up", "followup",
    "task", "todo", "asked_for", "asks_from", "next_step",
)

# Values a model returns to mean "there was nothing here". Treating these as
# real follow-ups is how a worklist fills up with the word "none".
_EMPTY_ANSWERS = frozenset({"none", "n/a", "na", "nothing", "-", "tbd", "unknown"})

_BODY_KEYS = ("text", "quote", "statement", "content", "what", "action", "item", "description")
_WHO_KEYS = ("speaker", "who", "owner", "party", "counterparty", "for", "with")
_WHEN_KEYS = ("when", "due", "date", "deadline", "by")

# The state file is bound to this string rather than to a recording id, because
# it belongs to no single recording.
_STATE_AAD = "followups"
_STATE_VERSION = 1

_NON_WORD = re.compile(r"[^a-z0-9]+")


class FollowUpError(RuntimeError):
    """Raised when a follow-up operation cannot be completed honestly."""


# =========================================================================
# The item
# =========================================================================
@dataclass
class FollowUp:
    """
    One thing somebody owes somebody, and everything needed to chase it.

    `id` is derived from the wording, so the same promise restated in three
    recordings is one item mentioned three times rather than three items. That
    is the whole reason this is not just a filter over the digest.
    """

    id: str
    text: str
    profile_id: str
    recording_id: str
    field_key: str = ""
    counterparty: str = ""
    due: str = ""
    first_seen: str = ""          # YYYY-MM-DD of the earliest recording mentioning it
    last_seen: str = ""           # YYYY-MM-DD of the most recent one
    status: str = "open"
    mentions: int = 1
    recording_ids: list[str] = field(default_factory=list)
    source_name: str = ""

    @property
    def age_days(self) -> int:
        """How long this has been outstanding. The number the sort is built on."""
        return _days_since(self.first_seen)

    @property
    def is_open(self) -> bool:
        return self.status == "open"

    @property
    def short_id(self) -> str:
        """Enough of the id to type. `set_status` accepts any unique prefix."""
        return self.id[:11]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "profile_id": self.profile_id,
            "recording_id": self.recording_id,
            "field_key": self.field_key,
            "counterparty": self.counterparty,
            "due": self.due,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "status": self.status,
            "mentions": self.mentions,
            "recording_ids": list(self.recording_ids),
            "source_name": self.source_name,
            "age_days": self.age_days,
        }


def stable_id(text: str, profile_id: str) -> str:
    """
    An id derived from what was said, not from when it was said.

    Punctuation, casing, and whitespace are thrown away first, so "I'll email
    you tonight." and "I'll email you tonight" are the same commitment. The
    profile is part of the hash because the same sentence in a client call and
    in a conversation at home is not the same obligation and should not merge.
    """
    normal = _NON_WORD.sub(" ", text.lower()).strip()
    digest = hashlib.sha256(f"{profile_id}|{normal}".encode()).hexdigest()
    return f"fu_{digest[:12]}"


def sort_key(item: FollowUp) -> tuple[int, int, str]:
    """Open before closed, oldest first. An eleven-day-old promise goes on top."""
    return (0 if item.is_open else 1, -item.age_days, item.text.lower())


# =========================================================================
# Which fields count
# =========================================================================
def commitment_fields(cfg, profile) -> list[str]:
    """
    The extraction fields of one profile that hold commitments.

    An explicit `followups.fields.<profile_id>` list wins outright, for the case
    where a field is called something the tokens will never match. Otherwise the
    key and the label are both checked, so "What They Asked For" is found by the
    `asked_for` token even though its key is `asks_from_them`.
    """
    explicit = cfg.get(f"followups.fields.{profile.id}")
    if explicit:
        known = set(profile.field_keys)
        chosen = [str(k) for k in explicit if str(k) in known]
        missing = [str(k) for k in explicit if str(k) not in known]
        if missing:
            log.warning(
                "followups.fields.%s names %s, which profile '%s' does not extract; "
                "ignoring those",
                profile.id, ", ".join(missing), profile.id,
            )
        return chosen

    tokens = [str(t).lower() for t in (cfg.get("followups.field_tokens") or DEFAULT_FIELD_TOKENS)]
    chosen = []
    for spec in profile.fields:
        haystack = f"{spec.key.lower()} {spec.label.lower().replace(' ', '_')}"
        if any(token in haystack for token in tokens):
            chosen.append(spec.key)
    return chosen


# =========================================================================
# Collecting
# =========================================================================
def collect(cfg, db, archive, *, profile: str | None = None, days: int | None = None,
            status: str | None = None, include_personal: bool = False,
            recording_id: str | None = None, vault: Vault | None = None,
            limit: int | None = None) -> list[FollowUp]:
    """
    Every follow-up in the window, deduplicated and aged, worst first.

    `status` filters to one of open/done/dropped; None returns all of them so a
    caller can show what was closed as well as what is outstanding. Personal
    profiles are omitted unless asked for by name or with `include_personal`,
    the same rule the combined digest follows and for the same reason.

    A recording whose analysis will not decrypt is reported and skipped rather
    than guessed at. An incomplete worklist is bad; an invented one is worse.
    """
    if status is not None and status not in STATUSES:
        raise FollowUpError(f"unknown status '{status}'. Use one of: {', '.join(STATUSES)}")

    vault = vault or getattr(archive, "vault", None) or Vault(cfg.path("vault"))
    saved = _load_state(cfg, vault)

    personal = {p.id for p in cfg.profiles.values() if p.exclude_from_combined_export}
    max_rows = limit or int(cfg.get("followups.max_recordings", 500))
    max_chars = int(cfg.get("followups.max_text_chars", 400))

    rows = db.query(profile_id=profile, since_days=days, limit=max_rows)
    if recording_id:
        rows = [r for r in rows if r["id"] == recording_id]
        if not rows:
            raise FollowUpError(
                f"no recording with id '{recording_id}' in the last "
                f"{days if days is not None else 'unbounded'} day window. "
                "A follow-up has to trace to a recording that exists."
            )

    merged: dict[str, FollowUp] = {}
    unopened: list[str] = []

    for row in rows:
        record = archive.full_record(row)
        if record is None:
            unopened.append(f"{row['id']}  {row['source_name']}")
            continue

        when = ((row["recorded_at"] or row["ingested_at"] or "") or "")[:10]
        for analysis in record.get("analyses", []):
            pid = str(analysis.get("profile_id", ""))
            if pid not in cfg.profiles:
                continue
            if profile is not None and pid != profile:
                # `db.query(profile_id=...)` joins on routes, so a recording
                # co-routed to the asked-for profile but GOVERNED by a stricter
                # one comes back carrying that stricter profile's analysis too.
                # Reading it here is how `followups --profile insurance_agent`
                # surfaces a Husband or Father commitment. Filtering to the
                # requested profile keeps a scoped query scoped.
                continue
            if pid in personal and profile is None and not include_personal:
                continue
            if analysis.get("error"):
                # The extraction failed for this profile. Its field dict is the
                # schema's defaults, so anything read out of it would be an
                # artifact of the failure rather than something anyone said.
                continue

            prof = cfg.profile(pid)
            fields = analysis.get("fields") or {}
            for key in commitment_fields(cfg, prof):
                for raw in _candidates(fields.get(key)):
                    item = _to_followup(raw, pid, key, row, when, max_chars)
                    if item is None:
                        continue
                    _merge(merged, item)

    if unopened:
        log.warning(
            "%d recording(s) could not be opened and were not searched for "
            "follow-ups: %s. Set PLAUD_BRIDGE_PASSPHRASE if they are encrypted.",
            len(unopened), "; ".join(unopened[:5]),
        )

    items = list(merged.values())
    for item in items:
        entry = saved.get(item.id)
        if entry:
            item.status = str(entry.get("status", "open"))

    if status is not None:
        items = [i for i in items if i.status == status]

    items.sort(key=sort_key)
    return items


def _candidates(value: Any) -> list[Any]:
    """One field value, flattened into the things it might contain."""
    if value is None or value == "" or value == [] or value == {}:
        return []
    if isinstance(value, bool):
        # A boolean is a flag, not an obligation. `requires_human_attention`
        # would otherwise turn into a follow-up that reads "yes".
        return []
    return list(value) if isinstance(value, list) else [value]


def _to_followup(raw: Any, profile_id: str, field_key: str, row: dict[str, Any],
                 when: str, max_chars: int) -> FollowUp | None:
    text = _body(raw)[:max_chars].strip()
    if not text or text.lower().strip(".") in _EMPTY_ANSWERS:
        return None

    counterparty, due = "", ""
    if isinstance(raw, dict):
        counterparty = _first_string(raw, _WHO_KEYS)
        due = _first_string(raw, _WHEN_KEYS)

    return FollowUp(
        id=stable_id(text, profile_id),
        text=text,
        profile_id=profile_id,
        recording_id=row["id"],
        field_key=field_key,
        counterparty=counterparty,
        due=due,
        first_seen=when,
        last_seen=when,
        mentions=1,
        recording_ids=[row["id"]],
        source_name=row["source_name"],
    )


def _body(raw: Any) -> str:
    """
    The words of the commitment, without the timestamp wrapper.

    The digest renders a quote as "[00:45] Sasson: ...", which is right for
    reading and wrong here: the timestamp differs between two recordings of the
    same promise, and anything in the text goes into the id. So the body keys
    are read directly, and `fmt_value` is the fallback for a shape nothing here
    recognises -- same principle the digest applies, an ugly line beats a
    silently dropped one.
    """
    if isinstance(raw, dict):
        body = _first_string(raw, _BODY_KEYS)
        if body:
            return body
        rendered = fmt_value(raw, limit=1)
        return rendered[0] if rendered else ""
    return str(raw).strip()


def _first_string(item: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _merge(merged: dict[str, FollowUp], item: FollowUp) -> None:
    """Fold a repeated commitment into the one already collected."""
    existing = merged.get(item.id)
    if existing is None:
        merged[item.id] = item
        return

    existing.mentions += 1
    if item.recording_id not in existing.recording_ids:
        existing.recording_ids.append(item.recording_id)
    if item.first_seen and (not existing.first_seen or item.first_seen < existing.first_seen):
        # The recording it traces to is the one where the promise was made, not
        # the most recent time it came up again.
        existing.first_seen = item.first_seen
        existing.recording_id = item.recording_id
        existing.source_name = item.source_name
    if item.last_seen > existing.last_seen:
        existing.last_seen = item.last_seen
    existing.counterparty = existing.counterparty or item.counterparty
    existing.due = existing.due or item.due


def _days_since(stamp: str) -> int:
    if not stamp:
        return 0
    try:
        seen = date.fromisoformat(stamp[:10])
    except ValueError:
        return 0
    return max(0, (datetime.now(timezone.utc).date() - seen).days)


# =========================================================================
# Status, persisted
# =========================================================================
def state_path(cfg) -> Path:
    """
    Where the status file lives.

    Beside the SQLite index rather than inside the vault or the outbox, because
    `verify` walks those two directories and reports every file the artifact
    table does not know about as an orphan. A status file is neither an artifact
    nor an orphan, and having it appear in every verify run as something
    possibly-wrong is how you train yourself to skim that list.
    """
    return cfg.path("database").parent / "followups.state"


def _load_state(cfg, vault: Vault) -> dict[str, dict[str, Any]]:
    """
    Read the persisted statuses.

    Raises rather than returning empty when the file exists and will not open.
    An unreadable state file silently treated as "no statuses" marks every
    finished item open again, which is exactly the resurrection this file
    exists to prevent, and it happens on the day you forgot the passphrase.
    """
    path = state_path(cfg)
    if not path.exists():
        return {}

    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise FollowUpError(f"cannot read {path}: {exc}") from exc

    if raw.startswith(_VAULT_MAGIC):
        ok, why = vault.ready()
        if not ok:
            raise FollowUpError(
                f"{path} is encrypted and {why} Follow-up statuses cannot be read, "
                "and showing every completed item as open again would be worse "
                "than showing nothing."
            )
        try:
            text = vault.decrypt_bytes(raw, _STATE_AAD.encode("utf-8")).decode("utf-8")
        except VaultError as exc:
            raise FollowUpError(f"{path} will not decrypt: {exc}") from exc
    else:
        text = raw.decode("utf-8", "replace")

    try:
        data = json.loads(text)
    except ValueError as exc:
        raise FollowUpError(
            f"{path} is not valid JSON ({exc}). Move it aside to start again; "
            "you will lose which follow-ups were marked done, and nothing else."
        ) from exc

    items = data.get("items") if isinstance(data, dict) else None
    return items if isinstance(items, dict) else {}


def _save_state(cfg, vault: Vault, items: dict[str, dict[str, Any]]) -> Path:
    """
    Write the statuses back, encrypted when a passphrase is available.

    What is stored is ids, statuses, dates, and the recording each item came
    from. Never the wording. The id is a one-way hash of the text, so the
    plaintext fallback -- for someone running without a passphrase, who has no
    vault at all -- still leaks nothing about what anyone said. The encrypted
    form is used whenever it can be, because the recording ids alone say more
    than they look like they do.
    """
    path = state_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {"version": _STATE_VERSION, "items": items}, indent=2, ensure_ascii=False,
    )

    ok, why = vault.ready()
    if ok:
        blob = vault.encrypt_bytes(payload.encode("utf-8"), _STATE_AAD.encode("utf-8"))
    else:
        log.warning(
            "writing follow-up statuses unencrypted: %s The file holds hashed ids "
            "and recording ids, never the wording of a commitment.", why,
        )
        blob = payload.encode("utf-8")

    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(blob)
    os.replace(tmp, path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return path


def set_status(cfg, vault: Vault, followup_id: str, status: str, *,
               items: list[FollowUp] | None = None) -> FollowUp:
    """
    Mark one follow-up done, dropped, or open again, and remember it.

    `followup_id` may be any unique prefix of a real id. Passing `items` -- the
    list `collect` just returned -- lets the answer come back fully populated
    and is how the CLI calls this. Without it the id is resolved against the
    state file, which deliberately holds no wording, so the returned item has
    an empty `text`.

    An id that has never been collected is refused. Recording a status against
    something no recording produced would be the first invented follow-up in
    the system.
    """
    if status not in STATUSES:
        raise FollowUpError(f"unknown status '{status}'. Use one of: {', '.join(STATUSES)}")

    saved = _load_state(cfg, vault)
    match = _resolve(followup_id, items or [], saved)
    now = datetime.now(timezone.utc)

    entry = dict(saved.get(match.id) or {})
    entry.update({
        "status": status,
        "recording_id": match.recording_id,
        # The full set of recordings this commitment was seen in, not just the
        # primary. Forgetting one of several must not delete a done/dropped
        # status and let the commitment re-derive as open from the recordings
        # that remain -- so the whole set is stored, and forget_recording trims
        # rather than drops while any of them survive.
        "recording_ids": list(match.recording_ids) or [match.recording_id],
        "profile_id": match.profile_id,
        "first_seen": match.first_seen or entry.get("first_seen", ""),
        "last_seen": match.last_seen or entry.get("last_seen", ""),
        "updated_at": now.isoformat(),
    })
    saved[match.id] = entry
    _save_state(cfg, vault, saved)

    match.status = status
    log.info("follow-up %s marked %s (from %s)", match.id, status, match.recording_id)
    return match


def forget_recording(cfg, vault: Vault, recording_id: str) -> list[str]:
    """
    Drop every status entry tied to one recording. Returns the ids removed.

    `forget` promises a recording leaves no trace. The status file keeps, per
    item, the id of the recording a commitment came from and the profile it
    belonged to, so an entry left behind still names a recording that is meant
    to be gone -- and records that a promise from it was marked done. This
    removes every such entry and rewrites the file.

    The caller (`archive.forget`) refuses the whole operation when the vault is
    locked and any encrypted store exists, so reaching here with an unreadable
    encrypted state file is a bug worth surfacing rather than swallowing --
    which `_load_state` does by raising. A plaintext state file (the no-vault
    fallback) is read and rewritten in place, no passphrase required.
    """
    path = state_path(cfg)
    if not path.exists():
        return []

    saved = _load_state(cfg, vault)
    removed: list[str] = []
    changed = False
    for fid, entry in list(saved.items()):
        ids = entry.get("recording_ids") or (
            [entry["recording_id"]] if entry.get("recording_id") else []
        )
        if recording_id not in ids:
            continue
        remaining = [r for r in ids if r != recording_id]
        changed = True
        if not remaining:
            # Every recording this status came from is gone; the commitment
            # cannot re-derive, so the status goes with it.
            saved.pop(fid, None)
            removed.append(fid)
        else:
            # The commitment still exists via another recording, so its status
            # is preserved -- it just stops naming the forgotten one.
            entry["recording_ids"] = remaining
            if str(entry.get("recording_id", "")) == recording_id:
                entry["recording_id"] = remaining[0]
            saved[fid] = entry

    if not changed:
        return []
    _save_state(cfg, vault, saved)
    if removed:
        log.info(
            "follow-ups: dropped %d status entr%s tied to %s",
            len(removed), "y" if len(removed) == 1 else "ies", recording_id,
        )
    return removed


def _resolve(followup_id: str, items: list[FollowUp],
             saved: dict[str, dict[str, Any]]) -> FollowUp:
    """Find one follow-up by exact id or unique prefix, in the list or the state."""
    wanted = followup_id.strip()
    if not wanted:
        raise FollowUpError("no follow-up id given")

    hits = [i for i in items if i.id == wanted] or [i for i in items if i.id.startswith(wanted)]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise FollowUpError(
            f"'{wanted}' matches {len(hits)} follow-ups: "
            + ", ".join(sorted(i.id for i in hits))
            + ". Use more of the id."
        )

    known = [fid for fid in saved if fid == wanted] or [
        fid for fid in saved if fid.startswith(wanted)
    ]
    if len(known) == 1:
        entry = saved[known[0]]
        return FollowUp(
            id=known[0],
            text="",
            profile_id=str(entry.get("profile_id", "")),
            recording_id=str(entry.get("recording_id", "")),
            first_seen=str(entry.get("first_seen", "")),
            last_seen=str(entry.get("last_seen", "")),
            status=str(entry.get("status", "open")),
        )
    if len(known) > 1:
        raise FollowUpError(
            f"'{wanted}' matches {len(known)} known follow-ups. Use more of the id."
        )

    raise FollowUpError(
        f"no follow-up with id '{wanted}'. Run the follow-up list to see the "
        "current ids; they are derived from the wording, so they change if the "
        "commitment was re-extracted differently."
    )


# =========================================================================
# Rendering the worklist
# =========================================================================
def render(items: list[FollowUp], *, fmt: str = "markdown", title: str | None = None) -> str:
    """
    The worklist, in the digest's shape.

    Markdown is the source of truth and HTML is rendered from it by the digest's
    own converter, so the two formats cannot drift into saying different things.
    """
    if fmt not in ("markdown", "html"):
        raise FollowUpError(f"unknown format '{fmt}'. Use markdown or html.")

    heading = title or "Follow-ups"
    body = _render_markdown(items, heading)
    return to_html(body, title=heading) if fmt == "html" else body


def _render_markdown(items: list[FollowUp], heading: str) -> str:
    now = datetime.now(timezone.utc)
    ordered = sorted(items, key=sort_key)
    open_items = [i for i in ordered if i.is_open]
    closed = [i for i in ordered if not i.is_open]

    out: list[str] = [f"# {heading}", ""]

    if not ordered:
        out += [
            "Nothing outstanding. Either everything is done or nothing has been "
            "recorded in this window.",
            "",
        ]
        return "\n".join(out)

    oldest = max((i.age_days for i in open_items), default=0)
    summary = [f"{len(open_items)} open"]
    if open_items:
        summary.append(f"oldest {oldest} day{'' if oldest == 1 else 's'}")
    if closed:
        summary.append(f"{len(closed)} closed")
    summary.append(f"generated {now:%Y-%m-%d %H:%M}")
    out += ["`" + " · ".join(summary) + "`", ""]

    if open_items:
        out += ["## Still open", ""]
        out += ["| Follow-up | Open for | Profile |", "|---|---:|---|"]
        for item in open_items:
            out.append(
                f"| {_cell(item.text)} | {item.age_days}d | {_cell(item.profile_id)} |"
            )
        out.append("")

        for item in open_items:
            out += [f"### {item.text}", ""]
            meta = [
                f"{item.age_days}d open",
                item.profile_id,
                f"first seen {item.first_seen or 'unknown'}",
            ]
            if item.last_seen and item.last_seen != item.first_seen:
                meta.append(f"last mentioned {item.last_seen}")
            if item.mentions > 1:
                meta.append(f"{item.mentions} mentions")
            out += ["`" + " · ".join(meta) + "`", ""]

            if item.counterparty:
                out.append(f"- Said by: {item.counterparty}")
            if item.due:
                out.append(f"- Due: {item.due}")
            out.append(f"- From: `{item.recording_id}` ({item.source_name})")
            out.append(f"- Mark it done: `run.py followups --done {item.short_id}`")
            out.append("")

    if closed:
        out += ["## Closed", ""]
        for item in closed:
            out.append(f"- **{item.status}** — {item.text}  (`{item.short_id}`)")
        out.append("")

    out += [
        "---",
        "",
        "<sub>Collected from analyses already on disk. Nothing here has been sent "
        "to anyone, and this tool cannot send it. `run.py followups --draft <id>` "
        "writes a draft into the outbox for you to read, edit, and send yourself.</sub>",
        "",
    ]
    return "\n".join(out)


def _cell(text: str) -> str:
    """
    A table cell that cannot end its own row.

    The pipe is replaced rather than backslash-escaped: the digest's markdown
    converter splits a row on literal pipes and knows nothing about escapes, so
    an escaped one would render correctly as markdown and open a fourth column
    in the HTML. The whole point of sharing that converter is that the two
    formats say the same thing.
    """
    return text.replace("|", "/").replace("\n", " ").strip()


# =========================================================================
# Drafting
# =========================================================================
def draft(target: list[FollowUp] | FollowUp | str, cfg, *, db=None, archive=None,
          vault: Vault | None = None, out: str | Path | None = None,
          fmt: str = "markdown", include_personal: bool = False,
          use_llm: bool | None = None) -> Path:
    """
    Write a draft message for a follow-up, or for everything one recording owes.

    `target` is a list of follow-ups, a single follow-up, or a recording id, in
    which case `db` and `archive` are needed to look it up. The result is a file
    in the outbox whose name says DRAFT twice, and nothing else happens. No mail
    client is opened, no address is looked up, no message is sent.

    Phrasing goes through an LLM when one is reachable, under the same locality
    rule as everything else: if any profile in the set forbids a cloud model,
    the whole draft is phrased locally or falls back to the template. The
    template path is not a degraded mode -- it is deterministic, offline, and
    frequently the better output.
    """
    if fmt not in ("markdown", "text"):
        raise FollowUpError(f"unknown format '{fmt}'. Use markdown or text.")

    recording_id = target if isinstance(target, str) else None
    items = _draft_items(target, cfg, db, archive, vault, include_personal)

    # Redaction first, before a model and before the file. Everything below
    # this line works from the redacted copy only, so there is no path where
    # the raw wording reaches either.
    patterns = cfg.get("compliance.redact_patterns") or {}
    counts: dict[str, int] = {}
    redacted: list[FollowUp] = []
    for item in items:
        # Field by field rather than one joined blob: a commitment whose text
        # already contains a newline would otherwise come back out of the join
        # misaligned, with the due date landing in the body.
        text, counterparty, due = (
            _redact_field(value, patterns, counts)
            for value in (item.text, item.counterparty, item.due)
        )
        redacted.append(FollowUp(
            id=item.id, text=text, profile_id=item.profile_id,
            recording_id=item.recording_id, field_key=item.field_key,
            counterparty=counterparty, due=due, first_seen=item.first_seen,
            last_seen=item.last_seen, status=item.status, mentions=item.mentions,
            recording_ids=list(item.recording_ids), source_name=item.source_name,
        ))

    local_only = draft_local_only(cfg, redacted)
    wants_llm = bool(cfg.get("followups.draft_with_llm", True)) if use_llm is None else use_llm

    subject, body, phrased_by = _template_message(redacted)
    if wants_llm:
        written = _llm_message(cfg, redacted, local_only, db)
        if written is not None:
            subject, body, phrased_by = written

    path = Path(out) if out else _draft_path(cfg, redacted, recording_id, fmt)
    path.parent.mkdir(parents=True, exist_ok=True)
    if out is None:
        # Never overwrite a draft silently. The one on disk may already have
        # been edited by the person this was written for.
        path = _unclaimed(path)

    path.write_text(
        _render_draft(redacted, subject, body, phrased_by, counts, local_only, fmt),
        encoding="utf-8",
    )

    if db is not None:
        db.audit(
            "followup_draft",
            f"{len(redacted)} follow-up(s) drafted to {path.name}; nothing was sent",
            recording_id, actor="human",
        )
    log.info("wrote draft %s (%d follow-up(s), local_only=%s)", path, len(redacted), local_only)
    return path


def _redact_field(value: str, patterns: dict[str, str], counts: dict[str, int]) -> str:
    """Redact one string, accumulating what fired into the caller's tally."""
    clean, report = redact_text(value, patterns, enabled=True)
    for key, hits in report.counts.items():
        counts[key] = counts.get(key, 0) + hits
    return clean


def draft_local_only(cfg, items: list[FollowUp]) -> bool:
    """
    True when no cloud model may see this set.

    The strictest profile in the set governs the whole draft, exactly as the
    strictest matched profile governs a whole recording. A draft mixing a
    locked profile with a cloud-permitting one is still one document, and there
    is no honest way to send half of it somewhere.
    """
    for item in items:
        if item.profile_id not in cfg.profiles:
            return True     # an unknown profile is not evidence that cloud is fine
        profile = cfg.profile(item.profile_id)
        if profile.hard_local_only or not profile.allow_cloud_llm:
            return True
    return False


def _draft_items(target, cfg, db, archive, vault, include_personal: bool) -> list[FollowUp]:
    if isinstance(target, FollowUp):
        items = [target]
    elif isinstance(target, str):
        if db is None or archive is None:
            raise FollowUpError(
                "drafting from a recording id needs the index and the archive; "
                "pass db= and archive=, or pass the follow-ups themselves."
            )
        items = collect(
            cfg, db, archive, recording_id=target, status="open",
            include_personal=include_personal, vault=vault,
        )
    else:
        items = list(target)

    if not items:
        raise FollowUpError(
            "nothing to draft: no open follow-ups were found for that target."
        )

    personal = {p.id for p in cfg.profiles.values() if p.exclude_from_combined_export}
    if not include_personal:
        kept = [i for i in items if i.profile_id not in personal]
        if not kept:
            names = ", ".join(sorted({i.profile_id for i in items if i.profile_id in personal}))
            raise FollowUpError(
                f"every follow-up here belongs to a personal profile ({names}). "
                "A draft is a document meant to leave this machine, so personal "
                "profiles are excluded by default. Pass include_personal if that "
                "is genuinely what you want."
            )
        if len(kept) != len(items):
            log.info("omitting %d personal follow-up(s) from this draft", len(items) - len(kept))
        items = kept

    cap = int(cfg.get("followups.max_per_draft", 25))
    if len(items) > cap:
        log.warning(
            "drafting the %d oldest of %d follow-ups (followups.max_per_draft)",
            cap, len(items),
        )
        items = sorted(items, key=sort_key)[:cap]
    return sorted(items, key=sort_key)


def _template_message(items: list[FollowUp]) -> tuple[str, str, str]:
    """The draft nobody's model wrote. Deterministic, offline, perfectly usable."""
    lines = ["Following up on what we said:", ""]
    for item in items:
        detail = []
        if item.due:
            detail.append(f"by {item.due}")
        if item.first_seen:
            detail.append(f"from {item.first_seen}")
        suffix = f"  ({', '.join(detail)})" if detail else ""
        lines.append(f"- {item.text}{suffix}")
    lines += ["", "Let me know if I have any of that wrong."]

    subject = "Following up"
    if len(items) == 1:
        subject = f"Following up: {items[0].text[:60].rstrip('.')}"
    return subject, "\n".join(lines), "template (no model involved)"


def _llm_message(cfg, items: list[FollowUp], local_only: bool,
                 db=None) -> tuple[str, str, str] | None:
    """
    Ask a model to phrase the draft. Returns None when it could not, which is
    not an error: the template is a complete answer on its own.
    """
    system = (
        "You are drafting a short follow-up message for the person who made or "
        "received the commitments listed below. The message will be read and "
        "edited by that person before anyone else sees it.\n\n"
        "HARD CONSTRAINTS. You must follow all of these:\n"
        "- Use only the commitments in the list. Do not add, merge, split, or "
        "infer any others.\n"
        "- Do not invent names, dates, amounts, or context that is not in the list.\n"
        "- Text of the form [SOMETHING_REDACTED] was removed on purpose. Leave "
        "the marker exactly as it is; do not guess what it was.\n"
        "- Do not open with a name unless the list contains one.\n"
        "- Plain, direct, and short. No marketing language.\n\n"
        "OUTPUT CONTRACT:\n"
        '- Respond with a single JSON object {"subject": "...", "body": "..."} '
        "and nothing else. No preamble, no code fences."
    )
    listing = "\n".join(
        f"{n}. {i.text}"
        + (f" (due {i.due})" if i.due else "")
        + (f" (said by {i.counterparty})" if i.counterparty else "")
        for n, i in enumerate(items, start=1)
    )
    user = f"COMMITMENTS:\n{listing}\n\nReturn the JSON object now."

    try:
        data, response = complete_json(
            cfg, system, user, local_only=local_only,
            max_tokens=int(cfg.get("followups.draft_max_tokens", 1200)),
        )
    except LLMError as exc:
        log.info("drafting with the template instead of a model: %s", exc)
        return None

    subject = str(data.get("subject", "")).strip()
    body = str(data.get("body", "")).strip()
    if not body:
        log.warning("the model returned no draft body; using the template instead")
        return None
    # ADR-014: phrasing a draft costs money and has no recording to charge it
    # to, so it is recorded against the run or it is spent invisibly.
    if db is not None:
        try:
            db.record_spend("draft", response.cost_usd, response.provider, response.model)
        except Exception as exc:  # noqa: BLE001 - an unrecorded cost must not lose the draft
            log.warning("could not record what this draft cost: %s", exc)

    provider = f"{response.provider}/{response.model}" if response.provider else "model"
    return subject or "Following up", body, f"{provider}, local_only={local_only}"


def _render_draft(items: list[FollowUp], subject: str, body: str, phrased_by: str,
                  counts: dict[str, int], local_only: bool, fmt: str) -> str:
    sources = sorted({i.recording_id for i in items})
    found = (
        "Redacted before this draft was written: "
        + ", ".join(f"{k} ({v})" for k, v in sorted(counts.items()))
        if counts
        else "No redaction pattern matched anything in this draft"
    )
    # The caveat prints whether or not anything matched, for the same reason the
    # export footer does: "no pattern fired" is not "nothing sensitive is here",
    # and a draft with no note reads as a draft something cleared.
    trailer = [
        f"Built from {len(items)} follow-up(s), traced to: {', '.join(sources)}.",
        f"Phrased by: {phrased_by}. Processing was "
        + ("local-only." if local_only else "permitted to use a cloud model."),
        f"{found}. Redaction is pattern matching, not a guarantee.",
        "Nothing has been sent. Read it, fix it, and send it yourself.",
    ]

    if fmt == "text":
        return "\n".join([
            "DRAFT - NOT SENT",
            "=" * 40,
            "",
            f"Subject: {subject}",
            "",
            body,
            "",
            "-" * 40,
            *trailer,
            "",
        ])

    return "\n".join([
        f"# DRAFT — {subject}",
        "",
        "**Nothing has been sent.** This file is a draft for you to read, edit, "
        "and send yourself.",
        "",
        f"**Subject:** {subject}",
        "",
        body,
        "",
        "---",
        "",
        *[f"<sub>{line}</sub>" for line in trailer],
        "",
    ])


def _draft_path(cfg, items: list[FollowUp], recording_id: str | None, fmt: str) -> Path:
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if recording_id:
        stem = recording_id
    elif len(items) == 1:
        stem = f"{items[0].profile_id}-{_slug(items[0].text)}"
    else:
        stem = "-".join(sorted({i.profile_id for i in items})) or "followups"
    extension = "md" if fmt == "markdown" else "txt"
    return cfg.path("outbox") / "drafts" / f"DRAFT-{day}-{_slug(stem)}.draft.{extension}"


def _slug(text: str, limit: int = 48) -> str:
    slug = _NON_WORD.sub("-", text.lower()).strip("-")
    return slug[:limit].strip("-") or "followup"


def _unclaimed(path: Path) -> Path:
    """The first filename in this series nobody has written to yet."""
    if not path.exists():
        return path
    for n in range(2, 100):
        candidate = path.with_name(path.name.replace(".draft.", f"-{n}.draft."))
        if not candidate.exists():
            return candidate
    raise FollowUpError(
        f"there are already 99 drafts named like {path.name}. Clear some out of "
        f"{path.parent} before writing another."
    )
