"""
One page per person, assembled from everywhere they were heard.

The archive knows every conversation, but a person is scattered across it:
Marcus is three minutes of a Tuesday call, a promise inside one analysis, a
quote inside another, and nothing ties those together. No note-taker on the
market can answer "show me Marcus" -- every time he was heard, what he told
you, what you owe each other, when you last spoke. This module builds that
page, from artifacts already on disk, without asking a model anything.

Two honesty rules shape everything here, and both are baked into the data
rather than left to the rendering:

  - **A speaker label is attribution, not verified identity.** Diarization
    clusters voices and a "Name:" prefix in an imported transcript is whatever
    the exporter wrote. So a person carries `voice_verified` -- true only when
    the name matches a voiceprint somebody deliberately enrolled -- and every
    rendering says which kind of name it is showing. Placeholder labels
    ("SPEAKER", "Speaker 2") are not people at all: they are grouped under one
    "(unidentified speakers)" bucket, because presenting a clustering artifact
    as a person is inventing someone.
  - **Nothing is invented.** Quotes come only from the extraction fields the
    pipeline already verified verbatim against the transcript; commitments are
    the same items `followups.collect` traces to real recordings; a recording
    that will not decrypt is reported and skipped, exactly as the follow-up
    collector does. An incomplete roster is honest, an invented one is not.

Personal-profile recordings stay out unless asked for, the same rule the
combined digest follows and for the same reason: a roster is the page most
likely to be read over a shoulder, and the kid's bedtime does not belong on it
by default. The owner of the recorder is listed -- their minutes are real --
but marked, because you are not a contact in your own address book.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .digest import to_html
from .episodes import _STOP
from .followups import FollowUp, FollowUpError, sort_key
from .followups import collect as collect_followups
from .logging_setup import get
from .storage import Vault, VaultError

log = get("people")

# The display name of the bucket every placeholder label falls into. The
# parentheses are deliberate: it sorts and reads as a category, not a name.
UNIDENTIFIED = "(unidentified speakers)"

# Labels diarization emits when it can cluster a voice but not name it. The
# same shape memory.py refuses to file as a person, for the same reason: a
# clustering artifact presented as a person is an invented person.
_PLACEHOLDER = re.compile(r"^(?:speaker|spk)[_ ]?\d*$", re.IGNORECASE)

_WORD = re.compile(r"[a-z0-9']+")


class PeopleError(RuntimeError):
    """Raised when a roster or dossier cannot be built honestly."""


# =========================================================================
# The shapes
# =========================================================================
@dataclass
class Appearance:
    """One recording this person was heard in, and what of theirs was kept."""

    recording_id: str
    when: str                # YYYY-MM-DD, recorded_at falling back to ingested_at
    source_name: str
    minutes: float           # minutes THEY spoke, not the recording's length
    profile_id: str          # the governing profile the conversation fell under
    said: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recording_id": self.recording_id,
            "when": self.when,
            "source_name": self.source_name,
            "minutes": round(self.minutes, 1),
            "profile_id": self.profile_id,
            "said": list(self.said),
        }


@dataclass
class Person:
    """
    Everything the archive knows about one speaker label.

    `label` is the attribution as heard; `voice_verified` says whether that
    label matches an enrolled voiceprint. The two are separate fields on
    purpose -- collapsing them is how a tool starts presenting a guess as an
    identity, and the one thing this page must never do is tell you Marcus
    said something a stranger with his name said.
    """

    label: str
    display_name: str
    is_bucket: bool = False       # the "(unidentified speakers)" group
    is_owner: bool = False        # diarization.owner_label; you, not a contact
    voice_verified: bool = False  # the name matches an enrolled voiceprint
    appearances: list[Appearance] = field(default_factory=list)
    things_they_said: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    commitments_from_them: list[FollowUp] = field(default_factory=list)
    commitments_to_them: list[FollowUp] = field(default_factory=list)

    @property
    def conversations(self) -> int:
        return len(self.appearances)

    @property
    def minutes_heard(self) -> float:
        return sum(a.minutes for a in self.appearances)

    @property
    def first_heard(self) -> str:
        return min((a.when for a in self.appearances if a.when), default="")

    @property
    def last_heard(self) -> str:
        return max((a.when for a in self.appearances if a.when), default="")

    @property
    def profiles(self) -> list[str]:
        seen: list[str] = []
        for appearance in self.appearances:
            if appearance.profile_id and appearance.profile_id not in seen:
                seen.append(appearance.profile_id)
        return seen

    @property
    def open_items(self) -> int:
        return len([
            i for i in (*self.commitments_from_them, *self.commitments_to_them)
            if i.is_open
        ])

    @property
    def identity(self) -> str:
        """How this row's name should be trusted, in words the table can print."""
        if self.is_bucket:
            return "placeholder labels"
        if self.is_owner:
            return "you (the owner)"
        return "voice-verified" if self.voice_verified else "label only"

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "display_name": self.display_name,
            "is_bucket": self.is_bucket,
            "is_owner": self.is_owner,
            "voice_verified": self.voice_verified,
            "identity": self.identity,
            "conversations": self.conversations,
            "minutes_heard": round(self.minutes_heard, 1),
            "first_heard": self.first_heard,
            "last_heard": self.last_heard,
            "profiles": self.profiles,
            "open_items": self.open_items,
            "topics": list(self.topics),
            "things_they_said": list(self.things_they_said),
            "commitments_from_them": [i.to_dict() for i in self.commitments_from_them],
            "commitments_to_them": [i.to_dict() for i in self.commitments_to_them],
            "appearances": [a.to_dict() for a in self.appearances],
        }


def _norm(label: str) -> str:
    return label.strip().lower()


def _is_placeholder(label: str) -> bool:
    return not label.strip() or bool(_PLACEHOLDER.match(label.strip()))


# =========================================================================
# Collecting
# =========================================================================
def collect_people(cfg, db, archive, *, include_personal: bool = False,
                   days: int | None = None, vault: Vault | None = None) -> list[Person]:
    """
    Everyone the archive has heard in the window, most present first.

    Aggregation is per speaker label across recordings, which is the honest
    unit available: two recordings that both say "Marcus" are merged, and
    whether that name means one human is exactly what `voice_verified` reports.

    Personal-profile recordings are governed by the digest's rule: out unless
    `include_personal`. A recording whose content will not open is reported and
    skipped the way `followups.collect` does -- an incomplete roster names its
    gaps in the log; an invented one would not know it had any.
    """
    vault = vault or getattr(archive, "vault", None) or Vault(cfg.path("vault"))
    owner = _norm(str(cfg.get("diarization.owner_label", "")))
    personal = {p.id for p in cfg.profiles.values() if p.exclude_from_combined_export}
    enrolled = _enrolled_names(vault)
    max_rows = int(cfg.get("people.max_recordings", 500))
    max_quotes = int(cfg.get("people.max_quotes", 12))
    per_appearance = int(cfg.get("people.max_quotes_per_recording", 6))
    max_topics = int(cfg.get("people.max_topics", 8))

    people: dict[str, Person] = {}
    words: dict[str, Counter] = {}
    unopened: list[str] = []

    for row in db.query(since_days=days, limit=max_rows):
        governing = row["governing_profile"] or ""
        if governing in personal and not include_personal:
            continue

        record = archive.full_record(row)
        if record is None:
            unopened.append(f"{row['id']}  {row['source_name']}")
            continue

        segments = (record.get("transcript") or {}).get("segments") or []
        if not segments:
            # Quarantined or empty: there is nobody here to hear.
            continue

        when = ((row["recorded_at"] or row["ingested_at"] or "") or "")[:10]
        quotes = _kept_quotes(cfg, record, personal, include_personal)

        spoken: dict[str, float] = {}
        display: dict[str, str] = {}
        for segment in segments:
            label = str(segment.get("speaker", "") or "")
            key = "" if _is_placeholder(label) else _norm(label)
            display.setdefault(key, label)
            seconds = max(0.0, float(segment.get("end", 0.0)) - float(segment.get("start", 0.0)))
            spoken[key] = spoken.get(key, 0.0) + seconds
            if key:
                tally = words.setdefault(key, Counter())
                for word in _WORD.findall(str(segment.get("text", "")).lower()):
                    if len(word) > 3 and word not in _STOP:
                        tally[word] += 1

        for key, seconds in spoken.items():
            person = people.get(key)
            if person is None:
                label = UNIDENTIFIED if key == "" else display[key]
                person = people[key] = Person(
                    label=label,
                    display_name=label,
                    is_bucket=(key == ""),
                    is_owner=bool(key) and key == owner,
                    voice_verified=bool(key) and key in enrolled,
                )
            said = [text for who, text in quotes if who == key][:per_appearance]
            person.appearances.append(Appearance(
                recording_id=row["id"],
                when=when,
                source_name=row["source_name"],
                minutes=seconds / 60.0,
                profile_id=governing,
                said=said,
            ))
            for text in said:
                if text not in person.things_they_said and len(person.things_they_said) < max_quotes:
                    person.things_they_said.append(text)

    if unopened:
        log.warning(
            "%d recording(s) could not be opened and are missing from the "
            "roster: %s. Set PLAUD_BRIDGE_PASSPHRASE if they are encrypted.",
            len(unopened), "; ".join(unopened[:5]),
        )

    for key, person in people.items():
        person.appearances.sort(key=lambda a: (a.when, a.recording_id))
        tally = words.get(key)
        if tally:
            person.topics = [w for w, _n in tally.most_common(max_topics)]

    _attach_commitments(cfg, db, archive, vault, people, owner,
                        days=days, include_personal=include_personal)

    ordered = sorted(
        people.values(),
        key=lambda p: (p.is_bucket, -p.conversations, -p.minutes_heard, p.display_name.lower()),
    )
    return ordered


def _enrolled_names(vault: Vault) -> set[str]:
    """
    The names a voiceprint was deliberately enrolled for.

    A store that will not open verifies nobody, and that is what the roster
    then says -- `voice_verified` stays false everywhere, which is the honest
    reading of "could not check" rather than a claim anything is wrong.
    """
    from .diarize.voiceprint import VoiceprintError, VoiceprintStore

    try:
        return {_norm(p.name) for p in VoiceprintStore(vault).people()}
    except (VoiceprintError, VaultError) as exc:
        log.warning("the voiceprint store could not be opened (%s); "
                    "nobody is shown as voice-verified", exc)
        return set()


def _kept_quotes(cfg, record: dict[str, Any], personal: set[str],
                 include_personal: bool) -> list[tuple[str, str]]:
    """
    (normalised speaker, text) for every quote worth keeping in this record.

    Only fields the profile schema types as quotes are read, because those are
    the ones the extractor verified verbatim against the transcript before
    storing -- anything else with a "speaker" key is a model's summary wearing
    an attribution. Sensitive and suppressed fields are skipped for the same
    reason the memory ledger skips them: this page is rendered plaintext, and a
    health disclosure belongs in the vault artifact, not on a roster.
    """
    out: list[tuple[str, str]] = []
    for analysis in record.get("analyses") or []:
        pid = str(analysis.get("profile_id", ""))
        if pid not in cfg.profiles or analysis.get("error"):
            continue
        if pid in personal and not include_personal:
            # A work recording can be co-routed to a personal profile; the
            # recording-level rule alone would let that analysis's quotes in.
            continue
        prof = cfg.profile(pid)
        fields = analysis.get("fields") or {}
        for spec in prof.fields:
            if "quote" not in spec.type.lower():
                continue
            if spec.sensitive or spec.key in prof.suppress_fields:
                continue
            value = fields.get(spec.key)
            for item in value if isinstance(value, list) else []:
                if not isinstance(item, dict):
                    continue
                speaker = str(item.get("speaker") or "").strip()
                text = str(item.get("text") or "").strip()
                if speaker and text and not _is_placeholder(speaker):
                    out.append((_norm(speaker), text))
    return out


def _attach_commitments(cfg, db, archive, vault, people: dict[str, Person],
                        owner: str, *, days: int | None, include_personal: bool) -> None:
    """
    File each follow-up onto the person it belongs to.

    The split is attribution first, presence second:

      - `commitments_from_them`: the item's counterparty (who said it, as the
        extraction recorded) is this person. The strongest signal there is.
      - `commitments_to_them`: somebody else -- the owner, or an item the
        schema wrote in the owner's voice and left unattributed, like
        `promises_i_made` -- made it in a conversation this person was part
        of. That inference is presence-based and says so in the rendering.

    An unattributed item defaults to the wearer of the recorder, which is the
    same working assumption the diarization config makes about whose voice
    dominates. The bucket gets no commitments at all: chasing a promise with
    "Speaker 2" is not something this tool should pretend is possible.
    """
    try:
        items = collect_followups(cfg, db, archive, days=days,
                                  include_personal=include_personal, vault=vault)
    except FollowUpError as exc:
        raise PeopleError(f"could not read the follow-up worklist: {exc}") from exc

    for person in people.values():
        if person.is_bucket:
            continue
        me = _norm(person.label)
        their_recordings = {a.recording_id for a in person.appearances}
        for item in items:
            who = _norm(item.counterparty)
            if who == me or (person.is_owner and not who and
                             set(item.recording_ids) & their_recordings):
                person.commitments_from_them.append(item)
            elif (not person.is_owner and (who == owner or not who)
                  and set(item.recording_ids) & their_recordings):
                person.commitments_to_them.append(item)
        person.commitments_from_them.sort(key=sort_key)
        person.commitments_to_them.sort(key=sort_key)


# =========================================================================
# One person
# =========================================================================
def person_detail(people: list[Person], name: str) -> Person:
    """
    Find one person by name, exactly or by unique prefix.

    Refuses an ambiguous prefix and names the candidates, the same manners
    `followups.set_status` has about ids: guessing which Marcus was meant is
    how the wrong dossier gets read.
    """
    wanted = _norm(name)
    if not wanted:
        raise PeopleError("no name given")

    exact = [p for p in people if _norm(p.display_name) == wanted or _norm(p.label) == wanted]
    if len(exact) == 1:
        return exact[0]

    prefix = [p for p in people if _norm(p.display_name).startswith(wanted)]
    if len(prefix) == 1:
        return prefix[0]
    if len(prefix) > 1:
        raise PeopleError(
            f"'{name}' matches {len(prefix)} people: "
            + ", ".join(sorted(p.display_name for p in prefix))
            + ". Use more of the name."
        )

    known = ", ".join(p.display_name for p in people) or "nobody yet"
    raise PeopleError(
        f"nobody here is called '{name}'. Heard so far: {known}. "
        "Names come from speaker labels, so check `run.py people` for the "
        "exact spelling the archive uses."
    )


# =========================================================================
# Rendering
# =========================================================================
def render_roster(people: list[Person], *, fmt: str = "markdown",
                  title: str | None = None) -> str:
    """
    The roster, in the digest's shape: markdown is the source of truth and
    HTML is rendered from it, so the two formats cannot drift apart.
    """
    if fmt not in ("markdown", "html"):
        raise PeopleError(f"unknown format '{fmt}'. Use markdown or html.")
    heading = title or "People"
    body = _roster_markdown(people, heading)
    return to_html(body, title=heading) if fmt == "html" else body


def render_person(person: Person, *, fmt: str = "markdown") -> str:
    """One person's whole page: the dossier the module docstring promises."""
    if fmt not in ("markdown", "html"):
        raise PeopleError(f"unknown format '{fmt}'. Use markdown or html.")
    body = _person_markdown(person)
    return to_html(body, title=person.display_name) if fmt == "html" else body


def _roster_markdown(people: list[Person], heading: str) -> str:
    now = datetime.now(timezone.utc)
    out: list[str] = [f"# {heading}", ""]

    if not people:
        out += [
            "Nobody has been heard in this window. Either nothing was recorded "
            "or nothing has been processed yet.",
            "",
        ]
        return "\n".join(out)

    named = [p for p in people if not p.is_bucket]
    verified = len([p for p in named if p.voice_verified])
    summary = [f"{len(named)} name(s)"]
    if verified:
        summary.append(f"{verified} voice-verified")
    if any(p.is_bucket for p in people):
        summary.append("plus unidentified speakers")
    summary.append(f"generated {now:%Y-%m-%d %H:%M}")
    out += ["`" + " · ".join(summary) + "`", ""]

    out += [
        "| Person | Identity | Conversations | Last heard | Minutes | Open items |",
        "|---|---|---:|---|---:|---:|",
    ]
    for person in people:
        out.append(
            f"| {_cell(person.display_name)} | {person.identity} "
            f"| {person.conversations} | {person.last_heard or '-'} "
            f"| {person.minutes_heard:.0f} | {person.open_items} |"
        )
    out += [
        "",
        '`run.py people --name "..."` shows one person\'s full page.',
        "",
        "---",
        "",
        "<sub>A name here is a speaker label -- attribution, not verified "
        "identity -- unless marked voice-verified, which means it matches a "
        "voiceprint you enrolled yourself. Placeholder labels are grouped as "
        "(unidentified speakers) rather than presented as people.</sub>",
        "",
    ]
    return "\n".join(out)


def _person_markdown(person: Person) -> str:
    out: list[str] = [f"# {person.display_name}", ""]

    meta = [person.identity]
    meta.append(f"{person.conversations} conversation(s)")
    meta.append(f"{person.minutes_heard:.0f} min heard")
    if person.first_heard:
        meta.append(f"first heard {person.first_heard}")
    if person.last_heard and person.last_heard != person.first_heard:
        meta.append(f"last heard {person.last_heard}")
    if person.profiles:
        meta.append("profiles: " + ", ".join(person.profiles))
    out += ["`" + " · ".join(meta) + "`", ""]

    if person.is_bucket:
        out += [
            "These are voices diarization could separate but nobody has named: "
            "placeholder labels, grouped here rather than presented as a "
            'person. Enroll a voice (`run.py speakers enroll "Name" --audio '
            "clip.wav`) to start putting names on them.",
            "",
        ]
    elif not person.voice_verified and not person.is_owner:
        out += [
            "This name is a speaker label, not a verified identity. Enroll "
            "their voice to have future recordings matched acoustically.",
            "",
        ]

    if person.things_they_said:
        out += ["## What they said, worth keeping", ""]
        out += [f"- \"{_cell(text)}\"" for text in person.things_they_said]
        out.append("")

    if person.topics:
        out += ["## What they talk about", "", "`" + " · ".join(person.topics) + "`", ""]

    if person.commitments_from_them:
        out += ["## What they owe", ""]
        out += [_commitment_line(i) for i in person.commitments_from_them]
        out.append("")
    if person.commitments_to_them:
        out += ["## What is owed to them", ""]
        out += [_commitment_line(i) for i in person.commitments_to_them]
        out += [
            "",
            "<sub>Owed-to-them items are inferred from presence: a commitment "
            "voiced by you, or left unattributed by the extraction, in a "
            "conversation this person was part of.</sub>",
            "",
        ]

    out += ["## Every time they were heard", ""]
    if not person.appearances:
        out.append("Never, in this window.")
        out.append("")
    for appearance in person.appearances:
        out.append(
            f"- **{appearance.when or 'undated'}** — {_cell(appearance.source_name)} "
            f"(`{appearance.recording_id}`), {appearance.minutes:.1f} min"
            + (f", {appearance.profile_id}" if appearance.profile_id else "")
        )
        out += [f"    - \"{_cell(text)}\"" for text in appearance.said]
    out.append("")

    return "\n".join(out)


def _commitment_line(item: FollowUp) -> str:
    detail = [item.status]
    if item.is_open and item.age_days:
        detail.append(f"{item.age_days}d")
    if item.due:
        detail.append(f"due {item.due}")
    return (f"- **{'/'.join(detail)}** — {_cell(item.text)}  "
            f"(`{item.recording_id}`)")


def _cell(text: str) -> str:
    """
    A table cell that cannot end its own row -- the same replacement the
    follow-up worklist uses, because the shared markdown converter splits rows
    on literal pipes and knows nothing about escapes.
    """
    return text.replace("|", "/").replace("\n", " ").strip()
