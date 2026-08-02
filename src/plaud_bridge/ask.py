"""
Asking the archive a question.

`search --content` tells you where a phrase occurs. It does not tell you what
you promised somebody. "What did I tell the Hendersons about their term policy?"
is the question actually being asked in a car park before the next appointment,
and answering it from a list of twenty keyword hits means reading all twenty and
reconstructing the answer yourself. That is the gap this closes.

Retrieval first, generation second. The retrieval half is deterministic
keyword-and-recency ranking over the index, the stored analyses, and the
transcripts, and it runs with no model configured at all. When there is no
usable provider you get the excerpts back with a sentence saying plainly that
this is search output and not an answer, because a tool that quietly turns into
a worse version of itself is how you end up trusting something you should not.

The generation half only ever sees excerpts this module chose, and every
citation that comes back is checked against them before it is shown. A
fabricated recording id in an answer about your own archive is the worst failure
this feature has: it looks exactly like a real one, and you would act on it.

The pipeline's rules govern here too. The strictest profile in the bundle
decides whether a cloud provider may be used at all, personal profiles stay out
unless you ask for them by name, redaction happens to the copy the model sees
and never to the copy you read, and an answer written to disk goes into the
vault or is not written.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .compliance.redact import redact_text

# One stoplist, not two. The episode segmenter already had to decide which words
# say something about what is being discussed; a question is the same problem
# read backwards, and a second list would drift away from the first.
from .episodes import _STOP as STOPWORDS
from .llm import complete_json
from .llm.base import LLMError
from .logging_setup import get
from .models import format_stamp
from .profiles.extractor import _flatten, quote_is_present
from .runtime import is_offline
from .storage import Vault, VaultError

log = get("ask")

_WORD = re.compile(r"[a-z0-9']+")
_PHRASE = re.compile(r'"([^"]{3,})"')
_CONFIDENCE = ("high", "medium", "low")

# The keys an extracted quote can arrive under, and the ones that describe it
# rather than being it. Same set the digest renderer works from: a model asked
# for {"timestamp", "speaker", "text"} will sometimes send "who" and "quote"
# instead, and an item rendered as a blank line is an item silently lost.
_QUOTE_KEYS = ("text", "quote", "statement", "content")
_META_KEYS = ("timestamp", "time", "speaker", "who")


_SUFFIXES = ("ies", "ing", "ed", "es", "s")


def _stem(word: str) -> str:
    """
    Fold the endings that would otherwise make a question miss its own answer.

    "What did I promise" against a transcript that says "I promised" is the
    common case, not the exotic one, and an exact-match ranker scores that pair
    at zero. This is crude on purpose: a real stemmer is a dependency (ADR-008),
    and the failure mode of over-folding here is an extra excerpt in the bundle
    rather than a wrong answer.

    The trailing vowel strip at the end is what makes the fold converge.
    Removing "d" from "promised" alone gives "promis" while "promise" stays put,
    so the two forms of the same word still miss each other -- which is the bug
    this looked like it had already fixed.
    """
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    if len(word) > 4 and word[-1] in ("e", "y"):
        word = word[:-1]
    return word


def _tokens(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for word in _WORD.findall(text.lower()):
        stem = _stem(word)
        counts[stem] = counts.get(stem, 0) + 1
    return counts


def question_terms(question: str, min_len: int = 3) -> list[str]:
    """The words in a question that carry any information about the answer."""
    terms: list[str] = []
    for word in _WORD.findall(question.lower()):
        if len(word) < min_len or word in STOPWORDS:
            continue
        stem = _stem(word)
        if stem not in terms:
            terms.append(stem)
    return terms


# =========================================================================
# What retrieval produces
# =========================================================================
@dataclass
class Excerpt:
    """
    One piece of real, retrieved material, carrying everything a citation needs.

    `stamp` is always the timestamp of something that was actually said. When an
    excerpt spans several segments the stamp belongs to the first of them, so it
    points at where the quoted stretch begins rather than at its middle.
    """

    recording_id: str
    source_name: str = ""
    when: str = ""
    profile_id: str = ""
    stamp: str = ""
    speaker: str = ""
    text: str = ""
    kind: str = "transcript"     # transcript | analysis
    label: str = ""              # which analysis field this came from
    sensitive: bool = False      # the profile marks this field suppressed

    def header(self) -> str:
        where = f"{self.recording_id} @ {self.stamp}" if self.stamp else self.recording_id
        what = f" ({self.label})" if self.label else ""
        who = f" {self.speaker}:" if self.speaker else ""
        return f"[{where}]{what}{who}"


@dataclass
class Candidate:
    recording_id: str
    source_name: str = ""
    when: str = ""
    governing_profile: str = ""
    profile_ids: list[str] = field(default_factory=list)
    score: float = 0.0
    age_days: float = 0.0
    matched_terms: list[str] = field(default_factory=list)
    excerpts: list[Excerpt] = field(default_factory=list)


@dataclass
class Retrieval:
    """
    What the deterministic half found, and what it did not look at.

    `unopened` and `truncated_scan` exist for the same reason they exist on
    SearchResult: an answer assembled from what happened to open, presented as
    an answer assembled from the archive, is a lie the reader cannot detect.
    """

    question: str = ""
    terms: list[str] = field(default_factory=list)
    phrases: list[str] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)
    considered: int = 0
    total: int = 0
    personal_skipped: int = 0
    unopened: list[str] = field(default_factory=list)
    truncated_scan: bool = False

    @property
    def complete(self) -> bool:
        return not self.truncated_scan and not self.unopened


@dataclass
class Citation:
    recording_id: str
    stamp: str = ""
    quote: str = ""
    source_name: str = ""
    when: str = ""

    def render(self) -> str:
        return f"[{self.recording_id} @ {self.stamp}]" if self.stamp else f"[{self.recording_id}]"


@dataclass
class Answer:
    text: str = ""
    citations: list[Citation] = field(default_factory=list)
    recordings_considered: int = 0
    bundle_chars: int = 0
    truncated: bool = False
    local_only: bool = True
    provider: str = ""
    cost_usd: float = 0.0
    degraded: bool = False
    # `degraded` covers three different situations and the caller has to tell
    # them apart to pick an exit code. A question nobody could parse is the
    # asker's mistake; an archive with nothing matching is a complete answer
    # that happens to be "nothing"; a model that could not be reached is a real
    # incompleteness. Only the last two are about the archive at all.
    usage_error: bool = False
    note: str = ""

    question: str = ""
    confidence: str = ""
    unanswered: str = ""
    profiles: list[str] = field(default_factory=list)
    excerpts: list[Excerpt] = field(default_factory=list)
    left_out: int = 0
    dropped_citations: list[str] = field(default_factory=list)
    repaired_citations: int = 0
    unopened: list[str] = field(default_factory=list)
    redactions: dict[str, int] = field(default_factory=dict)

    @property
    def recordings_used(self) -> list[str]:
        seen: list[str] = []
        for excerpt in self.excerpts:
            if excerpt.recording_id not in seen:
                seen.append(excerpt.recording_id)
        return seen

    def render(self) -> str:
        """Terminal-friendly. Citations read as [rec_id @ 00:12:34]."""
        lines: list[str] = []
        if self.question:
            lines.append(f"Q: {self.question}")
            lines.append("")
        lines.append(self.text.strip() or "(the model returned no answer text)")

        if self.citations:
            lines.append("")
            lines.append("Sources")
            for citation in self.citations:
                tail = f"  {citation.when}  {citation.source_name}".rstrip()
                lines.append(f"  {citation.render()}{tail}")
                if citation.quote.strip():
                    lines.append(f'      "{citation.quote.strip()}"')
        elif self.excerpts:
            lines.append("")
            lines.append(f"{len(self.excerpts)} excerpt(s), most relevant first")
            for excerpt in self.excerpts:
                tail = f"  {excerpt.when}  {excerpt.source_name}".rstrip()
                lines.append(f"  {excerpt.header()}{tail}")
                lines.append(f"      {excerpt.text.strip()[:300]}")

        if self.unanswered.strip():
            lines.append("")
            lines.append(f"Not answered: {self.unanswered.strip()}")
        if self.note.strip():
            lines.append("")
            lines.append(self.note.strip())

        lines.append("")
        lines.append(self._footer())
        return "\n".join(lines)

    def _footer(self) -> str:
        bits = [
            f"{self.recordings_considered} recording(s) searched",
            f"{len(self.recordings_used)} in the answer",
            f"{self.bundle_chars} chars of context",
            "local only" if self.local_only else "cloud permitted",
        ]
        if self.provider:
            bits.append(self.provider)
        if self.cost_usd:
            bits.append(f"${self.cost_usd:.4f}")
        return "  |  ".join(bits)


# =========================================================================
# Retrieval
# =========================================================================
def _profile_ids(record: dict[str, Any], row: dict[str, Any]) -> list[str]:
    """
    Every profile with a claim on this recording, not just the governing one.

    Locality has to be decided from all of them. A recording that is Sales
    Trainer and Husband at once is governed by Husband, but reading only the
    governing field would still be the wrong way to find that out: the gate can
    be disabled, and a row written by an older version may not carry one.
    """
    ids: list[str] = []
    for pid in (
        [str((record.get("compliance") or {}).get("governing_profile") or "")]
        + [str(row.get("governing_profile") or "")]
        + [str(r.get("profile_id") or "") for r in record.get("routes") or []]
        + [str(a.get("profile_id") or "") for a in record.get("analyses") or []]
    ):
        if pid and pid not in ids:
            ids.append(pid)
    return ids


def _is_personal(cfg, profile_ids: list[str]) -> bool:
    return any(
        pid in cfg.profiles and cfg.profile(pid).exclude_from_combined_export
        for pid in profile_ids
    )


def _match(counts: dict[str, int], terms: list[str]) -> tuple[list[str], int]:
    """(which terms appeared, how many times in total)."""
    found = [t for t in terms if counts.get(t)]
    return found, sum(counts.get(t, 0) for t in found)


def _when(row: dict[str, Any]) -> str:
    return (row.get("recorded_at") or row.get("ingested_at") or "")[:16].replace("T", " ")


def _age_days(row: dict[str, Any], now: datetime) -> float:
    raw = row.get("recorded_at") or row.get("ingested_at") or ""
    try:
        when = datetime.fromisoformat(str(raw))
    except ValueError:
        return 0.0
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (now - when).total_seconds() / 86400.0)


def _analysis_excerpts(cfg, record: dict[str, Any], row: dict[str, Any],
                       terms: list[str]) -> list[Excerpt]:
    """
    Pull matching fields out of the stored analyses.

    The analyses are the curated version of the recording -- the commitments,
    the open questions, the next action -- so a hit here is usually worth more
    than a hit in the raw transcript. Fields the profile suppresses are still
    collected, and marked; whether they may be sent anywhere is decided later,
    once it is known whether "anywhere" includes somebody else's server.
    """
    out: list[Excerpt] = []
    when, source = _when(row), str(row.get("source_name") or "")

    for analysis in record.get("analyses") or []:
        pid = str(analysis.get("profile_id") or "")
        profile = cfg.profiles.get(pid)
        for key, value in (analysis.get("fields") or {}).items():
            spec = profile.field_by_key(key) if profile else None
            sensitive = bool(spec and spec.sensitive) or bool(
                profile and key in profile.suppress_fields
            )
            label = spec.label if spec else key
            for stamp, text in _render_field(value):
                if not text.strip():
                    continue
                found, _ = _match(_tokens(text), terms)
                if not found:
                    continue
                out.append(Excerpt(
                    recording_id=str(row["id"]), source_name=source, when=when,
                    profile_id=pid, stamp=stamp, text=text.strip(),
                    kind="analysis", label=label, sensitive=sensitive,
                ))
    return out


def _render_field(value: Any) -> list[tuple[str, str]]:
    """
    Flatten one analysis field into (timestamp, text) pairs.

    Quote-shaped items carry their own timestamp from the extraction contract
    and become one entry each, so they can be cited individually. Everything
    else becomes a single untimed entry rather than being dropped, because a
    model will happily answer {"what": ..., "when": ...} for an object field and
    an empty bundle is worse than an ugly one.
    """
    if value is None or value == "" or value == [] or value is False:
        return []
    if isinstance(value, bool):
        return [("", "yes")]
    if isinstance(value, list):
        out: list[tuple[str, str]] = []
        for item in value:
            out.extend(_render_field(item))
        return out
    if isinstance(value, dict):
        stamp = str(value.get("timestamp") or value.get("time") or "").strip()
        body = ""
        for key in _QUOTE_KEYS:
            if value.get(key):
                body = str(value[key]).strip()
                break
        extras = "; ".join(
            f"{k}: {v}" for k, v in value.items()
            if k not in _QUOTE_KEYS + _META_KEYS
            and v not in (None, "", [], {}) and not isinstance(v, (dict, list))
        )
        speaker = str(value.get("speaker") or value.get("who") or "").strip()
        line = " ".join(p for p in (f"{speaker}:" if speaker else "", body, extras) if p)
        return [(stamp, line.strip())] if line.strip() else []
    return [("", str(value))]


def _transcript_excerpts(row: dict[str, Any], segments: list[dict[str, Any]],
                         terms: list[str], per_recording: int,
                         context: int) -> tuple[list[Excerpt], list[str], int]:
    """Best-matching stretches of what was actually said, in time order."""
    when, source = _when(row), str(row.get("source_name") or "")
    scored: list[tuple[int, int, int]] = []      # (-distinct, -hits, index)
    all_found: list[str] = []
    total_hits = 0

    for index, segment in enumerate(segments):
        found, hits = _match(_tokens(str(segment.get("text", ""))), terms)
        if not found:
            continue
        total_hits += hits
        for term in found:
            if term not in all_found:
                all_found.append(term)
        scored.append((-len(found), -hits, index))

    scored.sort()
    starts: list[int] = []
    for _, _, index in scored[: per_recording * 2]:
        start = max(0, index - context)
        if start not in starts:
            starts.append(start)
        if len(starts) >= per_recording:
            break

    out: list[Excerpt] = []
    for start in sorted(starts):
        window = segments[start : start + (2 * context) + 1]
        body, speakers = _window_text(window)
        if not body:
            continue
        out.append(Excerpt(
            recording_id=str(row["id"]), source_name=source, when=when,
            profile_id=str(row.get("governing_profile") or ""),
            stamp=format_stamp(float(window[0].get("start", 0.0) or 0.0)),
            # Only claim a speaker when the whole window belongs to one. An
            # excerpt that spans a turn and is labelled with the first speaker
            # attributes the reply to the wrong person, and the model would then
            # cite it that way. The words carry their own labels instead.
            speaker=speakers[0] if len(speakers) == 1 else "",
            text=body, kind="transcript",
        ))
    return out, all_found, total_hits


def _window_text(window: list[dict[str, Any]]) -> tuple[str, list[str]]:
    """Render a stretch of segments, labelling each change of speaker inline."""
    parts: list[str] = []
    speakers: list[str] = []
    current: str | None = None
    for segment in window:
        body = str(segment.get("text", "")).strip()
        if not body:
            continue
        speaker = str(segment.get("speaker", "")).strip()
        if speaker and speaker not in speakers:
            speakers.append(speaker)
        if speaker and speaker != current:
            parts.append(f"{speaker}: {body}")
            current = speaker
        else:
            parts.append(body)
    return " ".join(parts).strip(), speakers


def retrieve(question: str, cfg, db, archive, *, profile: str | None = None,
             days: int | None = None, limit: int | None = None,
             include_personal: bool = False) -> Retrieval:
    """
    Rank recordings against a question without asking a model anything.

    Deterministic, and the whole reason this feature can be tested: the ranking
    is term coverage first, term frequency as a tie-break, a bonus for hits in
    the curated analyses, and recency last. Recency is weighted low on purpose.
    A question about a client from March should not lose to last Tuesday's
    coffee because Tuesday is nearer.
    """
    min_len = int(cfg.get("ask.min_term_length", 3))
    limit = int(cfg.get("ask.limit", 6)) if limit is None else int(limit)
    days = int(cfg.get("ask.days", 90)) if days is None else int(days)
    scan_limit = int(cfg.get("ask.scan_limit", 200))
    per_recording = int(cfg.get("ask.excerpts_per_recording", 4))
    context = int(cfg.get("ask.excerpt_context", 1))
    half_life = max(1.0, float(cfg.get("ask.recency_half_life_days", 30)))
    recency_weight = float(cfg.get("ask.recency_weight", 0.25))
    analysis_weight = float(cfg.get("ask.analysis_weight", 0.5))
    phrase_bonus = float(cfg.get("ask.phrase_bonus", 2.0))

    window = days if days > 0 else None
    result = Retrieval(
        question=question,
        terms=question_terms(question, min_len),
        phrases=[p.strip() for p in _PHRASE.findall(question) if p.strip()],
    )
    if not result.terms and not result.phrases:
        return result

    result.total = db.count_recordings(profile_id=profile, since_days=window)
    rows = db.query(
        profile_id=profile, since_days=window,
        limit=scan_limit if scan_limit > 0 else result.total or 1,
    )
    result.truncated_scan = result.total > len(rows)

    # A quoted phrase is the one case where an exact substring beats term
    # overlap, and search_content already knows how to find one across encrypted
    # transcripts. It costs a second pass over the archive, so it only runs when
    # the question actually contains quotes.
    boosted: dict[str, list[Excerpt]] = {}
    for phrase in result.phrases:
        exact = archive.search_content(
            phrase, profile_id=profile, since_days=window, scan_limit=scan_limit, context=context,
        )
        for match in exact.matches:
            if match.personal and not include_personal:
                continue
            boosted.setdefault(match.recording_id, []).append(Excerpt(
                recording_id=match.recording_id, source_name=match.source_name,
                when=match.when, profile_id=match.profile_id, stamp=match.stamp,
                speaker=match.speaker, text=match.text, kind="transcript",
            ))
        for entry in exact.unopened:
            if entry not in result.unopened:
                result.unopened.append(entry)

    now = datetime.now(timezone.utc)
    candidates: list[Candidate] = []

    for row in rows:
        record = archive.full_record(row)
        if record is None:
            # Same rule as search: a file that would not open is reported, never
            # counted as a file with nothing in it.
            entry = f"{row['id']}  {row['source_name']}"
            if entry not in result.unopened:
                result.unopened.append(entry)
            continue
        result.considered += 1

        profile_ids = _profile_ids(record, row)
        if _is_personal(cfg, profile_ids) and not include_personal:
            result.personal_skipped += 1
            continue

        segments = (record.get("transcript") or {}).get("segments") or []
        excerpts, found, hits = _transcript_excerpts(
            row, segments, result.terms, per_recording, context,
        )
        analysis = _analysis_excerpts(cfg, record, row, result.terms)
        for excerpt in analysis:
            for term in _match(_tokens(excerpt.text), result.terms)[0]:
                if term not in found:
                    found.append(term)

        extra = boosted.get(str(row["id"]), [])
        stamps = {e.stamp for e in excerpts}
        excerpts.extend(e for e in extra if e.stamp not in stamps)

        if not found and not extra:
            continue

        age = _age_days(row, now)
        score = (
            len(found)
            + 0.05 * min(hits, 40)
            + (analysis_weight if analysis else 0.0)
            + recency_weight / (1.0 + age / half_life)
            + (phrase_bonus if extra else 0.0)
        )
        candidates.append(Candidate(
            recording_id=str(row["id"]),
            source_name=str(row.get("source_name") or ""),
            when=_when(row),
            governing_profile=str(row.get("governing_profile") or ""),
            profile_ids=profile_ids,
            score=score,
            age_days=age,
            matched_terms=found,
            # Analysis first: it is the curated read of the same conversation,
            # so when the budget runs out it is the part worth keeping.
            excerpts=analysis + excerpts,
        ))

    # Score, then newest, then id. The last key is not cosmetic: without it two
    # recordings with identical scores and timestamps would come back in
    # whatever order SQLite felt like, and the same question asked twice would
    # produce two different answers.
    candidates.sort(key=lambda c: (-c.score, c.age_days, c.recording_id))
    result.candidates = candidates[:limit]
    return result


# =========================================================================
# The context bundle
# =========================================================================
@dataclass
class Bundle:
    """
    The excerpts the model is allowed to see, and everything it is not seeing.

    `left_out` and `withheld` are not bookkeeping. Both are stated in the prompt
    and again in the answer, because silently truncating the context and then
    answering with confidence is how a summary comes to be wrong in a way the
    reader cannot detect.
    """

    text: str = ""
    used: list[Excerpt] = field(default_factory=list)
    left_out: int = 0
    withheld: int = 0
    redactions: dict[str, int] = field(default_factory=dict)


def _bundle(candidates: list[Candidate], *, local_only: bool, redact: bool,
            patterns: dict[str, str], max_chars: int) -> Bundle:
    """Assemble the context, in ranked order, up to the character budget."""
    pieces: list[str] = []
    used: list[Excerpt] = []
    counts: dict[str, int] = {}
    chars = 0
    left_out = 0
    withheld = 0
    stop = False

    for candidate in candidates:
        allowed = []
        for excerpt in candidate.excerpts:
            # A suppressed field is one the profile said never renders into a
            # shareable document. A local model is not a shareable document; a
            # cloud provider is. So the field travels only when nothing leaves.
            if excerpt.sensitive and not local_only:
                withheld += 1
                continue
            allowed.append(excerpt)
        if not allowed:
            continue

        header = (
            f"RECORDING {candidate.recording_id}  "
            f"({candidate.when or 'undated'}, profile: "
            f"{candidate.governing_profile or 'unfiled'}, file: {candidate.source_name})"
        )
        for index, excerpt in enumerate(allowed):
            if stop:
                left_out += 1
                continue
            body, report = redact_text(excerpt.text, patterns, enabled=redact)
            for name, count in report.counts.items():
                counts[name] = counts.get(name, 0) + count
            piece = ("\n" + header + "\n" if index == 0 else "") + f"  {excerpt.header()} {body}"
            if pieces and chars + len(piece) > max_chars:
                stop = True
                left_out += 1
                continue
            pieces.append(piece)
            used.append(excerpt)
            chars += len(piece)

    return Bundle(
        text="\n".join(pieces).strip(), used=used,
        left_out=left_out, withheld=withheld, redactions=counts,
    )


# =========================================================================
# Locality
# =========================================================================
def _locality(cfg, profile_ids: set[str], requested: bool | None) -> tuple[bool, str]:
    """
    Decide whether this question may reach a cloud provider, and say why not.

    Same rule as the compliance gate, for the same reason: one profile in the
    bundle that forbids cloud forbids it for the whole bundle, because the model
    sees the excerpts together and there is no such thing as sending half a
    prompt somewhere. `requested` is a request, never a permission -- passing
    False cannot unlock a profile that says no.
    """
    forbidding = sorted(
        pid for pid in profile_ids
        if pid in cfg.profiles
        and (cfg.profile(pid).hard_local_only or not cfg.profile(pid).allow_cloud_llm)
    )
    if profile_ids:
        policy = bool(forbidding)
    else:
        # Nothing was retrieved, so the question could be about anything. Same
        # conservatism as routing, which also runs before it knows what it has.
        policy = not cfg.cloud_llm_permitted_by_every_profile()

    offline = is_offline(cfg)
    local = policy or offline or bool(requested)

    if not local:
        return False, ""
    if offline:
        return True, "runtime.offline is on, so this ran locally."
    if forbidding:
        return True, (
            "local-only: profile(s) " + ", ".join(forbidding) + " forbid a cloud LLM, "
            "and the strictest profile in the bundle governs all of it."
        )
    if policy:
        return True, "local-only: nothing was retrieved, so cloud was not assumed to be safe."
    return True, "local-only was requested."


def _redaction_required(cfg, profile_ids: set[str]) -> bool:
    """True when any profile in the bundle wants its text scrubbed first."""
    involved = [cfg.profile(p) for p in profile_ids if p in cfg.profiles]
    if not involved:
        # Nothing retrieved, or profiles that no longer exist. Redact anyway:
        # the cost of scrubbing text that did not need it is a few tokens.
        return True
    return any(p.redact_before_llm for p in involved)


# =========================================================================
# Prompting
# =========================================================================
_SYSTEM = """\
You answer questions about one person's own recorded conversations, from
excerpts of those recordings and nothing else.

RULES:
- The excerpts are the whole of what you know. Do not use general knowledge, and
  do not fill a gap with what is usually true.
- If the excerpts do not answer the question, say so. A refusal is a correct
  answer; a plausible guess is not. Put what you could not establish, and what
  would answer it, in "unanswered".
- Cite the recording id and timestamp exactly as they appear in the excerpt
  header, character for character. Never adjust, interpolate, or invent either.
- "quote" must be the speaker's own words, copied from an excerpt.
- "confidence" is "high" only when an excerpt answers the question directly,
  "medium" when the answer is assembled from several, "low" when you are
  reading between the lines. Say so rather than rounding up.

OUTPUT CONTRACT:
- Respond with a single JSON object and nothing else. No preamble, no code
  fences, no trailing commentary.
- Shape:
  {"answer": "...",
   "citations": [{"recording_id": "...", "stamp": "00:12:34", "quote": "..."}],
   "confidence": "high" | "medium" | "low",
   "unanswered": "..."}
- "citations" may be empty. It may not contain a recording id that is not in the
  excerpts above.
"""


def _user_prompt(question: str, bundle: str, left_out: int, withheld: int,
                 truncated_scan: bool, unopened: int) -> str:
    parts = [f"QUESTION:\n{question}", f"EXCERPTS:\n{bundle}"]
    caveats: list[str] = []
    if left_out:
        caveats.append(
            f"{left_out} further excerpt(s) were left out to stay inside the "
            "context budget. If the answer depends on material you cannot see, "
            'say so in "unanswered".'
        )
    if withheld:
        caveats.append(
            f"{withheld} field(s) marked sensitive were withheld from you. Do "
            "not guess at their contents."
        )
    if truncated_scan:
        caveats.append("Not every recording in the window was searched.")
    if unopened:
        caveats.append(f"{unopened} recording(s) could not be opened and were not searched.")
    if caveats:
        parts.append("WHAT YOU ARE NOT SEEING:\n" + "\n".join(f"- {c}" for c in caveats))
    parts.append("Return the JSON object now.")
    return "\n\n".join(parts)


def _validate_citations(raw: Any, used: list[Excerpt]) -> tuple[list[Citation], list[str], int]:
    """
    Keep only citations whose words were actually in the material sent.

    Three checks, in order, because each is a different way a citation can be a
    fabrication:

    1. The recording must have been in the bundle. A citation naming a recording
       that was never sent is dropped and logged.
    2. The QUOTE must appear verbatim (up to case and punctuation) in one of that
       recording's excerpts. This is the check that was missing, and its absence
       was the whole point of the feature turning back into the thing it exists
       to prevent: the model could attach an invented sentence to a real
       recording id and a real timestamp, and it rendered as a sourced quote and
       was saved to the vault. A quote nobody said is dropped exactly like a
       recording nobody sent -- it is the same lie with better paperwork.
    3. A quote that IS present but carries a timestamp that was not sent has its
       stamp snapped to the excerpt it was actually found in. That is real
       retrieved data rather than a number the model chose. Snapping only ever
       happens to a quote that already passed check 2.

    `extractor._verify_quotes` makes the same call for extracted quotes, and the
    two share `quote_is_present` so they cannot drift.
    """
    by_id: dict[str, list[Excerpt]] = {}
    for excerpt in used:
        by_id.setdefault(excerpt.recording_id, []).append(excerpt)

    kept: list[Citation] = []
    dropped: list[str] = []
    repaired = 0

    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        rec_id = str(item.get("recording_id") or "").strip()
        stamp = str(item.get("stamp") or item.get("timestamp") or "").strip()
        quote = str(item.get("quote") or item.get("text") or "").strip()

        excerpts = by_id.get(rec_id)
        if not excerpts:
            dropped.append(rec_id or "(no recording id)")
            log.warning(
                "ask: dropped a citation naming '%s', which was not in the bundle "
                "sent to the model", rec_id or "(empty)",
            )
            continue

        # The quote has to be findable in THIS recording's excerpts. The one
        # whose words it matches becomes its source; if none match, the model
        # invented it.
        match = next(
            (e for e in excerpts if quote_is_present(quote, _flatten(e.text))), None
        )
        if match is None:
            dropped.append(f"{rec_id} (quote not in the excerpts)")
            log.warning(
                "ask: dropped a citation for %s whose quote was in none of the "
                "excerpts sent for it: %.60s", rec_id, quote,
            )
            continue

        if stamp != match.stamp:
            if stamp:
                repaired += 1
                log.info(
                    "ask: citation for %s used timestamp '%s', which was not where "
                    "the quote was found; snapped to '%s'", rec_id, stamp, match.stamp,
                )
            stamp = match.stamp

        kept.append(Citation(
            recording_id=rec_id, stamp=stamp, quote=quote,
            source_name=match.source_name, when=match.when,
        ))
    return kept, dropped, repaired


def _overlap(a: str, b: str) -> int:
    return len(set(_tokens(a)) & set(_tokens(b)))


# =========================================================================
# The command
# =========================================================================
def ask(question: str, cfg, db, archive, *, profile: str | None = None,
        days: int | None = None, limit: int | None = None,
        include_personal: bool = False, local_only: bool | None = None) -> Answer:
    """
    Answer a question from what was actually said, with citations.

    Never raises for a missing model, an empty archive, or a locked vault. Each
    of those produces an Answer that says what happened and what would fix it,
    because the caller is a person holding a phone rather than a stack trace
    reader.
    """
    question = (question or "").strip()
    answer = Answer(question=question)
    notes: list[str] = []

    if not question:
        answer.degraded = True
        answer.usage_error = True
        answer.text = "No question was asked."
        answer.note = 'Ask something, for example: run.py ask "what did I promise Marcus?"'
        return answer

    if profile and profile not in cfg.profiles:
        answer.degraded = True
        answer.usage_error = True
        answer.text = f"There is no profile called '{profile}'."
        answer.note = (
            "Known profiles: " + ", ".join(sorted(cfg.profiles)) + ". "
            "Run `run.py profiles` to see them with their settings."
        )
        return answer

    if (
        profile
        and cfg.profile(profile).exclude_from_combined_export
        and not include_personal
    ):
        notes.append(
            f"'{profile}' is a personal profile and stays out of an answer unless "
            "you ask for it with --include-personal."
        )

    found = retrieve(
        question, cfg, db, archive, profile=profile, days=days,
        limit=limit, include_personal=include_personal,
    )
    answer.recordings_considered = found.considered
    answer.unopened = list(found.unopened)

    if found.personal_skipped and not include_personal:
        notes.append(
            f"{found.personal_skipped} personal recording(s) were left out. "
            "Pass --include-personal to search them too."
        )
    if found.truncated_scan:
        notes.append(
            f"Only {found.considered} of {found.total} recording(s) in the window "
            "were opened (ask.scan_limit). Raise it, or set it to 0, to search "
            "everything."
        )
    if found.unopened:
        notes.append(
            f"{len(found.unopened)} recording(s) could not be opened and were NOT "
            "searched. Set PLAUD_BRIDGE_PASSPHRASE if they are encrypted."
        )

    profile_ids = {pid for c in found.candidates for pid in c.profile_ids}
    answer.profiles = sorted(profile_ids)
    answer.local_only, locality_note = _locality(cfg, profile_ids, local_only)
    if local_only is False and answer.local_only and not is_offline(cfg):
        notes.append("Cloud was permitted by the caller and refused by profile policy.")

    if not found.candidates:
        answer.degraded = True
        answer.text = (
            f'Nothing in the archive matched "{question}". This is search output, '
            "not an answer: no model was asked, because there was nothing to ask "
            "it about."
        )
        notes.append(
            "Try fewer or different words, widen the window with --days, or search "
            'the raw text with: run.py search "<phrase>" --content'
        )
        answer.note = " ".join(notes)
        _audit(db, answer, found)
        return answer

    bundle = _bundle(
        found.candidates,
        local_only=answer.local_only,
        redact=_redaction_required(cfg, profile_ids),
        patterns=cfg.get("compliance.redact_patterns") or {},
        max_chars=int(cfg.get("ask.max_context_chars", 24000)),
    )
    answer.excerpts = bundle.used
    answer.bundle_chars = len(bundle.text)
    answer.left_out = bundle.left_out
    answer.truncated = bool(bundle.left_out)
    answer.redactions = bundle.redactions

    if bundle.left_out:
        notes.append(
            f"{bundle.left_out} excerpt(s) did not fit in the context budget "
            "(ask.max_context_chars) and the answer was written without them."
        )
    if bundle.withheld:
        notes.append(
            f"{bundle.withheld} suppressed field(s) were withheld because this "
            "question was allowed to use a cloud provider."
        )
    if bundle.redactions:
        notes.append(
            "Redacted before the model saw it: "
            + ", ".join(f"{k}={v}" for k, v in sorted(bundle.redactions.items())) + "."
        )
    if locality_note:
        notes.append(locality_note)

    user = _user_prompt(
        question, bundle.text, bundle.left_out, bundle.withheld,
        found.truncated_scan, len(found.unopened),
    )

    try:
        data, response = complete_json(
            cfg, _SYSTEM, user,
            local_only=answer.local_only,
            max_tokens=int(cfg.get("ask.max_tokens", 2000)),
        )
    except LLMError as exc:
        answer.degraded = True
        answer.text = (
            f"{len(bundle.used)} excerpt(s) from {len(answer.recordings_used)} "
            "recording(s) match this question. They are listed below unsummarised: "
            "this is search output, not an answer, because no model could be reached."
        )
        # The chain's message lists one provider per line. The note is a
        # paragraph, so flatten it rather than shredding the paragraph.
        notes.append("The provider chain said: " + " ".join(str(exc).split()))
        notes.append(
            "Configure llm.local in pipeline.yaml (ollama or vLLM) if you want "
            "answers rather than excerpts. `run.py doctor` shows what is missing."
        )
        answer.note = " ".join(notes)
        _audit(db, answer, found)
        return answer

    answer.provider = response.provider
    answer.cost_usd = response.cost_usd
    # ADR-014: spend is counted wherever it is incurred. A question costs money
    # and has no recording to hang it on, so it goes to the spend table or it
    # goes nowhere, and `status` would keep reporting only what ingestion cost.
    try:
        db.record_spend("ask", response.cost_usd, response.provider, response.model)
    except Exception as exc:  # noqa: BLE001 - an unrecorded cost must not lose the answer
        log.warning("could not record what this question cost: %s", exc)
    answer.text = str(data.get("answer") or "").strip()
    answer.unanswered = str(data.get("unanswered") or "").strip()
    confidence = str(data.get("confidence") or "").strip().lower()
    answer.confidence = confidence if confidence in _CONFIDENCE else "low"

    answer.citations, answer.dropped_citations, answer.repaired_citations = _validate_citations(
        data.get("citations"), bundle.used,
    )
    if answer.dropped_citations:
        notes.append(
            f"{len(answer.dropped_citations)} citation(s) were dropped because they "
            "named recordings that were never sent to the model: "
            + ", ".join(answer.dropped_citations) + ". Treat the rest with care."
        )
    if answer.repaired_citations:
        notes.append(
            f"{answer.repaired_citations} citation(s) carried a timestamp that was "
            "not in the excerpts; each was moved to the nearest quote that was."
        )
    if not answer.text:
        answer.text = "The model returned no answer text."
        notes.append("That usually means the excerpts did not contain the answer.")

    answer.note = " ".join(notes)
    _audit(db, answer, found)
    return answer


def _audit(db, answer: Answer, found: Retrieval) -> None:
    """
    Record that a question was asked, without recording the question.

    The audit table lives in the plain SQLite index (ADR-013), and a question
    like "what did I promise the Hendersons about the biopsy" is every bit as
    identifying as the recording it is about. Counts and locality are what an
    audit trail needs here; the wording is not.
    """
    try:
        db.audit(
            "ask",
            f"{found.considered} recording(s) searched, {len(answer.recordings_used)} cited, "
            f"{answer.bundle_chars} context chars, local_only={answer.local_only}, "
            f"provider={answer.provider or 'none'}, dropped_citations="
            f"{len(answer.dropped_citations)}",
            actor="human",
        )
    except Exception as exc:  # noqa: BLE001 - an unwritable audit must not eat the answer
        log.warning("ask: could not write the audit entry (%s)", exc)


def save_answer(answer: Answer, cfg, vault: Vault | None = None) -> Path:
    """
    Write an answer to disk, encrypted, or not at all.

    An answer quotes what people said, which makes it speech content and gives
    it the same rule as a transcript: through the vault, or nowhere. Refusing is
    the correct outcome when the passphrase is missing -- you will notice a
    refusal, and you would not notice a plaintext file holding a client's
    financial disclosures.
    """
    vault = vault or Vault(cfg.path("vault"))
    ok, why = vault.ready()
    if not ok:
        raise VaultError(
            f"{why} Refusing to write an answer containing quoted speech in "
            "plaintext. Set PLAUD_BRIDGE_PASSPHRASE and ask again."
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = json.dumps(
        {
            "asked_at": datetime.now(timezone.utc).isoformat(),
            "question": answer.question,
            "answer": answer.text,
            "confidence": answer.confidence,
            "unanswered": answer.unanswered,
            "citations": [
                {
                    "recording_id": c.recording_id, "stamp": c.stamp, "quote": c.quote,
                    "source_name": c.source_name, "when": c.when,
                }
                for c in answer.citations
            ],
            "recordings_used": answer.recordings_used,
            "local_only": answer.local_only,
            "provider": answer.provider,
            "cost_usd": answer.cost_usd,
            "degraded": answer.degraded,
            "note": answer.note,
        },
        indent=2, ensure_ascii=False,
    )
    return vault.write(f"ask/{stamp}", payload)
