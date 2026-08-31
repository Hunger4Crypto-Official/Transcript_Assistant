"""
How you actually talk, measured rather than remembered.

Nobody hears themselves. A producer who talks through eighty percent of a
discovery call, never asks a question, and delivers four-minute monologues
cannot learn any of that from a summary -- a summary is about what was said,
not about who said it, for how long, over whom. But the numbers are already
sitting in the stored segments: every one carries a speaker, a start, an end,
and its words, and plain arithmetic over them answers questions no model
should be asked. Same reasoning as the glossary (ADR-006) and the episode
cutter: deterministic, local, auditable. Every figure this module produces can
be recomputed by hand from `run.py open <id>`.

Nothing is stored. Every metric is derived on demand from the archive, because
a second store of derived numbers would be a second thing `forget` has to
chase -- and a talk-share table still quoting a deleted conversation is
exactly the leak that command exists to close. Recomputing a month of
recordings is milliseconds of arithmetic; invalidating a cache is a career.

Honest limits, because a coach that overstates its evidence teaches the wrong
lesson:

- **The timestamps are the recogniser's.** Talk seconds and pace are only as
  good as the segment boundaries. An imported text transcript carries
  synthesised timestamps (a fixed words-per-second guess), so its pace reads
  as near-constant by construction and its silences are fiction.
- **Questions are detected by shape**, not meaning: a segment ending in "?"
  or opening with an interrogative. "Tell me about your coverage" is a
  question in spirit and counts as a statement; "Do it now" is an order and
  counts as a question. The rate is consistent across your own recordings,
  which is what a trend needs; it is not a semantic count.
- **`interruptions_approx` is overlap, not rudeness.** See `measure`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from .logging_setup import get
from .models import Segment, format_stamp

log = get("insights")

# Gaps shorter than this are conversational rhythm -- a breath, a nod, a lag
# in the recogniser's boundaries -- and calling them silence would make every
# recording read as one-third dead air. Longer than this, somebody stopped
# talking. Config: insights.silence_gap_seconds.
DEFAULT_SILENCE_GAP = 3.0
DEFAULT_TREND_DAYS = 90
# "This month against the one before" is the comparison a person can act on.
DELTA_WINDOW_DAYS = 30

# Words a question opens with. Aux-inversions ("did you", "can we") are the
# common spoken form once the recogniser has dropped the question mark, which
# it routinely does. Kept small on purpose: every entry also matches some
# declaratives, and a long list converts the rate into noise.
_QUESTION_OPENERS = frozenset("""
what when where who whom whose which why how
do does did am is are was were can could would will shall should
""".split())
_FIRST_WORD = re.compile(r"[a-z']+")


class InsightsError(RuntimeError):
    """A metrics request that cannot be answered, and why."""

    def __init__(self, message: str, *, unopened: bool = False):
        super().__init__(message)
        # True when the recording exists but would not decrypt -- a different
        # problem from "no such recording", and one the exit code should keep
        # distinct so a script can tell a typo from a locked vault.
        self.unopened = unopened


def is_question(text: str) -> bool:
    """
    Whether a segment reads as a question, by shape alone.

    Ends in "?", or opens with an interrogative. Both halves matter: ASR
    punctuation is unreliable enough that "?" alone undercounts badly, and
    openers alone miss "you free Thursday?". Deterministic and wrong at the
    edges in both directions -- documented in the module docstring -- but
    wrong the same way every time, which is what comparing yourself across
    months requires.
    """
    body = text.strip()
    if not body:
        return False
    if body.rstrip("\"')]* ").endswith("?"):
        return True
    match = _FIRST_WORD.match(body.lower())
    return bool(match) and match.group(0) in _QUESTION_OPENERS


def _is_owner(speaker: str, owner_label: str) -> bool:
    """Exact match, same rule and reason as the consent detector's."""
    a, b = speaker.strip().lower(), owner_label.strip().lower()
    return bool(a) and bool(b) and a == b


# =========================================================================
# Per-recording arithmetic
# =========================================================================
@dataclass
class SpeakerMetrics:
    """One speaker's numbers within one recording."""

    speaker: str
    is_owner: bool = False
    seconds: float = 0.0
    share: float = 0.0                  # of everything spoken, not of the clock
    words: int = 0
    words_per_minute: float = 0.0
    segment_count: int = 0
    questions: int = 0
    question_rate: float = 0.0          # questions / their segment count
    longest_monologue_seconds: float = 0.0
    interruptions_approx: int = 0

    @property
    def minutes(self) -> float:
        return self.seconds / 60.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker": self.speaker,
            "is_owner": self.is_owner,
            "seconds": round(self.seconds, 1),
            "minutes": round(self.minutes, 2),
            "share": round(self.share, 4),
            "words": self.words,
            "words_per_minute": round(self.words_per_minute, 1),
            "segment_count": self.segment_count,
            "questions": self.questions,
            "question_rate": round(self.question_rate, 4),
            "longest_monologue_seconds": round(self.longest_monologue_seconds, 1),
            "interruptions_approx": self.interruptions_approx,
        }


@dataclass
class RecordingMetrics:
    """One recording, measured. Everything here is arithmetic over segments."""

    recording_id: str = ""
    source_name: str = ""
    when: str = ""                      # YYYY-MM-DD
    profile_id: str = ""
    duration_seconds: float = 0.0       # wall clock, first start to last end
    spoken_seconds: float = 0.0
    silence_seconds: float = 0.0
    silence_share: float = 0.0          # of the wall clock, not of speech
    owner: str = ""                     # the matched speaker label, or ""
    speakers: list[SpeakerMetrics] = field(default_factory=list)

    @property
    def owner_metrics(self) -> SpeakerMetrics | None:
        return next((s for s in self.speakers if s.is_owner), None)

    @property
    def words(self) -> int:
        return sum(s.words for s in self.speakers)

    @property
    def segment_count(self) -> int:
        return sum(s.segment_count for s in self.speakers)

    @property
    def questions(self) -> int:
        return sum(s.questions for s in self.speakers)

    @property
    def longest_monologue_seconds(self) -> float:
        return max((s.longest_monologue_seconds for s in self.speakers), default=0.0)

    @property
    def interruptions_approx(self) -> int:
        return sum(s.interruptions_approx for s in self.speakers)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recording_id": self.recording_id,
            "source_name": self.source_name,
            "when": self.when,
            "profile_id": self.profile_id,
            "duration_seconds": round(self.duration_seconds, 1),
            "spoken_seconds": round(self.spoken_seconds, 1),
            "silence_seconds": round(self.silence_seconds, 1),
            "silence_share": round(self.silence_share, 4),
            "owner": self.owner,
            "words": self.words,
            "segment_count": self.segment_count,
            "questions": self.questions,
            "longest_monologue_seconds": round(self.longest_monologue_seconds, 1),
            "interruptions_approx": self.interruptions_approx,
            "speakers": [s.to_dict() for s in self.speakers],
        }


def metrics_dict(obj: Any) -> dict[str, Any]:
    """
    The serialisable shape, for anything that charts these numbers.

    One function rather than callers reaching for `.to_dict()` directly, so
    the app and the digest depend on a name this module promises to keep.
    """
    return obj.to_dict()


def measure(segments: list[Any], *, gap_seconds: float = DEFAULT_SILENCE_GAP,
            owner_label: str = "") -> RecordingMetrics:
    """
    Every metric for one recording, from its segments alone.

    Accepts stored segment dicts or Segment objects. Segments with no text are
    ignored: they carry no words and usually no speech.

    Division is guarded everywhere. A one-speaker memo is a normal input --
    share 1.0, zero interruptions, a possibly enormous monologue -- and an
    empty list returns zeros, never NaN, because these numbers feed charts and
    a chart that renders NaN renders nothing.

    On `interruptions_approx`: a speaker is charged one when their segment
    STARTS before the previous different-speaker segment ENDED. That is
    overlap in the recogniser's timeline, which is all we can actually see.
    It is an approximation of interruption, not proof of rudeness: diarization
    smears boundaries by a second or so either way, back-channel agreement
    ("mm-hm", "right") overlaps constantly and is the opposite of rude, and
    two people finishing each other's sentences overlap by design. The suffix
    is in the name so nobody quotes it without the caveat. Watch the trend,
    not the single count.

    Monologue runs break on a speaker change or on a silence gap longer than
    `gap_seconds` -- the same pause that would end a monologue in the room.
    Sub-threshold pauses stay inside the run, so its length is measured wall
    clock, first word to last.
    """
    segs: list[Segment] = []
    for raw in segments or []:
        seg = Segment.from_dict(raw) if isinstance(raw, dict) else raw
        if seg.text.strip():
            segs.append(seg)
    segs.sort(key=lambda s: (s.start, s.end))

    out = RecordingMetrics()
    if not segs:
        return out

    def blank() -> dict[str, Any]:
        return {"seconds": 0.0, "words": 0, "segments": 0, "questions": 0,
                "monologue": 0.0, "interruptions": 0}

    per: dict[str, dict[str, Any]] = {}
    spoken = silence = 0.0
    # Furthest end seen so far. Gaps are measured against this rather than the
    # previous segment's end, because overlapping segments would otherwise
    # manufacture "silence" in the middle of continuous speech.
    reach = segs[0].end
    run_speaker, run_start, run_end = segs[0].speaker, segs[0].start, segs[0].end

    for i, seg in enumerate(segs):
        bucket = per.setdefault(seg.speaker, blank())
        bucket["seconds"] += seg.duration
        bucket["words"] += len(seg.text.split())
        bucket["segments"] += 1
        if is_question(seg.text):
            bucket["questions"] += 1
        spoken += seg.duration

        if i:
            prev = segs[i - 1]
            gap = seg.start - reach
            if gap > gap_seconds:
                silence += gap
            if seg.speaker != prev.speaker and seg.start < prev.end:
                bucket["interruptions"] += 1
            if seg.speaker != run_speaker or gap > gap_seconds:
                run = per.setdefault(run_speaker, blank())
                run["monologue"] = max(run["monologue"], run_end - run_start)
                run_speaker, run_start, run_end = seg.speaker, seg.start, seg.end
            else:
                run_end = max(run_end, seg.end)
        reach = max(reach, seg.end)

    run = per.setdefault(run_speaker, blank())
    run["monologue"] = max(run["monologue"], run_end - run_start)

    duration = max(0.0, reach - segs[0].start)
    out.duration_seconds = duration
    out.spoken_seconds = spoken
    out.silence_seconds = silence
    out.silence_share = silence / duration if duration > 0 else 0.0

    for name, b in sorted(per.items(), key=lambda kv: (-kv[1]["seconds"], kv[0])):
        minutes = b["seconds"] / 60.0
        sm = SpeakerMetrics(
            speaker=name,
            is_owner=_is_owner(name, owner_label),
            seconds=b["seconds"],
            share=b["seconds"] / spoken if spoken > 0 else 0.0,
            words=b["words"],
            words_per_minute=b["words"] / minutes if minutes > 0 else 0.0,
            segment_count=b["segments"],
            questions=b["questions"],
            question_rate=b["questions"] / b["segments"] if b["segments"] else 0.0,
            longest_monologue_seconds=b["monologue"],
            interruptions_approx=b["interruptions"],
        )
        out.speakers.append(sm)
        if sm.is_owner:
            out.owner = name
    return out


def recording_metrics(cfg, db, archive, recording_id: str) -> RecordingMetrics:
    """
    One stored recording, measured. Raises InsightsError when it cannot be:
    no such recording, or one whose content will not open. The two are kept
    apart (`unopened`) because the fixes are different -- retype the id, or
    set PLAUD_BRIDGE_PASSPHRASE.
    """
    total = db.count_recordings()
    row = next((r for r in db.query(limit=total or 1) if r["id"] == recording_id), None)
    if row is None:
        raise InsightsError(
            f"no recording with id '{recording_id}'. `run.py status` shows what exists; "
            "`run.py search <name>` finds an id from a filename."
        )
    segments = archive.segments(row)
    if segments is None:
        raise InsightsError(
            f"{recording_id} exists but its content could not be opened. Set "
            "PLAUD_BRIDGE_PASSPHRASE if it is encrypted; `run.py verify` says what "
            "is actually wrong.",
            unopened=True,
        )
    m = measure(
        segments,
        gap_seconds=float(cfg.get("insights.silence_gap_seconds", DEFAULT_SILENCE_GAP)),
        owner_label=str(cfg.get("diarization.owner_label", "") or ""),
    )
    m.recording_id = row["id"]
    m.source_name = row["source_name"] or ""
    m.when = ((row["recorded_at"] or row["ingested_at"] or "") or "")[:10]
    m.profile_id = row["governing_profile"] or ""
    return m


# =========================================================================
# Windowed trends
# =========================================================================
@dataclass
class WindowAggregate:
    """
    One set of numbers over many recordings.

    `focus` says whose numbers these are: "owner" when the owner's speaker
    label was identifiable (coaching is about you, not about the average of
    everyone you met), falling back to "everyone" when it never was, because
    numbers about somebody are better than numbers about nobody -- as long as
    they say which they are.
    """

    focus: str = "owner"
    recordings: int = 0
    seconds: float = 0.0
    share: float = 0.0
    words: int = 0
    words_per_minute: float = 0.0
    segment_count: int = 0
    questions: int = 0
    question_rate: float = 0.0
    longest_monologue_seconds: float = 0.0
    interruptions_approx: int = 0

    @property
    def minutes(self) -> float:
        return self.seconds / 60.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "focus": self.focus,
            "recordings": self.recordings,
            "seconds": round(self.seconds, 1),
            "minutes": round(self.minutes, 2),
            "share": round(self.share, 4),
            "words": self.words,
            "words_per_minute": round(self.words_per_minute, 1),
            "segment_count": self.segment_count,
            "questions": self.questions,
            "question_rate": round(self.question_rate, 4),
            "longest_monologue_seconds": round(self.longest_monologue_seconds, 1),
            "interruptions_approx": self.interruptions_approx,
        }


def _aggregate(items: list[RecordingMetrics]) -> WindowAggregate:
    agg = WindowAggregate(recordings=0)
    owner_rows = [(m, m.owner_metrics) for m in items if m.owner_metrics is not None]

    if owner_rows:
        agg.focus = "owner"
        agg.recordings = len(owner_rows)
        total_spoken = sum(m.spoken_seconds for m, _sm in owner_rows)
        agg.seconds = sum(sm.seconds for _m, sm in owner_rows)
        agg.words = sum(sm.words for _m, sm in owner_rows)
        agg.segment_count = sum(sm.segment_count for _m, sm in owner_rows)
        agg.questions = sum(sm.questions for _m, sm in owner_rows)
        agg.longest_monologue_seconds = max(
            (sm.longest_monologue_seconds for _m, sm in owner_rows), default=0.0)
        agg.interruptions_approx = sum(sm.interruptions_approx for _m, sm in owner_rows)
    elif items:
        agg.focus = "everyone"
        agg.recordings = len(items)
        total_spoken = sum(m.spoken_seconds for m in items)
        agg.seconds = total_spoken
        agg.words = sum(m.words for m in items)
        agg.segment_count = sum(m.segment_count for m in items)
        agg.questions = sum(m.questions for m in items)
        agg.longest_monologue_seconds = max(
            (m.longest_monologue_seconds for m in items), default=0.0)
        agg.interruptions_approx = sum(m.interruptions_approx for m in items)
    else:
        return agg

    agg.share = agg.seconds / total_spoken if total_spoken > 0 else 0.0
    minutes = agg.seconds / 60.0
    agg.words_per_minute = agg.words / minutes if minutes > 0 else 0.0
    agg.question_rate = agg.questions / agg.segment_count if agg.segment_count else 0.0
    return agg


@dataclass
class TrendReport:
    """The window, measured, with the honest edges attached."""

    days: int = DEFAULT_TREND_DAYS
    owner_label: str = ""
    owner_recordings: int = 0           # recordings where the owner was identifiable
    generated: str = ""
    overall: WindowAggregate = field(default_factory=WindowAggregate)
    current: WindowAggregate = field(default_factory=WindowAggregate)
    prior: WindowAggregate = field(default_factory=WindowAggregate)
    deltas: dict[str, float] = field(default_factory=dict)
    by_profile: dict[str, WindowAggregate] = field(default_factory=dict)
    recordings: list[RecordingMetrics] = field(default_factory=list)
    unopened: list[str] = field(default_factory=list)
    excluded_personal: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "days": self.days,
            "owner_label": self.owner_label,
            "owner_recordings": self.owner_recordings,
            "generated": self.generated,
            "overall": self.overall.to_dict(),
            "current": self.current.to_dict(),
            "prior": self.prior.to_dict(),
            "deltas": {k: round(v, 4) for k, v in self.deltas.items()},
            "by_profile": {pid: agg.to_dict() for pid, agg in self.by_profile.items()},
            "recordings": [m.to_dict() for m in self.recordings],
            "unopened": list(self.unopened),
            "excluded_personal": self.excluded_personal,
        }


def trend(cfg, db, archive, *, profile: str | None = None, days: int = DEFAULT_TREND_DAYS,
          include_personal: bool = False) -> TrendReport:
    """
    The same metrics per recording over time, focused on the owner's numbers.

    Personal profiles are excluded unless asked for by name or with
    `include_personal` -- the same rule the combined digest follows and for
    the same reason: numbers about bedtime do not belong in a document about
    work, even in aggregate.

    A recording whose content will not open is reported and skipped rather
    than guessed at, and the report carries the list so a caller can say so.
    Deltas compare the last 30 days against the 30 before, and are only
    computed when both windows hold recordings measured on the same focus --
    a "you got quieter" claim built from you-in-one-month and everyone-in-the-
    other would be an artifact of the bookkeeping, not of your behaviour.
    """
    days = days if days and days > 0 else DEFAULT_TREND_DAYS
    owner = str(cfg.get("diarization.owner_label", "") or "")
    gap = float(cfg.get("insights.silence_gap_seconds", DEFAULT_SILENCE_GAP))
    personal = {p.id for p in cfg.profiles.values() if p.exclude_from_combined_export}

    report = TrendReport(
        days=days, owner_label=owner,
        generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
    )

    total = db.count_recordings(profile_id=profile, since_days=days)
    rows = db.query(profile_id=profile, since_days=days, limit=total or 1)

    for row in rows:
        pid = row["governing_profile"] or ""
        if pid in personal and profile is None and not include_personal:
            report.excluded_personal += 1
            continue
        segments = archive.segments(row)
        if segments is None:
            report.unopened.append(f"{row['id']}  {row['source_name']}")
            continue
        if not segments:
            # Quarantined, or empty: there is nothing to measure, which is
            # different from something unreadable.
            continue
        m = measure(segments, gap_seconds=gap, owner_label=owner)
        m.recording_id = row["id"]
        m.source_name = row["source_name"] or ""
        m.when = ((row["recorded_at"] or row["ingested_at"] or "") or "")[:10]
        m.profile_id = pid
        report.recordings.append(m)

    if report.unopened:
        log.warning(
            "%d recording(s) could not be opened and are not counted: %s. "
            "Set PLAUD_BRIDGE_PASSPHRASE if they are encrypted.",
            len(report.unopened), "; ".join(report.unopened[:5]),
        )

    report.owner_recordings = len([m for m in report.recordings if m.owner_metrics])
    report.overall = _aggregate(report.recordings)

    now = datetime.now(timezone.utc)
    edge = (now - timedelta(days=DELTA_WINDOW_DAYS)).date().isoformat()
    floor = (now - timedelta(days=2 * DELTA_WINDOW_DAYS)).date().isoformat()
    report.current = _aggregate([m for m in report.recordings if m.when >= edge])
    report.prior = _aggregate(
        [m for m in report.recordings if floor <= m.when < edge])

    if (report.current.recordings and report.prior.recordings
            and report.current.focus == report.prior.focus):
        report.deltas = {
            "talk_share": report.current.share - report.prior.share,
            "words_per_minute": (report.current.words_per_minute
                                 - report.prior.words_per_minute),
            "question_rate": report.current.question_rate - report.prior.question_rate,
            "longest_monologue_seconds": (report.current.longest_monologue_seconds
                                          - report.prior.longest_monologue_seconds),
        }

    by_profile: dict[str, list[RecordingMetrics]] = {}
    for m in report.recordings:
        by_profile.setdefault(m.profile_id or "unfiled", []).append(m)
    report.by_profile = {
        pid: _aggregate(items) for pid, items in sorted(by_profile.items())
    }
    return report


# =========================================================================
# Rendering
# =========================================================================
_FOOTNOTE = (
    "<sub>Computed from the stored segments; nothing here came from a model. "
    "Interruptions are overlap approximations -- a segment starting before the "
    "previous speaker's ended -- and back-channel agreement overlaps constantly, "
    "so read them as a trend, not an accusation. Question counting is by shape "
    "(ends in '?' or opens with an interrogative) and misses questions phrased "
    "as statements.</sub>"
)


def _cell(text: str) -> str:
    """A table cell that cannot end its own row; same rule as followups."""
    return text.replace("|", "/").replace("\n", " ").strip()


def _pct(value: float) -> str:
    return f"{value:.0%}"


def render_recording(m: RecordingMetrics, *, title: str | None = None) -> str:
    """One call's breakdown per speaker, in the digest's markdown shape."""
    heading = title or f"Insights — {m.source_name or m.recording_id or 'recording'}"
    out: list[str] = [f"# {heading}", ""]

    meta = [p for p in (m.recording_id, m.when, m.profile_id) if p]
    if meta:
        out += ["`" + " · ".join(meta) + "`", ""]

    if not m.speakers:
        out += ["Nothing to measure: this recording has no spoken segments.", ""]
        return "\n".join(out)

    out += [
        "`" + " · ".join([
            f"{m.duration_seconds / 60.0:.1f} min recorded",
            f"{m.spoken_seconds / 60.0:.1f} min spoken",
            f"{_pct(m.silence_share)} silence",
            f"{len(m.speakers)} speaker(s)",
        ]) + "`",
        "",
        "| Speaker | Talk share | Minutes | Pace | Questions | Longest monologue "
        "| Interruptions* |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in m.speakers:
        name = f"{s.speaker} (you)" if s.is_owner else s.speaker
        out.append(
            f"| {_cell(name)} | {_pct(s.share)} | {s.minutes:.1f} "
            f"| {s.words_per_minute:.0f} wpm "
            f"| {_pct(s.question_rate)} ({s.questions}/{s.segment_count}) "
            f"| {format_stamp(s.longest_monologue_seconds)} "
            f"| {s.interruptions_approx} |"
        )
    out += ["", _FOOTNOTE, ""]
    return "\n".join(out)


def _who(report: TrendReport) -> str:
    if report.overall.focus == "owner" and report.owner_label:
        return f"You ({report.owner_label})"
    return "Everyone (the owner's voice was not identifiable in this window)"


def _aggregate_lines(agg: WindowAggregate) -> list[str]:
    return [
        f"- Talk share: {_pct(agg.share)} of spoken time "
        f"({agg.minutes:.1f} min across {agg.recordings} recording(s))",
        f"- Pace: {agg.words_per_minute:.0f} words per minute",
        f"- Question rate: {_pct(agg.question_rate)} of segments "
        f"({agg.questions}/{agg.segment_count})",
        f"- Worst monologue: {format_stamp(agg.longest_monologue_seconds)}",
        f"- Interruptions*: {agg.interruptions_approx}",
    ]


def render_trend(report: TrendReport, *, title: str | None = None) -> str:
    """The coaching summary, in the digest's markdown shape."""
    heading = title or f"Insights — last {report.days} days"
    out: list[str] = [f"# {heading}", ""]

    if not report.recordings:
        out += [
            "Nothing to measure in this window. Either nothing was recorded, or "
            "everything in it was excluded or could not be opened."
        ]
        if report.excluded_personal:
            out.append(
                f"{report.excluded_personal} personal recording(s) were excluded; "
                "--include-personal counts them."
            )
        if report.unopened:
            out.append(
                f"{len(report.unopened)} recording(s) could not be opened. Set "
                "PLAUD_BRIDGE_PASSPHRASE if they are encrypted."
            )
        out.append("")
        return "\n".join(out)

    summary = [f"{len(report.recordings)} recording(s)"]
    if report.overall.focus == "owner":
        summary.append(f"owner identified in {report.owner_recordings}")
    summary.append(f"generated {report.generated}")
    out += ["`" + " · ".join(summary) + "`", ""]

    out += [f"## {_who(report)}", ""]
    out += _aggregate_lines(report.overall)
    out.append("")

    if report.deltas:
        cur, pri = report.current, report.prior
        out += [
            f"## Last {DELTA_WINDOW_DAYS} days vs the {DELTA_WINDOW_DAYS} before", "",
            "| Metric | Now | Before | Change |",
            "|---|---:|---:|---:|",
            f"| Talk share | {_pct(cur.share)} | {_pct(pri.share)} "
            f"| {report.deltas['talk_share']:+.0%} |",
            f"| Pace (wpm) | {cur.words_per_minute:.0f} | {pri.words_per_minute:.0f} "
            f"| {report.deltas['words_per_minute']:+.0f} |",
            f"| Question rate | {_pct(cur.question_rate)} | {_pct(pri.question_rate)} "
            f"| {report.deltas['question_rate']:+.0%} |",
            f"| Worst monologue | {format_stamp(cur.longest_monologue_seconds)} "
            f"| {format_stamp(pri.longest_monologue_seconds)} "
            f"| {report.deltas['longest_monologue_seconds']:+.0f}s |",
            "",
        ]

    if report.by_profile:
        out += [
            "## By profile", "",
            "| Profile | Recordings | Talk share | Pace | Questions | Worst monologue |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for pid, agg in report.by_profile.items():
            out.append(
                f"| {_cell(pid)} | {agg.recordings} | {_pct(agg.share)} "
                f"| {agg.words_per_minute:.0f} wpm | {_pct(agg.question_rate)} "
                f"| {format_stamp(agg.longest_monologue_seconds)} |"
            )
        out.append("")

    if report.excluded_personal:
        out += [
            f"{report.excluded_personal} personal recording(s) excluded, the same "
            "rule as the digest; --include-personal counts them.",
            "",
        ]
    if report.unopened:
        out += [
            f"**{len(report.unopened)} recording(s) could not be opened and are "
            "not counted:**",
        ]
        out += [f"- `{entry}`" for entry in report.unopened[:10]]
        out += ["", "Set PLAUD_BRIDGE_PASSPHRASE if they are encrypted; these "
                "numbers describe less than the whole archive.", ""]

    out += [_FOOTNOTE, ""]
    return "\n".join(out)
