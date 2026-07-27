"""
Cutting a long recording into episodes.

A day is not one conversation. Wear the device from 8am to 6pm and you capture a
client fact-find, a drive, a coaching call, lunch with your wife, and bedtime
with your kids -- and the old pipeline classified all of that as one thing, from
a 14,000 character sample of it. Whatever the sample happened to contain won.

So the day is cut into episodes first, each episode is routed on its own, and
each profile's analysis is built only from the episodes that belong to it. That
is what turns "one recording" into "a rundown per profile".

Segmentation is deterministic and local. No model, no network, no randomness --
same reasoning as the glossary (ADR-006) and the stitcher (ADR-009). Four
signals, in order of how much they mean:

1. **A silence gap.** People leave, meetings end, you get in the car. The
   strongest and least ambiguous boundary there is.
2. **The set of people changed.** A different room, a different conversation.
3. **The subject changed.** Measured as vocabulary overlap between the window
   before and after a candidate point -- lexical cohesion, not topic modelling.
4. **Length.** An episode that has run too long is split regardless, because an
   episode nobody can analyse is not useful however coherent it is.

Every boundary records which signal produced it, so a surprising split can be
explained rather than guessed at.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .logging_setup import get
from .models import RouteMatch, Segment, Transcript, format_stamp

log = get("episodes")

# Words too common to say anything about what is being discussed. Kept short on
# purpose: a long stoplist starts throwing away domain vocabulary.
_STOP = frozenset("""
a about after all also am an and any are as at be been but by can could did do
does for from get got had has have he her him his how i if in into is it its
just like me my no not of on one or our out she so than that the their them then
there these they this to too up us was we were what when which who will with
would you your yeah yes okay ok right well going know think really
""".split())

_WORD = re.compile(r"[a-z0-9']+")


@dataclass
class Episode:
    """One coherent stretch of a recording."""

    index: int
    segments: list[Segment] = field(default_factory=list)
    reason: str = "start of recording"
    routes: list[RouteMatch] = field(default_factory=list)
    # True when the boundary that opened this episode was independently
    # conclusive -- a long silence. Those survive the minimum-length rule; a
    # two-minute client call with five minutes of driving either side is a
    # separate conversation no matter how short it is.
    strong: bool = False

    @property
    def start(self) -> float:
        return self.segments[0].start if self.segments else 0.0

    @property
    def end(self) -> float:
        return self.segments[-1].end if self.segments else 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def speakers(self) -> list[str]:
        seen: list[str] = []
        for seg in self.segments:
            if seg.speaker not in seen:
                seen.append(seg.speaker)
        return seen

    @property
    def profile_ids(self) -> list[str]:
        return [r.profile_id for r in self.routes]

    def transcript(self, source: Transcript | None = None) -> Transcript:
        """This episode as a standalone transcript, for routing or analysis."""
        return Transcript(
            segments=list(self.segments),
            language=source.language if source else "en",
            asr_provider=source.asr_provider if source else "",
            asr_model=source.asr_model if source else "",
            duration_seconds=self.duration,
        )

    def label(self) -> str:
        return f"{format_stamp(self.start)}-{format_stamp(self.end)}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "reason": self.reason,
            "speakers": self.speakers,
            "segment_count": len(self.segments),
            "routes": [r.to_dict() for r in self.routes],
        }


def _keywords(segments: list[Segment]) -> set[str]:
    words: set[str] = set()
    for seg in segments:
        for word in _WORD.findall(seg.text.lower()):
            if len(word) > 3 and word not in _STOP:
                words.add(word)
    return words


def _overlap(before: set[str], after: set[str]) -> float:
    """Jaccard overlap. 1.0 means the same subject, 0.0 means unrelated."""
    if not before or not after:
        return 1.0        # not enough vocabulary to claim a change
    return len(before & after) / len(before | after)


def _speaker_change(before: list[Segment], after: list[Segment]) -> float:
    """How much the set of people talking changed, 0.0 to 1.0."""
    a = {s.speaker for s in before}
    b = {s.speaker for s in after}
    if not a or not b:
        return 0.0
    return 1.0 - (len(a & b) / len(a | b))


def segment_episodes(transcript: Transcript, cfg) -> list[Episode]:
    """
    Cut a transcript into episodes.

    Always returns at least one episode when there is anything to return, so
    callers never have to special-case a short recording -- a twenty minute
    client call is simply one episode and behaves exactly as it did before.
    """
    segments = [s for s in transcript.segments if s.text.strip()]
    if not segments:
        return []

    enabled = bool(cfg.get("episodes.enabled", True))
    min_seconds = float(cfg.get("episodes.min_seconds", 120))
    max_seconds = float(cfg.get("episodes.max_seconds", 1800))
    gap_seconds = float(cfg.get("episodes.silence_gap_seconds", 45))
    topic_drop = float(cfg.get("episodes.topic_overlap_below", 0.08))
    speaker_shift = float(cfg.get("episodes.speaker_change_above", 0.6))
    window = int(cfg.get("episodes.comparison_window_segments", 12))
    short_enough = float(cfg.get("episodes.skip_below_seconds", 900))

    total = segments[-1].end - segments[0].start
    if not enabled or total <= short_enough:
        # A recording shorter than the threshold is one conversation. Cutting it
        # up would multiply the routing cost for no gain.
        return [Episode(index=0, segments=segments, reason="whole recording")]

    boundaries: list[tuple[int, str, bool]] = []
    for i in range(1, len(segments)):
        previous, current = segments[i - 1], segments[i]

        gap = current.start - previous.end
        if gap >= gap_seconds:
            boundaries.append((i, f"{gap:.0f}s silence", True))
            continue

        before = segments[max(0, i - window):i]
        after = segments[i:i + window]
        if len(before) < 4 or len(after) < 4:
            continue

        if _speaker_change(before, after) >= speaker_shift:
            boundaries.append((i, "the people talking changed", False))
            continue

        if _overlap(_keywords(before), _keywords(after)) <= topic_drop:
            boundaries.append((i, "the subject changed", False))

    episodes = _cut(segments, boundaries, min_seconds, max_seconds)
    log.info(
        "segmented %.0f min into %d episode(s): %s",
        total / 60.0, len(episodes),
        ", ".join(f"{e.label()} ({e.reason})" for e in episodes[:8]),
    )
    return episodes


def _cut(segments: list[Segment], boundaries: list[tuple[int, str, bool]],
         min_seconds: float, max_seconds: float) -> list[Episode]:
    """Apply the boundaries, then enforce the length rules."""
    marks = {index: (why, strong) for index, why, strong in boundaries}
    episodes: list[Episode] = []
    current: list[Segment] = []
    reason, strong = "start of recording", True

    for i, seg in enumerate(segments):
        opening = marks.get(i)
        too_long = current and (seg.end - current[0].start) >= max_seconds

        if current and (opening or too_long):
            episodes.append(Episode(
                index=len(episodes), segments=current, reason=reason, strong=strong,
            ))
            current = []
            reason, strong = opening if opening else (
                f"ran past {max_seconds / 60:.0f} min", True
            )
        current.append(seg)

    if current:
        episodes.append(Episode(
            index=len(episodes), segments=current, reason=reason, strong=strong,
        ))

    # A short episode is usually a fragment -- a passing remark rather than a
    # conversation -- and folding it into its neighbour avoids routing and
    # analysing it on its own.
    #
    # Unless the boundary that opened it was a long silence. That is conclusive
    # on its own: a two minute client call with five minutes of driving either
    # side is a separate conversation however short it is, and merging it into
    # the coaching session before it would put both in one rundown.
    merged: list[Episode] = []
    for episode in episodes:
        fragment = episode.duration < min_seconds and not episode.strong
        if merged and fragment:
            merged[-1].segments.extend(episode.segments)
            continue
        merged.append(episode)

    # The opening episode can be a fragment too, with nothing before it to
    # absorb it, so it folds forward instead.
    if len(merged) > 1 and merged[0].duration < min_seconds and not merged[1].strong:
        merged[1].segments[:0] = merged[0].segments
        merged[1].reason = merged[0].reason
        merged.pop(0)

    for index, episode in enumerate(merged):
        episode.index = index
    return merged


def transcript_for(profile_id: str, episodes: list[Episode],
                   source: Transcript | None = None) -> Transcript:
    """
    Everything from this recording that belongs to one profile.

    This is what makes a day produce a rundown per profile instead of one
    summary of everything: the Insurance Agent analysis sees the client calls and
    nothing else, and the Father analysis sees bedtime and nothing else.
    """
    segments: list[Segment] = []
    for episode in episodes:
        if profile_id in episode.profile_ids:
            segments.extend(episode.segments)
    return Transcript(
        segments=segments,
        language=source.language if source else "en",
        asr_provider=source.asr_provider if source else "",
        asr_model=source.asr_model if source else "",
        duration_seconds=sum(
            e.duration for e in episodes if profile_id in e.profile_ids
        ),
    )
