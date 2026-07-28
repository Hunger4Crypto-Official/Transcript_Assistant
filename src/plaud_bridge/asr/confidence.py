"""
Whether the transcript is worth believing.

Whisper does not decline. Hand it a car radio, a restaurant, a recital, or ten
minutes of a device in a pocket, and it will return fluent, well-punctuated
English that nobody said. This is the best-documented failure mode of the model
family and it is not a rough transcript with mistakes in it -- it is invented
text, indistinguishable in shape from the real thing.

That matters more here than it would elsewhere. Everything downstream treats the
transcript as fact: the router files it, the extractor pulls promises out of it,
the memory ledger carries those promises into next month's prompt, and the
follow-up worklist puts them in front of you as things you owe somebody. A
hallucinated sentence does not stay a hallucinated sentence. It becomes a
commitment you believe you made.

Every segment has carried an average log probability since the first version and
nothing has ever read it. This module reads it, along with the model's own
estimate that a span contained no speech at all, and answers one question: how
much of this transcript is the model guessing?

It does not delete anything or refuse to process. Deciding a recording is
worthless is not a call to make automatically -- a quiet conversation in a car
scores badly and is still the conversation you wanted. What it does is say so,
in the log, in the digest, and in the extraction prompt, so that a bad
transcript is read as a bad transcript rather than as testimony.

The thresholds are guesses tuned to nothing in particular, which is why they are
config and why `verdict` distinguishes "some of this is shaky" from "most of
this is". Run it against your own recordings before trusting either number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..logging_setup import get
from ..models import Segment

log = get("asr.confidence")

# faster-whisper reports avg_logprob per segment. Clean speech sits above about
# -0.5; below -1.0 the model is largely guessing at what it heard. This is the
# threshold most likely to need moving for a particular microphone.
DEFAULT_MIN_LOGPROB = -1.0
# The model's own probability that a span held no speech. High no_speech with
# confident-looking text is the classic hallucination signature: the model knows
# there was nothing there and produced a sentence regardless.
DEFAULT_MAX_NO_SPEECH = 0.6
DEFAULT_SUSPECT_SHARE = 0.30
DEFAULT_UNRELIABLE_SHARE = 0.60
# Below this much speech there is not enough to judge, and calling a ten second
# note unreliable on two shaky segments is noise rather than a warning.
DEFAULT_MIN_SECONDS = 20.0

OK = "ok"
SUSPECT = "suspect"
UNRELIABLE = "unreliable"
UNKNOWN = "unknown"


@dataclass
class Assessment:
    """How much of a transcript the model appears to have been guessing at."""

    verdict: str = UNKNOWN
    reason: str = ""
    low_share: float = 0.0
    silent_share: float = 0.0
    mean_logprob: float | None = None
    scored: int = 0
    total: int = 0
    seconds: float = 0.0
    worst: list[str] = field(default_factory=list)

    @property
    def believable(self) -> bool:
        return self.verdict in (OK, UNKNOWN)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "reason": self.reason,
            "low_share": round(self.low_share, 3),
            "silent_share": round(self.silent_share, 3),
            "mean_logprob": (round(self.mean_logprob, 3)
                             if self.mean_logprob is not None else None),
            "scored": self.scored,
            "total": self.total,
            "seconds": round(self.seconds, 1),
            "worst": list(self.worst),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> Assessment:
        d = d or {}
        return cls(
            verdict=str(d.get("verdict") or UNKNOWN),
            reason=str(d.get("reason") or ""),
            low_share=float(d.get("low_share") or 0.0),
            silent_share=float(d.get("silent_share") or 0.0),
            mean_logprob=(float(d["mean_logprob"])
                          if d.get("mean_logprob") is not None else None),
            scored=int(d.get("scored") or 0),
            total=int(d.get("total") or 0),
            seconds=float(d.get("seconds") or 0.0),
            worst=[str(w) for w in d.get("worst") or []],
        )

    def line(self) -> str:
        """One sentence for a person, or "" when there is nothing to say."""
        if self.verdict == UNRELIABLE:
            return (
                f"This transcript is probably not trustworthy: "
                f"{self.low_share:.0%} of it scored below the confidence floor. "
                "Speech recognition invents fluent sentences when it cannot hear "
                "speech, so read the audio before acting on any of this."
            )
        if self.verdict == SUSPECT:
            return (
                f"Parts of this transcript are shaky: {self.low_share:.0%} of it "
                "scored below the confidence floor. Check anything surprising "
                "against the recording."
            )
        return ""


def assess(segments: list[Segment], cfg) -> Assessment:
    """
    Judge a finished transcript by the model's own confidence in it.

    Weighted by duration rather than by segment count, because a hallucination
    over four minutes of music is one segment and one honest "mm-hm" is another,
    and counting them equally would let the long invention hide behind a crowd of
    short real ones.
    """
    total = len(segments)
    if not segments:
        return Assessment(verdict=UNKNOWN, reason="there is no transcript to judge")

    floor = float(cfg.get("asr.confidence.min_avg_logprob", DEFAULT_MIN_LOGPROB))
    no_speech_ceiling = float(
        cfg.get("asr.confidence.max_no_speech_prob", DEFAULT_MAX_NO_SPEECH))
    suspect_at = float(cfg.get("asr.confidence.suspect_share", DEFAULT_SUSPECT_SHARE))
    unreliable_at = float(cfg.get("asr.confidence.unreliable_share", DEFAULT_UNRELIABLE_SHARE))
    min_seconds = float(cfg.get("asr.confidence.min_seconds", DEFAULT_MIN_SECONDS))

    scored_seconds = low_seconds = silent_seconds = 0.0
    weighted_logprob = 0.0
    scored = 0
    worst: list[tuple[float, str]] = []

    for seg in segments:
        duration = max(seg.duration, 0.01)
        if seg.confidence is None:
            continue
        scored += 1
        scored_seconds += duration
        weighted_logprob += seg.confidence * duration

        shaky = seg.confidence < floor
        silent = seg.no_speech is not None and seg.no_speech > no_speech_ceiling
        if silent:
            silent_seconds += duration
        if shaky or silent:
            low_seconds += duration
            worst.append((seg.confidence, f"[{seg.stamp()}] {seg.text[:80]}"))

    if not scored or scored_seconds <= 0:
        # Imported text has no confidence to report, and saying "unknown" is the
        # honest answer rather than implying the transcript was checked.
        return Assessment(
            verdict=UNKNOWN, total=total,
            reason="this transcript carries no confidence scores (imported text, or a "
                   "provider that does not report them)",
        )

    result = Assessment(
        low_share=low_seconds / scored_seconds,
        silent_share=silent_seconds / scored_seconds,
        mean_logprob=weighted_logprob / scored_seconds,
        scored=scored,
        total=total,
        seconds=scored_seconds,
        worst=[text for _score, text in sorted(worst)[:5]],
    )

    if scored_seconds < min_seconds:
        result.verdict = UNKNOWN
        result.reason = (
            f"only {scored_seconds:.0f}s of scored speech, which is too little to "
            f"judge (asr.confidence.min_seconds is {min_seconds:.0f})"
        )
        return result

    if result.low_share >= unreliable_at:
        result.verdict = UNRELIABLE
    elif result.low_share >= suspect_at:
        result.verdict = SUSPECT
    else:
        result.verdict = OK
    result.reason = (
        f"{result.low_share:.0%} of {scored_seconds:.0f}s scored below "
        f"{floor} avg_logprob or above {no_speech_ceiling} no-speech probability"
    )

    if result.verdict != OK:
        log.warning("transcript confidence %s: %s", result.verdict, result.reason)
    return result


def prompt_warning(assessment: Assessment) -> str:
    """
    What the extraction prompt is told, or "" when there is nothing to say.

    Phrased as an instruction rather than as context, because a model handed
    "this transcript may be unreliable" as background will note the caveat and
    then extract confidently from it anyway.
    """
    if assessment.verdict == UNRELIABLE:
        return (
            "TRANSCRIPT RELIABILITY: this transcript scored badly for confidence and "
            "may contain text the speech recogniser invented rather than heard. "
            "Extract only what is unambiguous. Prefer empty fields. Do not repair "
            "text that reads as garbled -- leaving it out is correct."
        )
    if assessment.verdict == SUSPECT:
        return (
            "TRANSCRIPT RELIABILITY: parts of this transcript scored badly for "
            "confidence. Where a passage reads as garbled or out of context, leave it "
            "out rather than interpreting it."
        )
    return ""
