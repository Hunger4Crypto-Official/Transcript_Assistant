"""
Consent detection.

Scans the opening window of a transcript for an announcement that the
conversation is being recorded and that the other party agreed.

Read this carefully, because the tool cannot do this part for you:

This is a detector, not a lawyer. It tells you whether you SAID the words. It
cannot tell you whether the consent was legally sufficient in the jurisdiction
where the other party was sitting. Nevada and Florida are both in your licence
footprint and both generally require all parties to consent for telephone
communications. Texas and Arizona are generally one-party. Statutes change and
get reinterpreted.

The operational rule that makes all of this moot: announce every time, get a
verbal yes on tape, every call, every state. It costs eight seconds and removes
the entire category of risk. This detector exists to catch the times you forgot.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..logging_setup import get
from ..models import Transcript

log = get("consent")

# Phrases where the producer announces recording.
_ANNOUNCE = [
    r"\brecord(?:ing|ings|ed|s)?\s+(?:this|these|those|our|the|all|every)\b",
    r"\bi'?(?:m|\s+am)?\s+record(?:ing)?\b",
    r"\bi\s+record\b",
    r"\bthis\s+(?:call|meeting|conversation)\s+is\s+being\s+recorded\b",
    r"\bbeing\s+recorded\b",
    r"\bon\s+the\s+record\b",
    r"\b(?:do\s+you\s+mind|is\s+it\s+ok(?:ay)?|any\s+objection)\b[^.?!]{0,40}\brecord",
    r"\bmind\s+if\s+i\s+record\b",
    r"\bpermission\s+to\s+record\b",
    r"\btaking\s+(?:a\s+)?record(?:ing)?\b",
    r"\bwith\s+your\s+permission\b[^.?!]{0,40}\brecord",
]

# Phrases where the other party agrees. Deliberately narrow: a bare "yeah"
# somewhere in the first ninety seconds is not consent.
_AGREE = [
    r"\b(?:yes|yeah|yep|sure|of course|absolutely|no problem|that'?s fine|go ahead|fine by me|no objection)\b",
    r"\bi\s+consent\b",
    r"\bi\s+agree\b",
    r"\bthat'?s\s+ok(?:ay)?\b",
]

# Phrases where somebody objects. These veto everything else in the window.
# Without them the announcement patterns match the objection itself -- "I really
# don't want this being recorded" contains "being recorded" -- and a sympathetic
# "yeah, of course" from the next speaker completes a consent that was in fact
# a refusal. That is the worst failure this module can have, so it is checked
# first and it wins.
_REFUSE = [
    r"\b(?:don'?t|do\s+not|would\s+rather\s+not|d?on'?t)\b[^.?!]{0,40}\brecord",
    r"\brecord[^.?!]{0,40}\b(?:without\s+my\s+consent|is\s+not\s+ok(?:ay)?)\b",
    r"\bnot\s+(?:ok(?:ay)?|comfortable|happy)\b[^.?!]{0,40}\brecord",
    r"\bstop\s+recording\b",
    r"\bturn\s+(?:that|it|the\s+recorder|the\s+recording)\s+off\b",
    r"\b(?:please|no,?)\s+don'?t\s+record",
    r"\bi\s+(?:object|refuse)\b",
    r"\bi'?d\s+prefer\s+(?:you\s+)?(?:did\s*n'?t|not)\b[^.?!]{0,40}\brecord",
    r"\brather\s+you\s+did\s*n'?t\s+record",
]

_ANNOUNCE_RE = [re.compile(p, re.IGNORECASE) for p in _ANNOUNCE]
_AGREE_RE = [re.compile(p, re.IGNORECASE) for p in _AGREE]
_REFUSE_RE = [re.compile(p, re.IGNORECASE) for p in _REFUSE]


@dataclass
class ConsentResult:
    announced: bool = False
    agreed: bool = False
    refused: bool = False
    announce_quote: str = ""
    agree_quote: str = ""
    refusal_quote: str = ""
    timestamp: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        """Your announcement, an affirmative response, and nobody objecting."""
        return self.announced and self.agreed and not self.refused


def _is_owner(speaker: str, owner_label: str) -> bool:
    """
    Whether a speaker label is the owner's, by exact match.

    Exact, not substring. Speaker labels are not trustworthy identity: an
    imported VTT or SRT names its own speakers, and even diarization's labels
    are only as good as the clustering. The old test was `owner_label in
    speaker`, which let "Not Sasson", "Sasson's assistant", or a spoofed
    "Sassonx" pass as the owner -- and the one thing this decides is whose "I'm
    recording this" counts as YOU announcing it. Diarization labels the owner's
    own segments with exactly `owner_label`, so nothing legitimate is lost by
    refusing the lookalikes.
    """
    a, b = speaker.strip().lower(), owner_label.strip().lower()
    return bool(a) and bool(b) and a == b


def detect_consent(transcript: Transcript, window_seconds: float = 90.0,
                   owner_label: str | None = None) -> ConsentResult:
    result = ConsentResult()
    window = [s for s in transcript.segments if s.start <= window_seconds]
    if not window:
        result.notes.append("transcript has no content in the consent window")
        return result

    # An objection anywhere in the window ends the question. Nothing said
    # afterwards converts a refusal into consent.
    for seg in window:
        if any(rx.search(seg.text) for rx in _REFUSE_RE):
            result.refused = True
            result.refusal_quote = seg.text.strip()[:300]
            result.notes.append(
                "someone objected to being recorded in the opening window: "
                f'"{result.refusal_quote}". Consent cannot be inferred from a '
                "refusal, and the answer to a refusal is not a config change."
            )
            log.warning("refusal detected in the consent window")
            return result

    # Whose announcement counts. COMPLIANCE section 2 requires that YOU announce
    # the recording; the other party stating that they are recording on their end
    # is not you obtaining their consent. Enforced only when the owner can
    # actually be identified among the speakers, because with diarization off
    # every segment carries the same placeholder label and no speaker-based rule
    # can mean anything.
    speakers = {s.speaker for s in window}
    owner_identified = bool(owner_label) and any(_is_owner(s, owner_label) for s in speakers)

    announce_idx: int | None = None
    announced_by_other = ""
    for idx, seg in enumerate(window):
        if not any(rx.search(seg.text) for rx in _ANNOUNCE_RE):
            continue
        if owner_identified and not _is_owner(seg.speaker, owner_label or ""):
            announced_by_other = announced_by_other or seg.text.strip()[:300]
            continue
        result.announced = True
        result.announce_quote = seg.text.strip()[:300]
        result.timestamp = seg.start
        announce_idx = idx
        break

    if announce_idx is None:
        if announced_by_other:
            result.notes.append(
                f'the recording was announced by someone other than {owner_label}: '
                f'"{announced_by_other}". That is them telling you they are '
                "recording, not you obtaining their consent."
            )
        else:
            result.notes.append("no recording announcement found in the opening window")
        return result

    if not owner_identified and owner_label:
        result.notes.append(
            f"could not confirm the announcement came from {owner_label}; no speaker "
            "in the opening window carries that label. Enable diarization or check "
            "diarization.owner_label if you want this verified."
        )

    announcer = window[announce_idx].speaker
    # Look only AFTER the announcement, and only at a different speaker.
    for seg in window[announce_idx + 1 : announce_idx + 9]:
        if seg.speaker == announcer:
            continue
        if any(rx.search(seg.text) for rx in _AGREE_RE):
            result.agreed = True
            result.agree_quote = seg.text.strip()[:300]
            break

    if not result.agreed:
        if len({s.speaker for s in window}) <= 1:
            result.notes.append(
                "only one speaker detected; cannot confirm the other party agreed"
            )
        else:
            result.notes.append(
                "announcement found but no affirmative response from another speaker"
            )

    return result
