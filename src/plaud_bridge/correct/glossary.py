"""
Deterministic post-ASR correction.

Whisper mangles insurance vocabulary in predictable ways. "Elimination period"
becomes "elimination. Period." IUL becomes "I, you, well." Rather than hoping a
bigger model fixes it, we fix it with a lookup table you own and grow.

Deterministic on purpose. An LLM correction pass would be more flexible and
would also occasionally rewrite something a client actually said, which in a
records context is unacceptable. Every change here is auditable and reversible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..logging_setup import get
from ..models import Segment

log = get("correct")


@dataclass
class CorrectionReport:
    applied: dict[str, int] = field(default_factory=dict)
    segments_touched: int = 0

    @property
    def total(self) -> int:
        return sum(self.applied.values())

    def summary(self) -> str:
        """
        Counts only -- deliberately no matched phrases.

        A rule is only counted when it actually fired, so the source term of
        every applied rule is a phrase that was really spoken in the recording.
        This string is written to the plaintext log and to the audit trail in
        the plaintext index, neither of which is meant to carry content. Redaction
        logs its pattern NAMES ("phone=2"), never the matched text, and this
        holds the same line. The per-rule breakdown is on `applied` for anyone
        who wants it in memory.
        """
        if not self.applied:
            return "no corrections applied"
        return f"{self.total} corrections across {self.segments_touched} segments"


def _compile(corrections: dict[str, str]) -> list[tuple[re.Pattern[str], str, str]]:
    compiled: list[tuple[re.Pattern[str], str, str]] = []
    # Longest source first so "attending physicians statement" wins over
    # any shorter overlapping rule.
    for src, dest in sorted(corrections.items(), key=lambda kv: -len(kv[0])):
        escaped = re.escape(src).replace(r"\ ", r"[\s\-,\.]+")
        # The optional trailing group cleans up the other half of the artifact
        # this module exists for. Whisper writes "The elimination. Period. is
        # ninety days"; fixing only the interior stop leaves "The elimination
        # period. is ninety days". The group is deliberately narrow -- it only
        # matches a full stop followed by a lowercase word, which is never a real
        # sentence ending, so genuine punctuation is untouched.
        pattern = re.compile(
            rf"(?<![\w]){escaped}(?![\w])(?P<trail>\.(?=\s+[a-z]))?",
            re.IGNORECASE,
        )
        compiled.append((pattern, dest, src))
    return compiled


def _match_case(original: str, replacement: str) -> str:
    """Keep sentence-initial capitalisation without shouting acronyms down."""
    if replacement.isupper() or any(c.isupper() for c in replacement[1:]):
        return replacement          # IUL, MEC, APS stay as authored
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def apply_corrections(segments: list[Segment], glossary) -> tuple[list[Segment], CorrectionReport]:
    report = CorrectionReport()
    if not glossary or not glossary.corrections:
        return segments, report

    rules = _compile(glossary.corrections)

    for seg in segments:
        before = seg.text
        text = before
        for pattern, dest, src in rules:
            def _sub(m: re.Match[str], dest=dest, src=src) -> str:
                matched = m.group(0)
                trail = m.groupdict().get("trail") or ""
                core = matched[: len(matched) - len(trail)] if trail else matched
                replacement = _match_case(core, dest)
                # Only count a rule that changed something. The glossary ships
                # identity entries that exist to protect a term from a broader
                # rule, and counting those produced audit lines reading
                # "3 corrections across 0 segments".
                if replacement != matched:
                    report.applied[src] = report.applied.get(src, 0) + 1
                return replacement

            text = pattern.sub(_sub, text)
        if text != before:
            seg.text = text
            report.segments_touched += 1

    if report.total:
        log.info("glossary: %s", report.summary())
    return segments, report
