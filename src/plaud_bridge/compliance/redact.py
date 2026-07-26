"""
PII redaction before anything leaves the machine.

Applied to the text handed to an LLM, never to the stored transcript. The
stored transcript is your record; redacting it would destroy the thing you are
trying to keep. The redacted copy is a travel document, not the original.

Regex redaction is a floor, not a ceiling. It will catch a spoken SSN in
standard form and miss one spoken as "five five five, twelve, thirty four
sixty seven". Treat it as defence in depth behind the real control, which is
not sending sensitive profiles to a cloud provider at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..logging_setup import get

log = get("redact")

# Spoken digit sequences that regex over transcribed words will otherwise miss.
_SPOKEN_DIGITS = re.compile(
    r"\b(?:(?:zero|oh|one|two|three|four|five|six|seven|eight|nine)[\s,-]+){8,}"
    r"(?:zero|oh|one|two|three|four|five|six|seven|eight|nine)\b",
    re.IGNORECASE,
)


@dataclass
class RedactionReport:
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def summary(self) -> str:
        if not self.counts:
            return "no redactions"
        return ", ".join(f"{k}={v}" for k, v in sorted(self.counts.items()))


def redact_text(text: str, patterns: dict[str, str],
                enabled: bool = True) -> tuple[str, RedactionReport]:
    report = RedactionReport()
    if not enabled or not text:
        return text, report

    out = text
    for name, pattern in (patterns or {}).items():
        try:
            rx = re.compile(pattern)
        except re.error as exc:
            log.warning("skipping invalid redaction pattern '%s': %s", name, exc)
            continue
        out, count = rx.subn(f"[{name.upper()}_REDACTED]", out)
        if count:
            report.counts[name] = report.counts.get(name, 0) + count

    out, spoken = _SPOKEN_DIGITS.subn("[SPOKEN_DIGITS_REDACTED]", out)
    if spoken:
        report.counts["spoken_digits"] = spoken

    if report.total:
        log.info("redacted before LLM: %s", report.summary())
    return out, report
