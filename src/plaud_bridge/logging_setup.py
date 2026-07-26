"""
Logging.

One rule that matters: transcript content never reaches a log file. Logs hold
identifiers, hashes, counts, and durations. If you ship a log to someone for
debugging, nothing private goes with it.
"""

from __future__ import annotations

import logging
import logging.handlers
import re
from pathlib import Path

_SECRET_PATTERNS = [
    re.compile(r"(sk-[A-Za-z0-9_\-]{8,})"),
    re.compile(r"(gsk_[A-Za-z0-9_\-]{8,})"),
    re.compile(r"(hf_[A-Za-z0-9_\-]{8,})"),
    re.compile(r"(Bearer\s+[A-Za-z0-9._\-]{8,})"),
]


class RedactingFilter(logging.Filter):
    """
    Strips API keys and personal identifiers from everything on its way out.

    Two things this has to get right that are easy to miss:

    **Tracebacks.** The formatter appends the exception text independently of
    the message, so scrubbing `record.msg` alone leaves the interesting part
    untouched. Provider errors fold the raw API response body and the full URL
    into the exception, so the traceback is where the secrets actually are.

    **Doing something by default.** An earlier version keyed redaction off an
    `extra={"content": True}` marker that no caller ever passed, which made
    `logging.redact_content` a switch wired to nothing. Content redaction now
    applies the same patterns the compliance gate uses, unconditionally, so the
    setting means what its name says.
    """

    def __init__(self, redact_content: bool = True, patterns: dict[str, str] | None = None):
        super().__init__()
        self.redact_content = redact_content
        self._content_res: list[tuple[str, re.Pattern[str]]] = []
        if redact_content:
            for name, pattern in (patterns or {}).items():
                try:
                    self._content_res.append((name, re.compile(str(pattern))))
                except re.error:
                    # config._validate reports bad patterns properly. Logging is
                    # not the place to fail a run over one.
                    continue

    def _scrub(self, text: str) -> str:
        for pat in _SECRET_PATTERNS:
            text = pat.sub("[redacted-key]", text)
        for name, pat in self._content_res:
            text = pat.sub(f"[redacted-{name}]", text)
        return text

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True

        if self.redact_content and getattr(record, "content", False):
            msg = "[content withheld from logs]"
        else:
            msg = self._scrub(msg)
        record.msg = msg
        record.args = ()

        # Pre-format the traceback so it can be scrubbed, then clear exc_info so
        # the formatter uses our cleaned copy instead of re-rendering the raw one.
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = logging.Formatter().formatException(record.exc_info)
            record.exc_info = None
        if record.exc_text:
            record.exc_text = self._scrub(record.exc_text)
        if record.stack_info:
            record.stack_info = self._scrub(record.stack_info)

        return True


def setup(log_dir: Path, level: str = "INFO", redact_content: bool = True,
          rotate_mb: int = 20, backups: int = 5,
          redact_patterns: dict[str, str] | None = None) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("plaud_bridge")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()
    root.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)-28s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    redactor = RedactingFilter(redact_content, redact_patterns)

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    console.addFilter(redactor)
    root.addHandler(console)

    fileh = logging.handlers.RotatingFileHandler(
        log_dir / "bridge.log",
        maxBytes=rotate_mb * 1024 * 1024,
        backupCount=backups,
        encoding="utf-8",
    )
    fileh.setFormatter(fmt)
    fileh.addFilter(redactor)
    root.addHandler(fileh)

    return root


def get(name: str) -> logging.Logger:
    return logging.getLogger(f"plaud_bridge.{name}")
