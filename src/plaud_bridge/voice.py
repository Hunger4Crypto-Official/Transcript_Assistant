"""
Voice: every user-facing string, in config.

The digest is the thing you actually read, and how it reads matters. Its
wording used to be hardcoded prose scattered through the renderer, which meant
changing "Needs You" to something that sounded like you was a source edit.

Two deliberate limits on how far this goes.

**This is not a template language.** The structure of the digest stays in code
because that is where the compliance rules live: suppressed fields, personal
profiles omitted from combined exports, encrypted analyses opened on demand. A
template that could reorder or re-emit sections could also defeat those, and a
digest is the document most likely to get forwarded. So config controls every
word and no control flow.

**Nothing can fail to render.** A voice pack is a set of overrides on top of the
defaults below, so a partial pack is fine and a missing key is impossible. A
placeholder that does not exist renders empty rather than raising: a typo in a
voice file should make the digest look slightly wrong, not lose you a week of
recordings at the last step.

Packs live in `config/voice/*.yaml`. Pick one with `voice.preset` in
pipeline.yaml and override individual keys with `voice.overrides`.
"""

from __future__ import annotations

import string
from pathlib import Path
from typing import Any

import yaml

from .logging_setup import get

log = get("voice")


# The full set of strings, and the fallback for every key. A pack overrides
# what it wants to change and inherits the rest, which is why a three-line
# voice file is a legitimate voice file.
DEFAULTS: dict[str, Any] = {
    "id": "plain",
    "name": "Plain",
    "description": "Neutral and factual. Says what happened and gets out of the way.",

    "digest": {
        "title": "Digest",
        "profile_title": "{profile_name}",
        "window": "**Window:** {start} to {end} ({days} days)  ",
        "generated": "**Generated:** {generated} UTC",
        "empty": "No recordings in this window.",
        "opening": "",

        "needs_you": {
            "heading": "Needs You",
            "intro": "",
            "why_flagged": "flagged for human review",
            "why_error": "error: {error}",
            "attention_line": "- **{section}** :: `{name}` {why}",
            "action_line": "- **{section}** :: {action}  \n  <sub>{name}</sub>",
            "empty": "",
        },

        "glance": {
            "heading": "At a Glance",
            "columns": "| Section | Recordings | Minutes |",
            "row": "| {section} | {count} | {minutes} |",
        },

        "section": {
            "intro": "",
            "suppressed_note": "> {note}",
        },

        "entry": {
            "meta_separator": "` · `",
            "attention_note": (
                "> This recording was flagged for human attention and was not "
                "summarised. Read it yourself."
            ),
            "error_note": "> Analysis error: {error}",
            "nothing_extracted": "_Nothing extracted for the highlighted fields._",
            "next_action": "**Next:** {action}",
            "withheld": "**{label}**: {count} item(s), withheld from this view.",
            "files": "<sub>Files: {links}</sub>",
            "encrypted": "<sub>Encrypted: {kinds} (open with: run.py open {id})</sub>",
        },

        "footer": {
            "costs": (
                "<sub>{minutes} minutes processed · ${cost} in API spend across "
                "this window.</sub>"
            ),
            "personal_hidden": (
                "<sub>Personal profiles ({names}) are omitted from the combined "
                "view. Use `--profile {example}` or `--include-personal` to see "
                "them.</sub>"
            ),
            "sign_off": "",
        },
    },

    # Prepended to every profile's extraction system prompt. The profile's own
    # prompt still follows and still wins on anything specific; this sets the
    # house style so five profiles do not drift into five different registers.
    "analysis": {
        "house_style": (
            "Write in plain, concrete language. Prefer the speaker's own words "
            "to a paraphrase. Do not editorialise, congratulate, or soften. "
            "An empty field is always better than a plausible invention."
        ),
    },
}


class _Blanks(dict):
    """Missing placeholders render empty instead of raising mid-digest."""

    def __missing__(self, key: str) -> str:
        log.debug("voice: no value for placeholder '%s'", key)
        return ""


def _merge(base: Any, over: Any) -> Any:
    """Deep merge. Dicts combine key by key; anything else is replaced."""
    if isinstance(base, dict) and isinstance(over, dict):
        out = dict(base)
        for key, value in over.items():
            out[key] = _merge(out.get(key), value) if key in out else value
        return out
    return over


class Voice:
    def __init__(self, data: dict[str, Any] | None = None):
        self._d = _merge(DEFAULTS, data or {})

    # ---- access ---------------------------------------------------------
    @property
    def id(self) -> str:
        return str(self._d.get("id", "plain"))

    @property
    def name(self) -> str:
        return str(self._d.get("name", "Plain"))

    def get(self, dotted: str, default: Any = "") -> Any:
        node: Any = self._d
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def text(self, dotted: str, **values: Any) -> str:
        """Render one string. Never raises; a broken template returns itself."""
        raw = self.get(dotted, "")
        if not isinstance(raw, str) or not raw:
            return ""
        try:
            return string.Formatter().vformat(raw, (), _Blanks(values))
        except (ValueError, IndexError, KeyError, TypeError) as exc:
            log.warning("voice: could not render '%s' (%s); using it verbatim", dotted, exc)
            return raw

    def lines(self, dotted: str, **values: Any) -> list[str]:
        """A rendered string as digest lines, or nothing when it is empty."""
        rendered = self.text(dotted, **values).strip("\n")
        return [rendered, ""] if rendered.strip() else []

    # ---- construction ---------------------------------------------------
    @classmethod
    def load(cls, voice_dir: Path, preset: str = "plain",
             overrides: dict[str, Any] | None = None) -> Voice:
        data: dict[str, Any] = {}
        path = Path(voice_dir) / f"{preset}.yaml"

        if path.exists():
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError as exc:
                log.warning("voice: %s is not valid YAML (%s); using defaults", path.name, exc)
                data = {}
        elif preset != "plain":
            # Not fatal. A missing voice pack should change how the digest
            # sounds, not stop it being written.
            log.warning(
                "voice: preset '%s' not found in %s; falling back to the built-in "
                "plain voice", preset, voice_dir,
            )

        return cls(_merge(data, overrides or {}))

    @classmethod
    def available(cls, voice_dir: Path) -> list[tuple[str, str, str]]:
        """(id, name, description) for every pack on disk, for `run.py voices`."""
        found: list[tuple[str, str, str]] = []
        directory = Path(voice_dir)
        if not directory.is_dir():
            return found
        for path in sorted(directory.glob("*.yaml")):
            try:
                raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError:
                continue
            found.append((
                str(raw.get("id", path.stem)),
                str(raw.get("name", path.stem.title())),
                str(raw.get("description", "")).strip(),
            ))
        return found
