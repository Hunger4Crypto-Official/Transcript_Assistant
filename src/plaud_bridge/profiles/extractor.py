"""
Schema-driven extraction.

The schema comes from the profile YAML, so adding a field to your Insurance
Agent output is a config edit, not a code change. The system prompt comes from
the profile too, including the hard constraints on the family profiles that
forbid psychological assessment, blame allocation, and relationship scoring.

Those constraints are not decoration. Read father.yaml and husband.yaml before
changing them.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..config import Profile
from ..llm import complete_json
from ..llm.base import LLMError
from ..logging_setup import get
from ..models import ProfileAnalysis, Transcript

log = get("extractor")

MAX_EXTRACTION_CHARS = 90000


class ExtractorError(RuntimeError):
    pass


def _max_tokens(cfg) -> int:
    """
    Output ceiling for an extraction call.

    Reads the first configured provider's setting as a stand-in for the chain,
    because the provider that ends up serving the call is not known until the
    registry has tried them. A missing or empty `llm.providers` returns the
    default rather than raising a TypeError three frames down.
    """
    providers = cfg.get("llm.providers") or []
    if not providers:
        return 8000
    return int(cfg.get(f"llm.{providers[0]}.max_tokens", 8000))


def _schema_block(profile: Profile) -> str:
    lines = []
    for spec in profile.fields:
        marker = ""
        if spec.priority == "critical":
            marker = "  [CRITICAL]"
        elif spec.priority == "high":
            marker = "  [HIGH PRIORITY]"
        sensitive = "  [SENSITIVE]" if spec.sensitive else ""
        lines.append(
            f'  "{spec.key}": {spec.type}{marker}{sensitive}\n'
            f"      // {spec.description}"
        )
    return "{\n" + "\n".join(lines) + "\n}"


def _type_default(type_name: str) -> Any:
    t = type_name.lower()
    if t.startswith("list"):
        return []
    if t == "boolean":
        return False
    if t in ("int", "integer"):
        return 0
    if t in ("float", "number"):
        return 0.0
    return ""


def _coerce(value: Any, type_name: str) -> Any:
    """Bring whatever the model returned into the shape the schema promised."""
    t = type_name.lower()
    if value is None:
        return _type_default(type_name)
    if t.startswith("list"):
        if isinstance(value, list):
            return value
        if isinstance(value, (str, dict)):
            return [value] if value else []
        return []
    if t == "boolean":
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("true", "yes", "1")
    if t in ("int", "integer"):
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0
    if t in ("float", "number"):
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


_WORD = re.compile(r"[^a-z0-9]+")


def _flatten(text: str) -> str:
    """
    Lowercase, punctuation-free, single-spaced, and boundary-padded.

    Comparison happens on this form so that a model returning smart quotes,
    different capitalisation, or an extra line break is not accused of making
    the quote up. What it does not forgive is different words.

    The leading and trailing spaces are load-bearing: they let `quote_is_present`
    require whole-word matches. Strip them off a needle and "pay" matches inside
    "payment" -- which is how a fabricated fragment slipped through the first
    version of the quote check.
    """
    return f" {_WORD.sub(' ', text.lower()).strip()} "


def quote_is_present(quote: str, haystack_flat: str) -> bool:
    """
    Whether a quote appears verbatim (up to case and punctuation) in the text.

    `haystack_flat` must already be `_flatten`ed. The needle is flattened here
    but NOT stripped, so its boundary spaces stay attached and the match is on
    whole words rather than substrings. A quote of only punctuation flattens to
    a bare " " and is treated as absent rather than matching everywhere.

    This is the one check `ask` and the extractor must agree on -- both decide
    whether the model invented a quote -- so they share it rather than keeping
    two subtly different copies.
    """
    needle = _flatten(quote)
    if not needle.strip():
        return False
    return needle in haystack_flat


def _quote_texts(value: Any) -> list[str]:
    """Every quoted string inside a coerced quote field."""
    items = value if isinstance(value, list) else [value]
    out = []
    for item in items:
        if isinstance(item, dict):
            body = str(item.get("text") or "")
        else:
            body = str(item or "")
        if body.strip():
            out.append(body)
    return out


def _verify_quotes(fields: dict[str, Any], profile: Profile,
                   body: str) -> tuple[dict[str, Any], list[str]]:
    """
    Drop any quote the transcript does not actually contain.

    The schema calls these fields `quote` and the prompt demands the speaker's
    exact words, so a passage that is not present verbatim is not a quote --
    it is the model's paraphrase wearing quotation marks and a timestamp.

    This matters more here than anywhere else in the pipeline. `ask` validates
    its citations because a fabricated one is believed; an extracted quote is
    believed harder. It is attributed to a named person, it flows into the
    memory ledger as something they said, and it can surface a year later in a
    digest as a thing your kid told you. Nothing downstream re-checks it, and
    the audio it supposedly came from may be gone by then.

    Dropping rather than flagging is deliberate, and it is the same call
    ADR-024 made for citations: a quote nobody can find is worse than no quote,
    because the empty field reads as "nothing worth keeping was said" while the
    invented one reads as testimony.
    """
    haystack = _flatten(body)
    dropped: list[str] = []
    cleaned = dict(fields)

    for spec in profile.fields:
        if "quote" not in spec.type.lower():
            continue
        value = cleaned.get(spec.key)
        if not value:
            continue

        items = value if isinstance(value, list) else [value]
        kept = []
        for item in items:
            # An item carrying no quoted text has nothing to check; it is the
            # ones claiming somebody said something that have to earn it.
            missing = [t for t in _quote_texts(item) if not quote_is_present(t, haystack)]
            if missing:
                dropped.extend(missing)
            else:
                kept.append(item)
        cleaned[spec.key] = kept if isinstance(value, list) else (kept[0] if kept else "")

    return cleaned, dropped


def extract(transcript: Transcript, profile: Profile, cfg,
            transcript_text: str | None = None,
            local_only: bool | None = None,
            prior: str = "",
            warning: str = "") -> ProfileAnalysis:
    """
    Run one profile's extraction schema over a transcript.

    `transcript_text` lets the caller pass a redacted rendering. When compliance
    requires redaction, the caller passes redacted text and the raw transcript
    never reaches the model.
    """
    body = transcript_text if transcript_text is not None else transcript.labelled_text()
    if len(body) > MAX_EXTRACTION_CHARS:
        body = body[:MAX_EXTRACTION_CHARS] + "\n\n[... transcript truncated ...]"

    if not body.strip():
        return ProfileAnalysis(
            profile_id=profile.id,
            fields={spec.key: _type_default(spec.type) for spec in profile.fields},
            error="empty transcript",
        )

    # Prompt layering, outermost first: house style sets the register for every
    # profile, persona narrows it for this one, and the profile's own prompt --
    # which carries the hard constraints on the family profiles -- comes last so
    # it is nearest the task and cannot be softened by anything above it.
    house_style = str(getattr(cfg, "voice", None) and cfg.voice.get("analysis.house_style") or "")
    preamble = "\n\n".join(p for p in (house_style, profile.persona) if p.strip())

    system = (
        (f"{preamble}\n\n" if preamble else "")
        + f"{profile.system_prompt}\n\n"
        "OUTPUT CONTRACT:\n"
        "- Respond with a single JSON object and nothing else. No preamble, no "
        "code fences, no trailing commentary.\n"
        "- Include every key from the schema, even when the value is empty.\n"
        "- For quote fields, return objects shaped "
        '{"timestamp": "MM:SS", "speaker": "...", "text": "..."} using the '
        "speaker's exact words.\n"
        "- Never invent content that is not in the transcript. An empty list is "
        "always a better answer than a plausible fabrication."
    )

    # The carry-forward brief sits between the schema and the transcript, never
    # inside it. It arrives already redacted and carries its own line saying it
    # is background rather than something that was said, because the one thing
    # this must not do is let last month's conversation be quoted as though it
    # happened today.
    user = (
        f"SCHEMA:\n{_schema_block(profile)}\n\n"
        + (f"{prior.strip()}\n\n" if prior.strip() else "")
        + f"TRANSCRIPT:\n{body}\n\n"
        # A reliability caveat placed before the transcript reads as background
        # and gets noted and then ignored. Last, next to the instruction, it is
        # part of the task.
        + (f"{warning.strip()}\n\n" if warning.strip() else "")
        + "Return the JSON object now."
    )

    # This profile's own policy is a floor, never a ceiling. The caller knows
    # which profile governs the whole recording and passes the stricter answer;
    # falling back to the profile alone would let a cloud-permitting profile
    # decide the fate of a transcript that a locked profile also matched.
    own_policy = profile.hard_local_only or not profile.allow_cloud_llm
    local_only = own_policy if local_only is None else (local_only or own_policy)

    try:
        data, response = complete_json(
            cfg, system, user, local_only=local_only,
            max_tokens=_max_tokens(cfg),
        )
    except LLMError as exc:
        log.error("extraction failed for profile %s: %s", profile.id, exc)
        return ProfileAnalysis(
            profile_id=profile.id,
            fields={spec.key: _type_default(spec.type) for spec in profile.fields},
            error=str(exc),
        )

    fields: dict[str, Any] = {}
    for spec in profile.fields:
        fields[spec.key] = _coerce(data.get(spec.key), spec.type)

    needs_attention = bool(fields.get("requires_human_attention", False))
    if needs_attention:
        # The family profiles instruct the model to stop and flag rather than
        # summarise conflict or anything concerning. Honour that by discarding
        # whatever else came back.
        log.warning("profile %s flagged this recording for human attention", profile.id)
        fields = {
            spec.key: (True if spec.key == "requires_human_attention" else _type_default(spec.type))
            for spec in profile.fields
        }

    # Every quote has to be findable in the text the model was actually shown.
    dropped: list[str] = []
    if not needs_attention and cfg.get("llm.verify_quotes", True):
        fields, dropped = _verify_quotes(fields, profile, body)
        if dropped:
            log.warning(
                "profile %s: dropped %d quote(s) that are not in the transcript: %s",
                profile.id, len(dropped), "; ".join(q[:60] for q in dropped[:3]),
            )

    unexpected = set(data) - set(profile.field_keys)
    if unexpected:
        log.debug("profile %s: ignoring unexpected keys %s", profile.id, sorted(unexpected))

    return ProfileAnalysis(
        profile_id=profile.id,
        fields=fields,
        llm_provider=response.provider,
        llm_model=response.model,
        cost_usd=response.cost_usd,
        requires_human_attention=needs_attention,
        unverified_quotes=len(dropped),
    )
