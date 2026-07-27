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


def extract(transcript: Transcript, profile: Profile, cfg,
            transcript_text: str | None = None,
            local_only: bool | None = None) -> ProfileAnalysis:
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

    user = (
        f"SCHEMA:\n{_schema_block(profile)}\n\n"
        f"TRANSCRIPT:\n{body}\n\n"
        "Return the JSON object now."
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
    )
