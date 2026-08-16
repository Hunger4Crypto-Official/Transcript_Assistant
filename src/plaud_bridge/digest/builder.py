"""
Digest rendering.

This is the thing you actually read. Two modes:

  combined  - every profile in one document, sectioned, work first
  filtered  - one profile only

Two guardrails you should know about before you share a digest with anyone:

1. Profiles marked `exclude_from_combined_export` (father, husband) are omitted
   from the combined digest unless you pass include_personal explicitly. The
   default assumption is that a combined digest might get forwarded, pasted
   into a message, or opened on a shared screen.

2. Fields marked `suppress_fields` in a profile never render into markdown at
   all. Client health and financial disclosures stay in the encrypted analysis
   file where they belong. The digest tells you they exist and how many; it
   does not print them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from ..logging_setup import get

log = get("digest")


@dataclass
class DigestOptions:
    profile_id: str | None = None
    days: int = 7
    include_personal: bool = False
    include_costs: bool = True
    include_links: bool = True
    max_items: int = 40
    title: str = ""


@dataclass
class DigestSection:
    profile_id: str
    heading: str
    priority: int
    entries: list[dict[str, Any]] = field(default_factory=list)
    suppressed_note: str = ""


_QUOTE_KEYS = ("text", "quote", "statement", "content")
_META_KEYS = ("timestamp", "time", "speaker", "who")


def _fmt_quote(item: Any) -> str:
    """
    Render one extracted item.

    Quote-shaped dicts get the "[00:12] Speaker: text" treatment. Everything
    else is rendered as key/value pairs rather than silently vanishing, because
    a model will happily return {"what": ..., "when": ...} for an object field
    and a blank bullet is worse than an ugly one.
    """
    if not isinstance(item, dict):
        return str(item).strip()

    ts = str(item.get("timestamp") or item.get("time") or "").strip()
    speaker = str(item.get("speaker") or item.get("who") or "").strip()
    body = ""
    for key in _QUOTE_KEYS:
        if item.get(key):
            body = str(item[key]).strip()
            break

    extras = [
        f"{k}: {v}"
        for k, v in item.items()
        if k not in _QUOTE_KEYS + _META_KEYS
        and v not in (None, "", [], {})
        and not isinstance(v, (dict, list))
    ]

    if body:
        prefix = " ".join(p for p in (f"[{ts}]" if ts else "", f"{speaker}:" if speaker else "") if p)
        line = f"{prefix} {body}".strip() if prefix else body
        return f"{line}  ({'; '.join(extras)})" if extras else line

    if extras:
        return "; ".join(extras)

    # Last resort: something nested we have no shape for. Show it rather than
    # dropping it, truncated so it cannot blow out the digest.
    import json as _json

    return _json.dumps(item, ensure_ascii=False)[:300]


def _fmt_value(value: Any, limit: int = 8) -> list[str]:
    if value is None or value == "" or value == []:
        return []
    if isinstance(value, bool):
        return ["yes"] if value else []
    if isinstance(value, list):
        return [_fmt_quote(v) for v in value[:limit] if str(v).strip()]
    return [_fmt_quote(value)]


class DigestBuilder:
    def __init__(self, cfg, db, vault=None):
        self.cfg = cfg
        self.db = db
        self._vault_override = vault
        self._vault_cache = None

    # ---- encrypted analyses ---------------------------------------------
    def _vault(self):
        if self._vault_override is not None:
            return self._vault_override
        if self._vault_cache is None:
            from ..storage import Vault

            self._vault_cache = Vault(self.cfg.path("vault"))
        return self._vault_cache

    def _open_withheld_analysis(self, recording_id: str, payload: dict, pid: str):
        """
        Fetch an analysis the index deliberately does not hold.

        High and maximum sensitivity recordings keep their extracted quotes in
        the vault rather than in the plain SQLite index, so rendering them here
        means decrypting. Failure is reported in the digest rather than shown as
        an empty section: "we could not open this" and "there was nothing in it"
        are very different statements to make to someone reading their week.
        """
        from pathlib import Path

        from ..storage import VaultError

        path = str(payload.get("artifact_paths", {}).get("analysis", ""))
        if not path.endswith(".enc") or not Path(path).exists():
            return None, "the encrypted analysis file is missing from disk"

        try:
            import json as _json

            full = _json.loads(self._vault().read_text(Path(path), recording_id))
        except VaultError as exc:
            return None, f"could not decrypt the analysis ({exc})"
        except (ValueError, OSError) as exc:
            return None, f"could not read the analysis ({exc})"

        found = next(
            (a for a in full.get("analyses", []) if a.get("profile_id") == pid), None
        )
        return found, "" if found else "the decrypted analysis had no entry for this profile"

    # ---- assembly -------------------------------------------------------
    def _collect(self, opts: DigestOptions) -> list[DigestSection]:
        order = self.cfg.get("digest.section_order", []) or sorted(self.cfg.profiles)
        wanted = [opts.profile_id] if opts.profile_id else order

        sections: list[DigestSection] = []
        for pid in wanted:
            if pid not in self.cfg.profiles:
                log.warning("digest: unknown profile '%s', skipping", pid)
                continue
            profile = self.cfg.profile(pid)

            # Personal profiles stay out of a combined digest by default.
            if (
                profile.exclude_from_combined_export
                and opts.profile_id is None
                and not opts.include_personal
            ):
                log.debug("digest: omitting personal profile '%s' from combined view", pid)
                continue

            rows = self.db.query(profile_id=pid, since_days=opts.days, limit=opts.max_items)
            if not rows:
                continue

            section = DigestSection(pid, profile.digest_heading, profile.digest_priority)
            if profile.suppress_fields:
                section.suppressed_note = (
                    "Suppressed from this view: "
                    + ", ".join(
                        (profile.field_by_key(f).label if profile.field_by_key(f) else f)
                        for f in profile.suppress_fields
                    )
                    + ". Held in the encrypted analysis file."
                )

            import json as _json

            for row in rows:
                payload = _json.loads(row["payload_json"])
                analysis = next(
                    (a for a in payload.get("analyses", []) if a.get("profile_id") == pid), None
                )
                if analysis is None:
                    continue

                unopened = ""
                if analysis.get("fields_withheld"):
                    restored, why = self._open_withheld_analysis(row["id"], payload, pid)
                    if restored is None:
                        unopened = why
                        log.warning("digest: %s for %s", why, row["id"])
                    else:
                        analysis = restored

                route = next(
                    (r for r in payload.get("routes", []) if r.get("profile_id") == pid), {}
                )
                section.entries.append(
                    {
                        "id": row["id"],
                        "name": row["source_name"],
                        "when": (row["recorded_at"] or row["ingested_at"] or "")[:16].replace("T", " "),
                        "minutes": round((row["duration_seconds"] or 0) / 60.0, 1),
                        "confidence": route.get("confidence", 0.0),
                        "consent": row["consent_status"],
                        "cost": row["total_cost_usd"] or 0.0,
                        "attention": bool(analysis.get("requires_human_attention")),
                        "fields": analysis.get("fields", {}),
                        "error": analysis.get("error") or unopened,
                        "artifacts": payload.get("artifact_paths", {}),
                    }
                )

            if section.entries:
                sections.append(section)

        sections.sort(key=lambda s: s.priority)
        return sections

    # ---- rendering ------------------------------------------------------
    def render_markdown(self, opts: DigestOptions,
                        sections: list[DigestSection] | None = None) -> str:
        # `sections` lets render_html reuse one _collect for both the text and
        # the charts, so the two cannot disagree about what the window held.
        # Left to default, behavior is exactly what it always was.
        if sections is None:
            sections = self._collect(opts)
        voice = self.cfg.voice
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=opts.days)

        if opts.title:
            title = opts.title
        elif opts.profile_id:
            profile = self.cfg.profile(opts.profile_id)
            title = voice.text(
                "digest.profile_title",
                profile_name=profile.name,
                profile_short=profile.short_name,
            )
        else:
            title = voice.text("digest.title")

        out: list[str] = [
            f"# {title}",
            "",
            voice.text("digest.window", start=f"{start:%Y-%m-%d}",
                       end=f"{now:%Y-%m-%d}", days=opts.days),
            voice.text("digest.generated", generated=f"{now:%Y-%m-%d %H:%M}"),
            "",
        ]
        out += voice.lines("digest.opening")

        if not sections:
            out += [self._empty_note(opts, voice), ""]
            return "\n".join(out)

        # --- what needs you first ----------------------------------------
        attention = [
            (s, e) for s in sections for e in s.entries if e["attention"] or e["error"]
        ]
        actions: list[tuple[str, str, str]] = []
        for section in sections:
            for entry in section.entries:
                na = entry["fields"].get("next_action")
                if isinstance(na, str) and na.strip() and na.strip().lower() not in ("none", "n/a"):
                    actions.append((section.heading, entry["name"], na.strip()))

        if attention or actions:
            out += [f"## {voice.text('digest.needs_you.heading')}", ""]
            out += voice.lines("digest.needs_you.intro")
            for section, entry in attention:
                why = (
                    voice.text("digest.needs_you.why_flagged")
                    if entry["attention"]
                    else voice.text("digest.needs_you.why_error", error=entry["error"][:120])
                )
                out.append(voice.text(
                    "digest.needs_you.attention_line",
                    section=section.heading, name=entry["name"], why=why,
                ))
            for heading, name, action in actions[:20]:
                out.append(voice.text(
                    "digest.needs_you.action_line",
                    section=heading, name=name, action=action,
                ))
            out.append("")

        # --- overview table ----------------------------------------------
        out += [
            f"## {voice.text('digest.glance.heading')}",
            "",
            voice.text("digest.glance.columns"),
            "|---|---:|---:|",
        ]
        for section in sections:
            mins = sum(e["minutes"] for e in section.entries)
            out.append(voice.text(
                "digest.glance.row",
                section=section.heading, count=len(section.entries), minutes=f"{mins:.0f}",
            ))
        out.append("")

        # --- per-section body ---------------------------------------------
        for section in sections:
            profile = self.cfg.profile(section.profile_id)
            out += [f"## {section.heading}", ""]
            # Per-profile intro wins over the voice pack's generic one, so a
            # section can explain itself in its own words.
            intro = profile.digest_intro or voice.text("digest.section.intro",
                                                       section=section.heading)
            if intro:
                out += [intro, ""]
            if section.suppressed_note:
                out += [voice.text("digest.section.suppressed_note",
                                   note=section.suppressed_note), ""]

            for entry in section.entries:
                header = f"### {entry['name']}"
                out += [header, ""]
                meta = [
                    f"{entry['when']}",
                    f"{entry['minutes']:.0f} min",
                    f"match {entry['confidence']:.2f}",
                ]
                if entry["consent"] and entry["consent"] != "not_required":
                    meta.append(f"consent: {entry['consent'].replace('_', ' ')}")
                separator = voice.get("digest.entry.meta_separator", "` · `")
                out += ["`" + separator.join(meta) + "`", ""]

                if entry["attention"]:
                    out += [voice.text("digest.entry.attention_note"), ""]
                    continue
                if entry["error"]:
                    out += [voice.text("digest.entry.error_note",
                                       error=entry["error"][:300]), ""]
                    continue

                highlight = profile.highlight_fields or profile.field_keys
                shown = 0
                for key in highlight:
                    if key in profile.suppress_fields or key == "next_action":
                        continue
                    spec = profile.field_by_key(key)
                    values = _fmt_value(entry["fields"].get(key))
                    if not values:
                        continue
                    out.append(f"**{spec.label if spec else key}**")
                    out += [f"- {v}" for v in values]
                    out.append("")
                    shown += 1

                for key in profile.suppress_fields:
                    raw = entry["fields"].get(key)
                    count = len(raw) if isinstance(raw, list) else (1 if raw else 0)
                    if count:
                        spec = profile.field_by_key(key)
                        out += [voice.text(
                            "digest.entry.withheld",
                            label=spec.label if spec else key, count=count,
                        ), ""]

                na = entry["fields"].get("next_action")
                if isinstance(na, str) and na.strip():
                    out += [voice.text("digest.entry.next_action", action=na.strip()), ""]

                if opts.include_links and entry["artifacts"]:
                    links = ", ".join(
                        f"[{k}]({v})" for k, v in entry["artifacts"].items() if not v.endswith(".enc")
                    )
                    encrypted = [k for k, v in entry["artifacts"].items() if v.endswith(".enc")]
                    if links:
                        out += [voice.text("digest.entry.files", links=links), ""]
                    if encrypted:
                        out += [voice.text(
                            "digest.entry.encrypted",
                            kinds=", ".join(encrypted), id=entry["id"],
                        ), ""]

                if shown == 0:
                    out += [voice.text("digest.entry.nothing_extracted"), ""]

        # --- footer -------------------------------------------------------
        if opts.include_costs:
            total = sum(e["cost"] for s in sections for e in s.entries)
            minutes = sum(e["minutes"] for s in sections for e in s.entries)
            out += ["---", ""]
            out += [voice.text("digest.footer.costs",
                               minutes=f"{minutes:.0f}", cost=f"{total:.4f}"), ""]

        if opts.profile_id is None and not opts.include_personal:
            hidden = [
                p for p in self.cfg.profiles.values() if p.exclude_from_combined_export
            ]
            if hidden:
                out += [voice.text(
                    "digest.footer.personal_hidden",
                    names=", ".join(p.name for p in hidden),
                    example=hidden[0].id,
                ), ""]

        out += voice.lines("digest.footer.sign_off")
        return "\n".join(out)

    def render_html(self, opts: DigestOptions, title: str = "Digest") -> str:
        """
        The digest as a self-contained page, with charts.

        The page body is still the markdown, converted (html.py), so the two
        formats cannot drift into saying different things. The charts are an
        HTML-only layer drawn from the same collected sections and slotted in
        after the At a Glance table — they re-plot numbers the text already
        prints and nothing more. See charts.py for the privacy reasoning.
        """
        from .charts import charts_html, inject_charts
        from .html import to_html

        sections = self._collect(opts)
        page = to_html(self.render_markdown(opts, sections=sections), title=title)
        return inject_charts(page, charts_html(sections, opts, self.cfg.voice))

    def _empty_note(self, opts: DigestOptions, voice) -> str:
        """What a section with nothing in it says. Per-profile wording wins."""
        if opts.profile_id and opts.profile_id in self.cfg.profiles:
            own = self.cfg.profile(opts.profile_id).digest_empty
            if own:
                return own
        return voice.text("digest.empty")
