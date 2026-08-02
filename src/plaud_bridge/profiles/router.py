"""
Multi-label profile routing.

The core insight: one recording can legitimately belong to several profiles.
A dinner conversation where a client calls is Husband and Insurance Agent at
the same time. Forcing a single label would either lose the client follow-up
or file a private conversation under work. So routing is multi-label and the
compliance gate handles the collision afterwards by letting the strictest
profile govern the whole file.

Two stages:
  1. Keyword prescore. Free, deterministic, narrows the candidate set and gives
     the LLM stage something to anchor on.
  2. One LLM call scoring every candidate at once. One call, not one per
     profile, because five calls per recording adds up and the model reasons
     better when it can see the alternatives side by side.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..llm import complete_json
from ..llm.base import LLMError
from ..logging_setup import get
from ..models import RouteMatch, Transcript

log = get("router")

MAX_ROUTING_CHARS = 14000


class RouterError(RuntimeError):
    pass


@dataclass
class RoutingResult:
    """
    What routing produced, and what it cost.

    The cost is carried out rather than discarded because routing is an LLM
    call on every single recording. Leaving it uncounted meant the spend
    guardrail in pipeline.yaml was blind to a per-recording charge that fires
    whether or not anything is ever analysed.
    """

    matches: list[RouteMatch] = field(default_factory=list)
    cost_usd: float = 0.0


@dataclass
class _Prescore:
    profile_id: str
    score: float
    hits: list[str]


def _keyword_prescore(text: str, profiles) -> list[_Prescore]:
    lowered = text.lower()
    results: list[_Prescore] = []

    for profile in profiles:
        if not profile.keywords:
            results.append(_Prescore(profile.id, 0.0, []))
            continue

        hits: list[str] = []
        for kw in profile.keywords:
            pattern = re.compile(rf"(?<!\w){re.escape(kw)}(?!\w)", re.IGNORECASE)
            found = len(pattern.findall(lowered))
            if found:
                hits.append(f"{kw}({found})")

        penalty = 0
        for nkw in profile.negative_keywords:
            pattern = re.compile(rf"(?<!\w){re.escape(nkw)}(?!\w)", re.IGNORECASE)
            penalty += len(pattern.findall(lowered))

        # Distinct terms matter more than repetition. Saying "policy" nine
        # times is weaker evidence than saying policy, premium, and beneficiary
        # once each.
        distinct = len(hits)
        raw = distinct / max(4.0, len(profile.keywords) * 0.30)
        score = max(0.0, min(1.0, raw - (penalty * 0.20)))
        results.append(_Prescore(profile.id, score, hits[:12]))

    return results


def _build_prompt(cfg, profiles, prescores: dict[str, _Prescore], transcript_text: str) -> tuple[str, str]:
    lines = []
    for profile in profiles:
        pre = prescores.get(profile.id)
        lines.append(
            f"- id: {profile.id}\n"
            f"  name: {profile.name}\n"
            f"  when_to_use: {profile.llm_hint or profile.description}\n"
            f"  keyword_prescore: {pre.score:.2f} (matched: {', '.join(pre.hits) or 'none'})"
        )
    catalogue = "\n".join(lines)

    system = (
        "You classify a recording transcript against a fixed set of life "
        "profiles. A recording can belong to more than one profile at the same "
        "time; score each one independently.\n\n"
        "Rules:\n"
        "- Score every profile from 0.0 to 1.0 for how much of this recording "
        "genuinely belongs to it.\n"
        "- The keyword prescore is a hint, not an answer. Override it when the "
        "content says otherwise.\n"
        "- Background presence is not membership. A child audible behind a "
        "sales call does not make it a parenting recording.\n"
        "- Give a short evidence phrase for any profile you score above 0.3.\n"
        "- Respond with JSON only. No preamble, no code fences, no commentary."
    )

    user = (
        f"PROFILES:\n{catalogue}\n\n"
        f"TRANSCRIPT:\n{transcript_text}\n\n"
        "Respond with exactly this JSON shape:\n"
        '{"scores": [{"profile_id": "...", "score": 0.0, "evidence": ["..."]}]}'
    )
    return system, user


def route(transcript: Transcript, cfg, local_only: bool = False) -> RoutingResult:
    """Return the profiles this recording belongs to, sorted by confidence."""
    profiles = cfg.routable_profiles()
    if not profiles:
        raise RouterError("no routable profiles configured")

    text = transcript.labelled_text(max_chars=MAX_ROUTING_CHARS)
    if not text.strip():
        return RoutingResult()

    prescores = {p.profile_id: p for p in _keyword_prescore(text, profiles)}
    kw_weight = float(cfg.get("routing.keyword_prescore_weight", 0.35))
    use_llm = bool(cfg.get("routing.llm_confirm", True))

    llm_scores: dict[str, float] = {}
    evidence: dict[str, list[str]] = {}
    cost_usd = 0.0

    if use_llm:
        system, user = _build_prompt(cfg, profiles, prescores, text)
        try:
            data, resp = complete_json(cfg, system, user, local_only=local_only, max_tokens=2000)
            cost_usd = resp.cost_usd
            for entry in data.get("scores", []) or []:
                if not isinstance(entry, dict):
                    continue
                pid = str(entry.get("profile_id", "")).strip()
                if pid not in cfg.profiles:
                    continue
                llm_scores[pid] = max(0.0, min(1.0, float(entry.get("score", 0.0))))
                evidence[pid] = [str(e) for e in (entry.get("evidence") or [])][:4]
        except (LLMError, AttributeError, ValueError, TypeError) as exc:
            log.warning("LLM routing failed, falling back to keywords only: %s", exc)
            use_llm = False

        # A reply that parsed but scored nothing usable is the same situation as
        # a reply that never arrived, and it has to be treated the same way.
        # Treating it as "the model scored everything zero" scaled every
        # confidence down by the keyword weight, which quietly dropped a family
        # recording that keywords had matched perfectly into the unfiled bucket
        # -- a profile with weaker retention and none of the family prompt's
        # safety constraints.
        if use_llm and not llm_scores:
            log.warning(
                "LLM routing returned no usable scores; falling back to keywords only"
            )
            use_llm = False

    matches: list[RouteMatch] = []
    for profile in profiles:
        kw = prescores[profile.id].score
        llm = llm_scores.get(profile.id, 0.0)
        # When the LLM stage is unavailable, keywords carry the full weight
        # rather than being scaled down into never clearing the threshold.
        confidence = (kw_weight * kw + (1 - kw_weight) * llm) if use_llm else kw

        # A keyword floor for the locked profiles. Missing a family or spousal
        # recording is the asymmetric failure this whole tool exists to prevent
        # -- it means that conversation could be handed to a cloud model or kept
        # in the clear. So for a profile that forbids cloud processing, a keyword
        # match strong enough to clear the bar on its own is not allowed to be
        # averaged away by a low LLM score: the model can add to a locked
        # profile's confidence but never score it below what its own keywords
        # earned. An adversarial transcript that talks the model down cannot use
        # that to smuggle a family conversation past the local-only gate.
        if (profile.hard_local_only or not profile.allow_cloud_llm) and kw >= profile.min_confidence:
            confidence = max(confidence, kw)

        if confidence >= profile.min_confidence:
            matches.append(
                RouteMatch(
                    profile_id=profile.id,
                    confidence=round(confidence, 4),
                    keyword_score=round(kw, 4),
                    llm_score=round(llm, 4),
                    evidence=evidence.get(profile.id, prescores[profile.id].hits[:4]),
                )
            )

    matches.sort(key=lambda m: -m.confidence)
    cap = int(cfg.get("routing.max_profiles_per_recording", 4))
    matches = matches[:cap]

    if not matches:
        fallback = cfg.get("routing.fallback_profile", "unfiled")
        log.info("no profile cleared its threshold; filing under '%s'", fallback)
        matches = [RouteMatch(profile_id=fallback, confidence=0.0, evidence=["no confident match"])]

    log.info("routed to %s", ", ".join(f"{m.profile_id}={m.confidence:.2f}" for m in matches))
    return RoutingResult(matches=matches, cost_usd=cost_usd)
