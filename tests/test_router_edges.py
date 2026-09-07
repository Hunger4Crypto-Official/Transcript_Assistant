"""
Routing edges: what happens when the model stage gives the router nothing.

The keyword floor and the fallback bucket are the two behaviours that decide
whether a family recording ends up under a profile with the family prompt's
constraints or in "unfiled". Each edge here pins the route that was actually
taken, not just that the call survived. The Transcript rendering the router
reads is pinned alongside, because an elided or empty rendering is the input
to all of it.
"""

from __future__ import annotations

import logging

import pytest

from _fixtures import CLIENT_CALL, build_sandbox
from plaud_bridge.llm.base import LLMResponse

# Dense enough in insurance vocabulary that the keyword prescore clears the
# insurance_agent threshold on its own (eight distinct terms). CLIENT_CALL
# does not: four hits of twenty-nine keywords lands under the 0.55 floor.
DENSE_WORK = """\
Sasson: So the client wants term life with a conversion rider on the policy.
Marcus: And we talked about the elimination period on the disability coverage.
Sasson: The premium is four hundred a month and the death benefit is one million.
"""
from plaud_bridge.models import ProfileAnalysis, Recording, Segment, Transcript
from plaud_bridge.profiles.router import (
    RouterError,
    RoutingResult,
    _keyword_prescore,
    route,
)


def _transcript(text: str) -> Transcript:
    segments = []
    for i, line in enumerate(text.strip().splitlines()):
        speaker, _, body = line.partition(":")
        segments.append(Segment(i * 4.0, i * 4.0 + 3.0, body.strip(), speaker.strip()))
    return Transcript(segments=segments)


def _install(monkeypatch, payload):
    calls = []

    def fake(cfg, system, user, local_only=False, max_tokens=None):
        calls.append(user)
        return payload, LLMResponse(provider="stub", model="stub", cost_usd=0.001)

    monkeypatch.setattr("plaud_bridge.profiles.router.complete_json", fake)
    return calls


# =========================================================================
# Prescore
# =========================================================================
def test_a_profile_with_no_keywords_prescores_zero_with_no_hits(tmp_path, monkeypatch):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    unfiled = cfg.profile("unfiled")
    assert unfiled.keywords == [], "the fixture profile is supposed to have no keywords"
    [pre] = _keyword_prescore("policy premium coverage", [unfiled])
    assert pre.profile_id == "unfiled"
    assert pre.score == 0.0
    assert pre.hits == []


# =========================================================================
# Refusals and empties
# =========================================================================
def test_no_routable_profiles_is_a_router_error(tmp_path, monkeypatch):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    monkeypatch.setattr(cfg, "routable_profiles", lambda: [])
    with pytest.raises(RouterError, match="no routable profiles configured"):
        route(_transcript(CLIENT_CALL), cfg)


def test_an_empty_transcript_routes_nowhere_and_costs_nothing(tmp_path, monkeypatch):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    calls = _install(monkeypatch, {"scores": []})
    result = route(Transcript(segments=[Segment(0, 1, "   ", "Sasson")]), cfg)
    assert result == RoutingResult()
    assert result.matches == [] and result.cost_usd == 0.0
    assert not calls, "an empty transcript was still sent to a model"


# =========================================================================
# Unusable model replies fall back to keywords, never to zero
# =========================================================================
def test_unusable_score_entries_are_ignored_and_keywords_carry_full_weight(
        tmp_path, monkeypatch, caplog):
    """
    A non-dict entry and an entry naming a profile that does not exist are both
    dropped. With nothing usable left the router treats the reply as absent:
    the keyword score IS the confidence rather than being scaled by the weight.
    """
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    _install(monkeypatch, {"scores": [
        "not an object",
        {"profile_id": "ghost_profile", "score": 1.0, "evidence": ["x"]},
    ]})
    with caplog.at_level(logging.WARNING, logger="plaud_bridge.router"):
        result = route(_transcript(DENSE_WORK), cfg)

    assert any("no usable scores" in r.getMessage() for r in caplog.records)
    agent = next(m for m in result.matches if m.profile_id == "insurance_agent")
    assert agent.llm_score == 0.0
    assert agent.confidence == agent.keyword_score, (
        "keywords were scaled down by the LLM weight even though the LLM stage produced nothing"
    )
    assert agent.keyword_score >= cfg.profile("insurance_agent").min_confidence
    # The evidence shown is the keyword hits, since no model phrase exists.
    assert agent.evidence and all("(" in e for e in agent.evidence)
    assert "ghost_profile" not in [m.profile_id for m in result.matches]
    # The call still happened, so its cost is still carried out.
    assert result.cost_usd == pytest.approx(0.001)


def test_nothing_clearing_its_threshold_files_under_the_fallback(tmp_path, monkeypatch, caplog):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    _install(monkeypatch, {"scores": [
        {"profile_id": p.id, "score": 0.0, "evidence": []} for p in cfg.routable_profiles()
    ]})
    with caplog.at_level(logging.INFO, logger="plaud_bridge.router"):
        result = route(_transcript("A: The weather was nice.\nB: It was."), cfg)

    [match] = result.matches
    assert match.profile_id == "unfiled"
    assert match.confidence == 0.0
    assert match.evidence == ["no confident match"]
    assert any("filing under 'unfiled'" in r.getMessage() for r in caplog.records)


# =========================================================================
# Transcript: the rendering the router reads
# =========================================================================
def test_labelled_text_skips_blank_segments_and_groups_by_speaker():
    tr = Transcript(segments=[
        Segment(0, 1, "hello", "A"),
        Segment(1, 2, "   ", "B"),
        Segment(2, 3, "again", "A"),
        Segment(3, 4, "reply", "B"),
    ])
    assert tr.labelled_text() == "[00:00] A:\nhello\nagain\n\n[00:03] B:\nreply"


def test_labelled_text_elides_the_middle_and_says_how_much():
    tr = Transcript(segments=[Segment(i, i + 1, f"line number {i:03d}", "A") for i in range(200)])
    full = tr.labelled_text()
    out = tr.labelled_text(max_chars=400)
    assert out.startswith(full[:280]), "the head must be the opening 70%"
    assert out.endswith(full[-100:]), "the tail must be the closing 25%"
    elided = len(full) - 280 - 100
    assert f"[... {elided} characters elided ...]" in out
    assert tr.labelled_text(max_chars=len(full)) == full, "a fitting transcript is untouched"


def test_the_opening_window_is_the_text_that_starts_inside_it():
    tr = Transcript(segments=[
        Segment(0, 5, "first", "A"), Segment(80, 85, "second", "B"), Segment(95, 99, "third", "A"),
    ])
    assert tr.window(90) == "first second"


def test_a_transcript_round_trips_through_its_dict_form():
    tr = Transcript(
        segments=[Segment(0, 1.5, "hi", "A", confidence=-0.2, no_speech=0.1)],
        language="fr", asr_provider="local", asr_model="large-v3",
        duration_seconds=1.5, cost_usd=0.01, confidence_report={"verdict": "ok"},
    )
    back = Transcript.from_dict(tr.to_dict())
    assert back == tr
    minimal = Transcript.from_dict({"segments": [{"start": 0, "end": 1}]})
    assert minimal.segments[0].text == "" and minimal.language == "en"
    assert minimal.confidence_report == {}


def test_analysis_for_finds_the_matching_profile_or_none():
    rec = Recording(analyses=[ProfileAnalysis(profile_id="father", fields={"x": 1})])
    assert rec.analysis_for("father").fields == {"x": 1}
    assert rec.analysis_for("husband") is None
