"""
What gets sent to the model, and what is believed when it answers.

Four things are pinned here. Three are about the request: no sampling
parameter reaches the current Anthropic models (sending one is a 400, not a
no-op), the profile's system prompt travels as a cached block, and the Groq
path is left exactly as it was because none of that applies to it.

The fourth is about the answer. A quote is attributed to a named person, flows
into the memory ledger as something they said, and can surface in a digest a
year later when the audio is gone. Nothing downstream re-checks it. So it is
checked here, against the text the model was actually shown.
"""

from __future__ import annotations

import pytest

from _fixtures import CLIENT_CALL, build_sandbox
from plaud_bridge.llm import anthropic_provider, openai_compat_provider
from plaud_bridge.llm.anthropic_provider import AnthropicLLM
from plaud_bridge.llm.openai_compat_provider import OpenAICompatLLM
from plaud_bridge.models import Transcript
from plaud_bridge.profiles import extractor


def capture(monkeypatch, module, response=None):
    """Intercept the HTTP call and hand back the payload that would have gone out."""
    seen: dict = {}

    def fake_post(url, payload, headers=None, timeout=None, max_retries=None):
        seen["url"], seen["payload"], seen["headers"] = url, payload, headers or {}
        return response or {
            "content": [{"type": "text", "text": '{"ok": true}'}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

    monkeypatch.setattr(module, "post_json", fake_post)
    return seen


# ---------------------------------------------------------------------------
# the request
# ---------------------------------------------------------------------------
def test_no_sampling_parameter_reaches_anthropic(sandbox, monkeypatch):
    """
    It was pinned to 0.0 for determinism it never actually provided, and the
    current models reject it outright rather than ignoring it.
    """
    cfg, _ = sandbox
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    seen = capture(monkeypatch, anthropic_provider)

    AnthropicLLM(cfg).complete("system text", "user text")
    payload = seen["payload"]
    for banned in ("temperature", "top_p", "top_k"):
        assert banned not in payload, f"{banned} would be a 400 on this model"


def test_the_system_prompt_is_sent_as_a_cached_block(sandbox, monkeypatch):
    cfg, _ = sandbox
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    seen = capture(monkeypatch, anthropic_provider)

    AnthropicLLM(cfg).complete("a profile's whole system prompt", "the transcript")
    system = seen["payload"]["system"]
    assert isinstance(system, list), "the system prompt is not a cacheable block"
    assert system[0]["cache_control"] == {"type": "ephemeral"}
    assert system[0]["text"] == "a profile's whole system prompt"


def test_the_transcript_never_travels_inside_the_cached_block(sandbox, monkeypatch):
    """
    Caching is a prefix match. Putting the one thing that changes every
    recording ahead of the marker would mean nothing is ever served from cache.
    """
    cfg, _ = sandbox
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    seen = capture(monkeypatch, anthropic_provider)

    AnthropicLLM(cfg).complete("stable instructions", "VOLATILE TRANSCRIPT TEXT")
    payload = seen["payload"]
    assert "VOLATILE TRANSCRIPT TEXT" not in str(payload["system"])
    assert "VOLATILE TRANSCRIPT TEXT" in str(payload["messages"])


def test_caching_can_be_turned_off_without_breaking_the_request(tmp_path, monkeypatch):
    cfg, _ = build_sandbox(tmp_path, monkeypatch,
                           overrides={"llm": {"anthropic": {"cache_system_prompt": False,
                                                            "enabled": True}}})
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    seen = capture(monkeypatch, anthropic_provider)

    AnthropicLLM(cfg).complete("system text", "user text")
    assert seen["payload"]["system"] == "system text"


def test_effort_is_sent_and_is_configurable(tmp_path, monkeypatch):
    cfg, _ = build_sandbox(tmp_path, monkeypatch,
                           overrides={"llm": {"anthropic": {"effort": "low", "enabled": True}}})
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    seen = capture(monkeypatch, anthropic_provider)

    AnthropicLLM(cfg).complete("system text", "user text")
    assert seen["payload"]["output_config"] == {"effort": "low"}


def test_max_tokens_leaves_room_for_thinking(sandbox):
    """
    Thinking is on by default on this generation and shares the ceiling with the
    response, so a budget sized around the JSON alone truncates mid-object.
    """
    cfg, _ = sandbox
    assert AnthropicLLM(cfg).max_tokens >= 16000


# ---------------------------------------------------------------------------
# the free Groq key is not collateral damage
# ---------------------------------------------------------------------------
def test_groq_still_gets_its_temperature(sandbox, monkeypatch):
    """
    The sampling parameters were removed on one vendor's models, not on every
    endpoint that speaks the OpenAI shape. Stripping it here would change Groq's
    behaviour for no reason.
    """
    cfg, _ = sandbox
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    seen = capture(monkeypatch, openai_compat_provider, response={
        "choices": [{"message": {"content": "{}"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2},
    })

    OpenAICompatLLM(cfg, "groq", "llm.groq").complete("system", "user")
    assert seen["payload"]["temperature"] == 0.0
    assert "cache_control" not in str(seen["payload"])


def test_groq_is_still_in_the_chain_for_both_jobs(sandbox):
    cfg, _ = sandbox
    assert "groq" in (cfg.get("llm.providers") or [])
    assert cfg.get("llm.groq.enabled") is True
    assert cfg.get("asr.groq.enabled") is True


# ---------------------------------------------------------------------------
# what the model claims somebody said
# ---------------------------------------------------------------------------
def quoted(cfg, monkeypatch, payload, body=CLIENT_CALL, profile="insurance_agent"):
    """Run one extraction with the model's answer stubbed to `payload`."""
    from plaud_bridge.llm.base import LLMResponse

    def fake(cfg_, system, user, local_only=False, max_tokens=None):
        return payload, LLMResponse(text="", provider="stub", model="stub-1")

    monkeypatch.setattr(extractor, "complete_json", fake)
    transcript = Transcript()
    return extractor.extract(transcript, cfg.profile(profile), cfg, body)


def _quote_field(cfg, profile_id="insurance_agent"):
    for spec in cfg.profile(profile_id).fields:
        if "quote" in spec.type.lower():
            return spec.key
    raise AssertionError("no quote field in this profile")


def test_a_quote_that_was_really_said_survives(sandbox, monkeypatch):
    cfg, _ = sandbox
    key = _quote_field(cfg)
    real = CLIENT_CALL.splitlines()[2].split(":", 1)[-1].strip()[:40]

    analysis = quoted(cfg, monkeypatch, {
        key: [{"timestamp": "00:10", "speaker": "Client", "text": real}],
    })
    assert len(analysis.fields[key]) == 1
    assert analysis.unverified_quotes == 0


def test_a_quote_nobody_said_is_dropped_and_counted(sandbox, monkeypatch):
    cfg, _ = sandbox
    key = _quote_field(cfg)

    analysis = quoted(cfg, monkeypatch, {
        key: [{"timestamp": "00:10", "speaker": "Client",
               "text": "I have decided to buy the largest policy you offer today."}],
    })
    assert analysis.fields[key] == []
    assert analysis.unverified_quotes == 1


def test_the_real_quote_survives_alongside_the_invented_one(sandbox, monkeypatch):
    """Verification is per quote, not per field — one bad one is not contagious."""
    cfg, _ = sandbox
    key = _quote_field(cfg)
    real = CLIENT_CALL.splitlines()[2].split(":", 1)[-1].strip()[:40]

    analysis = quoted(cfg, monkeypatch, {
        key: [
            {"timestamp": "00:10", "speaker": "Client", "text": real},
            {"timestamp": "00:20", "speaker": "Client", "text": "and I will pay in cash"},
        ],
    })
    assert len(analysis.fields[key]) == 1
    assert analysis.unverified_quotes == 1


def test_punctuation_and_case_are_forgiven_but_different_words_are_not(sandbox, monkeypatch):
    cfg, _ = sandbox
    key = _quote_field(cfg)
    real = CLIENT_CALL.splitlines()[2].split(":", 1)[-1].strip()[:40]

    loud = quoted(cfg, monkeypatch, {
        key: [{"timestamp": "00:10", "speaker": "Client", "text": f"  {real.upper()}!!  "}],
    })
    assert len(loud.fields[key]) == 1, "a case change should not read as a fabrication"

    reworded = quoted(cfg, monkeypatch, {
        key: [{"timestamp": "00:10", "speaker": "Client",
               "text": "the client indicated interest in coverage options"}],
    })
    assert reworded.fields[key] == [], "a paraphrase is not a quote"


def test_it_checks_against_what_the_model_was_shown_not_the_raw_transcript(sandbox, monkeypatch):
    """
    When compliance redacts before the model sees it, the model can only quote
    the redacted text. Checking against the raw transcript would flag every
    redacted quote as invented.
    """
    cfg, _ = sandbox
    key = _quote_field(cfg)
    redacted = "my policy number is [ACCOUNT_REDACTED] and I want to raise the limit"

    analysis = quoted(cfg, monkeypatch, {
        key: [{"timestamp": "00:10", "speaker": "Client",
               "text": "policy number is [ACCOUNT_REDACTED] and I want"}],
    }, body=redacted)
    assert len(analysis.fields[key]) == 1
    assert analysis.unverified_quotes == 0


def test_verification_can_be_turned_off(tmp_path, monkeypatch):
    cfg, _ = build_sandbox(tmp_path, monkeypatch, overrides={"llm": {"verify_quotes": False}})
    key = _quote_field(cfg)

    analysis = quoted(cfg, monkeypatch, {
        key: [{"timestamp": "00:10", "speaker": "X", "text": "never said anywhere at all"}],
    })
    assert len(analysis.fields[key]) == 1
    assert analysis.unverified_quotes == 0


def test_non_quote_fields_are_left_alone(sandbox, monkeypatch):
    """
    A next action or an open question is the model's own summary and is supposed
    to be. Only fields the schema calls quotes are held to the transcript.
    """
    cfg, _ = sandbox
    key = _quote_field(cfg)

    analysis = quoted(cfg, monkeypatch, {
        key: [],
        "next_action": "Send two quote options by Thursday, phrased however you like",
    })
    assert analysis.fields["next_action"].startswith("Send two quote options")
    assert analysis.unverified_quotes == 0


def test_a_flagged_recording_is_not_second_guessed(sandbox, monkeypatch):
    """
    The family profiles discard everything when they flag, so there is nothing
    left to verify and no count to report.
    """
    cfg, _ = sandbox
    analysis = quoted(cfg, monkeypatch, {"requires_human_attention": True}, profile="father")
    assert analysis.requires_human_attention
    assert analysis.unverified_quotes == 0


def test_the_count_survives_into_the_stored_record(sandbox, monkeypatch):
    cfg, _ = sandbox
    key = _quote_field(cfg)
    analysis = quoted(cfg, monkeypatch, {
        key: [{"timestamp": "00:10", "speaker": "X", "text": "definitely never uttered"}],
    })
    assert analysis.to_dict()["unverified_quotes"] == 1


@pytest.mark.parametrize("profile_id", ["insurance_agent", "sales_trainer", "father", "husband"])
def test_every_shipped_profile_has_a_quote_field_to_protect(sandbox, profile_id):
    """
    If a profile stops declaring quote fields, this file silently stops testing
    anything for it. Better to fail here than to believe the coverage.
    """
    cfg, _ = sandbox
    types = [spec.type.lower() for spec in cfg.profile(profile_id).fields]
    assert any("quote" in t for t in types), f"{profile_id} declares no quote field"
