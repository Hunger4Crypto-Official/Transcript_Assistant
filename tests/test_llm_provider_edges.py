"""
The LLM backends and their chain, driven against a loopback server.

`test_llm_brain` pins what the request *contains* by intercepting `post_json`.
These tests go one layer down: the bytes on the wire, the retry-without-schema
fallback that only fires on a real 400, what a malformed or non-JSON answer
turns into, what a rejected request looks like from the outside, and -- since
the key rides in a header on every call -- that it never lands in an error or
a log line.

The registry tests cover the two rules in its docstring: a cloud provider is
removed under a local veto rather than deprioritised, and a chain that runs
out of providers fails loudly rather than reaching for one it excluded.
"""

from __future__ import annotations

import json
import logging

import pytest
import yaml

from _fixtures import build_sandbox
from plaud_bridge import http_util
from plaud_bridge.config import Config
from plaud_bridge.llm.anthropic_provider import AnthropicLLM
from plaud_bridge.llm.base import LLMError
from plaud_bridge.llm.openai_compat_provider import OpenAICompatLLM
from plaud_bridge.llm.registry import build_llm_chain, complete_json
from test_http_util import SECRET, ScriptedServer

KEY = SECRET


@pytest.fixture
def server():
    stub = ScriptedServer()
    try:
        yield stub
    finally:
        stub.close()


def _reconfigure(tmp_path, **blocks) -> Config:
    """Rewrite the sandbox config, merging two levels deep, and reload it."""
    path = tmp_path / "config" / "pipeline.yaml"
    raw = yaml.safe_load(path.read_text())
    for block, values in blocks.items():
        raw.setdefault(block, {})
        for key, value in values.items():
            if isinstance(value, dict) and isinstance(raw[block].get(key), dict):
                raw[block][key].update(value)
            else:
                raw[block][key] = value
    path.write_text(yaml.safe_dump(raw))
    return Config.load(tmp_path / "config", root=tmp_path)


def _chat_reply(content: str, prompt_tokens: int = 100, completion_tokens: int = 20) -> dict:
    return {
        "id": "stub",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


@pytest.fixture
def groq_cfg(tmp_path, monkeypatch, server):
    """Groq's chat block pointed at the loopback stub, key in the env, backoff off."""
    build_sandbox(tmp_path, monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", KEY)
    monkeypatch.setattr(http_util, "_sleep_backoff", lambda attempt, **kw: None)
    return _reconfigure(tmp_path, llm={"groq": {"base_url": f"{server.url}/groq", "max_retries": 1}})


@pytest.fixture
def local_cfg(tmp_path, monkeypatch, server):
    """A local ollama-shaped server: enabled, no key, free."""
    build_sandbox(tmp_path, monkeypatch)
    monkeypatch.setattr(http_util, "_sleep_backoff", lambda attempt, **kw: None)
    return _reconfigure(tmp_path, llm={
        "local": {"enabled": True, "base_url": f"{server.url}/local", "api_key_env": "", "max_retries": 0},
    })


# =========================================================================
# OpenAI-compatible: availability
# =========================================================================
def test_openai_compat_is_unavailable_when_disabled(sandbox, monkeypatch):
    cfg, _ = sandbox
    assert OpenAICompatLLM(cfg, "local", is_cloud=False).available() == (False, "disabled in config")


def test_openai_compat_is_unavailable_without_a_base_url_or_model(tmp_path, monkeypatch):
    build_sandbox(tmp_path, monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", KEY)

    cfg = _reconfigure(tmp_path, llm={"groq": {"base_url": ""}})
    assert OpenAICompatLLM(cfg, "groq", is_cloud=True).available() == (
        False, "base_url or model missing from config",
    )

    cfg = _reconfigure(tmp_path, llm={"groq": {"base_url": "http://127.0.0.1:1", "model": ""}})
    assert OpenAICompatLLM(cfg, "groq", is_cloud=True).available() == (
        False, "base_url or model missing from config",
    )


def test_openai_compat_is_unavailable_without_its_key_and_names_the_variable(sandbox, monkeypatch):
    cfg, _ = sandbox
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert OpenAICompatLLM(cfg, "groq", is_cloud=True).available() == (False, "GROQ_API_KEY not set")

    monkeypatch.setenv("GROQ_API_KEY", " \t")
    assert OpenAICompatLLM(cfg, "groq", is_cloud=True).available() == (False, "GROQ_API_KEY not set")


def test_a_local_server_needs_no_key_at_all(local_cfg):
    assert OpenAICompatLLM(local_cfg, "local", is_cloud=False).available() == (True, "ready")


def test_openai_compat_refuses_to_complete_when_unavailable(sandbox, monkeypatch):
    cfg, _ = sandbox
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(LLMError, match="groq unavailable: GROQ_API_KEY not set"):
        OpenAICompatLLM(cfg, "groq", is_cloud=True).complete("s", "u")


# =========================================================================
# OpenAI-compatible: the wire
# =========================================================================
def test_openai_compat_sends_the_chat_request_the_endpoint_expects(groq_cfg, server):
    server.respond_json(_chat_reply('{"ok": true}'))

    response = OpenAICompatLLM(groq_cfg, "groq", is_cloud=True).complete("be terse", "hello", max_tokens=321)

    assert response.text == '{"ok": true}'
    seen = server.seen[0]
    assert seen.path == "/groq/chat/completions"
    assert seen.headers["authorization"] == f"Bearer {KEY}"
    assert seen.headers["content-type"] == "application/json"
    assert seen.json == {
        "model": groq_cfg.get("llm.groq.model"),
        "max_tokens": 321,
        "temperature": 0.0,
        "messages": [{"role": "system", "content": "be terse"}, {"role": "user", "content": "hello"}],
        "response_format": {"type": "json_object"},
    }


def test_openai_compat_uses_the_configured_token_ceiling_when_none_is_given(groq_cfg, server):
    server.respond_json(_chat_reply("{}"))

    OpenAICompatLLM(groq_cfg, "groq", is_cloud=True).complete("s", "u")

    assert server.seen[0].json["max_tokens"] == int(groq_cfg.get("llm.groq.max_tokens"))


def test_a_local_server_gets_no_authorization_header(local_cfg, server):
    server.respond_json(_chat_reply("{}"))

    OpenAICompatLLM(local_cfg, "local", is_cloud=False).complete("s", "u")

    assert "authorization" not in server.seen[0].headers
    assert server.seen[0].path == "/local/chat/completions"


def test_openai_compat_bills_the_reported_usage_at_the_configured_rates(groq_cfg, server):
    server.respond_json(_chat_reply("{}", prompt_tokens=1_000_000, completion_tokens=500_000))

    response = OpenAICompatLLM(groq_cfg, "groq", is_cloud=True).complete("s", "u")

    rate_in = float(groq_cfg.get("llm.groq.usd_per_million_input_tokens"))
    rate_out = float(groq_cfg.get("llm.groq.usd_per_million_output_tokens"))
    assert rate_in > 0 and rate_out > 0, "the shipped config has no groq rates"
    assert response.input_tokens == 1_000_000
    assert response.output_tokens == 500_000
    assert response.cost_usd == pytest.approx(rate_in + rate_out / 2)
    assert response.provider == "groq"
    assert response.model == groq_cfg.get("llm.groq.model")
    assert response.raw["id"] == "stub"


def test_openai_compat_treats_a_reply_without_choices_or_usage_as_empty_and_free(groq_cfg, server):
    server.respond_json({"id": "odd"})

    response = OpenAICompatLLM(groq_cfg, "groq", is_cloud=True).complete("s", "u")

    assert response.text == ""
    assert response.input_tokens == 0
    assert response.output_tokens == 0
    assert response.cost_usd == 0.0


# =========================================================================
# OpenAI-compatible: the schema-hint fallback
# =========================================================================
def test_a_400_makes_the_provider_retry_once_without_the_response_format_hint(groq_cfg, server):
    """Not every server knows response_format. One schema hint must not sink a recording."""
    server.respond(400, '{"error": "response_format is not supported"}')
    server.respond_json(_chat_reply('{"answer": 42}'))

    response = OpenAICompatLLM(groq_cfg, "groq", is_cloud=True).complete("s", "u")

    assert response.text == '{"answer": 42}'
    assert len(server.seen) == 2
    assert "response_format" in server.seen[0].json
    assert "response_format" not in server.seen[1].json
    assert server.seen[1].json["messages"] == server.seen[0].json["messages"]


def test_a_422_also_triggers_the_schema_hint_fallback(local_cfg, server):
    server.respond(422, "unprocessable")
    server.respond_json(_chat_reply("{}"))

    OpenAICompatLLM(local_cfg, "local", is_cloud=False).complete("s", "u")

    assert len(server.seen) == 2
    assert "response_format" not in server.seen[1].json


def test_a_second_rejection_after_dropping_the_hint_is_reported_with_its_body(groq_cfg, server):
    server.respond(400, "first: no response_format")
    server.respond(400, '{"error": "second: model not found"}')
    server.respond_json(_chat_reply("{}"))   # must never be reached

    with pytest.raises(LLMError) as info:
        OpenAICompatLLM(groq_cfg, "groq", is_cloud=True).complete("s", "u")

    message = str(info.value)
    assert message.startswith("groq request failed: HTTP 400")
    assert "second: model not found" in message
    assert "first: no response_format" not in message
    assert len(server.seen) == 2


def test_a_non_schema_rejection_is_reported_without_retrying(groq_cfg, server, caplog):
    server.respond(401, '{"error": {"message": "Invalid API Key"}}')

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(LLMError) as info:
            OpenAICompatLLM(groq_cfg, "groq", is_cloud=True).complete("s", "u")

    message = str(info.value)
    assert message.startswith("groq request failed: HTTP 401")
    assert "Invalid API Key" in message
    assert KEY not in message
    assert KEY not in caplog.text
    assert len(server.seen) == 1


def test_a_transient_failure_is_retried_within_the_configured_budget(groq_cfg, server):
    server.respond(503, "overloaded")
    server.respond_json(_chat_reply('{"ok": 1}'))

    assert OpenAICompatLLM(groq_cfg, "groq", is_cloud=True).complete("s", "u").text == '{"ok": 1}'
    assert len(server.seen) == 2


def test_a_server_that_returns_html_instead_of_json_is_a_request_failure(groq_cfg, server):
    """A captive portal or a proxy error page. The provider says so; nothing is guessed."""
    server.respond(200, "<html>Sign in to the hotel wifi</html>", ctype="text/html")

    with pytest.raises(LLMError) as info:
        OpenAICompatLLM(groq_cfg, "groq", is_cloud=True).complete("s", "u")

    assert "groq request failed: non-JSON response" in str(info.value)
    assert len(server.seen) == 1


def test_a_reply_whose_content_is_prose_is_returned_verbatim_for_the_caller_to_judge(groq_cfg, server):
    """The provider carries text; deciding whether it is JSON is the registry's job."""
    server.respond_json(_chat_reply("I'm sorry, I can't help with that."))

    response = OpenAICompatLLM(groq_cfg, "groq", is_cloud=True).complete("s", "u")

    assert response.text == "I'm sorry, I can't help with that."


# =========================================================================
# Anthropic
# =========================================================================
def test_anthropic_is_unavailable_when_disabled(tmp_path, monkeypatch):
    build_sandbox(tmp_path, monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    cfg = _reconfigure(tmp_path, llm={"anthropic": {"enabled": False}})

    assert AnthropicLLM(cfg).available() == (False, "disabled in config")


def test_anthropic_refuses_to_complete_when_unavailable(sandbox, monkeypatch):
    cfg, _ = sandbox
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(LLMError, match="anthropic unavailable: ANTHROPIC_API_KEY not set"):
        AnthropicLLM(cfg).complete("s", "u")


def test_anthropic_sends_its_key_and_version_as_headers_over_the_wire(tmp_path, monkeypatch, server):
    build_sandbox(tmp_path, monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", f" {KEY} ")
    cfg = _reconfigure(tmp_path, llm={"anthropic": {"base_url": server.url}})
    server.respond_json({
        "content": [{"type": "text", "text": '{"ok": true}'}],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    })

    response = AnthropicLLM(cfg).complete("system", "user")

    assert response.text == '{"ok": true}'
    seen = server.seen[0]
    assert seen.path == "/messages"
    assert seen.headers["x-api-key"] == KEY
    assert seen.headers["anthropic-version"] == cfg.get("llm.anthropic.version_header")
    assert "authorization" not in seen.headers
    assert seen.json["model"] == cfg.get("llm.anthropic.model")


def test_anthropic_wraps_a_rejected_request_with_the_status_and_body_and_never_the_key(
    tmp_path, monkeypatch, server, caplog,
):
    build_sandbox(tmp_path, monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    monkeypatch.setattr(http_util, "_sleep_backoff", lambda attempt, **kw: None)
    cfg = _reconfigure(tmp_path, llm={"anthropic": {"base_url": server.url, "max_retries": 3}})
    server.respond(401, json.dumps({"type": "error", "error": {"type": "authentication_error",
                                                             "message": "invalid x-api-key"}}))

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(LLMError) as info:
            AnthropicLLM(cfg).complete("s", "u")

    message = str(info.value)
    assert message.startswith("anthropic request failed: HTTP 401")
    assert "invalid x-api-key" in message
    assert KEY not in message
    assert KEY not in caplog.text
    assert len(server.seen) == 1


# =========================================================================
# Registry: chain construction
# =========================================================================
def test_chain_follows_the_configured_order(sandbox):
    cfg, _ = sandbox
    assert [p.name for p in build_llm_chain(cfg)] == ["anthropic", "groq"]


def test_an_unknown_provider_name_is_skipped_with_a_warning(tmp_path, monkeypatch, caplog):
    build_sandbox(tmp_path, monkeypatch)
    cfg = _reconfigure(tmp_path, llm={"providers": ["mistral", "groq"]})

    with caplog.at_level(logging.WARNING):
        chain = build_llm_chain(cfg)

    assert [p.name for p in chain] == ["groq"]
    assert "unknown LLM provider 'mistral'" in caplog.text


def test_a_local_veto_removes_cloud_providers_and_reaches_for_the_unlisted_local_block(sandbox, caplog):
    cfg, _ = sandbox
    assert "local" not in cfg.get("llm.providers")

    with caplog.at_level(logging.INFO):
        chain = build_llm_chain(cfg, local_only=True)

    assert [p.name for p in chain] == ["local"]
    assert all(not p.is_cloud for p in chain)
    assert "excluding cloud LLM 'anthropic'" in caplog.text
    assert "excluding cloud LLM 'groq'" in caplog.text


def test_a_provider_block_is_cloud_unless_it_says_otherwise(tmp_path, monkeypatch):
    """A block that forgets is_cloud is treated as reaching the network."""
    build_sandbox(tmp_path, monkeypatch)
    cfg = _reconfigure(tmp_path, llm={
        "providers": ["vague"],
        "vague": {"enabled": True, "base_url": "http://127.0.0.1:1", "model": "m"},
    })

    assert [p.name for p in build_llm_chain(cfg)] == ["vague"]
    assert [p.name for p in build_llm_chain(cfg, local_only=True)] == ["local"]


# =========================================================================
# Registry: complete_json
# =========================================================================
class _NoLLMCfg:
    """No providers at all, a shape the validator refuses to load."""

    def get(self, dotted, default=None):
        return default


def test_complete_json_fails_loudly_when_the_chain_is_empty_and_says_why():
    with pytest.raises(LLMError) as generic:
        complete_json(_NoLLMCfg(), "s", "u", local_only=False)
    with pytest.raises(LLMError) as vetoed:
        complete_json(_NoLLMCfg(), "s", "u", local_only=True)

    assert "Check llm.providers in pipeline.yaml" in str(generic.value)
    assert "Enable llm.local in pipeline.yaml" in str(vetoed.value)
    assert "Compliance requires local processing" in str(vetoed.value)


def test_complete_json_falls_through_to_the_next_provider_when_a_request_fails(
    tmp_path, monkeypatch, server, caplog,
):
    build_sandbox(tmp_path, monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", KEY)
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    monkeypatch.setattr(http_util, "_sleep_backoff", lambda attempt, **kw: None)
    cfg = _reconfigure(tmp_path, llm={
        "providers": ["anthropic", "groq", "local"],
        "anthropic": {"base_url": f"{server.url}/anthropic", "max_retries": 0},
        "groq": {"base_url": f"{server.url}/groq", "max_retries": 0},
        "local": {"enabled": True, "base_url": f"{server.url}/local", "api_key_env": "", "max_retries": 0},
    })
    server.respond(529, "overloaded")               # anthropic
    server.respond(401, "bad key")                  # groq
    server.respond_json(_chat_reply('{"answer": "from local"}'))

    with caplog.at_level(logging.WARNING):
        data, response = complete_json(cfg, "s", "u")

    assert data == {"answer": "from local"}
    assert response.provider == "local"
    assert [e.path for e in server.seen] == [
        "/anthropic/messages", "/groq/chat/completions", "/local/chat/completions",
    ]
    assert "LLM provider anthropic failed, trying next" in caplog.text
    assert "LLM provider groq failed, trying next" in caplog.text


def test_complete_json_reports_every_problem_when_all_providers_fail(tmp_path, monkeypatch, server):
    build_sandbox(tmp_path, monkeypatch)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", KEY)
    cfg = _reconfigure(tmp_path, llm={"groq": {"base_url": f"{server.url}/groq", "max_retries": 0}})
    server.respond(403, "forbidden")

    with pytest.raises(LLMError) as info:
        complete_json(cfg, "s", "u")

    message = str(info.value)
    assert message.startswith("all LLM providers failed:")
    assert "- anthropic: ANTHROPIC_API_KEY not set" in message
    assert "- groq: groq request failed: HTTP 403" in message
    assert KEY not in message


def test_complete_json_moves_on_from_prose_and_charges_the_wasted_call_to_the_answer(
    tmp_path, monkeypatch, server,
):
    build_sandbox(tmp_path, monkeypatch)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", KEY)
    cfg = _reconfigure(tmp_path, llm={
        "providers": ["groq", "local"],
        "groq": {"base_url": f"{server.url}/groq", "max_retries": 0},
        "local": {"enabled": True, "base_url": f"{server.url}/local", "api_key_env": "", "max_retries": 0},
    })
    server.respond_json(_chat_reply("Sorry, I cannot produce that.", prompt_tokens=1_000_000, completion_tokens=0))
    server.respond_json(_chat_reply('{"answer": 1}'))

    data, response = complete_json(cfg, "s", "u")

    assert data == {"answer": 1}
    assert response.provider == "local"
    rate_in = float(cfg.get("llm.groq.usd_per_million_input_tokens"))
    assert response.cost_usd == pytest.approx(rate_in)


def test_complete_json_fails_when_the_only_answer_is_prose(local_cfg, server):
    cfg = _reconfigure(local_cfg.root, llm={"providers": ["local"]})
    server.respond_json(_chat_reply("Absolutely! Here is nothing."))

    with pytest.raises(LLMError) as info:
        complete_json(cfg, "s", "u")

    assert "- local: model did not return parseable JSON" in str(info.value)
