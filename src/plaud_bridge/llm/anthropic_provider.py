"""
Anthropic Messages API backend.

Two things here are model-generation specific and worth knowing before editing.

Sampling parameters are not sent. `temperature` used to be pinned to 0.0 for
determinism, which the current models reject outright -- they removed the
sampling parameters, so sending one is a 400 rather than a no-op. A zero
temperature never guaranteed identical output anyway; the output contract in the
extraction prompt does that work.

The system prompt is sent as a cached block. A profile's system prompt, persona,
and schema are byte-identical across every recording and every episode of every
recording, so without this the same few thousand tokens are paid for at full
price forever. Cache reads bill at a fraction of fresh input.

Caching is a prefix match, which is the constraint to respect when changing this
file: the stable half of the prompt has to come first, and a single changed byte
ahead of the marker invalidates everything after it. That is why the transcript
travels in the user turn and never inside the system block.
"""

from __future__ import annotations

import os
from typing import Any

from ..http_util import HttpError, post_json
from ..logging_setup import get
from .base import LLMError, LLMProvider, LLMResponse

log = get("llm.anthropic")


class AnthropicLLM(LLMProvider):
    name = "anthropic"
    is_cloud = True

    def __init__(self, cfg):
        super().__init__(cfg)
        self.base_url = cfg.get("llm.anthropic.base_url", "https://api.anthropic.com/v1")
        self.model = cfg.get("llm.anthropic.model", "claude-opus-5")
        self.key_env = cfg.get("llm.anthropic.api_key_env", "ANTHROPIC_API_KEY")
        self.version = cfg.get("llm.anthropic.version_header", "2023-06-01")
        # Thinking is on by default on the current models and shares this ceiling
        # with the response text, so a budget sized around the answer alone
        # truncates mid-JSON.
        self.max_tokens = int(cfg.get("llm.anthropic.max_tokens", 16000))
        self.effort = str(cfg.get("llm.anthropic.effort", "medium") or "").strip()
        self.cache_system = bool(cfg.get("llm.anthropic.cache_system_prompt", True))
        self.timeout = int(cfg.get("llm.anthropic.timeout_seconds", 180))
        self.retries = int(cfg.get("llm.anthropic.max_retries", 4))

    def available(self) -> tuple[bool, str]:
        if not self.cfg.get("llm.anthropic.enabled", False):
            return False, "disabled in config"
        if not os.environ.get(self.key_env, "").strip():
            return False, f"{self.key_env} not set"
        return True, "ready"

    def complete(self, system: str, user: str, max_tokens: int | None = None) -> LLMResponse:
        ok, why = self.available()
        if not ok:
            raise LLMError(f"anthropic unavailable: {why}")

        system_block: Any = system
        if self.cache_system and system.strip():
            system_block = [{
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }]

        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "system": system_block,
            "messages": [{"role": "user", "content": user}],
        }
        if self.effort:
            # Depth, and the main cost lever now that there is no token budget to
            # set. Extraction is a read-and-fill task rather than a reasoning
            # one, so it does not want the ceiling.
            payload["output_config"] = {"effort": self.effort}

        try:
            data = post_json(
                f"{self.base_url}/messages",
                payload,
                headers={
                    "x-api-key": os.environ[self.key_env].strip(),
                    "anthropic-version": self.version,
                },
                timeout=self.timeout,
                max_retries=self.retries,
            )
        except HttpError as exc:
            raise LLMError(f"anthropic request failed: {exc} :: {exc.body[:300]}") from exc

        text = "".join(
            block.get("text", "")
            for block in data.get("content", [])
            if block.get("type") == "text"
        )
        usage = data.get("usage", {}) or {}
        # Cache reads and writes are billed separately and are not the same rate
        # as fresh input. Counting them as plain input tokens is closer to the
        # truth than dropping them, and this is a guardrail, not an invoice.
        input_tokens = (
            int(usage.get("input_tokens", 0))
            + int(usage.get("cache_creation_input_tokens", 0))
            + int(usage.get("cache_read_input_tokens", 0))
        )
        output_tokens = int(usage.get("output_tokens", 0))
        cached = int(usage.get("cache_read_input_tokens", 0))
        if cached:
            log.debug("%d input token(s) served from cache", cached)
        return LLMResponse(
            text=text,
            provider=self.name,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=self.price(input_tokens, output_tokens),
            raw=data,
        )
