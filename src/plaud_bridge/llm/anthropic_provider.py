"""Anthropic Messages API backend."""

from __future__ import annotations

import os

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
        self.model = cfg.get("llm.anthropic.model", "claude-sonnet-4-6")
        self.key_env = cfg.get("llm.anthropic.api_key_env", "ANTHROPIC_API_KEY")
        self.version = cfg.get("llm.anthropic.version_header", "2023-06-01")
        self.max_tokens = int(cfg.get("llm.anthropic.max_tokens", 8000))
        self.temperature = float(cfg.get("llm.anthropic.temperature", 0.0))
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

        payload = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": self.temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
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
        return LLMResponse(
            text=text,
            provider=self.name,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=self.price(input_tokens, output_tokens),
            raw=data,
        )
