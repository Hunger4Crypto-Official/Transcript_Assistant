"""
OpenAI-compatible chat completions backend.

Covers Groq and any local server that speaks the same shape (ollama, vLLM,
llama.cpp server). One class, two config blocks, because the wire format is
identical and duplicating it would just be two places to fix a bug.
"""

from __future__ import annotations

import os

from ..http_util import HttpError, post_json
from ..logging_setup import get
from .base import LLMError, LLMProvider, LLMResponse

log = get("llm.openai_compat")


class OpenAICompatLLM(LLMProvider):
    def __init__(self, cfg, key: str, is_cloud: bool):
        super().__init__(cfg, config_prefix=f"llm.{key}")
        self.name = key
        self.is_cloud = is_cloud
        prefix = f"llm.{key}"
        self.base_url = cfg.get(f"{prefix}.base_url", "")
        self.model = cfg.get(f"{prefix}.model", "")
        self.key_env = cfg.get(f"{prefix}.api_key_env", "")
        self.max_tokens = int(cfg.get(f"{prefix}.max_tokens", 8000))
        self.temperature = float(cfg.get(f"{prefix}.temperature", 0.0))
        self.timeout = int(cfg.get(f"{prefix}.timeout_seconds", 180))
        self.retries = int(cfg.get(f"{prefix}.max_retries", 4))
        self._enabled = bool(cfg.get(f"{prefix}.enabled", False))

    def available(self) -> tuple[bool, str]:
        if not self._enabled:
            return False, "disabled in config"
        if not self.base_url or not self.model:
            return False, "base_url or model missing from config"
        if self.key_env and not os.environ.get(self.key_env, "").strip():
            return False, f"{self.key_env} not set"
        return True, "ready"

    def complete(self, system: str, user: str, max_tokens: int | None = None) -> LLMResponse:
        ok, why = self.available()
        if not ok:
            raise LLMError(f"{self.name} unavailable: {why}")

        headers = {}
        if self.key_env:
            headers["Authorization"] = f"Bearer {os.environ[self.key_env].strip()}"

        payload = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
        }
        try:
            data = post_json(
                f"{self.base_url}/chat/completions",
                payload,
                headers=headers,
                timeout=self.timeout,
                max_retries=self.retries,
            )
        except HttpError as exc:
            # Not every server supports response_format. Retry once without it
            # rather than failing the whole recording over a schema hint.
            if exc.status in (400, 422):
                payload.pop("response_format", None)
                try:
                    data = post_json(
                        f"{self.base_url}/chat/completions", payload,
                        headers=headers, timeout=self.timeout, max_retries=1,
                    )
                except HttpError as exc2:
                    raise LLMError(f"{self.name} request failed: {exc2} :: {exc2.body[:300]}") from exc2
            else:
                raise LLMError(f"{self.name} request failed: {exc} :: {exc.body[:300]}") from exc

        choices = data.get("choices") or []
        text = choices[0].get("message", {}).get("content", "") if choices else ""
        usage = data.get("usage", {}) or {}
        input_tokens = int(usage.get("prompt_tokens", 0))
        output_tokens = int(usage.get("completion_tokens", 0))
        return LLMResponse(
            text=text,
            provider=self.name,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=self.price(input_tokens, output_tokens),
            raw=data,
        )
