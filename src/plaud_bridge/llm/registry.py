"""
LLM provider chain.

Same rule as ASR: when compliance says local-only, cloud providers are removed
from the chain, not merely deprioritised. If nothing local is configured, the
analysis fails loudly rather than quietly shipping a private conversation to a
third party.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..logging_setup import get
from .anthropic_provider import AnthropicLLM
from .base import LLMError, LLMProvider, LLMResponse, extract_json
from .openai_compat_provider import OpenAICompatLLM

log = get("llm")


def _build(cfg, key: str) -> LLMProvider | None:
    if key == "anthropic":
        return AnthropicLLM(cfg)
    if cfg.get(f"llm.{key}") is None:
        log.warning("unknown LLM provider '%s' in config; skipping", key)
        return None
    return OpenAICompatLLM(cfg, key, is_cloud=bool(cfg.get(f"llm.{key}.is_cloud", True)))


def build_llm_chain(cfg, local_only: bool = False) -> list[LLMProvider]:
    chain: list[LLMProvider] = []
    names = list(cfg.get("llm.providers", []) or [])
    if local_only and "local" not in names and cfg.get("llm.local") is not None:
        names.append("local")   # make the local option reachable even if unlisted
    for name in names:
        provider = _build(cfg, name)
        if provider is None:
            continue
        if local_only and provider.is_cloud:
            log.info("excluding cloud LLM '%s': compliance requires local processing", name)
            continue
        chain.append(provider)
    return chain


def complete_json(cfg, system: str, user: str, local_only: bool = False,
                  max_tokens: int | None = None) -> tuple[dict[str, Any], LLMResponse]:
    chain = build_llm_chain(cfg, local_only)
    if not chain:
        raise LLMError(
            "no LLM provider available. "
            + ("Compliance requires local processing. Enable llm.local in "
               "pipeline.yaml and point it at a local server (ollama, vLLM)."
               if local_only else "Check llm.providers in pipeline.yaml.")
        )

    problems: list[str] = []
    # A provider that answered but returned unparseable JSON still billed for the
    # call. If the chain then falls through to another provider, that earlier
    # spend has to travel with it or the recording's cost silently undercounts
    # every wasted attempt -- which is exactly the spend a guardrail watching for
    # a runaway loop needs to see.
    wasted = 0.0
    for provider in chain:
        ok, why = provider.available()
        if not ok:
            problems.append(f"{provider.name}: {why}")
            continue
        try:
            response = provider.complete(system, user, max_tokens)
        except LLMError as exc:
            problems.append(f"{provider.name}: {exc}")
            log.warning("LLM provider %s failed, trying next: %s", provider.name, exc)
            continue
        try:
            data = extract_json(response.text)
        except LLMError as exc:
            # The call succeeded and was billed; only the body was unusable.
            wasted += response.cost_usd
            problems.append(f"{provider.name}: {exc}")
            log.warning("LLM provider %s returned unparseable JSON, trying next: %s",
                        provider.name, exc)
            continue
        if wasted:
            response = replace(response, cost_usd=response.cost_usd + wasted)
        return data, response

    raise LLMError("all LLM providers failed:\n  - " + "\n  - ".join(problems))
