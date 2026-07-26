"""LLM provider interface plus tolerant JSON extraction."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..logging_setup import get

_log = get("llm.pricing")


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResponse:
    text: str = ""
    provider: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json(text: str) -> dict[str, Any]:
    """
    Pull a JSON object out of a model response.

    Models add preambles, fences, and trailing commentary no matter how firmly
    you instruct otherwise. Rather than pretending they will not, we handle it:
    try the whole string, then fenced blocks, then the outermost brace pair.
    """
    candidates: list[str] = [text.strip()]

    for match in _FENCE.finditer(text):
        candidates.append(match.group(1).strip())

    first, last = text.find("{"), text.rfind("}")
    if first != -1 and last > first:
        candidates.append(text[first : last + 1])

    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    raise LLMError(
        "model did not return parseable JSON. First 400 chars: " + text[:400].replace("\n", " ")
    )


_unpriced_warned: set[str] = set()


class LLMProvider(ABC):
    name: str = "base"
    is_cloud: bool = True

    def __init__(self, cfg, config_prefix: str | None = None):
        self.cfg = cfg
        # Where this provider's own settings live. OpenAI-compatible providers
        # share one class across several config blocks, so it cannot be derived
        # from the class alone.
        self.config_prefix = config_prefix or f"llm.{self.name}"

    def price(self, input_tokens: int, output_tokens: int) -> float:
        """
        Cost in USD from the provider's own reported token usage.

        Rates live in config so a price change is a YAML edit. A provider with
        no configured rate returns 0.0 and says so once per process: a guessed
        number inside a spend guardrail is worse than a visible zero, because
        the zero is at least honest about not knowing.
        """
        rate_in = float(self.cfg.get(f"{self.config_prefix}.usd_per_million_input_tokens", 0) or 0)
        rate_out = float(self.cfg.get(f"{self.config_prefix}.usd_per_million_output_tokens", 0) or 0)

        if rate_in <= 0 and rate_out <= 0:
            if self.name not in _unpriced_warned:
                _unpriced_warned.add(self.name)
                _log.warning(
                    "LLM provider '%s' has no usd_per_million_input_tokens / "
                    "usd_per_million_output_tokens configured. Its spend will read as "
                    "$0.00 and will not count toward cost.halt_usd_per_run.",
                    self.name,
                )
            return 0.0

        return (max(0, input_tokens) / 1_000_000.0) * rate_in + (
            max(0, output_tokens) / 1_000_000.0
        ) * rate_out

    @abstractmethod
    def available(self) -> tuple[bool, str]:
        """Return (usable, reason). Never raises."""

    @abstractmethod
    def complete(self, system: str, user: str, max_tokens: int | None = None) -> LLMResponse:
        """Single-turn completion."""
