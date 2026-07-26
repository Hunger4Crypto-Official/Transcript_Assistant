"""ASR provider interface. One contract, swappable backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from ..models import Segment


class ASRError(RuntimeError):
    pass


@dataclass
class ASRResult:
    segments: list[Segment] = field(default_factory=list)
    language: str = "en"
    provider: str = ""
    model: str = ""
    cost_usd: float = 0.0


class ASRProvider(ABC):
    """
    Contract for every ASR backend.

    `is_cloud` matters more than it looks. The compliance gate uses it to veto
    providers for sensitive recordings, so getting it wrong on a new provider
    would silently defeat the whole local-only guarantee.
    """

    name: str = "base"
    is_cloud: bool = True

    def __init__(self, cfg, glossary=None):
        self.cfg = cfg
        self.glossary = glossary

    @abstractmethod
    def available(self) -> tuple[bool, str]:
        """Return (usable, reason). Never raises."""

    @abstractmethod
    def transcribe_file(self, path: Path, offset: float = 0.0,
                        language: str | None = None) -> ASRResult:
        """Transcribe one prepared audio file. Timestamps shifted by `offset`."""

    def prompt(self) -> str:
        return self.glossary.asr_prompt() if self.glossary else ""
