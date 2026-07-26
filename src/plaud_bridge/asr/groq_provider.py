"""
Groq Whisper backend.

Fast and cheap. Not for anything a compliance gate marks sensitive: this sends
audio to a third party, and no BAA is in play at the standard developer tier.
Verify current pricing at groq.com/pricing before trusting the cost estimate.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..http_util import HttpError, post_multipart
from ..logging_setup import get
from ..models import Segment
from .base import ASRError, ASRProvider, ASRResult

log = get("asr.groq")


class GroqASR(ASRProvider):
    name = "groq"
    is_cloud = True

    def __init__(self, cfg, glossary=None):
        super().__init__(cfg, glossary)
        self.base_url = cfg.get("asr.groq.base_url", "https://api.groq.com/openai/v1")
        self.model = cfg.get("asr.groq.model", "whisper-large-v3-turbo")
        self.key_env = cfg.get("asr.groq.api_key_env", "GROQ_API_KEY")
        self.timeout = int(cfg.get("asr.groq.timeout_seconds", 300))
        self.retries = int(cfg.get("asr.groq.max_retries", 4))
        self.temperature = float(cfg.get("asr.groq.temperature", 0.0))
        self.usd_per_hour = float(cfg.get("asr.groq.usd_per_audio_hour", 0.04))

    def available(self) -> tuple[bool, str]:
        if not self.cfg.get("asr.groq.enabled", False):
            return False, "disabled in config"
        if not os.environ.get(self.key_env, "").strip():
            return False, f"{self.key_env} not set"
        return True, "ready"

    def transcribe_file(self, path: Path, offset: float = 0.0,
                        language: str | None = None) -> ASRResult:
        ok, why = self.available()
        if not ok:
            raise ASRError(f"groq ASR unavailable: {why}")

        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > 24.5:
            raise ASRError(
                f"{path.name} is {size_mb:.1f}MB; the endpoint caps near 25MB. "
                "Lower audio.max_chunk_mb in pipeline.yaml."
            )

        fields = {
            "model": self.model,
            "response_format": "verbose_json",
            "temperature": str(self.temperature),
            "timestamp_granularities[]": "segment",
        }
        if language:
            fields["language"] = language
        hint = self.prompt()
        if hint:
            fields["prompt"] = hint

        try:
            data = post_multipart(
                f"{self.base_url}/audio/transcriptions",
                fields=fields,
                file_path=path,
                file_field="file",
                headers={"Authorization": f"Bearer {os.environ[self.key_env].strip()}"},
                timeout=self.timeout,
                max_retries=self.retries,
            )
        except HttpError as exc:
            raise ASRError(f"groq transcription failed: {exc} :: {exc.body[:300]}") from exc

        segments: list[Segment] = []
        for raw in data.get("segments") or []:
            text = str(raw.get("text", "")).strip()
            if not text:
                continue
            segments.append(
                Segment(
                    start=float(raw.get("start", 0.0)) + offset,
                    end=float(raw.get("end", 0.0)) + offset,
                    text=text,
                    confidence=raw.get("avg_logprob"),
                )
            )

        reported = float(data.get("duration", 0.0) or 0.0)

        if not segments and data.get("text"):
            # Whole-file fallback: the endpoint returned text but no segment
            # breakdown. A zero-width segment here would make the span
            # undiarizable and drive the billed duration to zero, so give it the
            # real length of the chunk.
            span = reported if reported > 0 else 0.0
            segments.append(Segment(offset, offset + span, str(data["text"]).strip()))

        duration = reported or (segments[-1].end - offset if segments else 0.0)
        cost = (duration / 3600.0) * self.usd_per_hour

        log.info("groq transcribed %s (%.0fs, ~$%.5f)", path.name, duration, cost)
        return ASRResult(
            segments=segments,
            language=str(data.get("language", language or "en")),
            provider=self.name,
            model=self.model,
            cost_usd=cost,
        )
