"""
ASR provider chain.

The chain is a fallback list, not a preference list. Two rules govern it:

1. If the compliance gate says local-only, cloud providers are removed from the
   chain entirely. Not deprioritised. Removed.
2. If every remaining provider is unavailable, we fail loudly. We do not fall
   back to a cloud provider that was excluded on compliance grounds. That
   would defeat the entire point of the exclusion.
"""

from __future__ import annotations

from ..logging_setup import get
from ..models import Segment, Transcript
from .base import ASRError, ASRProvider, ASRResult
from .groq_provider import GroqASR
from .local_provider import LocalWhisperASR
from .stitch import stitch

log = get("asr")

_PROVIDERS: dict[str, type[ASRProvider]] = {
    "groq": GroqASR,
    "local": LocalWhisperASR,
}


def build_asr_chain(cfg, glossary=None, local_only: bool = False) -> list[ASRProvider]:
    chain: list[ASRProvider] = []
    for name in cfg.get("asr.providers", []) or []:
        klass = _PROVIDERS.get(name)
        if klass is None:
            log.warning("unknown ASR provider '%s' in config; skipping", name)
            continue
        provider = klass(cfg, glossary)
        if local_only and provider.is_cloud:
            log.info("excluding cloud ASR provider '%s': compliance requires local processing", name)
            continue
        chain.append(provider)
    return chain


def transcribe(chunks, cfg, glossary=None, local_only: bool = False,
               language: str | None = None) -> Transcript:
    """Run the chunk list through the first usable provider in the chain."""
    chain = build_asr_chain(cfg, glossary, local_only)
    if not chain:
        raise ASRError(
            "no ASR provider available. "
            + ("Compliance requires local processing and no local provider is "
               "configured or installed. Run: pip install faster-whisper"
               if local_only else "Check asr.providers in pipeline.yaml.")
        )

    problems: list[str] = []
    # Survives the provider loop so a partial run that failed over to the next
    # provider still reports what the first one charged.
    spent: dict[str, float] = {}
    for provider in chain:
        ok, why = provider.available()
        if not ok:
            problems.append(f"{provider.name}: {why}")
            continue

        try:
            per_chunk: list[list[Segment]] = []
            overlaps: list[float] = []
            starts: list[float] = []
            detected = language or "en"

            for chunk in chunks:
                result: ASRResult = provider.transcribe_file(
                    chunk.path, offset=chunk.start, language=language
                )
                per_chunk.append(result.segments)
                overlaps.append(chunk.overlap_lead)
                starts.append(chunk.start)
                # Bill as we go. Initialising the total inside the try meant a
                # provider that failed on chunk 6 discarded what chunks 0-5 had
                # already cost, so the spend ceiling under-counted on exactly
                # the retry-heavy runs it exists to stop.
                spent[provider.name] = spent.get(provider.name, 0.0) + result.cost_usd
                detected = result.language or detected

            segments = stitch(per_chunk, overlaps, starts)
            duration = max((s.end for s in segments), default=0.0)
            total_cost = sum(spent.values())

            log.info("transcribed via %s: %d segments, %.0fs, $%.5f",
                     provider.name, len(segments), duration, total_cost)

            return Transcript(
                segments=segments,
                language=detected,
                asr_provider=provider.name,
                asr_model=getattr(provider, "model", getattr(provider, "model_name", "")),
                duration_seconds=duration,
                cost_usd=total_cost,
            )
        except ASRError as exc:
            problems.append(f"{provider.name}: {exc}")
            log.warning("ASR provider %s failed, trying next: %s", provider.name, exc)

    raise ASRError("all ASR providers failed:\n  - " + "\n  - ".join(problems))
