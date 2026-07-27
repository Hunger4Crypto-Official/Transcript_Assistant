"""
Local faster-whisper backend.

This is the one that matters. Every maximum-sensitivity profile depends on it,
so if it is not installed those recordings will not process at all, by design.

    pip install faster-whisper

On Apple Silicon, compute_type "int8" on CPU is usually the pragmatic default.
On an NVIDIA box, device "cuda" with "float16" is dramatically faster.
"""

from __future__ import annotations

from pathlib import Path

from ..logging_setup import get
from ..models import Segment
from ..runtime import is_offline, model_path, require_local
from .base import ASRError, ASRProvider, ASRResult

log = get("asr.local")

_MODEL_CACHE: dict[tuple[str, str, str], object] = {}


class LocalWhisperASR(ASRProvider):
    name = "local"
    is_cloud = False

    def __init__(self, cfg, glossary=None):
        super().__init__(cfg, glossary)
        self.model_name = cfg.get("asr.local.model", "large-v3")
        self.device = cfg.get("asr.local.device", "auto")
        self.compute_type = cfg.get("asr.local.compute_type", "auto")
        self.beam_size = int(cfg.get("asr.local.beam_size", 5))

    def available(self) -> tuple[bool, str]:
        if not self.cfg.get("asr.local.enabled", True):
            return False, "disabled in config"
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            return False, (
                "faster-whisper is not installed. Run: pip install faster-whisper. "
                "Local ASR is required for the father and husband profiles."
            )
        return True, "ready"

    def _resolve_device(self) -> tuple[str, str]:
        device, compute = self.device, self.compute_type
        if device == "auto":
            try:
                import torch  # type: ignore

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        if compute == "auto":
            compute = "float16" if device == "cuda" else "int8"
        return device, compute

    def _model(self):
        from faster_whisper import WhisperModel

        device, compute = self._resolve_device()
        # Resolves to a directory under runtime.models_dir when the weights are
        # on disk. Offline, this raises with the path it wanted rather than
        # quietly reaching for HuggingFace.
        target = require_local(self.cfg, self.model_name, "whisper", "speech recognition")
        offline = is_offline(self.cfg)

        key = (target, device, compute)
        if key not in _MODEL_CACHE:
            log.info("loading faster-whisper %s on %s/%s%s",
                     target, device, compute,
                     "" if offline else " (downloads weights if not cached)")
            _MODEL_CACHE[key] = WhisperModel(
                target, device=device, compute_type=compute,
                download_root=str(model_path(self.cfg, "whisper")),
                local_files_only=offline,
            )
        return _MODEL_CACHE[key]

    def transcribe_file(self, path: Path, offset: float = 0.0,
                        language: str | None = None) -> ASRResult:
        ok, why = self.available()
        if not ok:
            raise ASRError(f"local ASR unavailable: {why}")

        model = self._model()
        try:
            raw_segments, info = model.transcribe(
                str(path),
                language=language,
                beam_size=self.beam_size,
                initial_prompt=self.prompt() or None,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
                word_timestamps=False,
            )
        except Exception as exc:  # noqa: BLE001 - surface the real cause
            raise ASRError(f"faster-whisper failed on {path.name}: {exc}") from exc

        segments: list[Segment] = []
        for seg in raw_segments:
            text = (seg.text or "").strip()
            if not text:
                continue
            segments.append(
                Segment(
                    start=float(seg.start) + offset,
                    end=float(seg.end) + offset,
                    text=text,
                    confidence=getattr(seg, "avg_logprob", None),
                )
            )

        log.info("local transcribed %s (%d segments)", path.name, len(segments))
        return ASRResult(
            segments=segments,
            language=getattr(info, "language", language or "en"),
            provider=self.name,
            model=self.model_name,
            cost_usd=0.0,
        )
