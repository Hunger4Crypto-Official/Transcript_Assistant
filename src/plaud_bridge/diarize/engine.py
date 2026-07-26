"""
Speaker diarization.

This is the part that earns its keep. Whisper alone hands you an
undifferentiated wall of text, and for a two-person sales conversation "who
said what" is not a nicety, it is the entire value.

Runs locally via pyannote. Requires a HuggingFace token and acceptance of the
model licence on the model page. If it is unavailable the pipeline continues
with a single unlabelled speaker rather than failing, because a transcript
without speaker labels still beats no transcript.

Expect this to degrade on heavy crosstalk. Two people talking over each other
is genuinely hard, and no current system handles it cleanly.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..logging_setup import get
from ..models import Segment

log = get("diarize")

_PIPELINE_CACHE: dict[str, object] = {}


class DiarizationError(RuntimeError):
    pass


def _available(cfg) -> tuple[bool, str]:
    if not cfg.get("diarization.enabled", True):
        return False, "disabled in config"
    if cfg.get("diarization.provider", "pyannote") != "pyannote":
        return False, "no diarization provider configured"
    try:
        import pyannote.audio  # noqa: F401
    except ImportError:
        return False, "pyannote.audio is not installed (pip install pyannote.audio)"
    token_env = cfg.get("diarization.pyannote.hf_token_env", "HUGGINGFACE_TOKEN")
    if not os.environ.get(token_env, "").strip():
        return False, f"{token_env} is not set"
    return True, "ready"


def _load_pipeline(cfg):
    from pyannote.audio import Pipeline

    model = cfg.get("diarization.pyannote.model", "pyannote/speaker-diarization-3.1")
    token_env = cfg.get("diarization.pyannote.hf_token_env", "HUGGINGFACE_TOKEN")
    if model in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[model]

    log.info("loading diarization pipeline %s", model)
    pipe = Pipeline.from_pretrained(model, use_auth_token=os.environ[token_env].strip())

    device_pref = cfg.get("diarization.pyannote.device", "auto")
    try:
        import torch

        if device_pref == "auto":
            device_pref = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(torch.device(device_pref))
    except Exception as exc:  # noqa: BLE001
        log.debug("could not move diarization pipeline to device: %s", exc)

    _PIPELINE_CACHE[model] = pipe
    return pipe


def _dominant_speaker(segments: list[Segment]) -> str | None:
    totals: dict[str, float] = {}
    for seg in segments:
        totals[seg.speaker] = totals.get(seg.speaker, 0.0) + seg.duration
    return max(totals, key=totals.get) if totals else None


def diarize(audio_path: Path, segments: list[Segment], cfg) -> list[Segment]:
    """
    Assign a speaker label to every transcript segment.

    Returns the segments with `.speaker` populated. On any failure it returns
    them unchanged with a single default label; it never raises upward.
    """
    ok, why = _available(cfg)
    if not ok:
        log.info("diarization skipped: %s", why)
        for seg in segments:
            seg.speaker = "SPEAKER"
        return segments

    try:
        pipe = _load_pipeline(cfg)
        kwargs = {}
        min_s = cfg.get("diarization.min_speakers")
        max_s = cfg.get("diarization.max_speakers")
        if min_s:
            kwargs["min_speakers"] = int(min_s)
        if max_s:
            kwargs["max_speakers"] = int(max_s)

        annotation = pipe(str(audio_path), **kwargs)
        turns = [
            (turn.start, turn.end, str(label))
            for turn, _, label in annotation.itertracks(yield_label=True)
        ]
    except Exception as exc:  # noqa: BLE001
        log.warning("diarization failed, continuing without speaker labels: %s", exc)
        for seg in segments:
            seg.speaker = "SPEAKER"
        return segments

    if not turns:
        for seg in segments:
            seg.speaker = "SPEAKER"
        return segments

    # Assign each transcript segment the speaker whose turn overlaps it most.
    for seg in segments:
        best_label, best_overlap = "SPEAKER", 0.0
        for start, end, label in turns:
            overlap = min(seg.end, end) - max(seg.start, start)
            if overlap > best_overlap:
                best_label, best_overlap = label, overlap
        seg.speaker = best_label

    # The device sits on your body, so the most-present voice is almost always
    # the wearer. Relabel for readability.
    if cfg.get("diarization.assume_owner_is_dominant_speaker", True):
        owner = cfg.get("diarization.owner_label", "Owner")
        dominant = _dominant_speaker(segments)
        if dominant:
            counter = 1
            mapping: dict[str, str] = {dominant: owner}
            for seg in segments:
                if seg.speaker not in mapping:
                    mapping[seg.speaker] = f"Speaker {counter}"
                    counter += 1
            for seg in segments:
                seg.speaker = mapping[seg.speaker]

    log.info("diarized into %d speakers", len({s.speaker for s in segments}))
    return segments
