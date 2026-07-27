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
from ..runtime import is_offline, resolve_local_model
from .voiceprint import named_speakers

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
    # A token is only needed to DOWNLOAD the model. Once the weights are on
    # disk, requiring one would make offline diarization impossible for no
    # reason.
    model = cfg.get("diarization.pyannote.model", "pyannote/speaker-diarization-3.1")
    _target, local = resolve_local_model(cfg, model, "diarization")
    if local:
        return True, "ready (local weights)"
    if is_offline(cfg):
        return False, (
            f"runtime.offline is on and '{model}' is not in runtime.models_dir. "
            "Fetch it with scripts/fetch_models.py on a networked machine."
        )
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

    target, local = resolve_local_model(cfg, model, "diarization")
    log.info("loading diarization pipeline %s%s", target, " (local)" if local else "")
    if local:
        pipe = Pipeline.from_pretrained(target)
    else:
        pipe = Pipeline.from_pretrained(
            target, use_auth_token=os.environ.get(token_env, "").strip() or None
        )

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


def speaker_turns(audio_path: Path, cfg) -> list[Segment]:
    """
    Who spoke when, with no transcript involved.

    `speakers identify` needs the clusters but not the words: it is answering
    "is one of these voices Marcus", and transcribing an enrollment clip to find
    out would mean loading Whisper for nothing. The segments come back with
    empty text, which is exactly what the identification path reads.
    """
    ok, why = _available(cfg)
    if not ok:
        raise DiarizationError(why)

    pipe = _load_pipeline(cfg)
    kwargs = {}
    min_s = cfg.get("diarization.min_speakers")
    max_s = cfg.get("diarization.max_speakers")
    if min_s:
        kwargs["min_speakers"] = int(min_s)
    if max_s:
        kwargs["max_speakers"] = int(max_s)

    annotation = pipe(str(audio_path), **kwargs)
    return [
        Segment(start=turn.start, end=turn.end, text="", speaker=str(label))
        for turn, _, label in annotation.itertracks(yield_label=True)
    ]


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

    # Put names on the clusters we recognise. This runs before the owner and
    # "Speaker N" fallbacks below so that a person who has been enrolled keeps
    # their name even when they are the one doing most of the talking.
    try:
        mapping: dict[str, str] = dict(named_speakers(audio_path, segments, cfg))
    except Exception as exc:  # noqa: BLE001
        # Recognising nobody is a worse transcript, not a broken one.
        log.warning("speaker identification failed, continuing unnamed: %s", exc)
        mapping = {}

    # The device sits on your body, so the most-present voice is almost always
    # the wearer. Relabel for readability.
    if cfg.get("diarization.assume_owner_is_dominant_speaker", True):
        owner = cfg.get("diarization.owner_label", "Owner")
        dominant = _dominant_speaker(segments)
        # If the wearer is enrolled under their own name, that name already won
        # above and applying the owner label on top would rename them twice.
        if dominant and dominant not in mapping and owner not in mapping.values():
            mapping[dominant] = owner
        used = set(mapping.values())
        counter = 1
        for seg in segments:
            if seg.speaker in mapping:
                continue
            while f"Speaker {counter}" in used:
                counter += 1
            mapping[seg.speaker] = f"Speaker {counter}"
            used.add(f"Speaker {counter}")

    # Clusters nobody recognised and no rule renamed keep the label diarization
    # gave them, which is what this did before any of the naming existed.
    if mapping:
        for seg in segments:
            seg.speaker = mapping.get(seg.speaker, seg.speaker)

    log.info("diarized into %d speakers", len({s.speaker for s in segments}))
    return segments
