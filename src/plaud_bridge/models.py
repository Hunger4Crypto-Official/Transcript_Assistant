"""
Core data structures.

Deliberately stdlib-only dataclasses rather than pydantic. This tool should
still run in five years on a machine where you have not touched pip since.
Validation lives in config.py where it can produce useful error messages.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def format_stamp(seconds: float) -> str:
    """MM:SS, or HH:MM:SS once it runs past an hour. One copy, used everywhere."""
    minutes, secs = divmod(int(max(0.0, seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


class Sensitivity(str, Enum):
    """Ordered. Higher ordinal wins when profiles collide on one recording."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"

    @property
    def rank(self) -> int:
        return {"low": 0, "medium": 1, "high": 2, "maximum": 3}[self.value]


class Stage(str, Enum):
    INGESTED = "ingested"
    PREPARED = "prepared"
    TRANSCRIBED = "transcribed"
    DIARIZED = "diarized"
    CORRECTED = "corrected"
    ROUTED = "routed"
    ANALYZED = "analyzed"
    COMPLETE = "complete"
    QUARANTINED = "quarantined"
    FAILED = "failed"


class ConsentStatus(str, Enum):
    DETECTED = "detected"
    NOT_DETECTED = "not_detected"
    NOT_REQUIRED = "not_required"
    WAIVED = "waived"


@dataclass
class Segment:
    """One diarized, timestamped span of speech."""

    start: float
    end: float
    text: str
    speaker: str = "SPEAKER_00"
    # Average log probability from the recogniser. Clean speech sits above about
    # -0.5; well below -1.0 the model is guessing at what it heard.
    confidence: float | None = None
    # The recogniser's own probability that this span held no speech at all.
    # High no-speech alongside fluent text is the hallucination signature.
    no_speech: float | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def stamp(self) -> str:
        return format_stamp(self.start)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Segment:
        return cls(
            start=float(d["start"]),
            end=float(d["end"]),
            text=str(d.get("text", "")),
            speaker=str(d.get("speaker", "SPEAKER_00")),
            confidence=d.get("confidence"),
            no_speech=d.get("no_speech"),
        )


@dataclass
class Transcript:
    segments: list[Segment] = field(default_factory=list)
    language: str = "en"
    asr_provider: str = ""
    asr_model: str = ""
    duration_seconds: float = 0.0
    cost_usd: float = 0.0
    # How much of this the recogniser appears to have been guessing at. See
    # asr/confidence.py; empty for imported text, which has nothing to score.
    confidence_report: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n".join(s.text.strip() for s in self.segments if s.text.strip())

    @property
    def speakers(self) -> list[str]:
        seen: list[str] = []
        for s in self.segments:
            if s.speaker not in seen:
                seen.append(s.speaker)
        return seen

    def labelled_text(self, max_chars: int | None = None) -> str:
        """Speaker-attributed, timestamped rendering. This is what the LLM sees."""
        lines: list[str] = []
        current: str | None = None
        for seg in self.segments:
            body = seg.text.strip()
            if not body:
                continue
            if seg.speaker != current:
                lines.append(f"\n[{seg.stamp()}] {seg.speaker}:")
                current = seg.speaker
            lines.append(body)
        out = "\n".join(lines).strip()
        if max_chars and len(out) > max_chars:
            head = out[: int(max_chars * 0.7)]
            tail = out[-int(max_chars * 0.25) :]
            out = f"{head}\n\n[... {len(out) - len(head) - len(tail)} characters elided ...]\n\n{tail}"
        return out

    def window(self, seconds: float) -> str:
        """Opening window. Used by the consent detector."""
        return " ".join(s.text for s in self.segments if s.start <= seconds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "segments": [s.to_dict() for s in self.segments],
            "language": self.language,
            "asr_provider": self.asr_provider,
            "asr_model": self.asr_model,
            "duration_seconds": self.duration_seconds,
            "cost_usd": self.cost_usd,
            "confidence_report": self.confidence_report,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Transcript:
        return cls(
            segments=[Segment.from_dict(s) for s in d.get("segments", [])],
            language=d.get("language", "en"),
            asr_provider=d.get("asr_provider", ""),
            asr_model=d.get("asr_model", ""),
            duration_seconds=float(d.get("duration_seconds", 0.0)),
            cost_usd=float(d.get("cost_usd", 0.0)),
            confidence_report=d.get("confidence_report") or {},
        )


@dataclass
class RouteMatch:
    profile_id: str
    confidence: float
    keyword_score: float = 0.0
    llm_score: float = 0.0
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ComplianceVerdict:
    """The gate's decision. `allow` false means nothing downstream runs."""

    allow: bool = True
    consent: ConsentStatus = ConsentStatus.NOT_REQUIRED
    consent_quote: str = ""
    consent_timestamp: float | None = None
    force_local_processing: bool = False

    # Whether this recording's content is encrypted at rest, and therefore
    # whether the plain SQLite index may hold a copy of it.
    #
    # Defaults to True, which looks paranoid and is not. This verdict is only
    # filled in once the compliance gate has run, and a recording can fail
    # between transcription and the gate -- a provider outage, a malformed
    # profile, anything. With a False default, that failure wrote the complete
    # plaintext transcript of a family conversation into an unencrypted file
    # and left it there forever. Assuming encryption until a profile says
    # otherwise costs nothing and closes that window.
    encrypt_at_rest: bool = True
    redactions: dict[str, int] = field(default_factory=dict)
    governing_profile: str = ""
    governing_sensitivity: Sensitivity = Sensitivity.LOW
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["consent"] = self.consent.value
        d["governing_sensitivity"] = self.governing_sensitivity.value
        return d


@dataclass
class ProfileAnalysis:
    profile_id: str
    fields: dict[str, Any] = field(default_factory=dict)
    llm_provider: str = ""
    llm_model: str = ""
    cost_usd: float = 0.0
    requires_human_attention: bool = False
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Recording:
    """The unit of work. One source file, start to finish."""

    id: str = field(default_factory=lambda: new_id("rec"))
    source_path: str = ""
    source_name: str = ""
    content_hash: str = ""
    size_bytes: int = 0
    kind: str = "audio"  # audio | text
    recorded_at: datetime | None = None
    ingested_at: datetime = field(default_factory=utc_now)
    stage: Stage = Stage.INGESTED
    duration_seconds: float = 0.0

    transcript: Transcript | None = None
    routes: list[RouteMatch] = field(default_factory=list)
    # Set by the pipeline. Typed loosely to keep models.py free of imports from
    # the modules that depend on it.
    episodes: list[Any] = field(default_factory=list)
    compliance: ComplianceVerdict = field(default_factory=ComplianceVerdict)
    analyses: list[ProfileAnalysis] = field(default_factory=list)

    total_cost_usd: float = 0.0
    errors: list[str] = field(default_factory=list)
    artifact_paths: dict[str, str] = field(default_factory=dict)

    @property
    def profile_ids(self) -> list[str]:
        return [r.profile_id for r in self.routes]

    @property
    def is_encrypted(self) -> bool:
        """
        One answer, from one place.

        This used to be derived from sensitivity while `_persist` decided using
        the profile's `encrypt_at_rest` flag. Two sources of truth for the same
        question, and a perfectly legal profile -- high sensitivity with
        encryption turned off -- made them disagree: the artifacts were written
        in plaintext while the index withheld them as though encrypted, so the
        digest could never render that recording again.
        """
        return self.compliance.encrypt_at_rest

    def analysis_for(self, profile_id: str) -> ProfileAnalysis | None:
        return next((a for a in self.analyses if a.profile_id == profile_id), None)

    def to_dict(self, include_transcript: bool = True,
                include_analysis_fields: bool = True) -> dict[str, Any]:
        """
        Serialise the recording.

        The same payload is written to two very different places: the encrypted
        vault artifact, which should hold everything, and the SQLite index,
        which is a plain file on disk. Writing the verbatim transcript or the
        extracted quotes into the index would leave an unencrypted copy of a
        maximum-sensitivity conversation sitting beside the encrypted one, which
        makes the vault decorative.

        Both flags drop content while keeping the metadata, so the index stays
        useful for search, counts, and cost. Anything dropped is marked, so a
        reader can tell "nothing was extracted" apart from "this is withheld"
        and go open the vault copy.
        """
        if self.transcript is None:
            transcript: dict[str, Any] | None = None
        elif include_transcript:
            transcript = self.transcript.to_dict()
        else:
            transcript = {
                "segments": [],
                "segments_withheld": len(self.transcript.segments),
                "language": self.transcript.language,
                "asr_provider": self.transcript.asr_provider,
                "asr_model": self.transcript.asr_model,
                "duration_seconds": self.transcript.duration_seconds,
                "cost_usd": self.transcript.cost_usd,
                "speakers": self.transcript.speakers,
                # Whether the transcript is trustworthy is metadata about it,
                # not content, and the digest reads the index rather than
                # opening the vault -- so dropping this would mean the one
                # warning that has to be read before the summary is the one
                # warning an encrypted recording never shows.
                #
                # `worst` is the exception: those are verbatim passages, and a
                # sample of a maximum-sensitivity conversation sitting in a
                # plain file is the thing this whole method exists to prevent.
                "confidence_report": {
                    k: v for k, v in (self.transcript.confidence_report or {}).items()
                    if k != "worst"
                },
            }

        analyses: list[dict[str, Any]] = []
        for analysis in self.analyses:
            entry = analysis.to_dict()
            if not include_analysis_fields:
                entry["fields"] = {}
                entry["fields_withheld"] = True
            analyses.append(entry)

        return {
            "id": self.id,
            "source_path": self.source_path,
            "source_name": self.source_name,
            "content_hash": self.content_hash,
            "size_bytes": self.size_bytes,
            "kind": self.kind,
            "recorded_at": self.recorded_at.isoformat() if self.recorded_at else None,
            "ingested_at": self.ingested_at.isoformat(),
            "stage": self.stage.value,
            "duration_seconds": self.duration_seconds,
            "transcript": transcript,
            "routes": [r.to_dict() for r in self.routes],
            "episodes": [e.to_dict() for e in self.episodes],
            "compliance": self.compliance.to_dict(),
            "analyses": analyses,
            "total_cost_usd": self.total_cost_usd,
            "errors": self.errors,
            "artifact_paths": self.artifact_paths,
        }

    def to_json(self, indent: int = 2, include_transcript: bool = True,
                include_analysis_fields: bool = True) -> str:
        return json.dumps(
            self.to_dict(
                include_transcript=include_transcript,
                include_analysis_fields=include_analysis_fields,
            ),
            indent=indent,
            ensure_ascii=False,
        )


@dataclass
class RunStats:
    started_at: datetime = field(default_factory=utc_now)
    processed: int = 0
    skipped: int = 0
    quarantined: int = 0
    failed: int = 0
    audio_seconds: float = 0.0
    cost_usd: float = 0.0

    def summary(self) -> str:
        mins = self.audio_seconds / 60.0
        return (
            f"processed={self.processed} skipped={self.skipped} "
            f"quarantined={self.quarantined} failed={self.failed} "
            f"audio={mins:.1f}min cost=${self.cost_usd:.4f}"
        )
