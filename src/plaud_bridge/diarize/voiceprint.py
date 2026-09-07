"""
Named speakers.

Diarization tells you that three different people spoke. It cannot tell you
which one is Marcus, because it has never heard Marcus before. This module is
the part that has: you enroll a person once from a clip of them talking, and
from then on their diarized cluster comes back with their name on it.

    python run.py speakers enroll "Marcus" --audio clips/marcus.wav
    python run.py speakers list
    python run.py speakers identify data/inbox/2026-07-14.mp3   # dry run
    python run.py speakers forget "Marcus"

How it works: a speaker-embedding model turns a span of one person's speech
into a vector. Two spans of the same person land close together; two different
people land apart. Enrollment stores the vector. Identification embeds each
diarized cluster and takes the nearest enrolled person, subject to two guards
described at `identify` below.

Three things this deliberately does NOT do:

**It never uploads anything.** The embedding model runs locally, exactly like
diarization does. A voiceprint is biometric data about someone who is often not
the person running this tool, and the whole premise of this project is that
such material does not leave the machine.

**It refuses to store voiceprints in plaintext.** Enrollment requires a working
vault. There is no unencrypted fallback and no flag to ask for one -- a
plaintext voiceprint file is a biometric database sitting in a user directory.

**It never guesses.** An unconfident match stays `Speaker 2`. A wrong name on a
transcript is worse than no name, because a name is believed: it will be read
six months later as fact, quoted in a follow-up, and acted on. Silence about
who spoke is recoverable; a confident lie is not.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..logging_setup import get
from ..models import Segment
from ..runtime import is_offline, resolve_local_model
from ..storage import Vault, VaultError

log = get("voiceprint")

# Sits at the vault root next to the recordings, so one passphrase covers
# everything and there is exactly one thing to back up.
STORE_RELATIVE = "voiceprints"
STORE_AAD = "voiceprints"
STORE_VERSION = 1

DEFAULT_MODEL = "pyannote/embedding"
DEFAULT_THRESHOLD = 0.55
DEFAULT_MARGIN = 0.08
DEFAULT_MIN_SPEECH = 3.0
DEFAULT_MAX_CROPS = 8
# Anything shorter than this carries more room tone than voice.
MIN_CROP_SECONDS = 0.8


class VoiceprintError(RuntimeError):
    """Raised when enrollment or identification cannot proceed honestly."""


def slugify(name: str) -> str:
    """A stable id for a display name. "Marcus O'Neill" -> "marcus-o-neill"."""
    s = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return s


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# =========================================================================
# Vector maths. Pure Python on purpose: the store, the scoring, and every
# test around them stay importable on a machine with no numpy and no torch.
# The vectors are a few hundred floats, so this costs nothing that matters.
# =========================================================================
def normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm <= 0.0:
        raise VoiceprintError("embedding has zero magnitude; the clip is probably silence")
    return [v / norm for v in vector]


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two already-normalised vectors, clamped to [-1, 1]."""
    if len(a) != len(b):
        raise VoiceprintError(
            f"embedding size mismatch ({len(a)} vs {len(b)}). This happens when "
            "diarization.identify.model changed after enrollment. Re-enroll "
            "everyone, or put the old model back."
        )
    return max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b, strict=True))))


def average(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        raise VoiceprintError("no vectors to average")
    width = len(vectors[0])
    if any(len(v) != width for v in vectors):
        raise VoiceprintError("cannot average embeddings of different sizes")
    return normalise([sum(v[i] for v in vectors) / len(vectors) for i in range(width)])


# =========================================================================
# Storage
# =========================================================================
@dataclass
class Sample:
    """One enrollment clip's contribution to a person's voiceprint."""

    vector: list[float]
    source: str = ""
    seconds: float = 0.0
    added_at: str = field(default_factory=_now)

    def to_dict(self) -> dict:
        return {
            "vector": self.vector,
            "source": self.source,
            "seconds": round(self.seconds, 2),
            "added_at": self.added_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Sample:
        return cls(
            vector=[float(x) for x in d.get("vector") or []],
            source=str(d.get("source") or ""),
            seconds=float(d.get("seconds") or 0.0),
            added_at=str(d.get("added_at") or ""),
        )


@dataclass
class Person:
    """
    Everyone the archive can recognise by name.

    Several samples per person rather than one averaged vector, because a
    kitchen and a conference room do not sound alike and averaging across them
    produces a centroid that matches neither well. Scoring takes the best
    sample, so adding a clip can only ever help.
    """

    id: str
    name: str
    samples: list[Sample] = field(default_factory=list)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    @property
    def seconds(self) -> float:
        return sum(s.seconds for s in self.samples)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "samples": [s.to_dict() for s in self.samples],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Person:
        return cls(
            id=str(d.get("id") or ""),
            name=str(d.get("name") or ""),
            samples=[Sample.from_dict(s) for s in d.get("samples") or []],
            created_at=str(d.get("created_at") or ""),
            updated_at=str(d.get("updated_at") or ""),
        )


class VoiceprintStore:
    """
    The enrolled people, encrypted at rest under the vault passphrase.

    Reads are lazy and cached; writes are whole-file, which is fine for a
    household-sized list and keeps the format trivially auditable.
    """

    def __init__(self, vault: Vault):
        self._vault = vault
        self._people: dict[str, Person] = {}
        self._loaded = False

    # ---- lifecycle -------------------------------------------------------
    @property
    def path(self) -> Path:
        return self._vault.root / f"{STORE_RELATIVE}.enc"

    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> VoiceprintStore:
        if self._loaded:
            return self
        self._loaded = True
        if not self.path.exists():
            return self
        try:
            # Read through the vault rather than decrypting by hand, so the AAD
            # matches what save() wrote -- the vault binds a file to its basename
            # as well as to this label, and reconstructing that here by hand is
            # how the two drift apart.
            raw = self._vault.read(self.path, STORE_AAD)
            payload = json.loads(raw.decode("utf-8"))
        except VaultError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise VoiceprintError(f"the voiceprint store is unreadable: {exc}") from exc

        version = int(payload.get("version") or 0)
        if version > STORE_VERSION:
            raise VoiceprintError(
                f"the voiceprint store is version {version}, but this build only "
                f"understands {STORE_VERSION}. Upgrade rather than overwrite."
            )
        for entry in payload.get("people") or []:
            person = Person.from_dict(entry)
            if person.id:
                self._people[person.id] = person
        return self

    def save(self) -> Path:
        ok, why = self._vault.ready()
        if not ok:
            raise VoiceprintError(
                f"{why}\nVoiceprints are biometric data and are never written in "
                "plaintext. Set the passphrase and try again."
            )
        payload = {
            "version": STORE_VERSION,
            "updated_at": _now(),
            "people": [p.to_dict() for p in sorted(self._people.values(), key=lambda p: p.id)],
        }
        return self._vault.write(
            STORE_RELATIVE, json.dumps(payload, ensure_ascii=False), STORE_AAD
        )

    # ---- queries ---------------------------------------------------------
    def people(self) -> list[Person]:
        self.load()
        return sorted(self._people.values(), key=lambda p: p.name.lower())

    def is_empty(self) -> bool:
        return not self.people()

    def find(self, name_or_id: str) -> Person | None:
        self.load()
        key = slugify(name_or_id)
        if key in self._people:
            return self._people[key]
        matches = [p for p in self._people.values() if p.name.lower() == name_or_id.strip().lower()]
        return matches[0] if len(matches) == 1 else None

    # ---- mutation --------------------------------------------------------
    def enroll(self, name: str, vector: list[float], source: str = "",
               seconds: float = 0.0, replace: bool = False) -> Person:
        self.load()
        display = name.strip()
        if not display:
            raise VoiceprintError("a person needs a name")
        pid = slugify(display)
        if not pid:
            raise VoiceprintError(f"'{name}' does not reduce to a usable id")

        sample = Sample(vector=normalise(vector), source=source, seconds=seconds)
        person = self._people.get(pid)
        if person is None:
            person = Person(id=pid, name=display)
            self._people[pid] = person
        elif replace:
            person.samples = []
        # A later enrollment under a different capitalisation should not
        # silently keep the old spelling on every future transcript.
        person.name = display
        person.samples.append(sample)
        person.updated_at = _now()
        return person

    def forget(self, name_or_id: str) -> Person | None:
        person = self.find(name_or_id)
        if person is None:
            return None
        del self._people[person.id]
        return person


# =========================================================================
# Embedding
# =========================================================================
class Embedder:
    """
    Wraps the speaker-embedding model.

    Availability is checked the same way diarization checks its own, and for
    the same reason: this feature is optional, and a machine without the weights
    should get a sentence explaining that, not an ImportError from three
    libraries down.
    """

    _CACHE: dict[str, object] = {}

    def __init__(self, cfg):
        self.cfg = cfg
        self.model_name = cfg.get("diarization.identify.model", DEFAULT_MODEL)

    # ---- preflight -------------------------------------------------------
    @staticmethod
    def available(cfg) -> tuple[bool, str]:
        if not cfg.get("diarization.identify.enabled", True):
            return False, "disabled in config (diarization.identify.enabled)"
        try:
            import pyannote.audio  # noqa: F401
        except ImportError:
            return False, "pyannote.audio is not installed (pip install pyannote.audio)"

        model = cfg.get("diarization.identify.model", DEFAULT_MODEL)
        _target, local = resolve_local_model(cfg, model, "diarization")
        if local:
            return True, "ready (local weights)"
        if is_offline(cfg):
            return False, (
                f"runtime.offline is on and '{model}' is not in runtime.models_dir. "
                "Fetch it with scripts/fetch_models.py --embedding on a networked machine."
            )
        token_env = cfg.get("diarization.pyannote.hf_token_env", "HUGGINGFACE_TOKEN")
        if not os.environ.get(token_env, "").strip():
            return False, f"{token_env} is not set, and the weights are not on disk yet"
        return True, "ready"

    def require(self) -> None:
        ok, why = self.available(self.cfg)
        if not ok:
            raise VoiceprintError(f"speaker identification is unavailable: {why}")

    # ---- the model -------------------------------------------------------
    def _inference(self):
        if self.model_name in self._CACHE:
            return self._CACHE[self.model_name]

        from pyannote.audio import Inference, Model

        target, local = resolve_local_model(self.cfg, self.model_name, "diarization")
        log.info("loading embedding model %s%s", target, " (local)" if local else "")
        token_env = self.cfg.get("diarization.pyannote.hf_token_env", "HUGGINGFACE_TOKEN")
        if local:
            model = Model.from_pretrained(target)
        else:
            model = Model.from_pretrained(
                target, use_auth_token=os.environ.get(token_env, "").strip() or None
            )

        device_pref = self.cfg.get("diarization.pyannote.device", "auto")
        try:
            import torch

            if device_pref == "auto":
                device_pref = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(torch.device(device_pref))
        except Exception as exc:  # noqa: BLE001
            log.debug("could not move embedding model to device: %s", exc)

        inference = Inference(model, window="whole")
        self._CACHE[self.model_name] = inference
        return inference

    @staticmethod
    def _flatten(raw) -> list[float]:
        """
        Whatever the model returned, reduce it to one flat vector.

        `window="whole"` yields a 1-D array, but a model configured otherwise
        yields one row per sliding window; averaging those is the right
        summary rather than a reason to fail.
        """
        try:
            import numpy as np

            arr = np.asarray(raw, dtype="float64")
            if arr.ndim > 1:
                arr = arr.mean(axis=tuple(range(arr.ndim - 1)))
            values = [float(x) for x in arr.reshape(-1)]
        except ImportError:  # pragma: no cover - numpy ships with pyannote
            values = [float(x) for x in raw]
        if not values:
            raise VoiceprintError("the embedding model returned nothing")
        return values

    def embed(self, audio: Path, start: float | None = None,
              end: float | None = None) -> list[float]:
        """Embed a whole file, or one span of it, as a unit-length vector."""
        self.require()
        inference = self._inference()
        if start is None and end is None:
            return normalise(self._flatten(inference(str(audio))))

        from pyannote.core import Segment as PSegment

        span = PSegment(float(start or 0.0), float(end or 0.0))
        if span.duration < MIN_CROP_SECONDS:
            raise VoiceprintError(
                f"{span.duration:.1f}s of speech is too short to embed "
                f"(need at least {MIN_CROP_SECONDS}s)"
            )
        return normalise(self._flatten(inference.crop(str(audio), span)))


# =========================================================================
# Identification
# =========================================================================
@dataclass
class ClusterMatch:
    """What identification concluded about one diarized cluster, and why."""

    cluster: str
    seconds: float
    scores: list[tuple[str, float]]  # (display name, similarity), best first
    matched: str | None = None
    reason: str = ""

    @property
    def best(self) -> tuple[str, float] | None:
        return self.scores[0] if self.scores else None


def _spans_by_cluster(segments: list[Segment]) -> dict[str, list[tuple[float, float]]]:
    spans: dict[str, list[tuple[float, float]]] = {}
    for seg in segments:
        if seg.duration <= 0:
            continue
        spans.setdefault(seg.speaker, []).append((seg.start, seg.end))
    return spans


def _cluster_vector(embedder: Embedder, audio: Path, spans: list[tuple[float, float]],
                    max_crops: int) -> tuple[list[float] | None, float]:
    """
    One vector summarising a cluster, from its longest spans.

    Longest first because a two-second interjection is mostly onset and offset,
    and averaging several spans smooths over the one where somebody coughed.
    """
    usable = sorted(
        ((s, e) for s, e in spans if (e - s) >= MIN_CROP_SECONDS),
        key=lambda se: se[1] - se[0],
        reverse=True,
    )[:max_crops]
    if not usable:
        return None, sum(e - s for s, e in spans)

    vectors: list[list[float]] = []
    used = 0.0
    for start, end in usable:
        try:
            vectors.append(embedder.embed(audio, start, end))
            used += end - start
        except VoiceprintError as exc:
            log.debug("skipped span %.1f-%.1f: %s", start, end, exc)
        except Exception as exc:  # noqa: BLE001
            log.debug("embedding failed for span %.1f-%.1f: %s", start, end, exc)
    if not vectors:
        return None, used
    return average(vectors), used


def identify(audio: Path, segments: list[Segment], cfg,
             store: VoiceprintStore) -> list[ClusterMatch]:
    """
    Score every diarized cluster against every enrolled person.

    Two guards stand between a score and a name on a transcript:

    **Threshold.** The similarity has to clear `diarization.identify.threshold`
    outright. Nobody in the room being enrolled is the common case, not the
    exception, and the nearest of five strangers is still a stranger.

    **Margin.** The best person has to beat the runner-up by
    `diarization.identify.margin`. Relatives sound alike, and a father and son
    both scoring 0.61 means the model cannot tell them apart on this audio --
    which is a reason to stay quiet, not to flip a coin.

    A person is then used at most once per recording: the same voice cannot be
    two people in the same room, and the higher score wins the tie.

    Returns a match per cluster with the full score table, so `speakers
    identify` can show the near misses. Tuning a threshold you cannot see the
    scores behind is guesswork.
    """
    people = store.people()
    matches: list[ClusterMatch] = []
    if not people:
        return matches

    threshold = float(cfg.get("diarization.identify.threshold", DEFAULT_THRESHOLD))
    margin = float(cfg.get("diarization.identify.margin", DEFAULT_MARGIN))
    min_speech = float(cfg.get("diarization.identify.min_speech_seconds", DEFAULT_MIN_SPEECH))
    max_crops = int(cfg.get("diarization.identify.max_crops", DEFAULT_MAX_CROPS))

    embedder = Embedder(cfg)
    embedder.require()

    for cluster, spans in sorted(_spans_by_cluster(segments).items()):
        total = sum(e - s for s, e in spans)
        if total < min_speech:
            matches.append(ClusterMatch(
                cluster, total, [], None,
                f"only {total:.1f}s of speech (need {min_speech:.0f}s)",
            ))
            continue

        vector, used = _cluster_vector(embedder, audio, spans, max_crops)
        if vector is None:
            matches.append(ClusterMatch(cluster, total, [], None, "no span was long enough to embed"))
            continue

        scores: list[tuple[str, float]] = []
        for person in people:
            best = max((cosine(vector, s.vector) for s in person.samples), default=None)
            if best is not None:
                scores.append((person.name, best))
        scores.sort(key=lambda ns: ns[1], reverse=True)
        matches.append(ClusterMatch(cluster, total, scores, None, ""))

    _decide(matches, threshold, margin)
    return matches


def _decide(matches: list[ClusterMatch], threshold: float, margin: float) -> None:
    """Apply threshold, margin, and one-name-per-recording, in place."""
    candidates: list[tuple[float, ClusterMatch, str]] = []
    for match in matches:
        if match.reason or not match.scores:
            if not match.reason:
                match.reason = "nobody is enrolled to compare against"
            continue
        name, top = match.scores[0]
        runner_up = match.scores[1][1] if len(match.scores) > 1 else 0.0
        if top < threshold:
            match.reason = f"best score {top:.2f} is below the {threshold:.2f} threshold"
            continue
        if len(match.scores) > 1 and (top - runner_up) < margin:
            match.reason = (
                f"{name} {top:.2f} and {match.scores[1][0]} {runner_up:.2f} are "
                f"within the {margin:.2f} margin"
            )
            continue
        candidates.append((top, match, name))

    candidates.sort(key=lambda c: c[0], reverse=True)
    claimed: dict[str, ClusterMatch] = {}
    for score, match, name in candidates:
        if name in claimed:
            match.reason = (
                f"{name} already matched cluster {claimed[name].cluster} at a "
                f"higher score than {score:.2f}"
            )
            continue
        match.matched = name
        match.reason = f"matched {name} at {score:.2f}"
        claimed[name] = match


def named_speakers(audio: Path, segments: list[Segment], cfg,
                   vault: Vault | None = None) -> dict[str, str]:
    """
    Cluster label -> person's name, for confident matches only.

    The pipeline's entry point. Returns `{}` for every ordinary reason this
    might not apply -- nobody enrolled, no vault passphrase, no model on disk --
    because none of those is a reason to fail a transcript that is otherwise
    fine. Only the store being corrupt propagates, since silently ignoring a
    damaged biometric file is not something to do quietly.
    """
    if not cfg.get("diarization.identify.enabled", True):
        return {}

    vault = vault or Vault(cfg.path("vault"))
    store = VoiceprintStore(vault)
    if not store.exists():
        return {}

    ok, why = vault.ready()
    if not ok:
        log.info("speaker identification skipped: %s", why)
        return {}

    if store.is_empty():
        return {}

    ok, why = Embedder.available(cfg)
    if not ok:
        log.info("speaker identification skipped: %s", why)
        return {}

    matches = identify(audio, segments, cfg, store)
    named = {m.cluster: m.matched for m in matches if m.matched}
    if named:
        log.info("identified %d speaker(s): %s", len(named), ", ".join(sorted(named.values())))
    for match in matches:
        if not match.matched:
            log.debug("cluster %s unnamed: %s", match.cluster, match.reason)
    return named
