"""
Pipeline orchestrator.

Order matters and is not arbitrary:

    ingest -> prepare -> transcribe -> diarize -> correct
           -> route -> COMPLIANCE GATE -> analyse -> persist

Routing runs BEFORE the compliance gate because the gate needs to know which
profiles matched in order to decide how strictly to treat the file. That
creates a bootstrapping problem: routing itself uses an LLM. It is solved by
routing conservatively, local-only, whenever the keyword prescore shows any
signal at all for a maximum-sensitivity profile. Better to route a business
call locally than to ship a family conversation to a third party because the
router had not classified it yet.
"""

from __future__ import annotations

import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .asr import transcribe
from .asr.base import ASRError
from .audio import AudioError, AudioPreparer
from .compliance import ComplianceGate, RetentionSweeper
from .config import Config
from .correct import apply_corrections
from .db import Database
from .diarize import diarize
from .logging_setup import get
from .models import (
    ProfileAnalysis,
    Recording,
    RunStats,
    Segment,
    Stage,
    Transcript,
)
from .profiles import extract, route
from .profiles.router import _keyword_prescore
from .storage import Vault, VaultError

log = get("pipeline")


class PipelineError(RuntimeError):
    pass


class Pipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        cfg.ensure_dirs()
        self.db = Database(cfg.path("database"))
        self.vault = Vault(cfg.path("vault"))
        self.audio = AudioPreparer(cfg)
        self.gate = ComplianceGate(cfg)
        self.retention = RetentionSweeper(cfg, self.db)
        self.warn_at = float(cfg.get("cost.warn_usd_per_run", 2.0))
        self.halt_at = float(cfg.get("cost.halt_usd_per_run", 10.0))

    # =====================================================================
    # Discovery
    # =====================================================================
    def discover(self) -> list[Path]:
        inbox = self.cfg.path("inbox")
        audio_ext = {e.lower() for e in self.cfg.get("ingest.audio_extensions", [])}
        text_ext = {e.lower() for e in self.cfg.get("ingest.text_extensions", [])}
        allowed = audio_ext | text_ext
        max_mb = float(self.cfg.get("ingest.max_file_mb", 512))

        archive = inbox / "_processed"

        found: list[Path] = []
        for path in sorted(inbox.rglob("*")):
            if not path.is_file() or path.name.startswith("."):
                continue
            # Skip the archive folder. It lives under the inbox for convenience,
            # but rescanning and rehashing every previously processed file on
            # every run wastes real time once the archive gets large.
            if archive in path.parents:
                continue
            if path.suffix.lower() not in allowed:
                continue
            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb > max_mb:
                log.warning("skipping %s: %.0fMB exceeds ingest.max_file_mb (%.0f)",
                            path.name, size_mb, max_mb)
                continue
            found.append(path)
        return found

    # =====================================================================
    # Single recording
    # =====================================================================
    def process_file(self, path: Path, stats: RunStats, force: bool = False) -> Recording:
        audio_ext = {e.lower() for e in self.cfg.get("ingest.audio_extensions", [])}
        kind = "audio" if path.suffix.lower() in audio_ext else "text"

        content_hash = Vault.fingerprint(path)
        existing = self.db.hash_exists(content_hash)
        if existing and not force and self.cfg.get("ingest.dedupe", True):
            log.info("skipping %s: already processed as %s", path.name, existing)
            stats.skipped += 1
            rec = Recording(id=existing, source_name=path.name, content_hash=content_hash)
            rec.stage = Stage.COMPLETE
            return rec

        stat = path.stat()
        rec = Recording(
            source_path=str(path),
            source_name=path.name,
            content_hash=content_hash,
            size_bytes=stat.st_size,
            kind=kind,
            recorded_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
        )
        self.db.audit("ingest", f"{path.name} ({stat.st_size} bytes)", rec.id)
        log.info("processing %s (%s, %.1fMB)", path.name, kind, stat.st_size / 1_048_576)

        work_dir = self.cfg.path("work") / rec.id
        try:
            if kind == "audio":
                self._transcribe_audio(rec, path, work_dir)
            else:
                self._load_text(rec, path)

            if rec.transcript is None or not rec.transcript.segments:
                raise PipelineError("no transcribable content found")

            rec.stage = Stage.CORRECTED
            self._route(rec)
            self._gate(rec)

            if not rec.compliance.allow:
                self._quarantine(rec, path)
                stats.quarantined += 1
                return rec

            self._analyse(rec)
            self._persist(rec)

            rec.stage = Stage.COMPLETE
            stats.processed += 1
            stats.audio_seconds += rec.duration_seconds

            if self.cfg.get("ingest.archive_originals", True):
                self._archive(rec, path)

        except (AudioError, ASRError, PipelineError, VaultError) as exc:
            rec.stage = Stage.FAILED
            rec.errors.append(str(exc))
            stats.failed += 1
            log.error("failed on %s: %s", path.name, exc)
            self.db.audit("failure", str(exc)[:500], rec.id)
        except Exception as exc:  # noqa: BLE001 - never lose the queue to one bad file
            rec.stage = Stage.FAILED
            rec.errors.append(f"unexpected: {exc}")
            stats.failed += 1
            log.exception("unexpected failure on %s", path.name)
            self.db.audit("failure", f"unexpected: {exc}"[:500], rec.id)
        finally:
            # Every exit path counts, not just the successful one. A recording
            # that quarantined or crashed still burned the ASR and routing calls
            # that got it that far, and a guardrail that only sees successes
            # cannot stop a loop that produces nothing but failures.
            stats.cost_usd += rec.total_cost_usd
            self.db.upsert(rec)
            AudioPreparer.cleanup(work_dir)

        return rec

    # ---- stages ---------------------------------------------------------
    def _transcribe_audio(self, rec: Recording, path: Path, work_dir: Path) -> None:
        normalised, duration = self.audio.normalise(path, work_dir)
        rec.duration_seconds = duration
        rec.stage = Stage.PREPARED

        local_only, why = self._asr_local_only(path)
        log.info("ASR locality for %s: %s (%s)",
                 path.name, "local-only" if local_only else "cloud permitted", why)
        self.db.audit("asr_locality", f"local_only={local_only}: {why}", rec.id)

        chunks = self.audio.chunk(normalised, work_dir, duration)
        rec.transcript = transcribe(chunks, self.cfg, self.cfg.glossary, local_only=local_only)
        rec.total_cost_usd += rec.transcript.cost_usd
        rec.stage = Stage.TRANSCRIBED

        rec.transcript.segments = diarize(normalised, rec.transcript.segments, self.cfg)
        rec.stage = Stage.DIARIZED

        rec.transcript.segments, report = apply_corrections(
            rec.transcript.segments, self.cfg.glossary
        )
        if report.total:
            self.db.audit("glossary", report.summary(), rec.id)

    def _load_text(self, rec: Recording, path: Path) -> None:
        """Accept a Plaud-exported transcript directly, skipping ASR entirely."""
        raw = path.read_text(encoding="utf-8", errors="replace")
        segments = _parse_text_transcript(raw, path.suffix.lower())
        rec.transcript = Transcript(
            segments=segments,
            asr_provider="imported",
            asr_model=path.suffix.lstrip("."),
            duration_seconds=max((s.end for s in segments), default=0.0),
        )
        rec.duration_seconds = rec.transcript.duration_seconds
        rec.transcript.segments, _ = apply_corrections(rec.transcript.segments, self.cfg.glossary)
        rec.stage = Stage.TRANSCRIBED

    def _asr_local_only(self, path: Path) -> tuple[bool, str]:
        """
        Decide whether this file may be transcribed by a cloud provider.

        ASR runs before there is a transcript to route on, so the only thing
        available is the filename, and a filename tells you almost nothing. A
        Plaud export is called REC0042.wav whether it holds a client fact-find
        or a conversation with a child.

        So cloud ASR is opt-in per file, not opt-out: it is permitted only when
        the filename positively names a profile that allows it AND no locked
        profile is implicated at the same time. Everything else stays local.

        This is the inverse of the obvious rule, deliberately. Sending a
        business call to a local model is slow and free. Sending a family
        conversation to a third party cannot be undone.
        """
        stem = path.stem.replace("_", " ").replace("-", " ")
        lowered = stem.lower()

        def names(profile) -> bool:
            pre = _keyword_prescore(stem, [profile])[0]
            return (
                pre.score > 0
                or profile.id.replace("_", " ") in lowered
                or profile.short_name.lower() in lowered
            )

        locked = [p for p in self.cfg.routable_profiles() if p.hard_local_only or not p.allow_cloud_asr]
        for profile in locked:
            if names(profile):
                return True, f"filename implicates '{profile.id}', which cannot use cloud ASR"

        permitted = [p for p in self.cfg.routable_profiles() if p.allow_cloud_asr and names(p)]
        if permitted:
            return False, (
                f"filename names '{permitted[0].id}', which permits cloud ASR, "
                "and no locked profile matched"
            )

        return True, (
            "filename does not identify a profile that permits cloud ASR; "
            "defaulting to local transcription"
        )

    def _route(self, rec: Recording) -> None:
        assert rec.transcript is not None
        # The routing call sees the whole transcript before anything is known
        # about it, so it cannot be governed by the profile it is about to
        # discover. The only sound rule is the policy every profile shares:
        # unless every routable profile permits a cloud LLM, this call is local.
        #
        # Keying it off a keyword coincidence, as this once did, meant a client
        # fact-find full of health and financial disclosures went to a cloud
        # provider whenever it happened not to mention anyone's family.
        local_only = not self.cfg.cloud_llm_permitted_by_every_profile()
        routing = route(rec.transcript, self.cfg, local_only=local_only)
        rec.routes = routing.matches
        rec.total_cost_usd += routing.cost_usd
        rec.stage = Stage.ROUTED
        self.db.audit(
            "route",
            ", ".join(f"{r.profile_id}={r.confidence:.2f}" for r in rec.routes),
            rec.id,
        )

    def _gate(self, rec: Recording) -> None:
        rec.compliance = self.gate.evaluate(rec)
        detail = "; ".join(rec.compliance.reasons)[:500]
        self.db.audit(
            "compliance",
            f"allow={rec.compliance.allow} consent={rec.compliance.consent.value} :: {detail}",
            rec.id,
        )
        for warning in rec.compliance.warnings:
            log.warning("%s: %s", rec.source_name, warning)

    def _analyse(self, rec: Recording) -> None:
        assert rec.transcript is not None

        # ADR-002. Every analysis on this recording sees the SAME transcript, so
        # every analysis is bound by the SAME policy: the governing profile's.
        # Deciding locality per profile meant a recording that matched both
        # Husband and Sales Trainer kept the Husband analysis local and then
        # handed the identical marital transcript to a cloud provider for the
        # Sales Trainer one. The gate already worked this out; the job here is
        # to obey it.
        governing = self.cfg.profile(
            rec.compliance.governing_profile or self.cfg.get("routing.fallback_profile", "unfiled")
        )
        local_only = rec.compliance.force_local_processing or governing.hard_local_only \
            or not governing.allow_cloud_llm

        body = rec.transcript.labelled_text()
        redacted, counts = self.gate.redact_for_llm(body, governing)
        for name, count in counts.items():
            rec.compliance.redactions[name] = rec.compliance.redactions.get(name, 0) + count

        for match in rec.routes:
            if match.profile_id not in self.cfg.profiles:
                continue
            profile = self.cfg.profile(match.profile_id)

            analysis: ProfileAnalysis = extract(
                rec.transcript, profile, self.cfg, redacted, local_only=local_only
            )
            rec.analyses.append(analysis)
            rec.total_cost_usd += analysis.cost_usd

            if rec.total_cost_usd > self.halt_at:
                raise PipelineError(
                    f"run cost ${rec.total_cost_usd:.2f} exceeded "
                    f"cost.halt_usd_per_run (${self.halt_at:.2f})"
                )

        rec.stage = Stage.ANALYZED

    # ---- persistence ----------------------------------------------------
    def _persist(self, rec: Recording) -> None:
        governing = self.cfg.profile(rec.compliance.governing_profile or "unfiled")
        encrypt = governing.encrypt_at_rest
        day = (rec.recorded_at or rec.ingested_at).strftime("%Y/%m/%d")
        stem = f"{day}/{rec.id}"

        assert rec.transcript is not None
        transcript_md = self._render_transcript(rec)
        analysis_json = rec.to_json()

        if encrypt:
            ok, why = self.vault.ready()
            if not ok:
                raise VaultError(
                    f"profile '{governing.id}' requires encryption at rest but {why}"
                )
            rec.artifact_paths["transcript"] = str(
                self.vault.write(f"{stem}.transcript.md", transcript_md, rec.id)
            )
            rec.artifact_paths["analysis"] = str(
                self.vault.write(f"{stem}.analysis.json", analysis_json, rec.id)
            )
        else:
            out = self.cfg.path("outbox") / stem
            out.parent.mkdir(parents=True, exist_ok=True)
            (out.parent / f"{rec.id}.transcript.md").write_text(transcript_md, encoding="utf-8")
            (out.parent / f"{rec.id}.analysis.json").write_text(analysis_json, encoding="utf-8")
            rec.artifact_paths["transcript"] = str(out.parent / f"{rec.id}.transcript.md")
            rec.artifact_paths["analysis"] = str(out.parent / f"{rec.id}.analysis.json")

        # The artifacts table has a foreign key onto recordings, so the parent
        # row has to exist before we can index its files. The run loop upserts
        # again at the end; this one is cheap and keeps the constraint honest.
        self.db.upsert(rec)

        # Retention runs from when the conversation happened, not from when it
        # was processed. Importing a three-year-old backlog should not restart
        # the clock on every file in it -- the whole point of the 180-day window
        # on a personal profile is how old the conversation is.
        created = rec.recorded_at or rec.ingested_at
        for kind, key in (("transcript", "transcript"), ("analysis", "analysis")):
            self.db.record_artifact(
                rec.id, kind, rec.artifact_paths[key], encrypt,
                self.retention.expires_at(kind, governing, created),
            )

    def _render_transcript(self, rec: Recording) -> str:
        assert rec.transcript is not None
        t = rec.transcript
        header = [
            f"# {rec.source_name}",
            "",
            f"- Recording ID: `{rec.id}`",
            f"- Recorded: {(rec.recorded_at or rec.ingested_at):%Y-%m-%d %H:%M} UTC",
            f"- Duration: {t.duration_seconds / 60:.1f} min",
            f"- ASR: {t.asr_provider} / {t.asr_model}",
            f"- Speakers: {', '.join(t.speakers)}",
            f"- Profiles: {', '.join(f'{r.profile_id} ({r.confidence:.2f})' for r in rec.routes)}",
            f"- Consent: {rec.compliance.consent.value}",
            f"- Governing profile: {rec.compliance.governing_profile} "
            f"({rec.compliance.governing_sensitivity.value})",
            "",
        ]
        if rec.compliance.consent_quote:
            header += [f"> Consent captured: {rec.compliance.consent_quote}", ""]
        header += ["---", "", "## Transcript", ""]
        return "\n".join(header) + t.labelled_text()

    def _quarantine(self, rec: Recording, path: Path) -> None:
        rec.stage = Stage.QUARANTINED
        qdir = self.cfg.path("quarantine") / rec.id
        qdir.mkdir(parents=True, exist_ok=True)
        (qdir / "WHY.md").write_text(
            "# Quarantined\n\n"
            f"- File: `{rec.source_name}`\n"
            f"- Recording ID: `{rec.id}`\n"
            f"- Profiles matched: {', '.join(rec.profile_ids)}\n\n"
            "## Reasons\n\n"
            + "\n".join(f"- {r}" for r in rec.compliance.reasons)
            + "\n\n## What to do\n\n"
            "1. Listen to the opening of the recording and confirm whether consent "
            "was actually obtained.\n"
            "2. If it was, release it: `python run.py release " + rec.id + "`\n"
            "3. If it was not, delete it. That is the correct outcome, not an "
            "inconvenience.\n",
            encoding="utf-8",
        )
        try:
            shutil.copy2(path, qdir / path.name)
        except OSError as exc:
            log.error("could not copy quarantined file: %s", exc)
        log.warning("quarantined %s -> %s", rec.source_name, qdir)
        self.db.audit("quarantine", "; ".join(rec.compliance.reasons)[:500], rec.id)

    def _archive(self, rec: Recording, path: Path) -> None:
        """
        Move the original out of the inbox and put it under retention.

        Two things this has to do beyond moving a file. The archive is flat
        while the inbox is scanned recursively, so two `recording.wav` files in
        different subfolders would land on the same name and `shutil.move`
        would overwrite the first without an error; the recording id prefix
        makes that impossible. And the audio has to be indexed as an artifact,
        or `raw_audio_days` describes a sweep that never happens -- audio would
        be the one thing that never expires, which is exactly backwards.
        """
        archive = self.cfg.path("inbox") / "_processed"
        archive.mkdir(parents=True, exist_ok=True)
        dest = archive / f"{rec.id}_{path.name}"
        try:
            shutil.move(str(path), str(dest))
        except (OSError, shutil.Error) as exc:
            log.warning("could not archive %s: %s", path.name, exc)
            return

        if rec.kind != "audio":
            return

        governing = self.cfg.profile(
            rec.compliance.governing_profile or self.cfg.get("routing.fallback_profile", "unfiled")
        )
        rec.artifact_paths["audio"] = str(dest)
        self.db.upsert(rec)
        self.db.record_artifact(
            rec.id, "audio", str(dest), False,
            self.retention.expires_at("audio", governing, rec.recorded_at or rec.ingested_at),
        )

    # =====================================================================
    # Batch
    # =====================================================================
    def run(self, force: bool = False, limit: int | None = None) -> RunStats:
        stats = RunStats()
        files = self.discover()
        if limit:
            files = files[:limit]

        if not files:
            log.info("inbox is empty: %s", self.cfg.path("inbox"))
            return stats

        log.info("found %d file(s) to process", len(files))
        warned = False
        for path in files:
            self.process_file(path, stats, force=force)
            if not warned and stats.cost_usd > self.warn_at:
                log.warning("run cost has passed $%.2f (warn threshold)", self.warn_at)
                warned = True
            if stats.cost_usd > self.halt_at:
                log.error("halting: run cost $%.2f exceeded $%.2f",
                          stats.cost_usd, self.halt_at)
                break

        self.db.audit("run_complete", stats.summary())
        log.info("run complete: %s", stats.summary())
        return stats

    def close(self) -> None:
        self.db.close()


# =========================================================================
# Text transcript import
# =========================================================================
_STAMP_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
)
_SPEAKER_RE = re.compile(r"^(?P<speaker>[A-Za-z][\w .'\-]{0,30}?):\s+(?P<text>\S.*)$")
_TS_PREFIX_RE = re.compile(r"^\[(?P<ts>\d{1,2}:\d{2}(?::\d{2})?)\]\s*(?P<rest>.*)$")

# Words that start sentences and are followed by a colon often enough to be
# mistaken for speaker labels. Without this list, "Well: I told him no." loses
# the word "Well" and invents a speaker called Well, who then appears in the
# transcript header and the digest as if they had been in the room.
_NOT_A_SPEAKER = frozenset({
    "a", "actually", "also", "and", "answer", "as", "aside", "but", "caution",
    "conclusion", "context", "correction", "edit", "error", "example", "finally",
    "first", "however", "http", "https", "idea", "in", "issue", "look", "meaning",
    "n b", "nb", "no", "note", "notes", "now", "ok", "okay", "p s", "problem",
    "ps", "question", "re", "reason", "recap", "remember", "reminder", "result",
    "right", "second", "see", "so", "solution", "source", "summary", "then",
    "third", "tip", "to", "topic", "update", "warning", "well", "yes",
})

WORDS_PER_SECOND = 2.6


def _speaker_split(line: str) -> tuple[str, str] | None:
    """Return (speaker, text) if this line plausibly carries a speaker label."""
    match = _SPEAKER_RE.match(line.strip())
    if not match:
        return None
    speaker = match.group("speaker").strip()
    # "See https://example.com/x" is a URL, not See speaking.
    if "//" in match.group("text")[:2] or "://" in line:
        return None
    if speaker.lower() in _NOT_A_SPEAKER or len(speaker.split()) > 4:
        return None
    return speaker, match.group("text").strip()


def _confirmed_speakers(lines: list[str]) -> set[str]:
    """
    Decide which candidate labels are really speakers.

    Two signals, either of which is enough: the label appears on more than one
    line, which a stray "Note:" almost never does, or it reads like a name.
    Guessing per-line was what let single occurrences through.
    """
    counts: dict[str, int] = {}
    for line in lines:
        split = _speaker_split(line)
        if split:
            counts[split[0]] = counts.get(split[0], 0) + 1

    confirmed = set()
    for speaker, count in counts.items():
        looks_like_a_name = all(w[:1].isupper() for w in speaker.split() if w)
        if count > 1 or looks_like_a_name:
            confirmed.add(speaker)
    return confirmed


def _parse_text_transcript(raw: str, suffix: str) -> list[Segment]:
    """
    Parse a Plaud-exported transcript.

    SRT gives real timestamps. Plain text does not, so we synthesise a rough
    timeline at an average speaking rate. Those timestamps are approximations
    and are labelled as such; do not quote them as evidence of when something
    was said.
    """
    segments: list[Segment] = []

    if suffix == ".srt":
        blocks = re.split(r"\n\s*\n", raw.strip())

        bodies: list[tuple[float, float, str]] = []
        for block in blocks:
            match = _STAMP_RE.search(block)
            if not match:
                continue
            g = [int(x) for x in match.groups()]
            start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000.0
            end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000.0

            lines = block.splitlines()
            # The cue index is the first line and only the first line. Dropping
            # every numeric line instead deletes real content: a caption reading
            # "1035" or a dollar figure on a line of its own vanishes entirely.
            if lines and lines[0].strip().isdigit():
                lines = lines[1:]
            lines = [ln for ln in lines if not _STAMP_RE.search(ln)]

            text = " ".join(ln.strip() for ln in lines).strip()
            if text:
                bodies.append((start, end, text))

        confirmed = _confirmed_speakers([b[2] for b in bodies])
        for start, end, text in bodies:
            speaker = "SPEAKER"
            split = _speaker_split(text)
            if split and split[0] in confirmed:
                speaker, text = split
            segments.append(Segment(start, end, text, speaker))
        return segments

    # Plain text / markdown. Honour "Speaker: text" and "[00:12] Speaker:" forms.
    raw_lines = [ln.strip() for ln in raw.splitlines()]
    bodies_txt: list[tuple[float | None, str]] = []
    for line in raw_lines:
        if not line or line.startswith("#"):
            continue
        stamped: float | None = None
        prefix = _TS_PREFIX_RE.match(line)
        if prefix:
            parts = [int(p) for p in prefix.group("ts").split(":")]
            stamped = (
                parts[0] * 60 + parts[1] if len(parts) == 2
                else parts[0] * 3600 + parts[1] * 60 + parts[2]
            )
            line = prefix.group("rest").strip()
        if line:
            bodies_txt.append((stamped, line))

    confirmed = _confirmed_speakers([b[1] for b in bodies_txt])
    cursor = 0.0
    for stamped, line in bodies_txt:
        speaker = "SPEAKER"
        text = line
        split = _speaker_split(line)
        if split and split[0] in confirmed:
            speaker, text = split
        if not text:
            continue
        # An explicit timestamp moves the clock forward, never backward. The
        # synthesised durations routinely overshoot the next real stamp, and
        # letting that rewind the cursor produced segments starting before the
        # previous one ended -- which renders out of order and hands the consent
        # detector an incoherent opening window.
        if stamped is not None:
            cursor = max(cursor, float(stamped))
        duration = max(1.0, len(text.split()) / WORDS_PER_SECOND)
        segments.append(Segment(cursor, cursor + duration, text, speaker))
        cursor += duration

    segments.sort(key=lambda s: (s.start, s.end))
    return segments
