"""
Audio preparation: probe, normalise, chunk.

Two problems this solves that will otherwise bite you:

1. Cloud ASR endpoints cap uploads around 25MB. A 90 minute meeting exceeds
   that. We chunk with overlap and let the stitcher recover words cut in half
   at a boundary.
2. Plaud pin recordings run noticeably quieter than card recordings. Loudness
   normalisation removes a whole class of "the model missed a sentence"
   complaints before they happen.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..logging_setup import get

log = get("audio")


class AudioError(RuntimeError):
    pass


@dataclass
class AudioChunk:
    index: int
    path: Path
    start: float          # offset in the ORIGINAL timeline
    duration: float
    overlap_lead: float   # seconds of this chunk that repeat the previous one

    @property
    def size_mb(self) -> float:
        return self.path.stat().st_size / (1024 * 1024) if self.path.exists() else 0.0


def _run(cmd: list[str], timeout: int = 900) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except FileNotFoundError as exc:
        raise AudioError(f"{cmd[0]} not found on PATH. Install ffmpeg.") from exc
    except subprocess.TimeoutExpired as exc:
        raise AudioError(f"{cmd[0]} timed out after {timeout}s") from exc


def probe_duration(path: Path, ffprobe: str = "ffprobe") -> float:
    proc = _run([ffprobe, "-v", "error", "-show_entries", "format=duration",
                 "-of", "json", str(path)], timeout=120)
    if proc.returncode != 0:
        raise AudioError(f"ffprobe failed on {path.name}: {proc.stderr.strip()[:300]}")
    try:
        # TypeError belongs here: ffprobe reports "duration": null for streams
        # with no container duration, which some .m4a exports have. Without it
        # the float(None) escapes as an unexpected error and the user is told
        # "float() argument must be a string or a real number" instead of
        # anything they can act on.
        seconds = float(json.loads(proc.stdout)["format"]["duration"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise AudioError(
            f"could not read duration from {path.name}. The file may be truncated "
            "or missing container metadata; try re-exporting it."
        ) from exc
    # A duration drives the chunk arithmetic and the loop bound in chunk(). A
    # NaN, an infinity, or a negative -- any of which float() will happily
    # return from a crafted or corrupt container -- turns that arithmetic into a
    # non-terminating loop or a nonsense chunk count. Reject them here, once, so
    # nothing downstream has to defend against them.
    if not math.isfinite(seconds) or seconds <= 0:
        raise AudioError(
            f"{path.name} reports a duration of {seconds}, which is not a positive "
            "finite number of seconds. The file is corrupt or not really audio."
        )
    return seconds


class AudioPreparer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.ffmpeg = cfg.get("audio.ffmpeg_binary", "ffmpeg")
        self.ffprobe = cfg.get("audio.ffprobe_binary", "ffprobe")
        self.sample_rate = int(cfg.get("audio.target_sample_rate", 16000))
        self.channels = int(cfg.get("audio.target_channels", 1))
        self.normalize = bool(cfg.get("audio.normalize_loudness", True))
        self.lufs = float(cfg.get("audio.loudness_target_lufs", -16.0))
        self.chunk_seconds = float(cfg.get("audio.chunk_seconds", 600))
        self.overlap = float(cfg.get("audio.chunk_overlap_seconds", 8))
        self.max_chunk_mb = float(cfg.get("audio.max_chunk_mb", 20))
        # A ceiling on how long a single recording may be. normalise() decodes
        # the whole input to uncompressed 16k mono PCM -- ~115 MB/hour -- before
        # anything else looks at it, so an accidental or malicious 40-hour file
        # becomes gigabytes of scratch and a chunk fan-out to match. 14 hours is
        # well past a day-long wearable session and still bounds all of that.
        self.max_duration = float(cfg.get("audio.max_duration_seconds", 50400))

    def check_tools(self) -> None:
        for tool in (self.ffmpeg, self.ffprobe):
            if shutil.which(tool) is None:
                raise AudioError(
                    f"'{tool}' is not on PATH. Install ffmpeg: "
                    "macOS 'brew install ffmpeg', Ubuntu 'apt install ffmpeg'."
                )

    def normalise(self, src: Path, work_dir: Path) -> tuple[Path, float]:
        self.check_tools()
        work_dir.mkdir(parents=True, exist_ok=True)
        dest = work_dir / f"{src.stem}.norm.wav"

        # Refuse before decoding when the source duration is readable and over
        # budget: there is no reason to write gigabytes of scratch WAV only to
        # reject it afterwards. Some containers report no duration, and those
        # fall through to the -t ceiling below.
        try:
            src_duration = probe_duration(src, self.ffprobe)
        except AudioError:
            src_duration = None
        if src_duration is not None and src_duration > self.max_duration:
            raise AudioError(
                f"{src.name} is {src_duration / 3600:.1f} hours long, over the "
                f"audio.max_duration_seconds budget of {self.max_duration / 3600:.1f} "
                "hours. Raise the limit in config if that is genuinely one recording."
            )

        filters = []
        if self.normalize:
            # Single-pass loudnorm. Two-pass is more accurate but doubles wall
            # time for a difference no ASR model can hear.
            filters.append(f"loudnorm=I={self.lufs}:TP=-1.5:LRA=11")
        filter_arg = ["-af", ",".join(filters)] if filters else []

        # A hard ceiling on the decode itself, a minute past the budget. For a
        # recording within budget this changes nothing; for a container that
        # reports no duration it stops ffmpeg writing an unbounded WAV, and the
        # post-decode check below then rejects it rather than guessing.
        limit = f"{self.max_duration + 60:.3f}"
        proc = _run([
            self.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(src), "-t", limit,
            "-ac", str(self.channels), "-ar", str(self.sample_rate),
            *filter_arg, "-c:a", "pcm_s16le", str(dest),
        ])
        if proc.returncode != 0 or not dest.exists():
            raise AudioError(f"ffmpeg normalise failed for {src.name}: {proc.stderr.strip()[:400]}")

        duration = probe_duration(dest, self.ffprobe)
        if duration > self.max_duration:
            # Only reachable for a source whose duration could not be read up
            # front; the -t ceiling capped the decode near the budget.
            raise AudioError(
                f"{src.name} decodes to more than the audio.max_duration_seconds "
                f"budget of {self.max_duration / 3600:.1f} hours. Its container "
                "reports no duration, so it was capped at the budget and refused."
            )
        log.info("normalised %s -> %.1fs @ %dHz mono", src.name, duration, self.sample_rate)
        return dest, duration

    def chunk(self, src: Path, work_dir: Path, duration: float | None = None) -> list[AudioChunk]:
        self.check_tools()
        duration = duration if duration is not None else probe_duration(src, self.ffprobe)
        chunk_dir = work_dir / f"{src.stem}.chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        bps = self.sample_rate * self.channels * 2      # 16-bit PCM
        size_limited = (self.max_chunk_mb * 1024 * 1024) / bps if bps else self.chunk_seconds
        window = max(30.0, min(self.chunk_seconds, size_limited))

        if duration <= window:
            single = chunk_dir / f"{src.stem}.000.wav"
            shutil.copy2(src, single)
            return [AudioChunk(0, single, 0.0, duration, 0.0)]

        stride = window - self.overlap
        if stride <= 0:
            raise AudioError("chunk_overlap_seconds must be smaller than the chunk window")

        count = math.ceil((duration - self.overlap) / stride)
        chunks: list[AudioChunk] = []
        for i in range(count):
            start = max(0.0, i * stride)
            take = min(window, duration - start)
            if take <= 0.5:
                break
            dest = chunk_dir / f"{src.stem}.{i:03d}.wav"
            proc = _run([
                self.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-ss", f"{start:.3f}", "-t", f"{take:.3f}",
                "-i", str(src), "-c:a", "pcm_s16le", str(dest),
            ])
            if proc.returncode != 0 or not dest.exists():
                raise AudioError(f"ffmpeg chunk {i} failed: {proc.stderr.strip()[:300]}")
            chunks.append(AudioChunk(i, dest, start, take, 0.0 if i == 0 else self.overlap))

        log.info("chunked %s into %d pieces (window=%.0fs overlap=%.0fs)",
                 src.name, len(chunks), window, self.overlap)
        return chunks

    @staticmethod
    def cleanup(work_dir: Path) -> None:
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)
