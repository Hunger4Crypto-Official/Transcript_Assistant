#!/usr/bin/env python3
"""
Measure what local processing actually costs on THIS machine.

The decision this informs: someone sitting on a large backlog -- hundreds of
hours of recordings -- has to choose between the fully-private local pipeline
and a cloud key before committing days of compute to the wrong answer. Specs
do not answer that; a laptop with the same CPU can transcribe at 8x realtime
or 0.8x depending on weights, compute type and thermals. So this runs one real
recording through the same stages the pipeline uses (ffmpeg normalise, then
local transcription via the real ASR chain, local-only) and reports the
measured realtime factor plus a projection for the backlog.

    python scripts/bench.py recording.mp3
    python scripts/bench.py recording.mp3 --backlog-hours 400

Everything is written to a temp directory that is deleted afterwards; the
project's data/ directories are never touched, so benchmarking cannot pollute
the inbox, the vault, or the database. Model weights are the one deliberate
exception: they cache in the project's models/ directory, shared with normal
runs, because re-downloading gigabytes per benchmark would be the tool wasting
exactly the time it exists to estimate.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _fail(message: str, code: int) -> int:
    """One friendly line on stderr and a nonzero exit. Never a traceback."""
    print(f"bench: {message}", file=sys.stderr)
    return code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Time ffmpeg + local transcription on one recording, "
                    "then project a backlog.")
    parser.add_argument("audio", type=Path, help="one real recording to measure")
    parser.add_argument("--backlog-hours", type=float, default=100.0,
                        help="backlog size to project (default: 100)")
    args = parser.parse_args(argv)

    src = args.audio.expanduser()
    if not src.is_file():
        return _fail(f"{src} does not exist or is not a file. "
                     "Point this at one real recording.", 2)

    from plaud_bridge.asr.base import ASRError
    from plaud_bridge.asr.registry import build_asr_chain, transcribe
    from plaud_bridge.audio.prepare import AudioError, AudioPreparer
    from plaud_bridge.config import Config

    with tempfile.TemporaryDirectory(prefix="plaud-bench-") as tmp:
        work = Path(tmp)
        # root at the temp dir points every data path away from the user's
        # project; models_dir is then pinned back to the repo so weights cache
        # where normal runs already keep them (see the module docstring).
        cfg = Config.load(ROOT / "config", root=work)
        cfg._d.setdefault("runtime", {})["models_dir"] = str(ROOT / "models")

        prep = AudioPreparer(cfg)
        try:
            prep.check_tools()
        except AudioError as exc:
            return _fail(str(exc), 3)

        local_chain = build_asr_chain(cfg, cfg.glossary, local_only=True)
        problems = []
        for provider in local_chain:
            ok, why = provider.available()
            if ok:
                break
            problems.append(f"{provider.name}: {why}")
        else:
            return _fail("no local transcriber is usable -- "
                         + ("; ".join(problems) or "none configured in asr.providers")
                         + ". Run: pip install faster-whisper", 4)

        try:
            t0 = time.perf_counter()
            norm, audio_seconds = prep.normalise(src, work)
            t_norm = time.perf_counter() - t0

            t0 = time.perf_counter()
            chunks = prep.chunk(norm, work, audio_seconds)
            transcript = transcribe(chunks, cfg, cfg.glossary, local_only=True)
            t_asr = time.perf_counter() - t0
        except (AudioError, ASRError) as exc:
            return _fail(str(exc), 5)

    total = t_norm + t_asr
    rtf = audio_seconds / total if total else 0.0

    def line(stage: str, wall: float) -> str:
        factor = audio_seconds / wall if wall else float("inf")
        return f"  {stage:<22}{wall:>9.1f} s   {factor:>7.1f}x realtime"

    print(f"\nBenchmark: {src.name} "
          f"({audio_seconds / 60:.1f} min of audio, {len(chunks)} chunk(s), "
          f"transcribed by '{transcript.asr_provider}')\n")
    print(line("ffmpeg normalise", t_norm))
    print(line("local transcription", t_asr))
    print(line("total", total))
    if rtf > 0:
        compute_hours = args.backlog_hours / rtf
        print(f"\nAt this rate, {args.backlog_hours:g} hours of audio "
              f"≈ {compute_hours:.1f} hours of compute.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
