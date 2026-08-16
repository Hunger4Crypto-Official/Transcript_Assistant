"""
The benchmark script fails the way a person can act on.

`scripts/bench.py` exists to be run on a machine that may be missing half its
tooling -- that is the whole point of measuring before committing a backlog to
local compute. So its failure modes ARE its interface: a missing file or a
missing ffmpeg must come back as one plain sentence and a nonzero exit, never
a traceback. These drive the script the way a shell does, in a subprocess, and
assert exactly that. No audio is processed here; the happy path's stages are
covered by the audio and ASR unit suites.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "scripts" / "bench.py"


def _run_bench(args: list[str], env: dict[str, str] | None = None):
    return subprocess.run(
        [sys.executable, str(BENCH), *args],
        capture_output=True, text=True, timeout=120, env=env,
    )


def test_a_missing_file_is_one_sentence_not_a_traceback(tmp_path):
    proc = _run_bench([str(tmp_path / "nope.wav")])
    assert proc.returncode != 0
    assert "Traceback" not in proc.stdout + proc.stderr
    assert "nope.wav" in proc.stderr, "the message should name the file it did not find"


def test_a_machine_without_ffmpeg_is_told_to_install_it(tmp_path):
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"RIFF0000WAVE")  # never decoded; ffmpeg is checked first
    env = dict(os.environ)
    env["PATH"] = str(tmp_path / "no-tools-here")  # nothing findable on PATH
    proc = _run_bench([str(clip)], env=env)
    assert proc.returncode != 0
    assert "Traceback" not in proc.stdout + proc.stderr
    assert "ffmpeg" in proc.stderr.lower()
    assert "install" in proc.stderr.lower()


def test_help_says_what_the_numbers_are_for(tmp_path):
    proc = _run_bench(["--help"])
    assert proc.returncode == 0
    assert "backlog" in proc.stdout.lower()
