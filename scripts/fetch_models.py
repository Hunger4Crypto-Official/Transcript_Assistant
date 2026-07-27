#!/usr/bin/env python3
"""
Fetch everything the offline install needs. Run this ONCE, on a machine with a
network, then carry the result across.

    python scripts/fetch_models.py --all
    # copy models/ and wheels/ to the offline machine, next to run.py

    # on the offline machine:
    pip install --no-index --find-links wheels -r requirements.txt
    python run.py doctor --offline

What it collects:

  models/whisper/<name>/          speech recognition weights (faster-whisper)
  models/diarization/<name>/      speaker separation weights (pyannote)
  wheels/                         every Python dependency, as wheels

The diarization model needs a HuggingFace token AND acceptance of the model
licence on its page, both one-time and both on this machine, not the offline
one. Without them speaker separation is simply unavailable offline and the
transcript carries a single unlabelled speaker -- which is the documented
degraded mode, not a failure.

Nothing here runs on the air-gapped machine. That is the entire point.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WHISPER = "large-v3"
DEFAULT_DIARIZATION = "pyannote/speaker-diarization-3.1"


def _flat(name: str) -> str:
    """models/whisper/large-v3, models/diarization/pyannote__speaker-...-3.1"""
    return name.replace("/", "__")


def fetch_whisper(name: str, models_dir: Path) -> int:
    dest = models_dir / "whisper" / _flat(name)
    print(f"\n== speech recognition: {name}\n   -> {dest}")
    try:
        from faster_whisper import download_model
    except ImportError:
        print("   faster-whisper is not installed here. Run:")
        print("     pip install faster-whisper")
        return 1

    dest.mkdir(parents=True, exist_ok=True)
    try:
        download_model(name, output_dir=str(dest))
    except Exception as exc:  # noqa: BLE001 - this is a fetch tool, report and move on
        print(f"   FAILED: {exc}")
        return 1
    print(f"   done ({sum(f.stat().st_size for f in dest.rglob('*') if f.is_file()) / 1e6:.0f} MB)")
    return 0


def fetch_diarization(name: str, models_dir: Path, token: str | None) -> int:
    dest = models_dir / "diarization" / _flat(name)
    print(f"\n== speaker separation: {name}\n   -> {dest}")
    if not token:
        print("   no HUGGINGFACE_TOKEN set, and this model needs one to download.")
        print("   1. create a token at huggingface.co/settings/tokens")
        print(f"   2. accept the licence at huggingface.co/{name}")
        print("   3. export HUGGINGFACE_TOKEN=hf_... and run this again")
        print("   Skipping. Speaker separation will be unavailable offline.")
        return 1

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("   huggingface_hub is not installed here. Run:")
        print("     pip install huggingface_hub")
        return 1

    try:
        snapshot_download(repo_id=name, local_dir=str(dest), token=token)
    except Exception as exc:  # noqa: BLE001
        print(f"   FAILED: {exc}")
        print("   The most common cause is not having accepted the model licence.")
        return 1
    print("   done")
    return 0


def fetch_wheels(wheels_dir: Path, extras: list[str]) -> int:
    print(f"\n== python dependencies\n   -> {wheels_dir}")
    wheels_dir.mkdir(parents=True, exist_ok=True)
    spec = f".[{','.join(extras)}]" if extras else "."
    cmd = [sys.executable, "-m", "pip", "download", spec, "-d", str(wheels_dir)]
    print(f"   {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ROOT, check=False)
    if result.returncode != 0:
        print("   FAILED. Some packages have no wheel for this platform and must")
        print("   be downloaded on a machine matching the offline one.")
        return 1
    count = len(list(wheels_dir.glob("*")))
    print(f"   done ({count} files)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Collect models and wheels for an offline install.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--all", action="store_true", help="models and wheels, the usual case")
    ap.add_argument("--whisper", nargs="?", const=DEFAULT_WHISPER, default=None,
                    metavar="NAME", help=f"speech recognition weights (default {DEFAULT_WHISPER})")
    ap.add_argument("--diarization", nargs="?", const=DEFAULT_DIARIZATION, default=None,
                    metavar="NAME", help="speaker separation weights")
    ap.add_argument("--wheels", action="store_true", help="download every dependency as a wheel")
    ap.add_argument("--extras", default="local-asr,diarize",
                    help="optional dependency groups to include in the wheel set")
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    ap.add_argument("--wheels-dir", default=str(ROOT / "wheels"))
    args = ap.parse_args(argv)

    if not any([args.all, args.whisper, args.diarization, args.wheels]):
        ap.print_help()
        return 1

    models_dir = Path(args.models_dir)
    failures = 0

    if args.all or args.whisper:
        failures += fetch_whisper(args.whisper or DEFAULT_WHISPER, models_dir)
    if args.all or args.diarization:
        failures += fetch_diarization(
            args.diarization or DEFAULT_DIARIZATION, models_dir,
            os.environ.get("HUGGINGFACE_TOKEN", "").strip() or None,
        )
    if args.all or args.wheels:
        failures += fetch_wheels(
            Path(args.wheels_dir), [e for e in args.extras.split(",") if e]
        )

    print("\n" + "=" * 68)
    if failures:
        print(f"{failures} step(s) did not complete. Read the messages above; each one")
        print("says what to do. The offline machine will refuse to start rather than")
        print("silently downloading what is missing.")
    else:
        print("Everything collected. Copy these to the offline machine:")
        print(f"  {models_dir}")
        print(f"  {Path(args.wheels_dir)}")
        print("\nThen there:")
        print("  pip install --no-index --find-links wheels -e .")
        print("  # set runtime.offline: true in config/pipeline.yaml")
        print("  python run.py doctor --offline")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
