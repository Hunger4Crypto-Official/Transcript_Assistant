#!/usr/bin/env python3
"""
Plaud Bridge, without installing anything.

    python run.py doctor                        preflight every dependency and key
    python run.py run                           process everything in the inbox
    python run.py digest                        combined digest, last 7 days
    python run.py digest --profile husband      one profile only
    python run.py status                        index summary
    python run.py search "elimination period"   find recordings
    python run.py open <recording_id>           decrypt and print an artifact
    python run.py audit                         read the compliance audit log
    python run.py release <recording_id>        release a quarantined recording
    python run.py retention --execute           delete expired artifacts
    python run.py profiles                      show the routing table

The commands live in `src/plaud_bridge/cli.py`. This shim exists so the tool
works straight out of a `git clone` with nothing installed but PyYAML and
cryptography. `pip install -e .` gives you the same commands as `plaud-bridge`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from plaud_bridge.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
