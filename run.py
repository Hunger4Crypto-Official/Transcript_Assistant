#!/usr/bin/env python3
"""
Plaud Bridge, without installing anything.

    python run.py doctor                        preflight every dependency and key
    python run.py run                           process everything in the inbox
    python run.py watch                         keep processing on an interval
    python run.py digest                        combined digest, last 7 days
    python run.py digest --format html          self-contained page, prints cleanly
    python run.py review                        what the review cadence says is due
    python run.py followups                     commitments still open, oldest first
    python run.py status                        index summary
    python run.py search "own occupation" --content    search what was actually said
    python run.py ask "what did I promise Marcus?"     answer it, with citations
    python run.py open <recording_id>           decrypt and print an artifact
    python run.py verify                        confirm every artifact still opens
    python run.py backup                        everything worth keeping, one encrypted file
    python run.py restore <file>                bring it all back, passphrase required
    python run.py export                        redacted document for someone else
    python run.py forget <recording_id>         delete one recording, permanently
    python run.py memory                        what it has learned across recordings
    python run.py people                        everyone it has heard, by name
    python run.py audit                         read the compliance audit log
    python run.py release <recording_id>        release a quarantined recording
    python run.py quarantine                    triage everything in quarantine at once
    python run.py retention --execute           delete expired artifacts
    python run.py profiles                      show the routing table
    python run.py new-profile <id>              scaffold a profile from the template
    python run.py voices                        show installed voice packs
    python run.py speakers list                 who this archive can name

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
