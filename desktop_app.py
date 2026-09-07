#!/usr/bin/env python3
"""
The entry point the packaged app freezes around.

Double-clicking the built .exe runs this: it puts `src/` on the path the same way
`run.py` does (so a plain checkout also works with `python desktop_app.py`), then
hands off to the launcher, which starts the loopback server and opens a browser.
Kept to a few lines on purpose -- all behaviour lives in `plaud_bridge.desktop`.
"""

from __future__ import annotations

import sys
from pathlib import Path

# In a normal checkout the package lives under src/. In a PyInstaller build the
# package is importable directly, and this line is a harmless no-op.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from plaud_bridge.desktop.launch import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
