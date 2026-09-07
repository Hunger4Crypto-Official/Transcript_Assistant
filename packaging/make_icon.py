#!/usr/bin/env python3
"""
Draw the app icon at build time so no binary is committed to the repo.

Produces `packaging/icon.ico` from a simple rounded-square mark. Needs Pillow
(the CI installs it); if Pillow is missing this exits 0 without an icon, and the
build falls back to PyInstaller's default rather than failing.
"""

from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parent / "icon.ico"


def main() -> int:
    try:
        from PIL import Image, ImageDraw
    except Exception:  # noqa: BLE001
        print("Pillow not installed; skipping icon (build will use the default).")
        return 0

    size = 256
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # A deep-blue rounded square with a simple white waveform: a recording tool.
    d.rounded_rectangle([8, 8, size - 8, size - 8], radius=52, fill=(31, 78, 138, 255))
    mid = size // 2
    bars = [(70, 60), (104, 110), (138, 30), (172, 90), (206, 55)]
    for x, h in bars:
        d.rounded_rectangle([x - 12, mid - h, x + 12, mid + h], radius=12, fill=(255, 255, 255, 235))

    sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    img.save(OUT, format="ICO", sizes=sizes)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
