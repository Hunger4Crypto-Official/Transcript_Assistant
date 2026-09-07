# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller build recipe for the Windows desktop app.

Run on Windows (the CI does this for you):

    pyinstaller packaging/PlaudBridge.spec

It produces `dist/PlaudBridge/PlaudBridge.exe` plus its support files. Three
things have to travel with the code or the app is dead on arrival on a machine
that has never seen this project:

  1. The `config/` template -- profiles, glossary, voice packs. The app copies it
     into a per-user folder on first run.
  2. ffmpeg.exe and ffprobe.exe -- the app shells them for every audio file. The
     CI downloads a static build and drops them in `packaging/bin/` before this
     runs; if they are absent the build still succeeds and the app tells the
     person to install ffmpeg.
  3. faster-whisper / ctranslate2 native pieces, which PyInstaller finds via the
     collect helpers below.

One-folder, not one-file: ctranslate2 ships native DLLs that a one-file build has
to unpack to a temp dir on every launch, which antivirus loves to quarantine. A
folder the person unzips once is slower to ship and far more reliable to run.
"""

import os
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules

ROOT = os.path.abspath(os.getcwd())

datas = [(os.path.join(ROOT, "config"), "config")]

# Bundle ffmpeg/ffprobe if the build step placed them here.
_bin = os.path.join(ROOT, "packaging", "bin")
binaries = []
if os.path.isdir(_bin):
    for exe in ("ffmpeg.exe", "ffprobe.exe"):
        p = os.path.join(_bin, exe)
        if os.path.isfile(p):
            binaries.append((p, "."))

# Native libs for the local transcription model. Missing on a build without
# faster-whisper installed -- that only disables offline ASR, not the app.
for pkg in ("ctranslate2", "faster_whisper", "onnxruntime"):
    try:
        binaries += collect_dynamic_libs(pkg)
    except Exception:
        pass

hiddenimports = []
for pkg in ("faster_whisper", "ctranslate2", "plaud_bridge"):
    try:
        hiddenimports += collect_submodules(pkg)
    except Exception:
        pass

_icon = os.path.join(ROOT, "packaging", "icon.ico")
icon = _icon if os.path.isfile(_icon) else None

a = Analysis(
    [os.path.join(ROOT, "desktop_app.py")],
    pathex=[os.path.join(ROOT, "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "pytest"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PlaudBridge",
    console=True,          # a console window that prints the URL and how to stop
    icon=icon,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name="PlaudBridge",
)
