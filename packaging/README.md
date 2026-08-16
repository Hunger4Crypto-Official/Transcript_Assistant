# The clickable Windows app

This folder builds Plaud Bridge into a double-clickable Windows app, so you can
run everything from a window in your browser instead of typing commands.

## How to get the app (no dev tools needed)

A Windows `.exe` can only be built on Windows, so a GitHub Actions runner builds
it for you:

1. On GitHub, open the repo's **Actions** tab.
2. Pick **Build Windows app** on the left, then **Run workflow**.
3. When it finishes (a few minutes), open the run and download
   **PlaudBridge-windows.zip** from the *Artifacts* section.
   - Or push a tag like `v1.0.0` and the zip is attached to a **Release**.
4. Unzip it anywhere (Desktop is fine). Inside is **PlaudBridge.exe**.

## Running it

Double-click **PlaudBridge.exe**.

- The first time, Windows shows a blue **"Windows protected your PC"** box,
  because the app isn't code-signed (signing costs money). Click **More info →
  Run anyway**. You only do this once.
- A small black window opens and prints a web address. Your browser opens to the
  app by itself. **Keep the black window open** while you use it; closing it
  stops the app.
- In the browser: type your passphrase, pick Offline or the free cloud key, pick
  your recordings, and press **Process**. When it's done, **Open digest**.

Your recordings, transcripts, and the encrypted vault live in a per-user folder
(`%LOCALAPPDATA%\PlaudBridge`), not inside the app folder, so updating the app
never touches your data.

## Building it locally instead (optional)

On a Windows machine with Python installed:

```bat
pip install -r requirements.txt
pip install faster-whisper pyinstaller pillow
python packaging\make_icon.py
pyinstaller packaging\PlaudBridge.spec --noconfirm
```

The app appears at `dist\PlaudBridge\PlaudBridge.exe`. (ffmpeg isn't bundled by a
local build unless you drop `ffmpeg.exe` and `ffprobe.exe` into `packaging\bin\`
first; without it the app runs but tells you to install ffmpeg.)

## The two "brains"

- **Free cloud key** — get a free key at groq.com, paste it in. Works for work
  recordings.
- **Offline** — install [Ollama](https://ollama.com) and pull a model once; then
  nothing leaves your machine, and family/spousal recordings work too. (Those are
  forced offline and encrypted regardless of what you pick.)
