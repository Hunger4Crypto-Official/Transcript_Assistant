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

## Use it from your phone

Start the app in **phone mode** and your phone becomes a second screen for it —
same tabs, same engine, nothing in any cloud:

1. Make a shortcut to `PlaudBridge.exe` and add ` --phone` to its Target (or set
   the environment variable `PLAUD_BRIDGE_PHONE=1`). Windows Firewall will ask
   once — allow it on **Private networks only**.
2. The black window (and the Tools tab) shows an address like
   `http://192.168.1.23:54321/?token=...`. Open it on a phone connected to the
   **same Wi-Fi**.
3. In the phone's browser menu, choose **Add to Home Screen**. It installs like
   an app, icon and all.

Honest limits: the link carries the session's key and traffic on your Wi-Fi is
not encrypted, so home network only — never public Wi-Fi. The address changes
each launch (new port, new key), so re-open it from the Tools tab after a
restart. Without `--phone`, the app answers your own machine only, exactly as
before.

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

## Updates

The app checks for a newer release when it opens and shows an **Update
available** banner. Clicking **Update now** downloads the new build, verifies
it against the checksum the build published, closes the app, swaps the folder,
and reopens itself. Notes:

- Updates come from this repo's **Releases**, so they only exist for builds
  made from a version tag (`git tag v1.0.1 && git push --tags`). Manual
  Actions-tab builds are downloads, not updates.
- If this repo is **private**, the installed app can only see releases when a
  `GITHUB_TOKEN` environment variable is set on that machine. If the repo is
  public, it just works.
- The check is one HTTPS request to GitHub at launch. To turn it off entirely,
  set `PLAUD_BRIDGE_NO_UPDATE_CHECK=1`.
- Your data and tuned config live outside the app folder, so an update never
  touches them.

## The two "brains"

- **Free cloud key** — get a free key at groq.com, paste it in. Works for work
  recordings.
- **Offline** — install [Ollama](https://ollama.com) and pull a model once; then
  nothing leaves your machine, and family/spousal recordings work too. (Those are
  forced offline and encrypted regardless of what you pick.)
