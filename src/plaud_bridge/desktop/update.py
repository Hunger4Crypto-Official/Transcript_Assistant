"""
Updating the installed app, without ever surprising the person running it.

The shape is check -> tell -> one click, deliberately not silent replacement.
An app that swaps its own executable in the background is a supply-chain attack
with a release schedule, and this tool's whole posture is that nothing happens
to the archive that a person did not visibly choose. So: the page checks GitHub
Releases for a newer version, shows a banner, and only replaces anything when
the person clicks Update -- after the download's SHA-256 has been verified
against the checksum the build published next to it.

Two environmental facts shape the code:

  - The GitHub API needs a token when the repository is private. The check
    reads GITHUB_TOKEN (or PLAUD_BRIDGE_UPDATE_TOKEN) and sends it if present;
    without one, a private repo's check simply reports "cannot check", never a
    crash and never a scary error in the window.
  - A running Windows .exe cannot overwrite itself. The apply step therefore
    stages the verified zip, writes a small updater script that waits for this
    process to exit, swaps the install folder, and relaunches -- then the app
    shuts itself down and lets the script do the one thing it exists for.

Checking at all is a network call, which a privacy tool should be loud about:
it sends nothing but the request itself, and PLAUD_BRIDGE_NO_UPDATE_CHECK=1
turns it off entirely.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .. import __version__
from ..logging_setup import get

log = get("update")

DEFAULT_REPO = "Hunger4Crypto-Official/Transcript_Assistant"
ASSET_NAME = "PlaudBridge-windows.zip"
CHECKSUM_ASSET = "PlaudBridge-windows.zip.sha256"
_TIMEOUT = 6.0
_MAX_DOWNLOAD_BYTES = 4 * 1024 * 1024 * 1024   # the app bundle is large; 4 GiB ceiling


class UpdateError(RuntimeError):
    """Raised when an update cannot be performed honestly."""


@dataclass
class UpdateInfo:
    """One available release, as much as the banner and the apply step need."""

    version: str          # the tag, e.g. "v1.2.0"
    current: str          # what this build reports
    zip_url: str          # browser_download_url of the app bundle
    checksum_url: str     # browser_download_url of its .sha256, "" if absent
    notes: str = ""       # release body, first lines only


def _parse_version(raw: str) -> tuple[int, ...]:
    """
    "v1.2.3" -> (1, 2, 3). Anything unparseable compares as (0,), so a
    malformed tag can never masquerade as newer than a real version.
    """
    cleaned = raw.strip().lstrip("vV")
    parts: list[int] = []
    for piece in cleaned.split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) or (0,)


def is_newer(candidate: str, current: str) -> bool:
    return _parse_version(candidate) > _parse_version(current)


def _token() -> str:
    return (
        os.environ.get("PLAUD_BRIDGE_UPDATE_TOKEN", "").strip()
        or os.environ.get("GITHUB_TOKEN", "").strip()
    )


def checks_disabled() -> bool:
    return os.environ.get("PLAUD_BRIDGE_NO_UPDATE_CHECK", "").strip() not in ("", "0")


def _api_request(url: str, timeout: float) -> bytes:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "plaud-bridge"}
    token = _token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def check_for_update(repo: str | None = None, current: str | None = None,
                     api_base: str | None = None,
                     timeout: float = _TIMEOUT) -> UpdateInfo | None:
    """
    The newest release, when it is newer than this build. None otherwise.

    Soft-fails by design: no network, a private repo without a token, a repo
    with no releases yet -- all of those return None and log at debug. The
    banner's absence is the correct rendering of "could not check", and a
    launch must never be delayed or scared by an update endpoint.
    """
    if checks_disabled():
        return None
    repo = repo or os.environ.get("PLAUD_BRIDGE_UPDATE_REPO", "").strip() or DEFAULT_REPO
    current = current or __version__
    base = (api_base or "https://api.github.com").rstrip("/")

    try:
        raw = _api_request(f"{base}/repos/{repo}/releases/latest", timeout)
        release = json.loads(raw)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
            OSError, ValueError) as exc:
        log.debug("update check could not complete: %s", exc)
        return None

    tag = str(release.get("tag_name") or "")
    if not tag or not is_newer(tag, current):
        return None

    zip_url = checksum_url = ""
    for asset in release.get("assets") or []:
        name = str(asset.get("name") or "")
        if name == ASSET_NAME:
            zip_url = str(asset.get("browser_download_url") or "")
        elif name == CHECKSUM_ASSET:
            checksum_url = str(asset.get("browser_download_url") or "")
    if not zip_url:
        # A newer tag with no Windows bundle attached is a release for someone
        # else (or still building). Nothing to offer.
        return None

    notes = "\n".join(str(release.get("body") or "").strip().splitlines()[:6])
    return UpdateInfo(version=tag, current=current, zip_url=zip_url,
                      checksum_url=checksum_url, notes=notes)


def download_and_verify(info: UpdateInfo, dest_dir: Path,
                        timeout: float = 60.0) -> Path:
    """
    Fetch the bundle and prove it is the one the build published.

    The checksum comes from a separate asset the CI wrote at build time. A
    mismatch -- truncated download, tampered asset, wrong file -- deletes the
    download and raises; nothing unverified is ever handed to the apply step.
    A release without a checksum asset is refused outright rather than trusted.
    """
    if not info.checksum_url:
        raise UpdateError(
            f"release {info.version} has no {CHECKSUM_ASSET} beside the bundle; "
            "refusing to install an unverifiable download."
        )

    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / ASSET_NAME

    expected = _api_request(info.checksum_url, timeout).decode("utf-8").split()[0].strip().lower()
    if len(expected) != 64:
        raise UpdateError("the published checksum file is malformed; refusing to install.")

    digest = hashlib.sha256()
    written = 0
    req = urllib.request.Request(info.zip_url, headers={"User-Agent": "plaud-bridge"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(zip_path, "wb") as out:
        while chunk := resp.read(1024 * 1024):
            written += len(chunk)
            if written > _MAX_DOWNLOAD_BYTES:
                out.close()
                zip_path.unlink(missing_ok=True)
                raise UpdateError("the download exceeded the size ceiling; refusing it.")
            digest.update(chunk)
            out.write(chunk)

    if digest.hexdigest().lower() != expected:
        zip_path.unlink(missing_ok=True)
        raise UpdateError(
            "the downloaded bundle does not match its published checksum. "
            "Nothing was installed."
        )
    return zip_path


def _install_dir() -> Path:
    """Where the running app lives: the folder holding the frozen executable."""
    return Path(sys.executable).resolve().parent


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def make_updater_script(install_dir: Path, zip_path: Path, exe_name: str) -> str:
    """
    The batch script that finishes the job after this process exits.

    It waits until the exe unlocks (Windows keeps a running binary locked, so a
    successful rename IS the exit signal), unpacks the verified zip over the
    install folder, relaunches, and deletes itself. Kept dumb on purpose: by the
    time this runs there is no Python to report an error, so every step is the
    simplest thing that can work.
    """
    return "\r\n".join([
        "@echo off",
        "echo Updating Plaud Bridge... do not close this window.",
        f"set EXE={install_dir / exe_name}",
        ":wait",
        "timeout /t 2 /nobreak >nul",
        f'ren "%EXE%" "{exe_name}.old" >nul 2>&1',
        "if errorlevel 1 goto wait",
        f'del "{install_dir / (exe_name + ".old")}" >nul 2>&1',
        f'powershell -NoProfile -Command "Expand-Archive -Force \'{zip_path}\' \'{install_dir}\'"',
        'start "" "%EXE%"',
        f'del "{zip_path}" >nul 2>&1',
        '(goto) 2>nul & del "%~f0"',
        "",
    ])


def apply_update(info: UpdateInfo, *, spawn=None) -> str:
    """
    Download, verify, and hand off to the updater script. Returns a short
    human-readable line for the page. Refuses anywhere it cannot finish the
    job honestly: a source checkout updates with `git pull`, not a zip, and a
    non-Windows build has no updater script to run.

    `spawn` exists for tests: it receives the script path instead of
    subprocess starting a real detached cmd.exe.
    """
    if not is_frozen():
        raise UpdateError(
            "this is a source checkout, not the packaged app. Update with: git pull"
        )
    if not sys.platform.startswith("win"):
        raise UpdateError("self-update is only wired for the Windows build.")

    staging = Path(tempfile.mkdtemp(prefix="plaud-update-"))
    zip_path = download_and_verify(info, staging)

    install_dir = _install_dir()
    exe_name = Path(sys.executable).name
    script = staging / "apply-update.bat"
    script.write_text(
        make_updater_script(install_dir, zip_path, exe_name),
        encoding="utf-8",
    )

    if spawn is not None:
        spawn(script)
    else:  # pragma: no cover - real detached spawn, exercised only on Windows
        subprocess.Popen(
            ["cmd", "/c", "start", "", "/min", str(script)],
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "DETACHED_PROCESS", 0),
            close_fds=True,
        )
    log.info("update to %s staged; handing off to the updater script", info.version)
    return f"updating to {info.version}; the app will restart itself"
