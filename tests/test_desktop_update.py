"""
The updater: check, verify, hand off -- and refuse everywhere it cannot finish.

A loopback stub plays GitHub so no test touches the network. The one thing that
cannot run here is the final Windows folder swap; its script is tested for
content, and the apply path is tested up to the spawn it hands the script to.
"""

from __future__ import annotations

import hashlib
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from plaud_bridge.desktop import update
from plaud_bridge.desktop.update import (
    UpdateError,
    UpdateInfo,
    check_for_update,
    download_and_verify,
    is_newer,
    make_updater_script,
)


# =========================================================================
# Version arithmetic
# =========================================================================
@pytest.mark.parametrize("candidate,current,newer", [
    ("v1.0.1", "1.0.0", True),
    ("v2.0.0", "1.9.9", True),
    ("1.0.0", "1.0.0", False),
    ("v0.9.0", "1.0.0", False),
    ("v1.0", "1.0.0", False),          # shorter is not newer
    ("garbage", "1.0.0", False),       # malformed can never masquerade as newer
])
def test_version_comparison(candidate, current, newer):
    assert is_newer(candidate, current) is newer


# =========================================================================
# The check, against a loopback GitHub
# =========================================================================
@contextmanager
def _github(release: dict | None, status: int = 200, files: dict[str, bytes] | None = None):
    """A stub serving /repos/.../releases/latest and any asset files."""
    files = files or {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path.endswith("/releases/latest"):
                body = json.dumps(release or {}).encode()
                self.send_response(status)
            else:
                name = self.path.rsplit("/", 1)[-1]
                if name in files:
                    body = files[name]
                    self.send_response(200)
                else:
                    body = b"missing"
                    self.send_response(404)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def _release(tag: str, base: str = "", with_checksum: bool = True) -> dict:
    assets = [{"name": update.ASSET_NAME,
               "browser_download_url": f"{base}/dl/{update.ASSET_NAME}"}]
    if with_checksum:
        assets.append({"name": update.CHECKSUM_ASSET,
                       "browser_download_url": f"{base}/dl/{update.CHECKSUM_ASSET}"})
    return {"tag_name": tag, "body": "notes line", "assets": assets}


def test_a_newer_release_is_offered(monkeypatch):
    monkeypatch.delenv("PLAUD_BRIDGE_NO_UPDATE_CHECK", raising=False)
    with _github(_release("v99.0.0")) as base:
        info = check_for_update(repo="x/y", current="1.0.0", api_base=base)
    assert info is not None and info.version == "v99.0.0"
    assert info.notes.startswith("notes line")


def test_an_older_or_equal_release_is_not(monkeypatch):
    monkeypatch.delenv("PLAUD_BRIDGE_NO_UPDATE_CHECK", raising=False)
    with _github(_release("v1.0.0")) as base:
        assert check_for_update(repo="x/y", current="1.0.0", api_base=base) is None


def test_a_release_without_the_windows_bundle_is_not_offered(monkeypatch):
    monkeypatch.delenv("PLAUD_BRIDGE_NO_UPDATE_CHECK", raising=False)
    bare = {"tag_name": "v99.0.0", "assets": []}
    with _github(bare) as base:
        assert check_for_update(repo="x/y", current="1.0.0", api_base=base) is None


def test_check_failures_are_soft(monkeypatch):
    """Offline / private-without-token / no releases: no banner, never a crash."""
    monkeypatch.delenv("PLAUD_BRIDGE_NO_UPDATE_CHECK", raising=False)
    with _github(None, status=404) as base:
        assert check_for_update(repo="x/y", current="1.0.0", api_base=base) is None
    # A dead port entirely.
    assert check_for_update(repo="x/y", current="1.0.0",
                            api_base="http://127.0.0.1:9", timeout=0.3) is None


def test_the_kill_switch_disables_checking(monkeypatch):
    monkeypatch.setenv("PLAUD_BRIDGE_NO_UPDATE_CHECK", "1")
    assert check_for_update(repo="x/y", current="1.0.0") is None
    assert update.checks_disabled()


# =========================================================================
# Download verification
# =========================================================================
def _info(base: str, with_checksum: bool = True) -> UpdateInfo:
    return UpdateInfo(
        version="v99.0.0", current="1.0.0",
        zip_url=f"{base}/dl/{update.ASSET_NAME}",
        checksum_url=f"{base}/dl/{update.CHECKSUM_ASSET}" if with_checksum else "",
    )


def test_a_verified_download_lands(tmp_path, monkeypatch):
    monkeypatch.delenv("PLAUD_BRIDGE_NO_UPDATE_CHECK", raising=False)
    bundle = b"pretend this is the app zip"
    good = hashlib.sha256(bundle).hexdigest()
    files = {update.ASSET_NAME: bundle,
             update.CHECKSUM_ASSET: f"{good}  {update.ASSET_NAME}".encode()}
    with _github(_release("v99.0.0"), files=files) as base:
        path = download_and_verify(_info(base), tmp_path)
    assert path.read_bytes() == bundle


def test_a_checksum_mismatch_installs_nothing(tmp_path, monkeypatch):
    bundle = b"pretend this is the app zip"
    wrong = hashlib.sha256(b"something else").hexdigest()
    files = {update.ASSET_NAME: bundle,
             update.CHECKSUM_ASSET: f"{wrong}  {update.ASSET_NAME}".encode()}
    with _github(_release("v99.0.0"), files=files) as base:
        with pytest.raises(UpdateError, match="checksum"):
            download_and_verify(_info(base), tmp_path)
    assert not (tmp_path / update.ASSET_NAME).exists(), "a failed download was left on disk"


def test_a_release_without_a_checksum_is_refused(tmp_path):
    with pytest.raises(UpdateError, match="unverifiable"):
        download_and_verify(_info("http://127.0.0.1:9", with_checksum=False), tmp_path)


# =========================================================================
# Applying
# =========================================================================
def test_a_source_checkout_refuses_to_self_update():
    """Dev runs update with git pull; the zip-swap is for the frozen app only."""
    with pytest.raises(UpdateError, match="git pull"):
        update.apply_update(_info("http://127.0.0.1:9"))


def test_the_updater_script_swaps_waits_and_relaunches(tmp_path):
    script = make_updater_script(Path(r"C:\Apps\PlaudBridge"),
                                 tmp_path / "PlaudBridge-windows.zip", "PlaudBridge.exe")
    # The load-bearing lines: wait for the exe to unlock (rename-as-probe),
    # unpack the verified zip over the install dir, relaunch, self-delete.
    assert ":wait" in script and "goto wait" in script
    assert "Expand-Archive -Force" in script
    assert 'start "" "%EXE%"' in script
    assert 'del "%~f0"' in script


# =========================================================================
# The server routes
# =========================================================================
def test_server_check_and_apply_routes(tmp_path, monkeypatch):
    """The page's contract: check offers, apply refuses without an offer."""
    from plaud_bridge.desktop.controller import AppController
    from plaud_bridge.desktop.server import AppServer

    monkeypatch.setenv("PLAUD_BRIDGE_PASSPHRASE", "a-long-enough-desktop-passphrase")
    ROOT = Path(__file__).resolve().parents[1]
    app = AppServer(AppController(base_dir=tmp_path / "home", template_dir=ROOT / "config"))

    # Nothing offered yet: apply must refuse rather than install a stale click.
    refused = app.update_apply()
    assert refused["applied"] is False and "check first" in refused["error"]

    # A canned newer release: the check offers it, and apply then runs into the
    # honest source-checkout refusal (this is not the frozen exe), surfaced as
    # an error string for the page rather than an exception.
    monkeypatch.setattr(
        update, "check_for_update",
        lambda **_kw: _info("http://127.0.0.1:9"),
    )
    offered = app.update_check()
    assert offered["available"] is True and offered["version"] == "v99.0.0"
    result = app.update_apply()
    assert result["applied"] is False and "git pull" in result["error"]

    # The kill switch renders as disabled, not as "up to date".
    monkeypatch.setenv("PLAUD_BRIDGE_NO_UPDATE_CHECK", "1")
    assert app.update_check() == {"available": False, "disabled": True}
