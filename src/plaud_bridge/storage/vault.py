"""
Encrypted at-rest storage for high and maximum sensitivity artifacts.

Design decision worth understanding: if encryption is unavailable, this module
REFUSES to write rather than falling back to plaintext. A tool that silently
degrades its own security guarantee is worse than one that stops and tells you.
You will notice a crash. You will not notice a quiet plaintext write.

Key derivation is scrypt over a passphrase from the environment. The key is
never written to disk. Lose the passphrase and the vault is gone, which is the
correct behaviour for a vault.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
from pathlib import Path
from typing import Any

try:  # pragma: no cover - availability varies by machine
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    _AESGCM_AVAILABLE = True
except Exception:  # pragma: no cover
    AESGCM = None  # type: ignore[assignment]
    _AESGCM_AVAILABLE = False

MAGIC = b"PBV1"
# Streaming format, for artifacts too large to hold in memory twice. A full day
# of audio is several hundred megabytes, and encrypting it the one-shot way
# needs the plaintext and the ciphertext resident at the same time.
MAGIC_STREAM = b"PBS1"
STREAM_CHUNK = 4 * 1024 * 1024
SCRYPT_N = 2**15
SCRYPT_R = 8
SCRYPT_P = 1
KEY_LEN = 32
NONCE_LEN = 12
SALT_LEN = 16
MIN_PASSPHRASE_LEN = 16


class VaultError(RuntimeError):
    """Raised when the vault cannot honour its security guarantee."""


class Vault:
    def __init__(self, root: Path, passphrase_env: str = "PLAUD_BRIDGE_PASSPHRASE"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._passphrase_env = passphrase_env

    @staticmethod
    def available() -> bool:
        return _AESGCM_AVAILABLE

    def ready(self) -> tuple[bool, str]:
        if not _AESGCM_AVAILABLE:
            return False, "the 'cryptography' package is not installed. Run: pip install cryptography"
        raw = os.environ.get(self._passphrase_env, "").strip()
        if not raw:
            return False, (
                f"environment variable {self._passphrase_env} is not set. "
                "Encrypted profiles cannot be processed without it."
            )
        if len(raw) < MIN_PASSPHRASE_LEN:
            return False, f"{self._passphrase_env} must be at least {MIN_PASSPHRASE_LEN} characters."
        return True, "ready"

    def _passphrase(self) -> bytes:
        ok, why = self.ready()
        if not ok:
            raise VaultError(f"{why} Refusing to write sensitive artifacts in plaintext.")
        return os.environ[self._passphrase_env].strip().encode("utf-8")

    def _derive(self, salt: bytes) -> bytes:
        # OpenSSL enforces a default scrypt memory ceiling of 32MB, and these
        # parameters need exactly 128 * N * r bytes, which lands right on it and
        # raises "memory limit exceeded". Pass maxmem explicitly with headroom.
        maxmem = 128 * SCRYPT_N * SCRYPT_R * 2
        return hashlib.scrypt(
            self._passphrase(), salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P,
            dklen=KEY_LEN, maxmem=maxmem,
        )

    def encrypt_bytes(self, plaintext: bytes, aad: bytes = b"") -> bytes:
        if not _AESGCM_AVAILABLE:
            raise VaultError(
                "cryptography is not installed; refusing to write sensitive data "
                "unencrypted. Install it with: pip install cryptography"
            )
        salt = secrets.token_bytes(SALT_LEN)
        nonce = secrets.token_bytes(NONCE_LEN)
        key = self._derive(salt)
        blob = AESGCM(key).encrypt(nonce, plaintext, aad)
        return MAGIC + salt + nonce + blob

    def decrypt_bytes(self, payload: bytes, aad: bytes = b"") -> bytes:
        if not _AESGCM_AVAILABLE:
            raise VaultError("cryptography is not installed; cannot decrypt.")
        if not payload.startswith(MAGIC):
            raise VaultError("not a plaud-bridge vault file (bad magic header)")
        off = len(MAGIC)
        salt = payload[off : off + SALT_LEN]
        nonce = payload[off + SALT_LEN : off + SALT_LEN + NONCE_LEN]
        blob = payload[off + SALT_LEN + NONCE_LEN :]
        key = self._derive(salt)
        try:
            return AESGCM(key).decrypt(nonce, blob, aad)
        except Exception as exc:
            raise VaultError(
                "decryption failed. Either the passphrase is wrong or the file has been modified."
            ) from exc

    def write(self, relative: str, data: str | bytes, recording_id: str = "") -> Path:
        payload = data.encode("utf-8") if isinstance(data, str) else data
        # Bind ciphertext to its recording id so files cannot be silently
        # swapped between recordings.
        aad = recording_id.encode("utf-8")
        dest = self.root / f"{relative}.enc"
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".tmp")
        tmp.write_bytes(self.encrypt_bytes(payload, aad))
        os.replace(tmp, dest)
        try:
            os.chmod(dest, 0o600)
        except OSError:
            pass
        return dest

    # ---- streaming -------------------------------------------------------
    #
    # A day of audio does not fit comfortably in memory twice, so large
    # artifacts are encrypted a chunk at a time into a second format.
    #
    # Each chunk carries its own nonce and is bound by AAD to the recording id,
    # its own index, and whether it is the last one. That is what stops the
    # three attacks a naive chunked format invites: reordering chunks, dropping
    # chunks from the middle, and truncating the file. A reader that does not
    # see index 0, then 1, then 2, ending on a chunk marked final, refuses.
    def _stream_aad(self, recording_id: str, index: int, final: bool) -> bytes:
        return f"{recording_id}|{index}|{'end' if final else 'mid'}".encode()

    def write_stream(self, relative: str, source: Path, recording_id: str = "",
                     chunk_size: int = STREAM_CHUNK) -> Path:
        """Encrypt a file into the vault without loading it whole."""
        if not _AESGCM_AVAILABLE:
            raise VaultError(
                "cryptography is not installed; refusing to write sensitive data "
                "unencrypted. Install it with: pip install cryptography"
            )
        source = Path(source)
        salt = secrets.token_bytes(SALT_LEN)
        key = self._derive(salt)
        aes = AESGCM(key)

        dest = self.root / f"{relative}.enc"
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".tmp")

        try:
            with open(source, "rb") as fin, open(tmp, "wb") as fout:
                fout.write(MAGIC_STREAM)
                fout.write(salt)
                fout.write(chunk_size.to_bytes(4, "big"))

                index = 0
                block = fin.read(chunk_size)
                while True:
                    nxt = fin.read(chunk_size)
                    final = not nxt
                    nonce = secrets.token_bytes(NONCE_LEN)
                    body = aes.encrypt(nonce, block, self._stream_aad(recording_id, index, final))
                    fout.write(nonce)
                    fout.write(len(body).to_bytes(4, "big"))
                    fout.write(body)
                    if final:
                        break
                    block, index = nxt, index + 1
            os.replace(tmp, dest)
        except OSError:
            tmp.unlink(missing_ok=True)
            raise

        try:
            os.chmod(dest, 0o600)
        except OSError:
            pass
        return dest

    def read_stream(self, path: Path, dest: Path, recording_id: str = "") -> Path:
        """Decrypt a streamed artifact to a file, a chunk at a time."""
        if not _AESGCM_AVAILABLE:
            raise VaultError("cryptography is not installed; cannot decrypt.")
        path, dest = Path(path), Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "rb") as fin:
            if fin.read(len(MAGIC_STREAM)) != MAGIC_STREAM:
                # Not streamed. Small enough for the one-shot path by definition.
                dest.write_bytes(self.read(path, recording_id))
                return dest

            salt = fin.read(SALT_LEN)
            fin.read(4)                       # chunk size, informational
            key = self._derive(salt)
            aes = AESGCM(key)

            tmp = dest.with_name(dest.name + ".part")
            saw_final = False
            try:
                with open(tmp, "wb") as fout:
                    index = 0
                    while True:
                        nonce = fin.read(NONCE_LEN)
                        if not nonce:
                            break
                        raw = fin.read(4)
                        if len(nonce) != NONCE_LEN or len(raw) != 4:
                            raise VaultError("vault file is truncated mid-chunk")
                        body = fin.read(int.from_bytes(raw, "big"))

                        # Try "not final" first, then "final". The AAD is what
                        # makes a dropped or reordered chunk fail rather than
                        # silently producing a shorter file.
                        for final in (False, True):
                            try:
                                fout.write(aes.decrypt(
                                    nonce, body, self._stream_aad(recording_id, index, final)
                                ))
                                saw_final = final
                                break
                            except Exception:  # noqa: BLE001 - try the other marker
                                continue
                        else:
                            raise VaultError(
                                f"decryption failed at chunk {index}. Either the "
                                "passphrase is wrong or the file has been modified."
                            )
                        if saw_final:
                            break
                        index += 1

                if not saw_final:
                    raise VaultError(
                        "vault file ended without its final chunk; it has been "
                        "truncated. Refusing to hand back a partial recording."
                    )
                os.replace(tmp, dest)
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise
        return dest

    @staticmethod
    def is_streamed(path: Path) -> bool:
        try:
            with open(path, "rb") as fh:
                return fh.read(len(MAGIC_STREAM)) == MAGIC_STREAM
        except OSError:
            return False

    def read(self, path: Path, recording_id: str = "") -> bytes:
        return self.decrypt_bytes(Path(path).read_bytes(), recording_id.encode("utf-8"))

    def read_text(self, path: Path, recording_id: str = "") -> str:
        return self.read(path, recording_id).decode("utf-8")

    def read_json(self, path: Path, recording_id: str = "") -> Any:
        return json.loads(self.read_text(path, recording_id))

    @staticmethod
    def fingerprint(path: Path, chunk: int = 1 << 20) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            while block := fh.read(chunk):
                h.update(block)
        return h.hexdigest()

    @staticmethod
    def constant_time_equal(a: str, b: str) -> bool:
        return hmac.compare_digest(a, b)


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")
