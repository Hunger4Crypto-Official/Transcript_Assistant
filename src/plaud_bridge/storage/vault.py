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
