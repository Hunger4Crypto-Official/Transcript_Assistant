"""
The vault's refusals, one at a time.

The round-trip guarantees live in test_vault_and_digest.py. These pin the
edges: what happens when the cipher library is missing (refuse, never degrade),
when the filesystem will not take a permission bit (keep the file, it is still
encrypted), and what a reader does with a stream that has been cut short (raise,
and leave no partial plaintext behind). Each one fails if its guard is removed.
"""

from __future__ import annotations

import json
import os

import pytest

from plaud_bridge.storage import vault as vault_module
from plaud_bridge.storage.vault import (
    GCM_TAG_LEN,
    MAGIC_STREAM,
    NONCE_LEN,
    SALT_LEN,
    STREAM_CHUNK_MAX,
    Vault,
    VaultError,
    b64,
)

PASS = "a-sufficiently-long-test-passphrase"


@pytest.fixture
def vault(tmp_path, monkeypatch):
    monkeypatch.setenv("PLAUD_BRIDGE_PASSPHRASE", PASS)
    return Vault(tmp_path / "vault")


@pytest.fixture
def no_cipher(monkeypatch):
    """The machine where `pip install cryptography` never happened."""
    monkeypatch.setattr(vault_module, "_AESGCM_AVAILABLE", False)


def _files_under(root) -> list:
    return [p for p in root.rglob("*") if p.is_file()]


# =========================================================================
# Without the cipher library, everything refuses
# =========================================================================
def test_available_reports_whether_the_cipher_is_installed(monkeypatch):
    assert Vault.available() is True
    monkeypatch.setattr(vault_module, "_AESGCM_AVAILABLE", False)
    assert Vault.available() is False


def test_ready_names_the_missing_package_before_it_looks_at_the_passphrase(vault, no_cipher):
    """A correct passphrase is no help without the cipher, and the message says which."""
    ok, why = vault.ready()
    assert ok is False
    assert "'cryptography' package is not installed" in why
    assert "pip install cryptography" in why


def test_a_missing_cipher_refuses_to_write_and_leaves_no_file(vault, no_cipher):
    with pytest.raises(VaultError, match="refusing to write sensitive data unencrypted"):
        vault.write("day/rec.transcript.md", "the words that were said", "rec_1")
    assert _files_under(vault.root) == [], "a refusal wrote something anyway"


def test_a_missing_cipher_refuses_to_decrypt_a_file_it_could_have_read(
    vault, monkeypatch, tmp_path
):
    path = vault.write("x.md", "private", "rec_1")
    monkeypatch.setattr(vault_module, "_AESGCM_AVAILABLE", False)
    with pytest.raises(VaultError, match="cannot decrypt"):
        vault.read(path, "rec_1")


def test_a_missing_cipher_refuses_to_stream_a_file_in(vault, no_cipher, tmp_path):
    source = tmp_path / "audio.bin"
    source.write_bytes(b"\x00" * 100)
    with pytest.raises(VaultError, match="refusing to write sensitive data unencrypted"):
        vault.write_stream("day/rec.source.bin", source, "rec_1")
    assert _files_under(vault.root) == []


def test_a_missing_cipher_refuses_to_stream_a_file_out(vault, monkeypatch, tmp_path):
    source = tmp_path / "audio.bin"
    source.write_bytes(os.urandom(300))
    path = vault.write_stream("day/rec.source.bin", source, "rec_1", chunk_size=128)

    monkeypatch.setattr(vault_module, "_AESGCM_AVAILABLE", False)
    with pytest.raises(VaultError, match="cannot decrypt"):
        list(vault.iter_plaintext(path, "rec_1"))
    with pytest.raises(VaultError, match="cannot decrypt"):
        vault.verify_stream(path, "rec_1")


# =========================================================================
# A permission bit that will not stick is not a reason to lose the file
# =========================================================================
def test_a_chmod_failure_keeps_the_encrypted_artifact(vault, monkeypatch):
    """The 0600 mode is belt and braces; the ciphertext is the guarantee."""
    def refuse(*_a, **_k):
        raise OSError("chmod is not supported on this filesystem")

    monkeypatch.setattr(vault_module.os, "chmod", refuse)
    path = vault.write("x.md", "still encrypted", "rec_1")
    assert path.exists()
    assert b"still encrypted" not in path.read_bytes()
    assert vault.read_text(path, "rec_1") == "still encrypted"


def test_a_chmod_failure_keeps_the_streamed_artifact(vault, monkeypatch, tmp_path):
    def refuse(*_a, **_k):
        raise OSError("chmod is not supported on this filesystem")

    source = tmp_path / "audio.bin"
    body = os.urandom(500)
    source.write_bytes(body)

    monkeypatch.setattr(vault_module.os, "chmod", refuse)
    path = vault.write_stream("day/rec.source.bin", source, "rec_1", chunk_size=128)
    assert path.exists()
    assert b"".join(vault.iter_plaintext(path, "rec_1")) == body


# =========================================================================
# Small helpers that the rest of the tool leans on
# =========================================================================
def test_read_json_returns_the_structure_that_was_written(vault):
    payload = {"question": "what did I promise?", "citations": [{"stamp": "00:45"}]}
    path = vault.write("ask/answer", json.dumps(payload), "")
    assert vault.read_json(path, "") == payload


def test_constant_time_equal_compares_fingerprints():
    assert Vault.constant_time_equal("abc", "abc") is True
    assert Vault.constant_time_equal("abc", "abd") is False


def test_b64_is_plain_standard_base64():
    assert b64(b"\x00\xff") == "AP8="
    assert b64(b"") == ""


def test_a_file_without_the_magic_header_is_not_a_vault_file(vault, tmp_path):
    stray = tmp_path / "stray.enc"
    stray.write_bytes(b"not a vault file at all")
    with pytest.raises(VaultError, match="bad magic header"):
        vault.read(stray, "rec_1")


def test_is_streamed_is_false_for_a_file_that_is_not_there(tmp_path):
    assert Vault.is_streamed(tmp_path / "never.enc") is False


def test_iter_plaintext_serves_a_one_shot_artifact_in_a_single_chunk(vault):
    path = vault.write("day/rec.transcript.md", "the words that were said", "rec_1")
    assert Vault.is_streamed(path) is False
    assert list(vault.iter_plaintext(path, "rec_1")) == [b"the words that were said"]


def test_a_source_that_cannot_be_opened_leaves_no_half_written_stream(vault, tmp_path):
    with pytest.raises(OSError):
        vault.write_stream("day/rec.source.mp3", tmp_path / "never.mp3", "rec_1")
    assert _files_under(vault.root) == [], "a .tmp was left behind after a failed stream write"


# =========================================================================
# A stream that has been cut, padded or reshaped refuses whole
# =========================================================================
def _streamed(vault, tmp_path, size=1000, chunk=256):
    source = tmp_path / "audio.bin"
    body = os.urandom(size)
    source.write_bytes(body)
    return vault.write_stream("day/rec.source.bin", source, "rec_1", chunk_size=chunk), body


def test_a_truncated_stream_header_is_refused(vault, tmp_path):
    path, _ = _streamed(vault, tmp_path)
    path.write_bytes(MAGIC_STREAM + b"\x00" * (SALT_LEN - 1))
    with pytest.raises(VaultError, match="header is truncated"):
        vault.verify_stream(path, "rec_1")


def test_an_implausible_declared_chunk_size_is_refused_before_any_allocation(vault, tmp_path):
    """A 40-byte crafted header must not be able to ask for gigabytes."""
    path, _ = _streamed(vault, tmp_path)
    raw = bytearray(path.read_bytes())
    off = len(MAGIC_STREAM) + SALT_LEN
    raw[off:off + 4] = (STREAM_CHUNK_MAX + 1).to_bytes(4, "big")
    path.write_bytes(bytes(raw))
    with pytest.raises(VaultError, match="implausible chunk size"):
        vault.verify_stream(path, "rec_1")

    raw[off:off + 4] = (0).to_bytes(4, "big")
    path.write_bytes(bytes(raw))
    with pytest.raises(VaultError, match="implausible chunk size"):
        vault.verify_stream(path, "rec_1")


def test_a_chunk_length_outside_the_declared_bound_is_refused(vault, tmp_path):
    path, _ = _streamed(vault, tmp_path, chunk=256)
    raw = bytearray(path.read_bytes())
    off = len(MAGIC_STREAM) + SALT_LEN + 4 + NONCE_LEN     # first chunk's length field
    raw[off:off + 4] = (256 + GCM_TAG_LEN + 1).to_bytes(4, "big")
    path.write_bytes(bytes(raw))
    with pytest.raises(VaultError, match="chunk length is out of range"):
        vault.verify_stream(path, "rec_1")

    raw[off:off + 4] = (GCM_TAG_LEN - 1).to_bytes(4, "big")
    path.write_bytes(bytes(raw))
    with pytest.raises(VaultError, match="chunk length is out of range"):
        vault.verify_stream(path, "rec_1")


def test_a_stream_cut_off_before_its_final_chunk_is_refused_and_nothing_partial_is_kept(
    vault, tmp_path
):
    """
    The reader must not hand back the chunks that did arrive as if they were
    the recording. `read_stream` raises and leaves neither the destination nor
    its .part file behind.
    """
    path, _ = _streamed(vault, tmp_path, size=1000, chunk=256)
    raw = path.read_bytes()
    header = len(MAGIC_STREAM) + SALT_LEN + 4
    first_chunk = NONCE_LEN + 4 + 256 + GCM_TAG_LEN
    path.write_bytes(raw[: header + first_chunk])      # exactly one whole, non-final chunk

    dest = tmp_path / "out" / "audio.bin"
    with pytest.raises(VaultError, match="ended without its final chunk"):
        vault.read_stream(path, dest, "rec_1")
    assert not dest.exists()
    assert not dest.with_name(dest.name + ".part").exists()


def test_a_stream_cut_inside_a_chunk_is_refused(vault, tmp_path):
    path, _ = _streamed(vault, tmp_path, size=1000, chunk=256)
    raw = path.read_bytes()
    header = len(MAGIC_STREAM) + SALT_LEN + 4
    # Cut inside the second chunk's nonce, then inside its body.
    path.write_bytes(raw[: header + NONCE_LEN + 4 + 256 + GCM_TAG_LEN + 3])
    with pytest.raises(VaultError, match="truncated mid-chunk"):
        vault.verify_stream(path, "rec_1")
    path.write_bytes(raw[: header + NONCE_LEN + 4 + 256 + GCM_TAG_LEN + NONCE_LEN + 4 + 10])
    with pytest.raises(VaultError, match="truncated mid-chunk"):
        vault.verify_stream(path, "rec_1")


def test_bytes_appended_after_the_final_chunk_are_refused(vault, tmp_path):
    path, _ = _streamed(vault, tmp_path, size=100, chunk=256)
    path.write_bytes(path.read_bytes() + b"\x00")
    with pytest.raises(VaultError, match="data after its final chunk"):
        vault.verify_stream(path, "rec_1")


def test_a_stream_with_its_chunks_swapped_is_refused(vault, tmp_path):
    path, _ = _streamed(vault, tmp_path, size=512, chunk=256)      # two chunks
    raw = path.read_bytes()
    header = len(MAGIC_STREAM) + SALT_LEN + 4
    chunk_len = NONCE_LEN + 4 + 256 + GCM_TAG_LEN
    first, second = raw[header: header + chunk_len], raw[header + chunk_len:]
    assert len(second) == chunk_len, "the fixture did not produce two equal chunks"
    path.write_bytes(raw[:header] + second + first)
    with pytest.raises(VaultError, match="decryption failed at chunk 0"):
        vault.verify_stream(path, "rec_1")
