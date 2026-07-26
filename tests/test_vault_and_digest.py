"""Vault round-trip guarantees and digest field rendering."""


import pytest

from plaud_bridge.digest.builder import _fmt_quote, _fmt_value
from plaud_bridge.storage import Vault, VaultError

PASS = "a-sufficiently-long-test-passphrase"


@pytest.fixture
def vault(tmp_path, monkeypatch):
    monkeypatch.setenv("PLAUD_BRIDGE_PASSPHRASE", PASS)
    return Vault(tmp_path / "vault")


def test_round_trip(vault):
    path = vault.write("a/b/note.md", "client said the premium is too high", "rec_1")
    assert path.suffix == ".enc"
    assert vault.read_text(path, "rec_1") == "client said the premium is too high"


def test_ciphertext_is_not_plaintext(vault):
    secret = "SSN 123-45-6789 and a policy number"
    path = vault.write("x.md", secret, "rec_1")
    raw = path.read_bytes()
    assert b"123-45-6789" not in raw
    assert raw.startswith(b"PBV1")


def test_wrong_recording_id_fails_to_decrypt(vault):
    """AAD binds ciphertext to its recording; files cannot be swapped."""
    path = vault.write("x.md", "private", "rec_1")
    with pytest.raises(VaultError, match="decryption failed"):
        vault.read_text(path, "rec_2")


def test_tampering_is_detected(vault):
    path = vault.write("x.md", "private", "rec_1")
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 0x01
    path.write_bytes(bytes(raw))
    with pytest.raises(VaultError, match="decryption failed"):
        vault.read_text(path, "rec_1")


def test_short_passphrase_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("PLAUD_BRIDGE_PASSPHRASE", "short")
    v = Vault(tmp_path / "v")
    ok, why = v.ready()
    assert not ok and "16 characters" in why
    with pytest.raises(VaultError):
        v.write("x.md", "data", "rec_1")


def test_missing_passphrase_refuses_rather_than_writing_plaintext(tmp_path, monkeypatch):
    monkeypatch.delenv("PLAUD_BRIDGE_PASSPHRASE", raising=False)
    v = Vault(tmp_path / "v")
    with pytest.raises(VaultError, match="Refusing"):
        v.write("x.md", "data", "rec_1")
    assert not list((tmp_path / "v").rglob("*.md*"))


def test_fingerprint_is_stable_and_distinguishing(tmp_path):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text("same")
    b.write_text("same")
    assert Vault.fingerprint(a) == Vault.fingerprint(b)
    b.write_text("different")
    assert Vault.fingerprint(a) != Vault.fingerprint(b)


# ---- digest rendering ---------------------------------------------------
def test_quote_shaped_dict_renders_with_speaker_and_timestamp():
    out = _fmt_quote({"timestamp": "01:23", "speaker": "Marcus", "text": "price is my worry"})
    assert out == "[01:23] Marcus: price is my worry"


def test_object_dict_without_text_key_still_renders():
    """A model returning {'what':..,'when':..} must not produce a blank bullet."""
    out = _fmt_quote({"what": "sign the permission slip", "when": "tonight"})
    assert "sign the permission slip" in out and "tonight" in out


def test_quote_with_extra_keys_keeps_them():
    out = _fmt_quote({"type": "price", "quote": "too expensive"})
    assert "too expensive" in out and "price" in out


def test_plain_string_passes_through():
    assert _fmt_quote("  just a string  ") == "just a string"


def test_fmt_value_handles_empties_and_booleans():
    assert _fmt_value(None) == []
    assert _fmt_value([]) == []
    assert _fmt_value("") == []
    assert _fmt_value(False) == []
    assert _fmt_value(True) == ["yes"]
    assert _fmt_value(["a", "b"]) == ["a", "b"]


def test_fmt_value_truncates_long_lists():
    assert len(_fmt_value([f"item{i}" for i in range(50)], limit=8)) == 8
