"""
Playing the moment back: the media layer behind the app's player.

The whole reason this layer exists is the contract that plaintext never touches
disk on the playback path -- otherwise `open --out` already covered it. So the
tests here are mostly about honesty under that constraint: the encrypted stream
round-trips byte for byte, a wrong passphrase raises instead of truncating,
nothing plaintext appears anywhere on disk while a stream is consumed, and
"the original was not kept" comes back as None rather than as a guess.
"""

from pathlib import Path

import pytest

from _fixtures import CLIENT_CALL, build_sandbox, drop
from plaud_bridge.archive import Archive
from plaud_bridge.db import Database
from plaud_bridge.media import (
    MediaInfo,
    content_type_for,
    locate_original,
    read_range,
    stream_plaintext,
    transcript_lines,
)
from plaud_bridge.pipeline import Pipeline
from plaud_bridge.storage import Vault, VaultError

# Long and deliberately un-audio-like, so scanning the disk for it after an
# encrypted stream is a meaningful leak check: 16 KiB of a distinctive phrase
# does not appear in AES-GCM ciphertext by accident.
FAKE_AUDIO = b"TONE-IS-THE-CONTENT:" * 819  # ~16 KiB, > several 1 KiB chunks


def _processed(tmp_path, monkeypatch, overrides=None):
    """One CLIENT_CALL recording through the real pipeline, sandboxed."""
    cfg, _ = build_sandbox(tmp_path, monkeypatch, overrides=overrides)
    drop(cfg, "client-marcus.txt", CLIENT_CALL)
    pipe = Pipeline(cfg)
    try:
        pipe.run()
    finally:
        pipe.close()
    return cfg


def _vaulted_original(cfg, db, monkeypatch=None):
    """
    Stream a fake original into the vault and index it the way _archive does:
    write_stream bound to the recording id, then a `audio` artifact row. The
    tiny chunk_size forces multiple PBS1 chunks so the round-trip actually
    exercises the chunk walk, not just a one-block file.
    """
    rid = db.query()[0]["id"]
    vault = Vault(cfg.path("vault"))
    src = cfg.path("work") / "original.mp3"
    src.write_bytes(FAKE_AUDIO)
    dest = vault.write_stream(f"2026/01/01/{rid}.source.mp3", src, rid, chunk_size=1024)
    src.unlink()
    db.record_artifact(rid, "audio", str(dest), True, None)
    return rid, dest


# =========================================================================
# locate_original
# =========================================================================
def test_locate_finds_a_vault_streamed_original(tmp_path, monkeypatch):
    cfg = _processed(tmp_path, monkeypatch,
                     overrides={"ingest": {"archive_originals": False}})
    db = Database(cfg.path("database"))
    try:
        rid, dest = _vaulted_original(cfg, db)
        info = locate_original(cfg, db, rid)
        assert info is not None
        assert info.encrypted is True
        assert info.path == dest
        assert info.content_type == "audio/mpeg", "the .enc wrapper hid the real extension"
        assert info.size_bytes is None, (
            "reported the ciphertext size as if it were the plaintext size; "
            "a wrong Content-Length reads as a corrupt file to a player"
        )
    finally:
        db.close()


def test_locate_finds_a_processed_plaintext_original(tmp_path, monkeypatch):
    cfg = _processed(tmp_path, monkeypatch,
                     overrides={"ingest": {"archive_originals": False}})
    db = Database(cfg.path("database"))
    try:
        rid = db.query()[0]["id"]
        processed = cfg.path("inbox") / "_processed"
        processed.mkdir(parents=True, exist_ok=True)
        original = processed / f"{rid}_standup.wav"
        original.write_bytes(FAKE_AUDIO)

        info = locate_original(cfg, db, rid)
        assert info is not None
        assert info.encrypted is False
        assert info.path == original
        assert info.content_type == "audio/wav"
        assert info.size_bytes == len(FAKE_AUDIO)
    finally:
        db.close()


def test_locate_returns_none_when_the_original_was_not_kept(tmp_path, monkeypatch):
    """
    "The original is gone" must come back as None, not as a path that will 404
    or -- worse -- some other recording's file. Covers both a real recording
    whose original was never archived and an id that does not exist at all.
    """
    cfg = _processed(tmp_path, monkeypatch,
                     overrides={"ingest": {"archive_originals": False}})
    db = Database(cfg.path("database"))
    try:
        rid = db.query()[0]["id"]
        assert locate_original(cfg, db, rid) is None
        assert locate_original(cfg, db, "rec_doesnotexist") is None
    finally:
        db.close()


# =========================================================================
# stream_plaintext
# =========================================================================
def test_encrypted_stream_round_trips_byte_for_byte(tmp_path, monkeypatch):
    cfg = _processed(tmp_path, monkeypatch,
                     overrides={"ingest": {"archive_originals": False}})
    db = Database(cfg.path("database"))
    try:
        rid, _ = _vaulted_original(cfg, db)
        info = locate_original(cfg, db, rid)
        chunks = list(stream_plaintext(cfg, info))
        assert len(chunks) > 1, "a multi-chunk file came back as one block; the chunk walk is untested"
        assert b"".join(chunks) == FAKE_AUDIO
    finally:
        db.close()


def test_encrypted_stream_raises_on_a_wrong_passphrase(tmp_path, monkeypatch):
    """A wrong key must be a loud error, never a silent zero-byte 'recording'."""
    cfg = _processed(tmp_path, monkeypatch,
                     overrides={"ingest": {"archive_originals": False}})
    db = Database(cfg.path("database"))
    try:
        rid, _ = _vaulted_original(cfg, db)
        info = locate_original(cfg, db, rid)
        monkeypatch.setenv("PLAUD_BRIDGE_PASSPHRASE", "a-completely-different-passphrase")
        with pytest.raises(VaultError):
            list(stream_plaintext(cfg, info))
    finally:
        db.close()


def test_no_plaintext_lands_on_disk_during_an_encrypted_stream(tmp_path, monkeypatch):
    """
    The module's whole contract. Snapshot every file under the sandbox, consume
    a full encrypted stream, then prove the file set did not change and that no
    file anywhere -- work dir, vault, inbox, temp names -- contains the audio
    marker. If a future 'optimisation' stages a decrypted copy, this goes red.
    """
    cfg = _processed(tmp_path, monkeypatch,
                     overrides={"ingest": {"archive_originals": False}})
    db = Database(cfg.path("database"))
    try:
        rid, _ = _vaulted_original(cfg, db)
        info = locate_original(cfg, db, rid)

        before = {p for p in tmp_path.rglob("*") if p.is_file()}
        assert b"".join(stream_plaintext(cfg, info)) == FAKE_AUDIO
        after = {p for p in tmp_path.rglob("*") if p.is_file()}

        assert after == before, f"streaming created files: {sorted(after - before)}"
        marker = FAKE_AUDIO[:40]
        for path in after:
            if path.suffix == ".db":  # sqlite holds no audio; skip the lock dance
                continue
            assert marker not in path.read_bytes(), f"plaintext audio found in {path}"
    finally:
        db.close()


# =========================================================================
# read_range
# =========================================================================
def test_range_math_on_a_plaintext_original(tmp_path, monkeypatch):
    body = bytes(range(256)) * 40  # 10240 bytes, position-identifiable
    original = tmp_path / "clip.wav"
    original.write_bytes(body)
    info = MediaInfo("rec_x", original, False, "audio/wav", len(body))

    it, total = read_range(info, 1000, 1999, cfg=None)
    assert total == len(body)
    assert b"".join(it) == body[1000:2000]

    # The open-ended form browsers send first: bytes=9000-
    it, total = read_range(info, 9000, None, cfg=None)
    assert total == len(body)
    assert b"".join(it) == body[9000:]

    # An end past EOF is clamped, per RFC 7233, not an error.
    it, _ = read_range(info, len(body) - 10, len(body) + 5000, cfg=None)
    assert b"".join(it) == body[-10:]

    # A start past EOF is unsatisfiable and must not stream an empty 206.
    with pytest.raises(ValueError):
        read_range(info, len(body), None, cfg=None)


def test_range_on_an_encrypted_original_returns_the_full_stream(tmp_path, monkeypatch):
    """
    Seeking needs random access and the cipher stream is sequential by design;
    the contract forbids the decrypt-to-disk workaround. So a Range request on
    an encrypted original gets (whole stream, None) and the caller serves 200.
    """
    cfg = _processed(tmp_path, monkeypatch,
                     overrides={"ingest": {"archive_originals": False}})
    db = Database(cfg.path("database"))
    try:
        rid, _ = _vaulted_original(cfg, db)
        info = locate_original(cfg, db, rid)
        it, total = read_range(info, 1000, 1999, cfg)
        assert total is None, "claimed a total size it cannot honestly know"
        assert b"".join(it) == FAKE_AUDIO, "range-unsupported must still mean the full file"
    finally:
        db.close()


# =========================================================================
# content types
# =========================================================================
def test_content_types_come_from_the_original_extension():
    assert content_type_for(Path("a/rec_1.source.mp3.enc")) == "audio/mpeg"
    assert content_type_for(Path("rec_2_voice memo.m4a")) == "audio/mp4"
    assert content_type_for(Path("rec_3_standup.WAV")) == "audio/wav"
    assert content_type_for(Path("rec_4.source.opus.enc")) == "audio/opus"
    # Unknown gets the honest fallback, not a guessed audio type.
    assert content_type_for(Path("rec_5.source.xyz.enc")) == "application/octet-stream"
    assert content_type_for(Path("rec_6.source.txt.enc")) == "application/octet-stream"


# =========================================================================
# transcript_lines
# =========================================================================
def test_transcript_lines_returns_the_segments_with_speakers(tmp_path, monkeypatch):
    cfg = _processed(tmp_path, monkeypatch)
    db = Database(cfg.path("database"))
    try:
        rid = db.query()[0]["id"]
        lines = transcript_lines(cfg, db, Archive(cfg, db), rid)
        assert lines, "the player got no transcript to sync against"
        assert {line["speaker"] for line in lines} >= {"Sasson", "Marcus"}
        assert all(set(line) == {"start", "end", "speaker", "text"} for line in lines)
        assert all(isinstance(line["start"], float) and isinstance(line["end"], float)
                   for line in lines)
        assert any("elimination" in line["text"].lower() for line in lines)
    finally:
        db.close()


def test_transcript_lines_is_none_when_the_content_cannot_be_opened(tmp_path, monkeypatch):
    """None means 'could not read', which the UI must not render as 'silence'."""
    cfg = _processed(tmp_path, monkeypatch)
    monkeypatch.setenv("PLAUD_BRIDGE_PASSPHRASE", "a-completely-different-passphrase")
    db = Database(cfg.path("database"))
    try:
        rid = db.query()[0]["id"]
        assert transcript_lines(cfg, db, Archive(cfg, db), rid) is None
        assert transcript_lines(cfg, db, Archive(cfg, db), "rec_doesnotexist") is None
    finally:
        db.close()
