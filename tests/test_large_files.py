"""
Artifacts too big to hold in memory.

A full working day is roughly 450MB as a 128kbps mp3, and encrypting it the
one-shot way needs the plaintext and the ciphertext resident at the same time.
So originals are encrypted a chunk at a time.

A chunked format invites three attacks that the one-shot format could not have:
reordering chunks, dropping one from the middle, and truncating the file. Each
chunk is bound by AAD to the recording id, its own index, and whether it is the
last one, and the tests below are mostly about proving those three fail loudly
rather than silently handing back a shorter recording.
"""

import pytest

from plaud_bridge.storage import Vault, VaultError

PASSPHRASE = "a-long-enough-test-passphrase"


@pytest.fixture
def vault(tmp_path, monkeypatch):
    monkeypatch.setenv("PLAUD_BRIDGE_PASSPHRASE", PASSPHRASE)
    return Vault(tmp_path / "vault")


def _source(tmp_path, size: int, name: str = "day.mp3"):
    """A file of known content, large enough to span several chunks."""
    path = tmp_path / name
    body = bytes(range(256)) * (size // 256 + 1)
    path.write_bytes(body[:size])
    return path


# =========================================================================
# Round trip
# =========================================================================
@pytest.mark.parametrize("size", [0, 1, 4096, 5_000_000, 9_000_000])
def test_a_streamed_artifact_round_trips_at_any_size(vault, tmp_path, size):
    source = _source(tmp_path, size)
    stored = vault.write_stream("2026/07/27/rec_x.source.mp3", source, "rec_x")

    assert Vault.is_streamed(stored)
    out = vault.read_stream(stored, tmp_path / "out.mp3", "rec_x")
    assert out.read_bytes() == source.read_bytes()


def test_the_ciphertext_does_not_contain_the_plaintext(vault, tmp_path):
    source = tmp_path / "dinner.mp3"
    source.write_bytes(b"permission slip" * 100_000)
    stored = vault.write_stream("x/rec_y.source.mp3", source, "rec_y")
    assert b"permission slip" not in stored.read_bytes()


def test_streaming_spans_multiple_chunks(vault, tmp_path):
    """Otherwise the chunk-boundary tests below prove nothing."""
    source = _source(tmp_path, 3_000_000)
    stored = vault.write_stream("x/rec_z.source.mp3", source, "rec_z", chunk_size=64 * 1024)
    # Each chunk costs a 12-byte nonce, a 4-byte length, and a 16-byte GCM tag.
    # Roughly 46 chunks at this size, so the overhead is unmistakable.
    overhead = stored.stat().st_size - source.stat().st_size
    assert overhead >= 45 * 32, f"it did not chunk (overhead {overhead})"


def test_a_streamed_file_never_holds_the_whole_thing_in_memory(vault, tmp_path):
    """
    The chunk size is what bounds memory, so a file many times larger than it
    must still work. This is the property that makes a day of audio possible.
    """
    source = _source(tmp_path, 2_000_000)
    stored = vault.write_stream("x/rec_m.source.mp3", source, "rec_m", chunk_size=8192)
    out = vault.read_stream(stored, tmp_path / "back.mp3", "rec_m")
    assert out.read_bytes() == source.read_bytes()


# =========================================================================
# Tamper resistance
# =========================================================================
def test_truncation_is_detected(vault, tmp_path):
    """
    The failure this exists to prevent: half a recording handed back as though
    it were the whole thing.
    """
    source = _source(tmp_path, 500_000)
    stored = vault.write_stream("x/rec_t.source.mp3", source, "rec_t", chunk_size=16384)

    blob = stored.read_bytes()
    stored.write_bytes(blob[: len(blob) // 2])

    # Cutting mid-chunk fails the chunk's own tag; cutting on a boundary fails
    # the "must end on a chunk marked final" rule. Either way it refuses, and
    # either way it must not leave a half recording on disk.
    with pytest.raises(VaultError):
        vault.read_stream(stored, tmp_path / "out.mp3", "rec_t")
    assert not (tmp_path / "out.mp3").exists(), "a partial file was left behind"


def test_truncation_on_an_exact_chunk_boundary_is_detected(vault, tmp_path):
    """
    The subtle one. Every chunk that remains decrypts perfectly, so only the
    missing "this is the last chunk" marker gives it away.
    """
    chunk = 16384
    source = _source(tmp_path, 500_000)
    stored = vault.write_stream("x/rec_b.source.mp3", source, "rec_b", chunk_size=chunk)

    header = len(b"PBS1") + 16 + 4
    record = 12 + 4 + (chunk + 16)          # nonce + length + ciphertext with tag
    stored.write_bytes(stored.read_bytes()[: header + 3 * record])

    with pytest.raises(VaultError) as excinfo:
        vault.read_stream(stored, tmp_path / "out.mp3", "rec_b")
    assert "final chunk" in str(excinfo.value), str(excinfo.value)
    assert not (tmp_path / "out.mp3").exists()


def test_a_modified_chunk_is_detected(vault, tmp_path):
    source = _source(tmp_path, 200_000)
    stored = vault.write_stream("x/rec_c.source.mp3", source, "rec_c", chunk_size=16384)

    blob = bytearray(stored.read_bytes())
    blob[-20] ^= 0xFF
    stored.write_bytes(bytes(blob))

    with pytest.raises(VaultError):
        vault.read_stream(stored, tmp_path / "out.mp3", "rec_c")


def test_the_wrong_recording_id_will_not_open_it(vault, tmp_path):
    """Same binding the one-shot format has: files cannot be swapped between recordings."""
    source = _source(tmp_path, 100_000)
    stored = vault.write_stream("x/rec_a.source.mp3", source, "rec_a", chunk_size=16384)

    with pytest.raises(VaultError):
        vault.read_stream(stored, tmp_path / "out.mp3", "rec_b")


def test_the_wrong_passphrase_will_not_open_it(vault, tmp_path, monkeypatch):
    source = _source(tmp_path, 100_000)
    stored = vault.write_stream("x/rec_p.source.mp3", source, "rec_p", chunk_size=16384)

    monkeypatch.setenv("PLAUD_BRIDGE_PASSPHRASE", "a-completely-different-passphrase")
    with pytest.raises(VaultError):
        vault.read_stream(stored, tmp_path / "out.mp3", "rec_p")


def test_a_failed_write_leaves_no_partial_artifact(vault, tmp_path):
    with pytest.raises(OSError):
        vault.write_stream("x/rec_q.source.mp3", tmp_path / "does-not-exist.mp3", "rec_q")
    assert not list((vault.root).rglob("*.tmp"))


# =========================================================================
# Compatibility
# =========================================================================
def test_read_stream_still_opens_a_one_shot_artifact(vault, tmp_path):
    """Everything written before this format existed has to keep opening."""
    stored = vault.write("x/rec_old.transcript.md", "hello there", "rec_old")
    assert not Vault.is_streamed(stored)

    out = vault.read_stream(stored, tmp_path / "old.md", "rec_old")
    assert out.read_text() == "hello there"


def test_small_artifacts_are_still_written_one_shot(vault):
    """Transcripts and analyses are small; chunking them would gain nothing."""
    stored = vault.write("x/rec_s.transcript.md", "short", "rec_s")
    assert not Vault.is_streamed(stored)
    assert vault.read_text(stored, "rec_s") == "short"


def test_is_streamed_is_safe_on_a_missing_file(tmp_path):
    assert Vault.is_streamed(tmp_path / "nothing-here.enc") is False


# =========================================================================
# Through the pipeline
# =========================================================================
def test_verify_checks_a_streamed_artifact_without_loading_it(tmp_path, monkeypatch):
    from _fixtures import CLIENT_CALL, build_sandbox, drop
    from plaud_bridge.archive import Archive
    from plaud_bridge.db import Database
    from plaud_bridge.pipeline import Pipeline

    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    drop(cfg, "client.txt", CLIENT_CALL)
    pipe = Pipeline(cfg)
    try:
        pipe.run()
    finally:
        pipe.close()

    db = Database(cfg.path("database"))
    try:
        streamed = [a for a in db.all_artifacts() if Vault.is_streamed(a["path"])]
        assert streamed, "the archived original was not streamed"

        report = Archive(cfg, db).verify()
        assert report.healthy, report.render()
    finally:
        db.close()


def test_verify_catches_a_corrupted_streamed_original(tmp_path, monkeypatch):
    from _fixtures import CLIENT_CALL, build_sandbox, drop
    from plaud_bridge.archive import Archive
    from plaud_bridge.db import Database
    from plaud_bridge.pipeline import Pipeline

    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    drop(cfg, "client.txt", CLIENT_CALL)
    pipe = Pipeline(cfg)
    try:
        pipe.run()
    finally:
        pipe.close()

    db = Database(cfg.path("database"))
    try:
        from pathlib import Path

        target = Path(next(a["path"] for a in db.all_artifacts() if Vault.is_streamed(a["path"])))
        blob = bytearray(target.read_bytes())
        blob[-10] ^= 0xFF
        target.write_bytes(bytes(blob))

        report = Archive(cfg, db).verify()
        assert not report.healthy
        assert any(p.status == "undecryptable" for p in report.problems)
    finally:
        db.close()


def test_verify_does_not_write_the_decrypted_copy_anywhere(tmp_path, monkeypatch):
    """A verify that leaves plaintext behind is worse than no verify."""
    from _fixtures import CLIENT_CALL, build_sandbox, drop
    from plaud_bridge.archive import Archive
    from plaud_bridge.db import Database
    from plaud_bridge.pipeline import Pipeline

    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    drop(cfg, "client.txt", CLIENT_CALL)
    pipe = Pipeline(cfg)
    try:
        pipe.run()
    finally:
        pipe.close()

    before = {p for p in cfg.root.rglob("*") if p.is_file()}
    db = Database(cfg.path("database"))
    try:
        Archive(cfg, db).verify()
    finally:
        db.close()

    # Nothing new on disk at all, anywhere under the data root. The database
    # journal is the one legitimate exception: opening the index writes it.
    new = {p for p in cfg.root.rglob("*") if p.is_file()} - before
    new = {p for p in new if p.suffix not in (".db-wal", ".db-shm")}
    assert not new, f"verify wrote files it should not have: {sorted(new)}"

    leaked = [p for p in cfg.path("vault").rglob("*") if p.is_file() and p.suffix != ".enc"]
    assert not leaked, f"verify left decrypted files behind: {leaked}"


def test_verify_stream_needs_no_writable_destination(vault, tmp_path):
    """
    The bug this pins: verification used to decrypt to os.devnull, which meant
    it created os.devnull + ".part" alongside it. Root can do that. Nobody else
    can, so verification failed everywhere it mattered.
    """
    source = _source(tmp_path, 300_000)
    stored = vault.write_stream("x/rec_v.source.mp3", source, "rec_v", chunk_size=16384)

    before = sorted(p.name for p in tmp_path.rglob("*"))

    vault.verify_stream(stored, "rec_v")

    assert sorted(p.name for p in tmp_path.rglob("*")) == before

    blob = bytearray(stored.read_bytes())
    blob[-10] ^= 0xFF
    stored.write_bytes(bytes(blob))
    with pytest.raises(VaultError):
        vault.verify_stream(stored, "rec_v")


def test_verify_stream_also_handles_a_one_shot_artifact(vault):
    stored = vault.write("x/rec_o.transcript.md", "hello there", "rec_o")
    vault.verify_stream(stored, "rec_o")
    with pytest.raises(VaultError):
        vault.verify_stream(stored, "rec_wrong")
