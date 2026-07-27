"""
The inbox is the interface.

Almost everyone who uses this will never type a subcommand beyond `run`. They
drop a file in a folder. So the folder has to cope with everything a real
folder accumulates — the wrong file type, a half-copied file, a directory that
looks like a recording, a name with an apostrophe in it — and, when it declines
to process something, it has to say so. A run that reports "inbox is empty"
over a directory with three files in it is the same defect as a search that
reports "never said" without looking.
"""

from __future__ import annotations

import pytest

from _fixtures import CLIENT_CALL, build_sandbox
from plaud_bridge.cli import main
from plaud_bridge.db import Database
from plaud_bridge.pipeline import Pipeline


def run(cfg, *extra) -> int:
    return main(["--config", str(cfg.root / "config"), "run", *extra])


def ingested(cfg) -> list[str]:
    db = Database(cfg.path("database"))
    try:
        return [r["source_name"] for r in db.query(limit=50)]
    finally:
        db.close()


# =========================================================================
# Names a real folder produces
# =========================================================================
@pytest.mark.parametrize("name", [
    "réunion-クライアント.txt",
    "my call's \"notes\".txt",
    "-rf.txt",                          # looks like a flag
    "--help.txt",
    "call (2026-07-27) [final].txt",
    "LOUD.TXT",                         # extension matching is case-insensitive
    "a" * 120 + ".txt",                 # long but legal
    "call.with.many.dots.txt",
])
def test_an_awkward_filename_is_still_processed(tmp_path, monkeypatch, name):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    (cfg.path("inbox") / name).write_text(CLIENT_CALL, encoding="utf-8")
    assert run(cfg) == 0
    assert ingested(cfg) == [name]


def test_a_transcript_in_a_subdirectory_is_found(tmp_path, monkeypatch):
    """Plaud exports arrive in dated folders; nobody flattens them by hand."""
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    nested = cfg.path("inbox") / "2026-07-27" / "session-1"
    nested.mkdir(parents=True)
    (nested / "call.txt").write_text(CLIENT_CALL, encoding="utf-8")
    assert run(cfg) == 0
    assert ingested(cfg) == ["call.txt"]


# =========================================================================
# Things that are not recordings
# =========================================================================
@pytest.mark.parametrize("make,label", [
    (lambda i: (i / "notadir.mp3").mkdir(), "a directory named like a recording"),
    (lambda i: (i / "gone.txt").symlink_to(i / "does-not-exist"), "a broken symlink"),
    (lambda i: (i / ".DS_Store").write_bytes(b"junk"), "a dotfile"),
])
def test_something_that_is_not_a_file_is_ignored_without_crashing(tmp_path, monkeypatch,
                                                                  make, label):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    make(cfg.path("inbox"))
    assert run(cfg) == 0, label
    assert ingested(cfg) == [], label


def test_an_unsupported_file_is_named_rather_than_silently_ignored(tmp_path, monkeypatch,
                                                                   capsys):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    for name in ("notes.pdf", "photo.jpg", "recording"):
        (cfg.path("inbox") / name).write_bytes(b"not a recording")

    assert run(cfg) == 0
    out = capsys.readouterr().out
    assert "inbox is empty" not in out
    for name in ("notes.pdf", "photo.jpg", "recording"):
        assert name in out, f"{name} sat in the inbox and was never mentioned"
    # And it says what it would have accepted, so the next attempt succeeds.
    assert ".mp3" in out and ".txt" in out


def test_an_unsupported_file_alongside_a_real_one_does_not_hide_it(tmp_path, monkeypatch,
                                                                   capsys):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    (cfg.path("inbox") / "notes.pdf").write_bytes(b"%PDF")
    (cfg.path("inbox") / "call.txt").write_text(CLIENT_CALL, encoding="utf-8")

    assert run(cfg) == 0
    assert ingested(cfg) == ["call.txt"]
    assert "notes.pdf" in capsys.readouterr().out


def test_an_unsupported_file_is_never_deleted_or_moved(tmp_path, monkeypatch):
    """It might be the only copy of something. Declining to read it is not
    permission to touch it."""
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    stray = cfg.path("inbox") / "irreplaceable.pdf"
    stray.write_bytes(b"%PDF-1.4")
    assert run(cfg) == 0
    assert stray.read_bytes() == b"%PDF-1.4"


# =========================================================================
# Empty and malformed content
# =========================================================================
@pytest.mark.parametrize("body,label", [
    ("", "empty"),
    ("   \n\n  \n", "whitespace"),
    ("﻿\n", "a byte order mark"),
    ("​​\n‍\n", "zero-width characters"),
    ("﻿   \n​\n   ", "invisible characters and whitespace"),
])
def test_a_file_with_no_speech_fails_rather_than_becoming_a_recording(tmp_path, monkeypatch,
                                                                      body, label):
    """
    The bug this pins: a file holding nothing but a byte order mark parsed into
    one segment of nothing, passed the emptiness check, and was routed and
    analysed as though it were a conversation.
    """
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    (cfg.path("inbox") / "empty.txt").write_text(body, encoding="utf-8")

    # Exit 2 is "something failed", which is the honest answer.
    assert run(cfg) == 2, label

    db = Database(cfg.path("database"))
    try:
        complete = [r for r in db.query(limit=50) if r["stage"] == "complete"]
    finally:
        db.close()
    assert not complete, f"{label} was indexed as a finished recording"


def test_a_byte_order_mark_does_not_corrupt_the_first_speaker(tmp_path, monkeypatch):
    """A BOM in front of "Sasson:" is enough to stop it reading as a label."""
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    (cfg.path("inbox") / "bom.txt").write_text("﻿" + CLIENT_CALL, encoding="utf-8")
    assert run(cfg) == 0

    db = Database(cfg.path("database"))
    try:
        rid = db.query(limit=1)[0]["id"]
    finally:
        db.close()
    assert main(["--config", str(cfg.root / "config"), "open", rid]) == 0


def test_a_file_that_is_not_utf8_still_processes(tmp_path, monkeypatch):
    """Exports from other tools are not always utf-8, and a mojibake transcript
    beats a crash."""
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    body = CLIENT_CALL.replace("Marcus", "Café").encode("latin-1")
    (cfg.path("inbox") / "latin.txt").write_bytes(body)
    assert run(cfg) == 0
    assert ingested(cfg) == ["latin.txt"]


def test_a_file_with_null_bytes_does_not_take_the_run_down(tmp_path, monkeypatch):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    (cfg.path("inbox") / "nulls.txt").write_bytes(b"Sasson: hello\x00\x00 there\n")
    (cfg.path("inbox") / "good.txt").write_text(CLIENT_CALL, encoding="utf-8")
    assert run(cfg) in (0, 2)
    assert "good.txt" in ingested(cfg), "one bad file stopped a good one"


# =========================================================================
# Size and settling
# =========================================================================
def test_an_oversize_file_is_named_and_the_limit_is_quoted(tmp_path, monkeypatch, capsys):
    cfg, _ = build_sandbox(tmp_path, monkeypatch, overrides={"ingest": {"max_file_mb": 1}})
    big = cfg.path("inbox") / "day.txt"
    big.write_text("Sasson: hello\n" * 200_000, encoding="utf-8")
    assert big.stat().st_size > 1024 * 1024

    assert run(cfg) == 0
    out = capsys.readouterr().out
    assert "day.txt" in out
    assert "max_file_mb" in out
    assert ingested(cfg) == []
    assert big.exists(), "an oversize file was consumed rather than left alone"


def test_a_file_still_being_copied_is_skipped_and_explained(tmp_path, monkeypatch, capsys):
    cfg, _ = build_sandbox(tmp_path, monkeypatch, overrides={"ingest": {"settle_seconds": 300}})
    (cfg.path("inbox") / "copying.txt").write_text(CLIENT_CALL, encoding="utf-8")

    assert run(cfg) == 0
    out = capsys.readouterr().out
    assert "copying.txt" in out
    assert "settle_seconds" in out
    assert ingested(cfg) == []


def test_the_reason_is_reported_even_when_other_files_succeeded(tmp_path, monkeypatch, capsys):
    """
    The bug this pins: the explanation used to be printed only when nothing at
    all was processed, so a good file and a skipped one meant the skipped one
    was never mentioned.
    """
    cfg, _ = build_sandbox(tmp_path, monkeypatch, overrides={"ingest": {"max_file_mb": 1}})
    (cfg.path("inbox") / "call.txt").write_text(CLIENT_CALL, encoding="utf-8")
    (cfg.path("inbox") / "huge.txt").write_text("Sasson: hi\n" * 200_000, encoding="utf-8")
    (cfg.path("inbox") / "notes.pdf").write_bytes(b"%PDF")

    assert run(cfg) == 0
    out = capsys.readouterr().out
    assert ingested(cfg) == ["call.txt"]
    assert "huge.txt" in out, "an oversize file went unmentioned because another succeeded"
    assert "notes.pdf" in out, "an unsupported file went unmentioned because another succeeded"


# =========================================================================
# The discovery contract itself
# =========================================================================
def test_discover_accounts_for_every_file_it_saw(tmp_path, monkeypatch):
    """
    Every file in the inbox lands in exactly one bucket: processed, or one of
    the three named skip lists. A fourth, unnamed outcome is how a file goes
    missing without anyone noticing.
    """
    cfg, _ = build_sandbox(tmp_path, monkeypatch, overrides={"ingest": {"max_file_mb": 1}})
    inbox = cfg.path("inbox")
    (inbox / "call.txt").write_text(CLIENT_CALL, encoding="utf-8")
    (inbox / "notes.pdf").write_bytes(b"%PDF")
    (inbox / "huge.txt").write_text("Sasson: hi\n" * 200_000, encoding="utf-8")

    pipe = Pipeline(cfg)
    try:
        found = pipe.discover()
        accounted = (
            {p.name for p in found}
            | {p.name for p in pipe.unsupported}
            | {p.name for p in pipe.unsettled}
            | {p.name for p, _ in pipe.oversize}
        )
    finally:
        pipe.close()

    on_disk = {p.name for p in inbox.rglob("*") if p.is_file() and not p.name.startswith(".")}
    assert accounted == on_disk, f"unaccounted for: {sorted(on_disk - accounted)}"


def test_the_processed_archive_is_not_rescanned(tmp_path, monkeypatch):
    """Otherwise every run rehashes everything ever processed."""
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    archive = cfg.path("inbox") / "_processed"
    archive.mkdir(parents=True, exist_ok=True)
    (archive / "old.txt").write_text(CLIENT_CALL, encoding="utf-8")

    pipe = Pipeline(cfg)
    try:
        assert pipe.discover() == []
        assert pipe.unsupported == []
    finally:
        pipe.close()
