"""
Hear the moment: serve a recording's original audio without a plaintext copy.

A summary tells you what was said; sometimes you need the tone, and tone lives
only in the original audio. Until now the sole way back to it was
`open <id> --kind audio --out file.mp3`, which decrypts the whole recording to
a file and trusts you to delete it. That is acceptable for deliberate headless
recovery and unacceptable as the routine playback path of an app: every play
would mint another plaintext copy of the most sensitive artifact there is.

So this module's contract, which every function here honours: PLAINTEXT NEVER
TOUCHES DISK. No temp files, no decrypt-then-serve staging, no caching.
Encrypted originals are decrypted chunk by chunk out of the vault and yielded
straight to the caller, which hands them to the HTTP response. The only
plaintext that exists is the chunk currently in flight.

One consequence is documented rather than hidden: if decryption fails partway
(wrong passphrase discovered mid-file, a tampered chunk), some bytes have
already been yielded and, over HTTP, already sent -- streaming cannot un-send.
The vault raises its honest error at that point and the response dies visibly
mid-body, which every player surfaces as a failed load. What never happens is
silent truncation dressed up as a complete file.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .storage import Vault

# Plaintext originals are read in chunks this size. Large enough that a long
# recording is not ten thousand syscalls, small enough that a seek-happy
# player abandoning the connection wastes little.
PLAINTEXT_CHUNK = 256 * 1024

# The extensions this pipeline actually ingests, mapped explicitly. Guessing
# with mimetypes would depend on the OS's registry and quietly differ between
# machines; a wrong Content-Type makes some browsers refuse to play at all.
# Anything unrecognised is octet-stream -- the honest "bytes, you figure it
# out" -- rather than a confident lie.
_CONTENT_TYPES = {
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".mp4": "audio/mp4",
    ".aac": "audio/aac",
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".opus": "audio/opus",
    ".webm": "audio/webm",
    ".wma": "audio/x-ms-wma",
    ".aif": "audio/aiff",
    ".aiff": "audio/aiff",
    ".amr": "audio/amr",
}
_FALLBACK_TYPE = "application/octet-stream"


@dataclass
class MediaInfo:
    """
    Where a recording's original lives and how to serve it.

    `size_bytes` is None for encrypted originals on purpose: the file on disk
    is ciphertext, whose length includes salts, nonces, headers, and GCM tags.
    Reporting it as Content-Length would promise the player more (or fewer)
    bytes than the plaintext delivers, and a player that trusts the header
    treats the mismatch as a corrupt file. No number is better than a wrong one.
    """

    recording_id: str
    path: Path
    encrypted: bool
    content_type: str
    size_bytes: int | None


def content_type_for(path: Path | str) -> str:
    """
    The MIME type of the ORIGINAL file, seen through vault naming.

    An encrypted original keeps its real extension inside the vault name --
    `rec_x.source.mp3.enc` -- precisely so the type survives encryption. Strip
    the `.enc` wrapper first, then judge by what the writer preserved.
    """
    suffixes = [s.lower() for s in Path(path).suffixes]
    if suffixes and suffixes[-1] == ".enc":
        suffixes = suffixes[:-1]
    return _CONTENT_TYPES.get(suffixes[-1] if suffixes else "", _FALLBACK_TYPE)


def locate_original(cfg, db, recording_id: str) -> MediaInfo | None:
    """
    Find a recording's original wherever this pipeline put it, or say it is gone.

    Originals land in one of two places depending on the governing profile:
    encrypted profiles stream them into the vault (indexed as an `audio` or
    `source` artifact, `.enc` on disk), and plaintext profiles move them to
    inbox/_processed with the recording id as a filename prefix. Both the
    payload's artifact_paths and the artifacts table are consulted -- they
    should agree, but the index survives events the payload does not, and a
    player that cannot find audio the archive can verify is a bug.

    Returns None honestly when nothing is on disk: archiving was switched off,
    a retention sweep took the audio, or the id is unknown. "The original was
    not kept" is an answer the app must be able to give, not paper over.
    """
    candidates: list[Path] = []

    payload = db.load(recording_id) or {}
    paths = payload.get("artifact_paths") or {}
    # `audio` before `source`: for an audio recording they are the same file,
    # but if both somehow exist the actual audio is the one worth hearing.
    for kind in ("audio", "source"):
        if paths.get(kind):
            candidates.append(Path(str(paths[kind])))

    for row in db.all_artifacts():
        if row["recording_id"] == recording_id and row["kind"] in ("audio", "source"):
            candidates.append(Path(row["path"]))

    processed = cfg.path("inbox") / "_processed"
    if processed.is_dir():
        candidates.extend(sorted(processed.glob(f"{recording_id}_*")))

    for path in candidates:
        if not path.is_file():
            continue
        encrypted = path.suffix == ".enc"
        return MediaInfo(
            recording_id=recording_id,
            path=path,
            encrypted=encrypted,
            content_type=content_type_for(path),
            # Ciphertext size is not plaintext size; see MediaInfo.
            size_bytes=None if encrypted else path.stat().st_size,
        )
    return None


def stream_plaintext(cfg, info: MediaInfo, vault: Vault | None = None) -> Iterator[bytes]:
    """
    Yield the original's plaintext, chunk by chunk, never via disk.

    Plaintext originals are simply read in chunks. Encrypted ones go through
    the vault's streaming reader, which handles both formats the writers
    produce -- the chunked PBS1 stream `_archive` writes for audio and the
    one-shot PBV1 blob -- bound to the recording id exactly as they were
    written (the stream AAD folds recording_id|index|final per chunk; the
    one-shot AAD folds the id and the basename).

    A wrong passphrase or a tampered file raises VaultError from the vault
    itself, possibly after some chunks were already yielded; see the module
    docstring for why a loud mid-stream death beats silent truncation.
    """
    if not info.encrypted:
        with open(info.path, "rb") as fh:
            while block := fh.read(PLAINTEXT_CHUNK):
                yield block
        return

    vault = vault or Vault(cfg.path("vault"))
    yield from vault.iter_plaintext(info.path, info.recording_id)


def read_range(
    info: MediaInfo, start: int, end: int | None, cfg, vault: Vault | None = None
) -> tuple[Iterator[bytes], int | None]:
    """
    Serve an HTTP Range request: (bytes iterator, total size or None).

    Browsers seek by sending `Range: bytes=start-end`, and an audio element
    whose server ignores Range cannot scrub. For a PLAINTEXT original this is
    honoured exactly: the iterator yields bytes start..end inclusive (end=None
    means to EOF, matching the open-ended `bytes=start-` form) and the total
    size comes back for the Content-Range header.

    For an ENCRYPTED original the answer is (full stream, None): range is
    unsupported, and the caller should serve 200 with the whole body rather
    than 206. Why: byte-accurate seeking requires random access to plaintext,
    and the vault's cipher stream is sequential by design -- each chunk is
    authenticated in order, so byte N is only reachable by decrypting
    everything before it. The one implementation that would make seeking cheap
    -- decrypt once to a temp file and serve ranges from that -- is exactly
    what this module exists to never do. Players still work; they just
    buffer forward instead of jumping.

    A start at or past the end of a plaintext file raises ValueError so the
    caller can answer 416 instead of streaming an empty 206.
    """
    if info.encrypted:
        return stream_plaintext(cfg, info, vault), None

    total = info.size_bytes if info.size_bytes is not None else info.path.stat().st_size
    if start < 0 or start >= total:
        raise ValueError(f"range start {start} is outside the file ({total} bytes)")
    stop = total - 1 if end is None else min(end, total - 1)
    if stop < start:
        raise ValueError(f"range {start}-{end} is empty")

    def _iter() -> Iterator[bytes]:
        remaining = stop - start + 1
        with open(info.path, "rb") as fh:
            fh.seek(start)
            while remaining > 0:
                block = fh.read(min(PLAINTEXT_CHUNK, remaining))
                if not block:
                    # The file shrank under us. Stop rather than spin; the
                    # short body makes the failure visible to the player.
                    return
                remaining -= len(block)
                yield block

    return _iter(), total


def transcript_lines(cfg, db, archive, recording_id: str) -> list[dict[str, Any]] | None:
    """
    The stored segments as the player's synced-transcript payload.

    [{start, end, speaker, text}] with the numbers as floats, so the UI can
    highlight the line under the playhead and jump the audio when a line is
    clicked. Decryption goes through archive.full_record -- the one place that
    knows where a recording's words live -- and None means the recording is
    unknown or its content cannot be opened (locked vault, wrong passphrase),
    which the caller must report as such, not as an empty transcript.
    """
    payload = db.load(recording_id)
    if payload is None:
        return None
    # full_record wants an index row; build one from the payload we just
    # loaded rather than adding a second row-by-id query to the database.
    row = {
        "id": recording_id,
        "payload_json": json.dumps(payload),
        "stage": payload.get("stage"),
    }
    segments = archive.segments(row)
    if segments is None:
        return None
    return [
        {
            "start": float(s.get("start", 0.0) or 0.0),
            "end": float(s.get("end", 0.0) or 0.0),
            "speaker": str(s.get("speaker", "")),
            "text": str(s.get("text", "")),
        }
        for s in segments
    ]
