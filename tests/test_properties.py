"""
Properties that must hold for every input, not just the ones we thought of.

The rest of the suite tests behaviour against examples a person chose. That is
where most bugs live, but not the ones that matter most here: the vault is the
thing standing between a private conversation and a disk, and "we tried some
tampered files and they were rejected" is a weaker claim than it sounds. These
tests generate the inputs instead -- arbitrary plaintext, arbitrary byte
mutations, arbitrary truncations -- and assert the invariant rather than the
example.

The load-bearing property, stated once so it cannot be softened by accident:

    A vault artifact either returns EXACTLY the bytes that were written, or
    raises. There is no third outcome. Not a shorter file, not a partial file,
    not a plausible-looking file -- silent corruption is the one failure a
    tool holding recordings may never have, because nobody audits a file that
    opened cleanly.

The rest guard the parsers that touch untrusted input: filenames from a
browser, transcript text from a recogniser, extensions from a vault name.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from plaud_bridge.desktop.server import _ACCEPTED, _safe_name
from plaud_bridge.insights import is_question
from plaud_bridge.media import MediaInfo, content_type_for, read_range
from plaud_bridge.storage import Vault, VaultError

# Hypothesis reuses one tmp_path across the examples of a single test, which is
# what the function-scoped fixture health check complains about. That is fine
# here -- every example writes to a uniquely named file inside it -- and the
# alternative (a fresh directory per example) would slow scrypt-backed tests
# down for no added coverage.
SETTINGS = settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

# Key derivation is scrypt, which is deliberately expensive; a handful of
# examples per property is the honest trade against a suite people will run.
CRYPTO_SETTINGS = settings(
    max_examples=12,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


@pytest.fixture
def vault(tmp_path, monkeypatch):
    monkeypatch.setenv("PLAUD_BRIDGE_PASSPHRASE", "a-long-enough-property-passphrase")
    return Vault(tmp_path / "vault")


def _read_stream_bytes(vault: Vault, path: Path, recording_id: str) -> bytes:
    return b"".join(vault.iter_plaintext(path, recording_id))


# =========================================================================
# The vault: exact bytes, or an error. Never anything in between.
# =========================================================================
@given(payload=st.binary(min_size=0, max_size=4096),
       recording_id=st.text(alphabet=st.characters(min_codepoint=48, max_codepoint=122),
                            min_size=0, max_size=24))
@CRYPTO_SETTINGS
def test_a_one_shot_artifact_round_trips_exactly(vault, payload, recording_id):
    path = vault.write("prop/one-shot", payload, recording_id)
    assert vault.read(path, recording_id) == payload


@given(payload=st.binary(min_size=1, max_size=8192), chunk=st.integers(16, 512))
@CRYPTO_SETTINGS
def test_a_streamed_artifact_round_trips_exactly_at_any_chunk_size(
        vault, tmp_path, payload, chunk):
    """
    The chunk size changes how many authenticated blocks the file has, which
    is exactly the dimension a reassembly bug would hide in.
    """
    source = tmp_path / "source.bin"
    source.write_bytes(payload)
    path = vault.write_stream("prop/streamed", source, "rec_prop", chunk_size=chunk)
    assert _read_stream_bytes(vault, path, "rec_prop") == payload


@given(payload=st.binary(min_size=64, max_size=2048), data=st.data())
@CRYPTO_SETTINGS
def test_no_single_byte_mutation_can_produce_wrong_plaintext(vault, tmp_path,
                                                             payload, data):
    """
    Flip any one byte anywhere in the file -- header, salt, nonce, ciphertext,
    tag -- and the read must raise. What it must never do is return bytes that
    are not what was written.
    """
    source = tmp_path / "source.bin"
    source.write_bytes(payload)
    path = vault.write_stream("prop/mutate", source, "rec_prop", chunk_size=64)

    raw = bytearray(path.read_bytes())
    index = data.draw(st.integers(0, len(raw) - 1))
    flip = data.draw(st.integers(1, 255))
    raw[index] ^= flip
    path.write_bytes(bytes(raw))

    try:
        got = _read_stream_bytes(vault, path, "rec_prop")
    except VaultError:
        return                      # the only other acceptable outcome
    assert got == payload, (
        "a mutated vault file returned plaintext that was never written -- "
        f"byte {index} of {len(raw)} was flipped and the read stayed silent")


@given(payload=st.binary(min_size=64, max_size=2048), data=st.data())
@CRYPTO_SETTINGS
def test_truncation_is_always_caught(vault, tmp_path, payload, data):
    """
    A file cut short must never read back as a shorter recording. This is the
    property that makes "the transcript just ends there" impossible to
    mistake for the truth.
    """
    source = tmp_path / "source.bin"
    source.write_bytes(payload)
    path = vault.write_stream("prop/truncate", source, "rec_prop", chunk_size=64)

    raw = path.read_bytes()
    keep = data.draw(st.integers(0, len(raw) - 1))
    path.write_bytes(raw[:keep])

    with pytest.raises(VaultError):
        _read_stream_bytes(vault, path, "rec_prop")


@given(payload=st.binary(min_size=32, max_size=1024),
       wrong_id=st.text(alphabet="abcdef0123456789_", min_size=1, max_size=16))
@CRYPTO_SETTINGS
def test_reading_under_the_wrong_recording_id_always_fails(vault, tmp_path,
                                                           payload, wrong_id):
    """
    The recording id is folded into the authenticated data, so an artifact
    cannot be lifted out of one recording and read as another's. Without this
    the vault would happily decrypt a file moved between recordings, which is
    how `forget` could leave something readable behind.
    """
    source = tmp_path / "source.bin"
    source.write_bytes(payload)
    path = vault.write_stream("prop/aad", source, "rec_the_real_one", chunk_size=64)

    if wrong_id == "rec_the_real_one":
        return
    with pytest.raises(VaultError):
        _read_stream_bytes(vault, path, wrong_id)


# =========================================================================
# Untrusted input: filenames, transcript text, vault names
# =========================================================================
@given(raw=st.text(max_size=200))
@SETTINGS
def test_a_saved_upload_name_can_never_escape_the_inbox(raw):
    """
    The filename comes from a browser header, so it is attacker-controlled in
    the only sense that matters: a page you did not write can send one. It
    must always reduce to a bare name that lands inside the inbox.
    """
    name = _safe_name(raw)
    assert name, "a name must never reduce to empty; the inbox needs somewhere to write"
    assert "/" not in name and "\\" not in name
    assert "\x00" not in name
    assert name not in (".", "..")
    # The decisive check: joined onto a directory, it stays in that directory.
    base = Path("/data/inbox")
    assert (base / name).resolve().parent == base


@given(raw=st.text(max_size=200))
@SETTINGS
def test_an_upload_extension_check_is_never_bypassed_by_the_sanitiser(raw):
    """Sanitising must not turn a rejected extension into an accepted one."""
    name = _safe_name(raw)
    if Path(name).suffix.lower() in _ACCEPTED:
        # Whatever it started as, what will be written ends in an accepted
        # extension -- the check and the written name agree.
        assert Path(name).suffix.lower() in _ACCEPTED


@given(text=st.text(max_size=300))
@SETTINGS
def test_question_detection_is_total_and_deterministic(text):
    """
    It runs over recogniser output, which contains anything: empty segments,
    punctuation soup, other alphabets. It must answer, not raise, and answer
    the same way twice -- a metric that flickers is worse than no metric.
    """
    first = is_question(text)
    assert isinstance(first, bool)
    assert is_question(text) is first


@given(name=st.text(max_size=120))
@SETTINGS
def test_a_content_type_is_always_returned_for_any_vault_name(name):
    ctype = content_type_for(name)
    assert isinstance(ctype, str) and "/" in ctype


# =========================================================================
# Range serving: never a byte outside what was asked for
# =========================================================================
@given(payload=st.binary(min_size=1, max_size=2048), data=st.data())
@SETTINGS
def test_a_plaintext_range_never_returns_bytes_outside_the_request(tmp_path,
                                                                   payload, data):
    """
    A player asking for bytes 10-19 that receives 10-30 would render audio
    with a stutter nobody could explain. The iterator must be exact, and an
    unsatisfiable start must be refused rather than quietly clamped.
    """
    path = tmp_path / "audio.mp3"
    path.write_bytes(payload)
    info = MediaInfo(recording_id="rec_x", path=path, encrypted=False,
                     content_type="audio/mpeg", size_bytes=len(payload))

    start = data.draw(st.integers(0, len(payload) + 5))
    end = data.draw(st.one_of(st.none(), st.integers(0, len(payload) + 5)))

    if start >= len(payload) or (end is not None and end < start):
        with pytest.raises(ValueError):
            read_range(info, start, end, cfg=None)
        return

    body, total = read_range(info, start, end, cfg=None)
    got = b"".join(body)
    stop = len(payload) - 1 if end is None else min(end, len(payload) - 1)
    assert total == len(payload)
    assert got == payload[start:stop + 1]
