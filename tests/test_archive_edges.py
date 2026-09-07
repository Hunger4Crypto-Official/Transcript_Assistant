"""
The archive's edges: what verify says, what forget reaches, and what a locked
vault makes each of them refuse.

test_archive.py pins the headline promises. These pin the branches around
them: the rendered verify report, the ownership checks that stop a hand-edited
index deleting somebody else's file, every store `forget` has to reach (an
archived original, a draft named after the recording, a saved answer that
will not open), and each reason a locked vault refuses the whole operation --
or, for a plaintext state file, does not.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from _fixtures import CLIENT_CALL, build_sandbox, drop
from plaud_bridge.archive import Archive, VerifyReport, is_owned, owned_roots
from plaud_bridge.config import ConfigError
from plaud_bridge.db import Database
from plaud_bridge.followups import FollowUp, _load_state, set_status, stable_id, state_path
from plaud_bridge.models import Recording, RouteMatch, Segment, Transcript
from plaud_bridge.pipeline import Pipeline
from plaud_bridge.storage import Vault

# The consent exchange stripped from the client call, so the gate quarantines
# it for a missing announcement.
NO_ANNOUNCEMENT = CLIENT_CALL.split("\n", 2)[2]


def _processed(tmp_path, monkeypatch, files=(("client-marcus.txt", CLIENT_CALL),)):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    for name, body in files:
        drop(cfg, name, body)
    pipe = Pipeline(cfg)
    try:
        pipe.run()
    finally:
        pipe.close()
    return cfg


def _hand_indexed(cfg, db, rid="rec_hand", profile="sales_trainer") -> Path:
    """
    One unencrypted recording put straight into the index, with a plaintext
    transcript in the outbox for forget to have something to delete. Built by
    hand so no memory ledger exists -- that is what lets the tests below reach
    the other locked-vault branches.
    """
    rec = Recording(id=rid, source_name=f"{rid}.txt", source_path=f"/inbox/{rid}.txt",
                    content_hash=f"hash-{rid}", kind="text")
    rec.transcript = Transcript(segments=[Segment(0.0, 2.0, "spoken words", "Sasson")])
    rec.routes = [RouteMatch(profile_id=profile, confidence=0.9)]
    rec.compliance.governing_profile = profile
    rec.compliance.encrypt_at_rest = False
    out = cfg.path("outbox") / f"{rid}.transcript.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("spoken words", encoding="utf-8")
    rec.artifact_paths["transcript"] = str(out)
    db.upsert(rec)
    db.record_artifact(rid, "transcript", str(out), False, None)
    return out


def _set_payload_path(db, rec_id: str, key: str, path: Path) -> None:
    """Point one artifact_paths entry somewhere, the way a hand-edited index might."""
    payload = db.load(rec_id)
    payload["artifact_paths"][key] = str(path)
    with db.tx() as cur:
        cur.execute("UPDATE recordings SET payload_json=? WHERE id=?",
                    (json.dumps(payload), rec_id))


@pytest.fixture
def opened(tmp_path, monkeypatch):
    cfg = _processed(tmp_path, monkeypatch)
    db = Database(cfg.path("database"))
    try:
        yield cfg, db, Archive(cfg, db), db.query()[0]["id"]
    finally:
        db.close()


# =========================================================================
# What verify says
# =========================================================================
def test_the_verify_report_names_each_problem_with_its_path_and_why(opened):
    cfg, db, archive, _rec = opened
    missing = Path(db.all_artifacts()[0]["path"])
    missing.unlink()

    out = archive.verify().render()
    assert "[missing" in out
    assert str(missing) in out
    assert "the index points at a file that is not on disk" in out
    assert "Everything the index points at exists and opens" not in out


def test_the_verify_report_caps_the_orphan_list_and_says_how_many_more():
    report = VerifyReport(orphans=[Path(f"/vault/stray-{i:02d}.enc") for i in range(45)])
    out = report.render()
    assert "45 file(s) on disk that the index does not know about" in out
    assert "/vault/stray-39.enc" in out
    assert "/vault/stray-40.enc" not in out
    assert "... and 5 more" in out
    assert "not deleted automatically" in out
    assert report.healthy, "orphans are reported, not counted as breakage"


# =========================================================================
# Ownership: the check that stops a hand-edited index deleting anything
# =========================================================================
def test_owned_roots_skips_a_directory_the_config_cannot_name(tmp_path):
    class HalfConfigured:
        def path(self, name):
            if name == "work":
                raise ConfigError("paths.work is not configured")
            return tmp_path / name

    roots = owned_roots(HalfConfigured())
    assert len(roots) == 4
    assert is_owned(tmp_path / "vault" / "x.enc", roots)
    assert not is_owned(tmp_path / "work" / "x.wav", roots)


def test_a_path_that_cannot_be_resolved_is_not_owned(tmp_path):
    class Unresolvable(type(Path())):
        def resolve(self, strict=False):
            raise OSError("no such device")

    assert is_owned(Unresolvable(tmp_path / "vault" / "x.enc"), [tmp_path / "vault"]) is False


# =========================================================================
# Reading
# =========================================================================
def test_a_row_whose_payload_is_not_json_reads_as_empty_rather_than_locked(opened):
    """
    The current contract: an unparseable index row is "nothing here", not
    "cannot open". It is not reported in `unopened`, because the advice there
    (set the passphrase) could not help. This pins that so a change is a choice.
    """
    _cfg, _db, archive, _rec = opened
    row = {"id": "rec_corrupt", "source_name": "x.txt", "stage": "complete",
           "payload_json": "{this is not json"}
    assert archive.full_record(row) == {}
    assert archive.segments(row) == []


def test_a_blank_query_looks_at_nothing_and_says_so(opened):
    _cfg, _db, archive, _rec = opened
    result = archive.search_content("   ")
    assert result.matches == [] and result.scanned == 0 and result.total == 0
    assert result.complete


def test_a_quarantined_recording_is_reported_apart_from_an_unreadable_one(
    tmp_path, monkeypatch
):
    """
    Nothing was persisted for it, so there is nothing to search. Telling the
    user to check the passphrase -- what `unopened` means -- would send them
    after a problem they do not have.
    """
    cfg = _processed(tmp_path, monkeypatch, files=(("unannounced.txt", NO_ANNOUNCEMENT),))
    db = Database(cfg.path("database"))
    try:
        assert db.query(stage="quarantined"), "the fixture did not quarantine"
        result = Archive(cfg, db).search_content("elimination")
        assert result.matches == []
        assert result.unopened == []
        assert len(result.quarantined) == 1 and "unannounced.txt" in result.quarantined[0]
        assert result.complete
    finally:
        db.close()


def test_context_widens_a_hit_to_the_lines_around_it(opened):
    _cfg, _db, archive, _rec = opened
    narrow = archive.search_content("elimination period")
    wide = archive.search_content("elimination period", context=1)
    assert narrow.matches and wide.matches
    assert "What does that run" not in narrow.matches[0].text
    assert "What does that run" in wide.matches[0].text, "the previous line is missing"
    assert "price is my worry" in wide.matches[0].text, "the next line is missing"


# =========================================================================
# Everything forget has to reach
# =========================================================================
def test_forget_reaches_an_archived_original_and_leaves_other_originals_alone(opened):
    cfg, db, archive, rec = opened
    processed = cfg.path("inbox") / "_processed"
    processed.mkdir(parents=True, exist_ok=True)
    mine = processed / f"{rec}_client-marcus.txt"
    theirs = processed / "rec_someoneelse_call.txt"
    mine.write_text("the original")
    theirs.write_text("a different original")

    planned = archive.plan_forget(rec)
    assert mine in planned
    assert theirs not in planned

    _removed, failures = archive.forget(rec)
    assert failures == []
    assert not mine.exists()
    assert theirs.exists()


def test_forget_skips_an_artifact_the_index_names_but_disk_no_longer_holds(opened):
    _cfg, _db, archive, rec = opened
    planned = archive.plan_forget(rec)
    assert len(planned) >= 2
    already_gone = planned[0]
    already_gone.unlink()

    assert already_gone not in archive.plan_forget(rec)
    removed, failures = archive.forget(rec)
    assert removed == len(planned) - 1
    assert failures == []


def test_forget_skips_a_path_it_cannot_resolve_rather_than_crashing(opened, monkeypatch):
    cfg, db, archive, rec = opened

    class Flaky(type(Path())):
        def resolve(self, strict=False):
            if self.name == "boom":
                raise OSError("cannot resolve")
            return super().resolve(strict)

    ghost = cfg.path("vault") / "boom"
    ghost.write_bytes(b"x")
    _set_payload_path(db, rec, "ghost", ghost)

    monkeypatch.setattr("plaud_bridge.archive.Path", Flaky)
    assert ghost not in archive.plan_forget(rec)
    _removed, failures = archive.forget(rec)
    assert failures == []
    assert ghost.exists()


def test_a_draft_named_after_the_recording_goes_and_a_subfolder_is_ignored(opened):
    cfg, db, archive, rec = opened
    drafts = cfg.path("outbox") / "drafts"
    drafts.mkdir(parents=True, exist_ok=True)
    named = drafts / f"DRAFT-2026-01-01-{rec}.draft.md"
    named.write_text("# DRAFT\n\nnothing in the body names it")
    unrelated = drafts / "DRAFT-2026-01-01-other.draft.md"
    unrelated.write_text("# DRAFT\n\nabout somebody else")
    nested = drafts / "old"
    nested.mkdir()
    (nested / "archive.md").write_text(f"traced to: {rec}")

    planned = archive.plan_forget(rec)
    assert named in planned
    assert unrelated not in planned
    assert nested / "archive.md" not in planned

    archive.forget(rec)
    assert not named.exists()
    assert unrelated.exists()
    assert nested.is_dir()


def test_a_draft_that_cannot_be_read_is_left_alone_rather_than_crashing_forget(
    opened, monkeypatch
):
    cfg, _db, archive, rec = opened
    drafts = cfg.path("outbox") / "drafts"
    drafts.mkdir(parents=True, exist_ok=True)
    unreadable = drafts / "unreadable.md"
    unreadable.write_text(f"traced to: {rec}")

    real_read_text = Path.read_text

    def flaky(self, *a, **k):
        if self.name == "unreadable.md":
            raise OSError("input/output error")
        return real_read_text(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", flaky)
    assert unreadable not in archive.plan_forget(rec)
    _removed, failures = archive.forget(rec)
    assert failures == []
    assert unreadable.exists()


def test_a_saved_answer_that_will_not_open_is_left_for_verify_and_the_rest_still_go(opened):
    cfg, _db, archive, rec = opened
    vault = Vault(cfg.path("vault"))
    cites = vault.write("ask/cites-it", json.dumps({"recordings_used": [rec], "citations": []}))
    garbage = cfg.path("vault") / "ask" / "garbage.enc"
    garbage.write_bytes(b"PBV1 not really a vault file")

    planned = archive.plan_forget(rec)
    assert cites in planned
    assert garbage not in planned

    _removed, failures = archive.forget(rec)
    assert failures == []
    assert not cites.exists()
    assert garbage.exists()


# =========================================================================
# A locked vault: each store that makes forget refuse, and the one that does not
# =========================================================================
def _hand(tmp_path, monkeypatch):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    db = Database(cfg.path("database"))
    artifact = _hand_indexed(cfg, db)
    return cfg, db, Archive(cfg, db), "rec_hand", artifact


def _assert_refused(db, archive, rec, artifact):
    removed, failures = archive.forget(rec)
    assert removed == 0
    assert len(failures) == 1 and "locked" in failures[0] and "PLAUD_BRIDGE_PASSPHRASE" in failures[0]
    assert artifact.exists(), "the plaintext half was deleted despite the refusal"
    assert db.load(rec) is not None, "the index row was deleted despite the refusal"
    trail = db.audit_log(recording_id=rec, limit=10)
    assert [r["action"] for r in trail] == ["forget_refused"]
    assert all(r["actor"] == "human" for r in trail)


def test_a_locked_vault_with_a_saved_answer_on_disk_refuses_to_forget(tmp_path, monkeypatch):
    cfg, db, archive, rec, artifact = _hand(tmp_path, monkeypatch)
    try:
        Vault(cfg.path("vault")).write("ask/some-answer", "{}")
        monkeypatch.delenv("PLAUD_BRIDGE_PASSPHRASE")
        _assert_refused(db, archive, rec, artifact)
    finally:
        db.close()


def test_a_locked_vault_with_an_encrypted_follow_up_state_refuses_to_forget(
    tmp_path, monkeypatch
):
    cfg, db, archive, rec, artifact = _hand(tmp_path, monkeypatch)
    try:
        fu = FollowUp(id=stable_id("send it", "sales_trainer"), text="send it",
                      profile_id="sales_trainer", recording_id=rec)
        set_status(cfg, Vault(cfg.path("vault")), fu.id, "done", items=[fu])
        assert state_path(cfg).read_bytes().startswith(b"PBV1")

        monkeypatch.delenv("PLAUD_BRIDGE_PASSPHRASE")
        _assert_refused(db, archive, rec, artifact)
    finally:
        db.close()


def test_a_plaintext_follow_up_state_does_not_block_a_passphrase_free_forget(
    tmp_path, monkeypatch
):
    """
    Someone running with no vault at all has a plaintext state file. It can be
    read and purged without a passphrase, so it must not force the refusal --
    that would make forget impossible for them.
    """
    cfg, db, archive, rec, artifact = _hand(tmp_path, monkeypatch)
    try:
        monkeypatch.delenv("PLAUD_BRIDGE_PASSPHRASE")
        vault = Vault(cfg.path("vault"))
        fu = FollowUp(id=stable_id("send it", "sales_trainer"), text="send it",
                      profile_id="sales_trainer", recording_id=rec)
        set_status(cfg, vault, fu.id, "done", items=[fu])
        assert not state_path(cfg).read_bytes().startswith(b"PBV1")

        removed, failures = archive.forget(rec)
        assert removed == 1 and failures == []
        assert not artifact.exists()
        assert db.load(rec) is None
        assert _load_state(cfg, vault) == {}, "the plaintext state still names the recording"
        assert not db.audit_log(action="forget_refused", limit=5)
    finally:
        db.close()


def test_a_state_path_that_cannot_be_computed_is_not_treated_as_a_store(
    tmp_path, monkeypatch
):
    cfg, db, archive, rec, artifact = _hand(tmp_path, monkeypatch)
    try:
        def broken(_cfg):
            raise RuntimeError("paths.database is not configured")

        monkeypatch.setattr("plaud_bridge.followups.state_path", broken)
        monkeypatch.delenv("PLAUD_BRIDGE_PASSPHRASE")

        removed, failures = archive.forget(rec)
        assert removed == 1
        assert not artifact.exists()
        # The purge itself hits the same broken lookup and says so, rather
        # than pretending the follow-up state was cleaned.
        assert failures == ["follow-up state: paths.database is not configured"]
    finally:
        db.close()


def test_a_memory_store_that_cannot_be_built_is_reported_and_forget_still_completes(
    tmp_path, monkeypatch
):
    cfg, db, archive, rec, artifact = _hand(tmp_path, monkeypatch)
    try:
        class Broken:
            def __init__(self, *_a, **_k):
                raise RuntimeError("memory.dir is unwritable")

        monkeypatch.setattr("plaud_bridge.memory.MemoryStore", Broken)
        monkeypatch.delenv("PLAUD_BRIDGE_PASSPHRASE")

        removed, failures = archive.forget(rec)
        assert removed == 1
        assert not artifact.exists()
        assert db.load(rec) is None
        assert failures == ["memory ledgers: memory.dir is unwritable"]
    finally:
        db.close()


def test_a_follow_up_purge_that_raises_is_reported_in_failures(opened, monkeypatch):
    _cfg, db, archive, rec = opened

    def explode(*_a, **_k):
        raise RuntimeError("state file is corrupt")

    monkeypatch.setattr("plaud_bridge.followups.forget_recording", explode)
    _removed, failures = archive.forget(rec)
    assert "follow-up state: state file is corrupt" in failures
    assert db.load(rec) is None, "the index entry survived a reported failure"
    assert any(r["action"] == "forget_complete" for r in db.audit_log(recording_id=rec))


# =========================================================================
# When deletion itself fails
# =========================================================================
def test_a_target_that_will_not_unlink_is_reported_and_the_rest_still_go(opened):
    cfg, db, archive, rec = opened
    stubborn = cfg.path("vault") / "stubborn"
    stubborn.mkdir()
    _set_payload_path(db, rec, "stubborn", stubborn)
    others = [p for p in archive.plan_forget(rec) if p != stubborn]
    assert stubborn in archive.plan_forget(rec)

    removed, failures = archive.forget(rec)
    assert removed == len(others)
    assert len(failures) == 1 and str(stubborn) in failures[0]
    assert all(not p.exists() for p in others)
    assert stubborn.exists()
    assert db.load(rec) is None


def test_forget_clears_a_quarantine_folder_with_nested_directories(tmp_path, monkeypatch):
    cfg = _processed(tmp_path, monkeypatch, files=(("unannounced.txt", NO_ANNOUNCEMENT),))
    db = Database(cfg.path("database"))
    try:
        rec = db.query(stage="quarantined")[0]["id"]
        qdir = cfg.path("quarantine") / rec
        assert qdir.is_dir()
        deep = qdir / "nested" / "deeper"
        deep.mkdir(parents=True)
        (deep / "note.txt").write_text("left by hand")

        _removed, failures = Archive(cfg, db).forget(rec)
        assert failures == []
        assert not qdir.exists()
    finally:
        db.close()


def test_a_quarantine_folder_that_cannot_be_emptied_is_reported(tmp_path, monkeypatch):
    """A FIFO is neither a file nor a directory, so nothing here can remove it."""
    cfg = _processed(tmp_path, monkeypatch, files=(("unannounced.txt", NO_ANNOUNCEMENT),))
    db = Database(cfg.path("database"))
    try:
        rec = db.query(stage="quarantined")[0]["id"]
        qdir = cfg.path("quarantine") / rec
        os.mkfifo(qdir / "pipe")

        _removed, failures = Archive(cfg, db).forget(rec)
        assert len(failures) == 1 and str(qdir) in failures[0]
        assert qdir.is_dir()
        assert not (qdir / "WHY.md").exists(), "the regular files were not removed"
        assert db.load(rec) is None
    finally:
        db.close()
