"""
Regressions found by auditing the code against its own claims.

Every test here reproduces a defect that shipped. They are grouped by what the
defect actually cost, because that is what decides how hard to fight to keep
them passing:

  - content that should never have been in the clear, was
  - a search that answered "never said" without looking
  - a delete that reached outside the tool's own directories

None of these raised an exception. That is what made them worth writing down.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

import plaud_bridge.pipeline as pipeline_module
from _fixtures import CLIENT_CALL, FAMILY_DINNER, build_sandbox, drop
from plaud_bridge.archive import Archive, is_owned, owned_roots
from plaud_bridge.compliance import RetentionSweeper
from plaud_bridge.config import Config
from plaud_bridge.db import Database
from plaud_bridge.digest import DigestBuilder, DigestOptions
from plaud_bridge.models import ComplianceVerdict, Recording
from plaud_bridge.pipeline import Pipeline


# =========================================================================
# Plaintext that should not exist
# =========================================================================
def test_a_failure_before_the_gate_leaves_no_plaintext_in_the_index(
    tmp_path, monkeypatch
):
    """
    The gate decides whether content may sit in the clear. A recording can fail
    between transcription and the gate -- a provider outage, anything -- and the
    old default said "not encrypted", so the full transcript of a family
    conversation was written to the unencrypted index and stayed there.
    """
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    drop(cfg, "dinner-with-kid.txt", FAMILY_DINNER)

    def explode(*a, **k):
        raise RuntimeError("router outage")

    monkeypatch.setattr(pipeline_module, "route", explode)

    pipe = Pipeline(cfg)
    try:
        assert pipe.run().failed == 1
    finally:
        pipe.close()

    raw = cfg.path("database").read_bytes()
    assert b"permission slip" not in raw
    assert b"Coach said" not in raw


def test_the_verdict_defaults_to_encrypted():
    """The safe default is the one that survives a code path nobody thought of."""
    assert ComplianceVerdict().encrypt_at_rest is True
    assert Recording().is_encrypted is True


def test_encryption_has_one_source_of_truth(tmp_path, monkeypatch):
    """
    High sensitivity with encryption switched off is a legal profile. It used to
    make persistence and the index disagree: artifacts written in plaintext, the
    index withholding them as encrypted, and a digest that could never render
    that recording again.
    """
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    path = tmp_path / "config" / "profiles" / "insurance_agent.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["processing"]["encrypt_at_rest"] = False
    path.write_text(yaml.safe_dump(raw))

    cfg = Config.load(tmp_path / "config", root=tmp_path)
    cfg.ensure_dirs()
    drop(cfg, "client.txt", CLIENT_CALL)

    pipe = Pipeline(cfg)
    try:
        pipe.run()
        payload = json.loads(pipe.db.query()[0]["payload_json"])
        assert not payload["analyses"][0].get("fields_withheld")
        body = DigestBuilder(cfg, pipe.db).render_markdown(DigestOptions(days=30))
    finally:
        pipe.close()

    assert "Send two quote options" in body
    assert "missing from disk" not in body


def test_the_original_recording_is_encrypted_when_the_profile_encrypts(
    tmp_path, monkeypatch
):
    """
    The original is the actual voices, which makes it the most sensitive
    artifact of the lot. It used to be moved to _processed/ in the clear while
    the transcript derived from it sat encrypted beside it.
    """
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    drop(cfg, "dinner-with-kid.txt", FAMILY_DINNER)

    pipe = Pipeline(cfg)
    try:
        pipe.run()
        payload = json.loads(pipe.db.query()[0]["payload_json"])
    finally:
        pipe.close()

    assert payload["compliance"]["governing_profile"] == "father"
    leftovers = list((cfg.path("inbox") / "_processed").glob("*"))
    assert not leftovers, f"the original was left in the clear: {leftovers}"

    stored = Path(payload["artifact_paths"]["source"])
    assert stored.suffix == ".enc"
    assert b"permission slip" not in stored.read_bytes()

    from plaud_bridge.storage import Vault

    # Originals are streamed, so they come back out the same way -- a day of
    # audio must never need to be resident in memory to be read.
    assert Vault.is_streamed(stored)
    out = tmp_path / "recovered.txt"
    Vault(cfg.path("vault")).read_stream(stored, out, payload["id"])
    assert b"permission slip" in out.read_bytes(), "the original did not survive encryption"


# =========================================================================
# Searches that did not look
# =========================================================================
def test_content_search_scans_everything_by_default(tmp_path, monkeypatch):
    """
    The CLI's --limit means "results", and it was being handed to the row query
    as "recordings to open". An archive larger than the limit had its oldest
    recordings silently excluded, and the search reported nothing was found.
    """
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    for index in range(55):
        body = CLIENT_CALL if index == 0 else CLIENT_CALL.replace(
            "elimination. Period", "deductible"
        )
        drop(cfg, f"call-{index:03d}.txt", f"{body}\nSasson: filler {index}\n")

    pipe = Pipeline(cfg)
    try:
        pipe.run()
    finally:
        pipe.close()

    db = Database(cfg.path("database"))
    try:
        result = Archive(cfg, db).search_content("elimination period")
        assert result.scanned == result.total == 55
        assert result.matches, "the oldest recording was never opened"
        assert result.complete
    finally:
        db.close()


def test_a_bounded_search_says_it_was_bounded(tmp_path, monkeypatch):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    for index in range(5):
        drop(cfg, f"call-{index}.txt", f"{CLIENT_CALL}\nSasson: filler {index}\n")

    pipe = Pipeline(cfg)
    try:
        pipe.run()
    finally:
        pipe.close()

    db = Database(cfg.path("database"))
    try:
        result = Archive(cfg, db).search_content("elimination period", scan_limit=2)
        assert result.scanned == 2
        assert result.total == 5
        assert result.truncated
        assert not result.complete, "a partial search reported itself as complete"
    finally:
        db.close()


# =========================================================================
# Deletes that reached too far
# =========================================================================
def _point_index_at(db, recording_id: str, target: Path) -> None:
    payload = db.load(recording_id)
    payload["artifact_paths"]["transcript"] = str(target)
    with db.tx() as cur:
        cur.execute("UPDATE recordings SET payload_json=? WHERE id=?",
                    (json.dumps(payload), recording_id))
        cur.execute("UPDATE artifacts SET path=? WHERE recording_id=? AND kind='transcript'",
                    (str(target), recording_id))


def test_forget_will_not_delete_outside_the_data_directory(tmp_path, monkeypatch):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    drop(cfg, "client.txt", CLIENT_CALL)
    pipe = Pipeline(cfg)
    try:
        pipe.run()
        rec_id = pipe.db.query()[0]["id"]
    finally:
        pipe.close()

    outsider = tmp_path / "elsewhere" / "tax-return.pdf"
    outsider.parent.mkdir(parents=True)
    outsider.write_text("not ours")

    db = Database(cfg.path("database"))
    try:
        _point_index_at(db, rec_id, outsider)
        assert outsider not in Archive(cfg, db).plan_forget(rec_id)
        _removed, failures = Archive(cfg, db).forget(rec_id)
        assert outsider.exists(), "forget deleted a file outside its own directories"
        assert any("outside" in f for f in failures), "it deleted nothing but said nothing"
    finally:
        db.close()


def test_retention_will_not_delete_outside_the_data_directory(tmp_path, monkeypatch):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    drop(cfg, "client.txt", CLIENT_CALL)
    pipe = Pipeline(cfg)
    try:
        pipe.run()
        rec_id = pipe.db.query()[0]["id"]
    finally:
        pipe.close()

    outsider = tmp_path / "elsewhere" / "wedding-photos.zip"
    outsider.parent.mkdir(parents=True)
    outsider.write_text("not ours")

    db = Database(cfg.path("database"))
    try:
        _point_index_at(db, rec_id, outsider)
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        with db.tx() as cur:
            cur.execute("UPDATE artifacts SET expires_at=? WHERE recording_id=?", (past, rec_id))

        sweeper = RetentionSweeper(cfg, db)
        sweeper.execute(sweeper.plan(dry_run=False))
        assert outsider.exists(), "the retention sweep deleted somebody else's file"
        assert any(r["action"] == "retention_refused" for r in db.audit_log(limit=50))
    finally:
        db.close()


def test_owned_roots_covers_the_directories_we_write_to(tmp_path, monkeypatch):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    roots = owned_roots(cfg)
    for name in ("vault", "outbox", "inbox", "quarantine", "work"):
        assert is_owned(cfg.path(name) / "x.enc", roots)
    assert not is_owned(Path("/etc/passwd"), roots)
    assert not is_owned(tmp_path / "elsewhere" / "x", roots)


# =========================================================================
# Everything else the audit turned up
# =========================================================================
def test_a_file_still_being_written_is_not_ingested(tmp_path, monkeypatch):
    """Its partial content hash is what dedupe would remember forever."""
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    raw = yaml.safe_load((tmp_path / "config" / "pipeline.yaml").read_text())
    raw["ingest"]["settle_seconds"] = 3600
    (tmp_path / "config" / "pipeline.yaml").write_text(yaml.safe_dump(raw))
    cfg = Config.load(tmp_path / "config", root=tmp_path)
    cfg.ensure_dirs()

    drop(cfg, "still-copying.txt", CLIENT_CALL)
    pipe = Pipeline(cfg)
    try:
        assert pipe.discover() == []
        assert pipe.unsettled, "the skip happened but nothing recorded why"
    finally:
        pipe.close()


def test_audit_retention_is_applied_and_keeps_the_longest_window(tmp_path, monkeypatch):
    """`audit_log_days` was parsed and read by nothing."""
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    db = Database(cfg.path("database"))
    try:
        longest = max(p.audit_log_days for p in cfg.profiles.values())
        old = (datetime.now(timezone.utc) - timedelta(days=longest + 30)).isoformat()
        recent = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        with db.tx() as cur:
            for at, detail in ((old, "ancient"), (recent, "recent")):
                cur.execute(
                    "INSERT INTO audit(at,recording_id,action,detail,actor) "
                    "VALUES (?,NULL,'marker',?,'pipeline')", (at, detail))

        sweeper = RetentionSweeper(cfg, db)
        plan = sweeper.plan(dry_run=True)
        assert plan.audit_rows == 1
        assert not plan.empty

        sweeper.execute(sweeper.plan(dry_run=False))
        details = {r["detail"] for r in db.audit_log(action="marker", limit=50)}
        assert "recent" in details
        assert "ancient" not in details
    finally:
        db.close()


def test_a_dry_run_plan_still_cannot_delete(tmp_path, monkeypatch):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    db = Database(cfg.path("database"))
    try:
        sweeper = RetentionSweeper(cfg, db)
        with pytest.raises(ValueError):
            sweeper.execute(sweeper.plan(dry_run=True))
    finally:
        db.close()


@pytest.mark.parametrize("name", ["100% done.txt", "a_b.txt"])
def test_filename_search_treats_wildcards_as_literals(tmp_path, monkeypatch, name):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    drop(cfg, name, CLIENT_CALL)
    drop(cfg, "unrelated.txt", f"{CLIENT_CALL}\nSasson: different\n")

    pipe = Pipeline(cfg)
    try:
        pipe.run()
        hits = pipe.db.query(search=name.split(".")[0])
        assert len(hits) == 1, "a LIKE wildcard in the filename matched everything"
        assert hits[0]["source_name"] == name
    finally:
        pipe.close()


def test_duplicate_content_from_a_racing_process_is_not_a_failure(tmp_path, monkeypatch):
    """
    Two processes can both pass the dedupe check before either writes. That is a
    duplicate, not a broken recording, and it must not be reported as one.
    """
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    drop(cfg, "client.txt", CLIENT_CALL)
    pipe = Pipeline(cfg)
    try:
        pipe.run()
        original = pipe.db.load(pipe.db.query()[0]["id"])

        clash = Recording(source_name="other.txt", content_hash=original["content_hash"])
        clash.compliance.encrypt_at_rest = False
        pipe.db.upsert(clash)   # must not raise

        assert len(pipe.db.query()) == 1
    finally:
        pipe.close()
