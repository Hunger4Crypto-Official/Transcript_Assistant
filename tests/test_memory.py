"""
Learning across recordings, and the things that must not leak while it does.

Two kinds of test live here. The ordinary ones check that the ledger counts,
decays, and renders the way the module says it does. The rest correspond to
promises: a profile's memory never appearing in another profile's prompt, a
sensitive field never entering the ledger at all, a deleted recording leaving
nothing behind, and the ledger never being written in the clear because the
passphrase was missing. Those are written so that removing the guarantee makes
them fail rather than pass quietly.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest

from _fixtures import CLIENT_CALL, FAMILY_DINNER, build_sandbox, drop
from plaud_bridge.archive import Archive
from plaud_bridge.memory import (
    COMMITMENT,
    FACT,
    PERSON,
    TOPIC,
    MemoryStore,
    carry_forward_brief,
    render_ledger,
)
from plaud_bridge.pipeline import Pipeline
from plaud_bridge.storage import Vault, VaultError

NOW = datetime(2026, 7, 27, 9, 0, tzinfo=timezone.utc)


def _record(recording_id, profile_id, fields, *, when=NOW, source_name="rec.txt",
            attention=False):
    """One analysed recording in the shape the index stores it."""
    return {
        "id": recording_id,
        "source_name": source_name,
        "recorded_at": when.isoformat(),
        "ingested_at": when.isoformat(),
        "analyses": [{
            "profile_id": profile_id,
            "fields": fields,
            "requires_human_attention": attention,
            "error": "",
        }],
    }


FATHER_FIELDS = {
    "requires_human_attention": False,
    "worth_remembering": [{"timestamp": "00:03", "speaker": "Maya",
                           "text": "Coach said I'm starting on Saturday"}],
    "promises_i_made": [{"what": "sign the permission slip", "when": "tonight"}],
    "logistics": [{"what": "game", "when": "Saturday 10:00 AM"}],
    "asks_from_them": [{"timestamp": "00:12", "speaker": "Maya",
                        "text": "can we get pizza after the game?"}],
    "milestones": ["First time starting"],
    "next_action": "Sign the permission slip tonight",
}

AGENT_FIELDS = {
    "meeting_type": "fact_find",
    "participants": ["Sasson", "Marcus"],
    "stated_needs": [{"timestamp": "00:21", "speaker": "Marcus",
                      "text": "We just had our second kid."}],
    "financial_disclosures": [{"timestamp": "00:27", "speaker": "Marcus",
                               "text": "the mortgage is about four hundred thousand"}],
    "health_disclosures": [{"timestamp": "00:31", "speaker": "Marcus",
                            "text": "I had a biopsy last year"}],
    "objections": [{"type": "price", "quote": "The price is my worry honestly."}],
    "commitments_by_producer": [{"timestamp": "00:45", "speaker": "Sasson",
                                 "text": "I'll have them to you by Thursday."}],
    "statements_needing_review": [],
    "open_questions": ["What disability benefit amount fits the budget?"],
    "next_action": "Send two quote options by Thursday",
}


def _store(tmp_path, monkeypatch, overrides=None):
    cfg, _ = build_sandbox(tmp_path, monkeypatch, overrides=overrides)
    return cfg, MemoryStore(cfg)


# =========================================================================
# Idempotency
# =========================================================================
def test_the_same_recording_twice_does_not_double_count(tmp_path, monkeypatch):
    cfg, store = _store(tmp_path, monkeypatch)
    record = _record("rec_one", "father", FATHER_FIELDS)

    assert store.update_from_record(record) == ["father"]
    first = store.ledger("father").fingerprint()
    entries = len(store.ledger("father").entries)
    assert entries > 0, "nothing was filed, so this test would pass for the wrong reason"

    assert store.update_from_record(record) == []
    assert store.ledger("father").fingerprint() == first
    assert len(store.ledger("father").entries) == entries
    assert all(e.mentions == 1 for e in store.ledger("father").entries)


def test_reloading_from_disk_and_reapplying_is_still_idempotent(tmp_path, monkeypatch):
    """The seen-set has to survive the round trip, or every run double-counts."""
    cfg, store = _store(tmp_path, monkeypatch)
    store.update_from_record(_record("rec_one", "father", FATHER_FIELDS))
    before = store.ledger("father").fingerprint()

    reopened = MemoryStore(cfg)
    assert reopened.update_from_record(_record("rec_one", "father", FATHER_FIELDS)) == []
    assert reopened.ledger("father").fingerprint() == before


def test_a_reanalysed_recording_replaces_what_it_contributed(tmp_path, monkeypatch):
    cfg, store = _store(tmp_path, monkeypatch)
    store.update_from_record(_record("rec_one", "father", FATHER_FIELDS))

    changed = dict(FATHER_FIELDS)
    changed["promises_i_made"] = [{"what": "buy the goalkeeper gloves", "when": "Friday"}]
    store.update_from_record(_record("rec_one", "father", changed))

    ledger = store.ledger("father")
    assert ledger.entry(COMMITMENT, "buy the goalkeeper gloves when friday") is not None
    assert ledger.entry(COMMITMENT, "sign the permission slip when tonight") is None, (
        "the superseded analysis is still in the ledger alongside the new one"
    )
    assert all(e.mentions == 1 for e in ledger.entries)


def test_a_reanalysis_that_flags_a_recording_removes_what_it_filed(tmp_path, monkeypatch):
    """
    The flag exists to keep concerning content out of future prompts. A recording
    an earlier unflagged pass filed, then re-analysed and flagged, must have its
    old contribution REMOVED -- not left feeding carry_forward -- so the
    incremental ledger matches what rebuild() would produce from scratch (which
    never files a flagged analysis at all).
    """
    cfg, store = _store(tmp_path, monkeypatch)
    store.update_from_record(_record("rec_one", "father", FATHER_FIELDS))
    assert store.ledger("father").entries, "the first pass should have filed something"

    touched = store.update_from_record(
        _record("rec_one", "father", FATHER_FIELDS, attention=True)
    )
    assert "father" in touched, "flagging a previously-filed recording must change the ledger"

    ledger = store.ledger("father")
    assert ledger.entries == [], "the flagged re-analysis left the old contribution behind"
    assert "rec_one" not in ledger.seen, "the stale seen-signature survived the flag"


# =========================================================================
# Provenance
# =========================================================================
def test_every_entry_names_the_recordings_it_came_from(tmp_path, monkeypatch):
    cfg, store = _store(tmp_path, monkeypatch)
    store.update_from_record(_record("rec_one", "father", FATHER_FIELDS,
                                     source_name="dinner.txt"))
    store.update_from_record(_record("rec_two", "father", FATHER_FIELDS,
                                     when=NOW - timedelta(days=3), source_name="park.txt"))

    ledger = store.ledger("father")
    assert ledger.entries
    for entry in ledger.entries:
        assert entry.sightings, f"{entry.key} has no provenance at all"
        for sighting in entry.sightings:
            assert sighting.recording_id in ("rec_one", "rec_two")
            assert sighting.when
            assert sighting.field_key
            assert sighting.source_name in ("dinner.txt", "park.txt")

    repeated = ledger.entry(COMMITMENT, "sign the permission slip when tonight")
    assert repeated is not None, [e.key for e in ledger.entries]
    assert repeated.mentions == 2
    assert repeated.recording_ids == ["rec_two", "rec_one"]
    assert repeated.last_seen == NOW.isoformat(), "last seen is not the newest sighting"
    assert repeated.first_seen == (NOW - timedelta(days=3)).isoformat()


def test_the_ledger_carries_the_words_that_were_actually_said(tmp_path, monkeypatch):
    """Never invent: an entry's text has to be traceable to the stored analysis."""
    cfg, store = _store(tmp_path, monkeypatch)
    store.update_from_record(_record("rec_one", "father", FATHER_FIELDS))

    stored = json.dumps(FATHER_FIELDS)
    for entry in store.ledger("father").entries:
        head = entry.text.split(" (")[0]
        assert head in stored, f"the ledger holds text nothing in the analysis said: {entry.text}"


# =========================================================================
# Decay
# =========================================================================
def test_an_entry_older_than_the_decay_window_stops_being_surfaced(tmp_path, monkeypatch):
    cfg, store = _store(
        tmp_path, monkeypatch,
        overrides={"memory": {"decay_days": {"commitment": 30, "fact": 30, "topic": 30,
                                             "person": 30}}},
    )
    store.update_from_record(_record("rec_old", "father", FATHER_FIELDS,
                                     when=NOW - timedelta(days=200)))

    fresh = carry_forward_brief(cfg, "father", store, now=NOW - timedelta(days=199))
    assert "permission slip" in fresh, "the entry was never surfaced, so decay proves nothing"

    aged = carry_forward_brief(cfg, "father", store, now=NOW)
    assert aged == "", aged
    assert store.ledger("father").entries, "decay deleted the entry instead of hiding it"


def test_decay_is_per_kind(tmp_path, monkeypatch):
    cfg, store = _store(
        tmp_path, monkeypatch,
        overrides={"memory": {"decay_days": {"commitment": 400, "topic": 5, "fact": 5,
                                             "person": 5}}},
    )
    store.update_from_record(_record("rec_old", "father", FATHER_FIELDS,
                                     when=NOW - timedelta(days=60)))

    brief = carry_forward_brief(cfg, "father", store, now=NOW)
    assert "permission slip" in brief
    assert "Saturday 10:00 AM" not in brief, "a topic outlived its own decay window"


# =========================================================================
# Profile isolation
# =========================================================================
def test_one_profiles_memory_never_reaches_another_profiles_brief(tmp_path, monkeypatch):
    cfg, store = _store(tmp_path, monkeypatch)
    store.update_from_record(_record("rec_family", "father", FATHER_FIELDS))
    store.update_from_record(_record("rec_client", "insurance_agent", AGENT_FIELDS))

    father = carry_forward_brief(cfg, "father", store, budget=4000, now=NOW)
    agent = carry_forward_brief(cfg, "insurance_agent", store, budget=4000, now=NOW)
    assert father and agent, "both briefs have to be non-empty for this to mean anything"

    for private in ("permission slip", "Maya", "pizza", "First time starting"):
        assert private not in agent, (
            f"'{private}' came from the Father ledger and appeared in an Insurance "
            "Agent prompt"
        )
    for work in ("Marcus", "quote options", "fact_find", "second kid"):
        assert work not in father, (
            f"'{work}' came from the Insurance Agent ledger and appeared in a Father prompt"
        )


def test_a_ledger_file_does_not_open_under_another_profiles_name(tmp_path, monkeypatch):
    """Isolation at rest, not just in the code path that reads it."""
    cfg, store = _store(tmp_path, monkeypatch)
    store.update_from_record(_record("rec_family", "father", FATHER_FIELDS))

    father_bytes = store.path_for("father").read_bytes()
    vault = Vault(cfg.path("vault"))
    assert json.loads(vault.decrypt_bytes(father_bytes, b"memory:father"))["profile_id"] == "father"
    with pytest.raises(VaultError):
        vault.decrypt_bytes(father_bytes, b"memory:insurance_agent")

    store.path_for("insurance_agent").write_bytes(father_bytes)
    reopened = MemoryStore(cfg)
    assert reopened.ledger("insurance_agent").entries == []
    assert reopened.problems, "a ledger swapped between profiles was accepted silently"
    assert carry_forward_brief(cfg, "insurance_agent", reopened, now=NOW) == ""


# =========================================================================
# Sensitive fields
# =========================================================================
def test_a_sensitive_or_suppressed_field_never_enters_the_ledger(tmp_path, monkeypatch):
    """
    Config cannot talk memory into filing a sensitive field.

    The mapping below asks for exactly that, the way a careless edit would, and
    the profile's own `sensitive` and `suppress_fields` have to win anyway.
    """
    cfg, store = _store(tmp_path, monkeypatch, overrides={"memory": {"field_kinds": {
        "health_disclosures": FACT, "financial_disclosures": FACT,
    }}})
    store.update_from_record(_record("rec_client", "insurance_agent", AGENT_FIELDS))

    ledger = store.ledger("insurance_agent")
    assert ledger.entries, "nothing was filed at all, so nothing was excluded either"
    body = json.dumps(ledger.to_dict())
    assert "biopsy" not in body, "a field marked sensitive was written into the ledger"
    assert "four hundred thousand" not in body, "a suppressed field was written into the ledger"
    assert "Marcus" in body, "the non-sensitive fields were dropped too"

    brief = carry_forward_brief(cfg, "insurance_agent", store, budget=4000, now=NOW)
    assert "biopsy" not in brief and "four hundred thousand" not in brief


def test_the_brief_is_redacted_before_it_can_reach_a_model(tmp_path, monkeypatch):
    cfg, store = _store(tmp_path, monkeypatch)
    fields = dict(AGENT_FIELDS)
    fields["open_questions"] = ["Chase the paramed at marcus.reilly@example.com"]
    store.update_from_record(_record("rec_client", "insurance_agent", fields))

    brief = carry_forward_brief(cfg, "insurance_agent", store, budget=4000, now=NOW)
    assert "marcus.reilly@example.com" not in brief
    assert "[EMAIL_REDACTED]" in brief


def test_a_flagged_recording_is_not_filed(tmp_path, monkeypatch):
    """The family profiles stop and flag rather than summarise. Honour that."""
    cfg, store = _store(tmp_path, monkeypatch)
    flagged = dict(FATHER_FIELDS)
    flagged["requires_human_attention"] = True
    assert store.update_from_record(_record("rec_flag", "father", flagged, attention=True)) == []
    assert store.ledger("father").entries == []


def test_a_withheld_analysis_is_refused_rather_than_filed_as_empty(tmp_path, monkeypatch):
    """
    The index holds `fields: {}` for an encrypted recording. Filing that would
    mark the recording seen and its real contents would never be picked up.
    """
    cfg, store = _store(tmp_path, monkeypatch)
    record = _record("rec_enc", "father", {})
    record["analyses"][0]["fields_withheld"] = True

    assert store.update_from_record(record) == []
    assert "rec_enc" not in store.ledger("father").seen
    assert store.problems


# =========================================================================
# Budget
# =========================================================================
def test_the_brief_is_bounded_by_its_budget(tmp_path, monkeypatch):
    cfg, store = _store(tmp_path, monkeypatch)
    for index in range(12):
        fields = dict(FATHER_FIELDS)
        fields["worth_remembering"] = [
            {"timestamp": "00:03", "speaker": "Maya", "text": f"a sentence worth keeping {index}"}
        ]
        store.update_from_record(_record(f"rec_{index}", "father", fields,
                                         when=NOW - timedelta(days=index)))

    unbounded = carry_forward_brief(cfg, "father", store, budget=10000, now=NOW)
    assert len(unbounded) > 400, "the ledger is too small for the budget to bite"

    for budget in (120, 300, 600):
        brief = carry_forward_brief(cfg, "father", store, budget=budget, now=NOW)
        assert len(brief) <= budget, f"budget {budget} overrun by {len(brief) - budget}"

    tight = carry_forward_brief(cfg, "father", store, budget=300, now=NOW)
    assert 0 < len(tight) <= 300, (
        "300 characters bought nothing at all, which means the fixed preamble is "
        "eating the whole budget"
    )
    assert "Open commitments" in tight, "the highest-ranked section was not what survived"


def _fact_record(rid, text, when):
    return _record(rid, "father", {"worth_remembering": [{"text": text}]}, when=when)


def test_at_equal_recency_the_more_mentioned_entry_comes_first(tmp_path, monkeypatch):
    cfg, store = _store(tmp_path, monkeypatch)
    store.update_from_record(_fact_record("rec_once", "mentioned exactly once", NOW))
    for index in range(3):
        store.update_from_record(_fact_record(f"rec_often_{index}", "mentioned again and again",
                                              NOW))

    brief = carry_forward_brief(cfg, "father", store, budget=4000, now=NOW)
    assert brief.index("mentioned again and again") < brief.index("mentioned exactly once")


def test_at_equal_weight_the_newer_entry_comes_first(tmp_path, monkeypatch):
    cfg, store = _store(tmp_path, monkeypatch)
    store.update_from_record(_fact_record("rec_old", "said months ago",
                                          NOW - timedelta(days=200)))
    store.update_from_record(_fact_record("rec_new", "said on Tuesday", NOW))

    brief = carry_forward_brief(cfg, "father", store, budget=4000, now=NOW)
    assert brief.index("said on Tuesday") < brief.index("said months ago")


# =========================================================================
# Encryption at rest, and what happens without a passphrase
# =========================================================================
def test_the_ledger_is_encrypted_at_rest(tmp_path, monkeypatch):
    cfg, store = _store(tmp_path, monkeypatch)
    store.update_from_record(_record("rec_family", "father", FATHER_FIELDS))

    path = store.path_for("father")
    assert path.exists()
    raw = path.read_bytes()
    assert raw.startswith(b"PBV1"), "the ledger was not written through the vault"
    for private in (b"permission slip", b"Maya", b"pizza"):
        assert private not in raw, f"{private!r} is sitting in the clear in {path}"


def test_without_a_passphrase_nothing_is_written_and_it_says_why(tmp_path, monkeypatch):
    cfg, store = _store(tmp_path, monkeypatch)
    monkeypatch.delenv("PLAUD_BRIDGE_PASSPHRASE", raising=False)

    assert store.update_from_record(_record("rec_family", "father", FATHER_FIELDS)) == ["father"]
    assert not store.path_for("father").exists(), (
        "the ledger was written without a passphrase, which means it was written in plaintext"
    )
    assert store.problems, "the refusal to write happened silently"
    assert any("passphrase" in p.lower() or "PLAUD_BRIDGE_PASSPHRASE" in p
               for p in store.problems)

    # Nothing on disk to open, so a later run knows nothing rather than guessing.
    assert carry_forward_brief(cfg, "father", MemoryStore(cfg), now=NOW) == ""


def test_a_locked_vault_reports_the_existing_ledger_rather_than_reading_it(tmp_path, monkeypatch):
    cfg, store = _store(tmp_path, monkeypatch)
    store.update_from_record(_record("rec_family", "father", FATHER_FIELDS))
    assert store.path_for("father").exists()

    monkeypatch.delenv("PLAUD_BRIDGE_PASSPHRASE", raising=False)
    locked = MemoryStore(cfg)
    assert locked.ledger("father").entries == []
    assert locked.problems, "a ledger that could not be opened was reported as empty"
    assert carry_forward_brief(cfg, "father", locked, now=NOW) == ""


# =========================================================================
# Forgetting
# =========================================================================
def test_forget_recording_leaves_no_trace(tmp_path, monkeypatch):
    cfg, store = _store(tmp_path, monkeypatch)
    store.update_from_record(_record("rec_keep", "father", FATHER_FIELDS,
                                     when=NOW - timedelta(days=1)))
    gone = dict(FATHER_FIELDS)
    gone["worth_remembering"] = [{"text": "the thing that has to disappear"}]
    store.update_from_record(_record("rec_gone", "father", gone, source_name="gone.txt"))
    assert "the thing that has to disappear" in json.dumps(store.ledger("father").to_dict())

    assert store.forget_recording("rec_gone") == ["father"]

    for holder in (store, MemoryStore(cfg)):
        body = json.dumps(holder.ledger("father").to_dict())
        assert "rec_gone" not in body
        assert "gone.txt" not in body
        assert "the thing that has to disappear" not in body
        assert "rec_keep" in body, "forget took the other recording with it"
    assert "the thing that has to disappear" not in carry_forward_brief(
        cfg, "father", MemoryStore(cfg), budget=4000, now=NOW
    )


def test_forgetting_the_recording_that_closed_a_commitment_reopens_it(tmp_path, monkeypatch):
    cfg, store = _store(tmp_path, monkeypatch)
    store.update_from_record(_record("rec_promise", "father", FATHER_FIELDS,
                                     when=NOW - timedelta(days=2)))
    closer = _record("rec_closer", "father", {"completed": ["sign the permission slip"]})
    store.update_from_record(closer)

    entry = store.ledger("father").entry(COMMITMENT, "sign the permission slip when tonight")
    assert entry is not None and entry.closed_by == "rec_closer", "closure never happened"

    store.forget_recording("rec_closer")
    entry = store.ledger("father").entry(COMMITMENT, "sign the permission slip when tonight")
    assert entry is not None and entry.open, (
        "a deleted recording is still deciding that a promise was kept"
    )


def test_forget_recording_with_a_locked_vault_says_so_instead_of_claiming_success(
    tmp_path, monkeypatch
):
    cfg, store = _store(tmp_path, monkeypatch)
    store.update_from_record(_record("rec_gone", "father", FATHER_FIELDS))
    assert store.path_for("father").exists()

    monkeypatch.delenv("PLAUD_BRIDGE_PASSPHRASE", raising=False)
    locked = MemoryStore(cfg)
    assert locked.forget_recording("rec_gone") == []
    assert any("rec_gone" in p for p in locked.problems), (
        "forget reported nothing to do over a ledger it could not open"
    )

    monkeypatch.setenv("PLAUD_BRIDGE_PASSPHRASE", "a-long-enough-test-passphrase")
    assert "rec_gone" in json.dumps(MemoryStore(cfg).ledger("father").to_dict()), (
        "this test only means something if the ledger really did survive"
    )


def test_forget_profile_removes_the_whole_ledger(tmp_path, monkeypatch):
    cfg, store = _store(tmp_path, monkeypatch)
    store.update_from_record(_record("rec_family", "father", FATHER_FIELDS))
    assert store.path_for("father").exists()

    assert store.forget_profile("father") is True
    assert not store.path_for("father").exists()
    assert MemoryStore(cfg).ledger("father").entries == []
    assert store.forget_profile("father") is False


def test_a_closed_commitment_stops_being_carried_forward(tmp_path, monkeypatch):
    cfg, store = _store(tmp_path, monkeypatch)
    store.update_from_record(_record("rec_promise", "father", FATHER_FIELDS,
                                     when=NOW - timedelta(days=2)))
    assert "permission slip" in carry_forward_brief(cfg, "father", store, budget=4000, now=NOW)

    store.update_from_record(_record("rec_closer", "father",
                                     {"completed": ["sign the permission slip"]}))
    brief = carry_forward_brief(cfg, "father", store, budget=4000, now=NOW)
    assert "sign the permission slip (open since" not in brief


def test_close_commitment_by_hand(tmp_path, monkeypatch):
    cfg, store = _store(tmp_path, monkeypatch)
    store.update_from_record(_record("rec_promise", "father", FATHER_FIELDS))
    key = "sign the permission slip when tonight"

    assert store.close_commitment("father", key, "rec_manual", "done at bedtime") is True
    assert store.close_commitment("father", key, "rec_manual") is False
    assert store.ledger("father").entry(COMMITMENT, key).closed_note == "done at bedtime"


# =========================================================================
# Rebuild
# =========================================================================
def test_rebuild_reproduces_the_same_ledger(tmp_path, monkeypatch):
    """
    The end-to-end version: a real pipeline run, encrypted artifacts, and a
    ledger built by replaying what the archive holds.
    """
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    drop(cfg, "client-marcus.txt", CLIENT_CALL)
    drop(cfg, "dinner-with-kid.txt", FAMILY_DINNER)

    pipe = Pipeline(cfg)
    try:
        stats = pipe.run()
        assert stats.processed == 2, stats.summary()
        incremental = MemoryStore(cfg)
        archive = Archive(cfg, pipe.db)
        for row in pipe.db.query(limit=50):
            incremental.update_from_record(archive.full_record(row))
        before = {pid: incremental.ledger(pid).fingerprint() for pid in cfg.profiles}
        assert incremental.ledger("insurance_agent").entries
        assert incremental.ledger("father").entries

        rebuilt = MemoryStore(cfg)
        report = rebuilt.rebuild(pipe.db, archive)
    finally:
        pipe.close()

    assert report.complete, report.render()
    assert report.replayed == 2
    assert report.saved and "insurance_agent" in report.written

    for pid in cfg.profiles:
        assert rebuilt.ledger(pid).fingerprint() == before[pid], f"{pid} rebuilt differently"
        assert rebuilt.ledger(pid).to_dict()["entries"] == \
            incremental.ledger(pid).to_dict()["entries"]

    # And from a cold store, reading only what was written to disk.
    reopened = MemoryStore(cfg)
    assert reopened.ledger("insurance_agent").fingerprint() == before["insurance_agent"]


def test_rebuild_that_could_not_open_everything_does_not_overwrite(tmp_path, monkeypatch):
    """A ledger rebuilt from half the archive looks exactly like a whole one."""
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    drop(cfg, "client-marcus.txt", CLIENT_CALL)

    pipe = Pipeline(cfg)
    try:
        pipe.run()
        good = MemoryStore(cfg)
        good.rebuild(pipe.db, Archive(cfg, pipe.db))
        kept = good.ledger("insurance_agent").fingerprint()
        assert good.ledger("insurance_agent").entries

        monkeypatch.setenv("PLAUD_BRIDGE_PASSPHRASE", "a-completely-different-passphrase")
        blind = MemoryStore(cfg)
        report = blind.rebuild(pipe.db, Archive(cfg, pipe.db))
    finally:
        pipe.close()

    assert not report.complete
    assert not report.saved
    assert report.unopened

    monkeypatch.setenv("PLAUD_BRIDGE_PASSPHRASE", "a-long-enough-test-passphrase")
    assert MemoryStore(cfg).ledger("insurance_agent").fingerprint() == kept, (
        "a partial rebuild replaced a complete ledger"
    )


def test_the_brief_after_a_real_run_says_something_useful(tmp_path, monkeypatch):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    drop(cfg, "client-marcus.txt", CLIENT_CALL)

    pipe = Pipeline(cfg)
    try:
        pipe.run()
        store = MemoryStore(cfg)
        store.rebuild(pipe.db, Archive(cfg, pipe.db))
    finally:
        pipe.close()

    brief = carry_forward_brief(cfg, "insurance_agent", store, budget=2000)
    assert "WHAT YOU ALREADY KNOW" in brief
    assert "Marcus" in brief
    assert "Thursday" in brief
    assert "not part of the transcript" in brief, (
        "the brief does not tell the model what it is, so the model may quote it back"
    )


# =========================================================================
# Rendering for a person
# =========================================================================
def test_render_ledger_shows_provenance_and_marks_state(tmp_path, monkeypatch):
    cfg, store = _store(
        tmp_path, monkeypatch, overrides={"memory": {"decay_days": {"topic": 5}}},
    )
    store.update_from_record(_record("rec_one", "father", FATHER_FIELDS,
                                     when=NOW - timedelta(days=60), source_name="dinner.txt"))

    out = render_ledger(store.ledger("father"), cfg=cfg, now=NOW)
    assert "rec_one" in out and "dinner.txt" in out
    assert "sign the permission slip" in out
    assert "[open]" in out
    assert "[stale]" in out, "a topic well past its decay window is not marked"
    assert store.ledger("father").fingerprint()[:8] in out


def test_render_ledger_says_so_when_there_is_nothing(tmp_path, monkeypatch):
    cfg, store = _store(tmp_path, monkeypatch)
    out = render_ledger(store.ledger("husband"))
    assert "nothing recorded yet" in out
    assert "rebuild" in out


def test_kinds_are_filed_where_they_belong(tmp_path, monkeypatch):
    cfg, store = _store(tmp_path, monkeypatch)
    store.update_from_record(_record("rec_client", "insurance_agent", AGENT_FIELDS))
    ledger = store.ledger("insurance_agent")

    kinds = {e.kind for e in ledger.entries}
    assert {PERSON, COMMITMENT, TOPIC, FACT} <= kinds, sorted(kinds)
    assert ledger.entry(PERSON, "marcus") is not None
    assert not any(e.kind == PERSON and e.key == "sasson" for e in ledger.entries), (
        "the owner of the recorder was filed as a person he keeps meeting"
    )
