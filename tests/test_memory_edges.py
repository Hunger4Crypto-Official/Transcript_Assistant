"""
The memory ledger's edges: the branches test_memory.py does not reach.

Config shapes a person might reasonably write and get wrong, the ways a write
can fail after the passphrase check passed, a rebuild that has no archive to
lean on, and every early return that turns a bad input into "nothing filed"
rather than a stack trace. Each test asserts what the ledger, the report, or
the brief actually says afterwards -- executing a line is not the point.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from _fixtures import build_sandbox
from plaud_bridge import memory
from plaud_bridge.memory import (
    COMMITMENT,
    DEFAULT_CLOSURE_FIELDS,
    DEFAULT_DECAY_DAYS,
    FACT,
    PERSON,
    TOPIC,
    Entry,
    Ledger,
    MemoryLedgerError,
    MemoryStore,
    RebuildReport,
    Sighting,
    _closure_fields,
    _decay_days,
    _kind_for,
    _overlap,
    _parse_when,
    _phrase,
    _phrases,
    _score,
    _stale,
    carry_forward_brief,
    render_ledger,
)
from plaud_bridge.storage import Vault, VaultError

NOW = datetime(2026, 7, 27, 9, 0, tzinfo=timezone.utc)


def _record(recording_id, profile_id, fields, *, when=NOW, source_name="rec.txt",
            attention=False):
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
    "milestones": ["First time starting"],
    "next_action": "Sign the permission slip tonight",
}

PROMISE_KEY = "sign the permission slip when tonight"


def _store(tmp_path, monkeypatch, overrides=None):
    cfg, _ = build_sandbox(tmp_path, monkeypatch, overrides=overrides)
    return cfg, MemoryStore(cfg)


def _fact(rid, text, when=NOW):
    return _record(rid, "father", {"worth_remembering": [{"text": text}]}, when=when)


# =========================================================================
# Small helpers
# =========================================================================
def test_parse_when_accepts_datetimes_and_assumes_utc_for_naive_ones():
    aware = datetime(2026, 1, 2, 3, 4, tzinfo=timezone(timedelta(hours=-5)))
    assert _parse_when(aware) is aware
    naive = _parse_when(datetime(2026, 1, 2, 3, 4))
    assert naive == datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc)


def test_parse_when_returns_none_for_blank_or_unparsable_text():
    assert _parse_when("") is None
    assert _parse_when(None) is None
    assert _parse_when("   ") is None
    assert _parse_when("last Tuesday") is None
    assert _parse_when("2026-07-27T09:00:00Z") == NOW


def test_overlap_is_zero_when_either_side_has_no_words():
    assert _overlap("", "sign the slip") == 0.0
    assert _overlap("!!! ...", "sign the slip") == 0.0
    assert _overlap("sign the slip", "sign the permission slip") == 1.0


def test_a_bare_boolean_is_never_a_phrase():
    assert _phrase(True) == ""
    assert _phrase(False) == ""
    assert _phrases(True, 5) == []
    assert _phrases(None, 5) == []


def test_a_dict_without_a_quote_key_is_rendered_as_its_pairs():
    assert _phrase({"item": "gloves", "due": "Friday", "nested": {"x": 1}, "flag": True}) == (
        "item: gloves; due: Friday"
    )
    assert _phrase({"nested": {"x": 1}}) == ""


def test_a_ledger_lists_its_recordings_sorted():
    ledger = Ledger(profile_id="father", seen={"rec_b": "1", "rec_a": "2"})
    assert ledger.recordings == ["rec_a", "rec_b"]


# =========================================================================
# Config shapes
# =========================================================================
def test_a_single_number_for_decay_days_applies_to_every_kind(tmp_path, monkeypatch):
    cfg, store = _store(tmp_path, monkeypatch, overrides={"memory": {"decay_days": 10}})
    assert [_decay_days(cfg, k) for k in (COMMITMENT, PERSON, TOPIC, FACT)] == [10, 10, 10, 10]

    store.update_from_record(_fact("rec_old", "said eleven days ago", when=NOW - timedelta(days=11)))
    assert carry_forward_brief(cfg, "father", store, now=NOW) == ""
    assert "said eleven days ago" in carry_forward_brief(
        cfg, "father", store, now=NOW - timedelta(days=2)
    )


def test_a_non_numeric_decay_falls_back_to_the_built_in_default(tmp_path, monkeypatch):
    cfg, _ = _store(tmp_path, monkeypatch,
                    overrides={"memory": {"decay_days": {"topic": "soon", "default": 7}}})
    assert _decay_days(cfg, TOPIC) == DEFAULT_DECAY_DAYS[TOPIC]
    # A kind with no entry of its own takes the "default" key.
    assert _decay_days(cfg, FACT) == 7


def test_a_decay_of_zero_means_an_entry_is_never_stale(tmp_path, monkeypatch):
    cfg, store = _store(tmp_path, monkeypatch, overrides={"memory": {"decay_days": 0}})
    store.update_from_record(_fact("rec_ancient", "said years ago", when=NOW - timedelta(days=3000)))

    entry = store.ledger("father").entries[0]
    assert not _stale(cfg, entry, NOW)
    assert _score(cfg, entry, NOW) == pytest.approx(1.0), "freshness should not decay at all"
    assert "said years ago" in carry_forward_brief(cfg, "father", store, now=NOW)


def test_an_entry_whose_last_sighting_has_no_usable_date_is_stale_and_unranked(
    tmp_path, monkeypatch
):
    cfg, _ = _store(tmp_path, monkeypatch)
    entry = Entry(kind=FACT, key="x", sightings=[Sighting("rec", "not a date", "f", "x")])
    assert _stale(cfg, entry, NOW)
    assert _score(cfg, entry, NOW) == 0.0


def test_ignore_fields_keeps_a_field_out_of_the_ledger(tmp_path, monkeypatch):
    cfg, store = _store(tmp_path, monkeypatch,
                        overrides={"memory": {"ignore_fields": ["promises_i_made"]}})
    assert _kind_for(cfg, "promises_i_made") == ""

    store.update_from_record(_record("rec_one", "father", FATHER_FIELDS))
    ledger = store.ledger("father")
    assert ledger.entry(COMMITMENT, PROMISE_KEY) is None
    # The field was ignored, not the recording: next_action is still filed.
    assert ledger.entry(COMMITMENT, "sign the permission slip tonight") is not None


def test_a_field_kind_that_is_not_a_kind_drops_the_field_with_a_warning(tmp_path, monkeypatch, caplog):
    cfg, store = _store(tmp_path, monkeypatch,
                        overrides={"memory": {"field_kinds": {"worth_remembering": "banana"}}})
    with caplog.at_level("WARNING", logger="plaud_bridge.memory"):
        assert _kind_for(cfg, "worth_remembering") == ""
    assert any("banana" in r.getMessage() for r in caplog.records)

    store.update_from_record(_record("rec_one", "father", FATHER_FIELDS))
    body = json.dumps(store.ledger("father").to_dict())
    assert "Coach said" not in body
    assert "permission slip" in body


def test_closure_fields_default_when_the_config_leaves_them_out(tmp_path, monkeypatch):
    cfg, _ = _store(tmp_path, monkeypatch, overrides={"memory": {"closure_fields": None}})
    assert cfg.get("memory.closure_fields") is None
    assert _closure_fields(cfg) == set(DEFAULT_CLOSURE_FIELDS)


# =========================================================================
# Filing: what is left out, and what is refused
# =========================================================================
def test_memory_disabled_files_nothing_and_writes_nothing(tmp_path, monkeypatch):
    cfg, store = _store(tmp_path, monkeypatch, overrides={"memory": {"enabled": False}})
    assert store.update_from_record(_record("rec_one", "father", FATHER_FIELDS)) == []
    assert store.ledger("father").entries == []

    store._cache["father"] = Ledger(profile_id="father", seen={"rec_one": "sig"})
    assert store.save() == []
    assert not store.path_for("father").exists()
    assert carry_forward_brief(cfg, "father", store, now=NOW) == ""


def test_a_record_without_an_id_is_not_filed_and_is_reported(tmp_path, monkeypatch):
    cfg, store = _store(tmp_path, monkeypatch)
    record = _record("   ", "father", FATHER_FIELDS)
    assert store.update_from_record(record) == []
    assert store.ledger("father").entries == []
    assert any("no id" in p for p in store.problems)


def test_a_record_with_no_analyses_changes_nothing(tmp_path, monkeypatch):
    cfg, store = _store(tmp_path, monkeypatch)
    assert store.update_from_record({"id": "rec_bare", "analyses": []}) == []
    assert store.update_from_record({"id": "rec_bare", "analyses": ["not a dict"]}) == []
    assert store.problems == []
    assert store._cache == {}, "an empty record should not even have opened a ledger"


def test_an_analysis_for_an_unknown_profile_is_skipped(tmp_path, monkeypatch):
    cfg, store = _store(tmp_path, monkeypatch)
    assert store.update_from_record(_record("rec_x", "landlord", FATHER_FIELDS)) == []
    assert "landlord" not in store._cache
    assert not store.path_for("landlord").exists()


def test_a_field_the_profile_no_longer_declares_is_skipped(tmp_path, monkeypatch):
    cfg, store = _store(tmp_path, monkeypatch)
    fields = dict(FATHER_FIELDS)
    fields["promises_from_an_older_schema"] = ["repaint the fence"]
    store.update_from_record(_record("rec_one", "father", fields))

    body = json.dumps(store.ledger("father").to_dict())
    assert "repaint the fence" not in body
    assert "permission slip" in body


def test_a_phrase_with_no_words_in_it_makes_no_entry(tmp_path, monkeypatch):
    cfg, store = _store(tmp_path, monkeypatch)
    store.update_from_record(_record("rec_one", "father", {"worth_remembering": ["!!! ???"]}))
    assert store.ledger("father").entries == []
    assert "rec_one" in store.ledger("father").seen


def test_a_closure_note_that_matches_nothing_closes_nothing(tmp_path, monkeypatch):
    cfg, store = _store(tmp_path, monkeypatch)
    store.update_from_record(_record("rec_promise", "father", FATHER_FIELDS,
                                     when=NOW - timedelta(days=1)))
    store.update_from_record(_record("rec_closer", "father", {"completed": ["watered the plants"]}))

    entry = store.ledger("father").entry(COMMITMENT, PROMISE_KEY)
    assert entry is not None and entry.open
    assert entry.closed_by == ""


def test_filing_into_the_wrong_ledger_is_refused_outright(tmp_path, monkeypatch):
    cfg, store = _store(tmp_path, monkeypatch)
    analysis = {"profile_id": "father", "fields": FATHER_FIELDS}
    with pytest.raises(MemoryLedgerError, match="do not share memory"):
        store._apply(Ledger(profile_id="husband"), cfg.profiles["father"], analysis,
                     "rec_one", NOW.isoformat(), "rec.txt")


# =========================================================================
# Loading and saving: what happens after the passphrase check passes
# =========================================================================
def test_a_ledger_file_claiming_another_profile_inside_is_refused(tmp_path, monkeypatch):
    """The AAD is the first line of defence; the stored profile_id is the second."""
    cfg, store = _store(tmp_path, monkeypatch)
    vault = Vault(cfg.path("vault"))
    forged = {"version": 1, "profile_id": "husband", "seen": {"rec_h": "sig"},
              "entries": [Entry(kind=FACT, key="anniversary",
                                sightings=[Sighting("rec_h", NOW.isoformat(), "she_asked_for",
                                                    "anniversary dinner")]).to_dict()]}
    path = store.path_for("father")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(vault.encrypt_bytes(json.dumps(forged).encode(), b"memory:father"))

    reopened = MemoryStore(cfg)
    assert reopened.ledger("father").entries == []
    assert reopened.ledger("father").seen == {}
    assert any("belongs to 'husband'" in p for p in reopened.problems)
    assert "anniversary" not in carry_forward_brief(cfg, "father", reopened, now=NOW)


def test_saving_a_profile_that_was_never_loaded_writes_nothing(tmp_path, monkeypatch):
    cfg, store = _store(tmp_path, monkeypatch)
    assert store.save("father") == []
    assert not store.path_for("father").exists()
    assert store.problems == []


def test_a_ledger_cached_under_the_wrong_profile_is_never_written(tmp_path, monkeypatch):
    cfg, store = _store(tmp_path, monkeypatch)
    store._cache["father"] = Ledger(profile_id="husband")
    with pytest.raises(MemoryLedgerError, match="refusing to write the 'husband' ledger"):
        store.save("father")
    assert not store.path_for("father").exists()


def test_an_encryption_failure_leaves_no_file_and_says_so(tmp_path, monkeypatch):
    cfg, store = _store(tmp_path, monkeypatch)
    store.update_from_record(_record("rec_one", "father", FATHER_FIELDS), save=False)

    def refuse(*_a, **_k):
        raise VaultError("cipher unavailable")

    monkeypatch.setattr(store.vault, "encrypt_bytes", refuse)
    assert store.save() == []
    assert not store.path_for("father").exists()
    assert any("could not encrypt the father ledger" in p for p in store.problems)


def test_a_disk_failure_mid_write_leaves_no_temp_file_behind(tmp_path, monkeypatch):
    cfg, store = _store(tmp_path, monkeypatch)
    store.update_from_record(_record("rec_one", "father", FATHER_FIELDS), save=False)

    def fail(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(memory.os, "replace", fail)
    assert store.save() == []

    dest = store.path_for("father")
    assert not dest.exists()
    assert not dest.with_name(dest.name + ".tmp").exists(), "a half-written ledger was left on disk"
    assert any("could not write the father ledger" in p and "disk full" in p
               for p in store.problems)


# =========================================================================
# Forgetting when the write cannot happen
# =========================================================================
def test_forget_that_cannot_persist_says_the_disk_still_remembers(tmp_path, monkeypatch):
    cfg, store = _store(tmp_path, monkeypatch)
    store.update_from_record(_record("rec_gone", "father", FATHER_FIELDS))
    assert store.path_for("father").exists()

    def refuse(*_a, **_k):
        raise VaultError("cipher unavailable")

    monkeypatch.setattr(store.vault, "encrypt_bytes", refuse)
    assert store.forget_recording("rec_gone") == ["father"]
    assert store.ledger("father").entries == [], "the in-memory copy should be cleared"
    assert any("still remembers rec_gone on disk" in p for p in store.problems), (
        "forget claimed a clean deletion over a file it could not rewrite"
    )
    # And the file really does still hold it, which is why the message matters.
    assert "rec_gone" in json.dumps(MemoryStore(cfg).ledger("father").to_dict())


def test_forget_profile_reports_a_file_it_cannot_delete(tmp_path, monkeypatch):
    cfg, store = _store(tmp_path, monkeypatch)
    path = store.path_for("father")
    path.mkdir(parents=True)   # unlink() on a directory raises OSError

    assert store.forget_profile("father") is False
    assert path.exists()
    assert any("could not delete" in p for p in store.problems)


# =========================================================================
# Rebuild without an archive, and the report it renders
# =========================================================================
class FakeDB:
    def __init__(self, rows):
        self.rows = rows

    def count_recordings(self):
        return len(self.rows)

    def query(self, limit):
        assert limit >= len(self.rows)
        return list(reversed(self.rows))   # the index hands back newest first


def test_rebuild_from_stored_payloads_replays_in_recording_order_and_reports_the_rest(
    tmp_path, monkeypatch
):
    cfg, store = _store(tmp_path, monkeypatch)
    earlier = _record("rec_promise", "father", FATHER_FIELDS, when=NOW - timedelta(days=2))
    later = _record("rec_closer", "father", {"completed": ["sign the permission slip"]})
    rows = [
        {"id": "rec_promise", "source_name": "dinner.txt",
         "recorded_at": earlier["recorded_at"], "payload_json": json.dumps(earlier)},
        {"id": "rec_closer", "source_name": "bedtime.txt",
         "recorded_at": later["recorded_at"], "payload_json": json.dumps(later)},
        {"id": "rec_garbled", "source_name": "garbled.txt",
         "recorded_at": NOW.isoformat(), "payload_json": "{not json"},
        {"id": "rec_missing", "source_name": "missing.txt", "recorded_at": NOW.isoformat()},
    ]

    report = store.rebuild(FakeDB(rows))

    assert report.replayed == 2
    assert report.unopened == ["rec_garbled  garbled.txt", "rec_missing  missing.txt"]
    assert not report.complete
    assert not report.saved and report.written == []
    assert not store.path_for("father").exists(), "a partial rebuild was written without force"

    # Replay order was by when the conversations happened, so the later
    # recording closed the promise the earlier one made.
    entry = report.ledgers["father"].entry(COMMITMENT, PROMISE_KEY)
    assert entry is not None and entry.closed_by == "rec_closer"

    text = report.render()
    assert "replayed 2 recording(s)" in text
    assert "2 recording(s) could not be opened:" in text
    assert "  rec_garbled  garbled.txt" in text
    assert "Nothing was written" in text and "force" in text


def test_a_forced_partial_rebuild_is_written_and_says_how_many(tmp_path, monkeypatch):
    cfg, store = _store(tmp_path, monkeypatch)
    good = _record("rec_promise", "father", FATHER_FIELDS)
    rows = [
        {"id": "rec_promise", "source_name": "dinner.txt",
         "recorded_at": good["recorded_at"], "payload_json": json.dumps(good)},
        {"id": "rec_missing", "source_name": "missing.txt", "recorded_at": NOW.isoformat()},
    ]

    report = store.rebuild(FakeDB(rows), force=True)

    assert report.saved
    assert report.written == sorted(store._cache), "every profile's ledger is written, empty or not"
    assert store.path_for("father").exists()
    assert MemoryStore(cfg).ledger("father").entry(COMMITMENT, PROMISE_KEY) is not None
    text = report.render()
    assert f"wrote {len(report.written)} ledger(s)" in text
    assert "Nothing was written" not in text


def test_the_rebuild_report_truncates_a_long_list_of_unopened_recordings():
    report = RebuildReport(replayed=3, unopened=[f"rec_{i:02d}  file{i}.txt" for i in range(25)],
                           saved=True, written=["father"])
    text = report.render()
    assert "25 recording(s) could not be opened:" in text
    assert "rec_19  file19.txt" in text
    assert "rec_20" not in text
    assert "... and 5 more" in text
    assert "wrote 1 ledger(s)" in text


# =========================================================================
# The brief's early exits and its hard budget
# =========================================================================
def test_the_brief_is_empty_for_an_unknown_profile_or_a_zero_budget(tmp_path, monkeypatch):
    cfg, store = _store(tmp_path, monkeypatch)
    store.update_from_record(_record("rec_one", "father", FATHER_FIELDS))
    assert carry_forward_brief(cfg, "father", store, budget=4000, now=NOW), "the ledger is empty"

    assert carry_forward_brief(cfg, "landlord", store, now=NOW) == ""
    assert carry_forward_brief(cfg, "father", store, budget=0, now=NOW) == ""
    assert carry_forward_brief(cfg, "father", store, budget=-5, now=NOW) == ""


def test_the_brief_never_overruns_any_budget_by_a_single_character(tmp_path, monkeypatch):
    """
    The line-by-line accounting is one character short per section, so some
    budgets land where the assembled text is longer than what was counted. The
    final trim has to catch every one of those, because an overrun brief is a
    prompt that is longer than the caller was promised.
    """
    cfg, store = _store(tmp_path, monkeypatch)
    for index in range(6):
        fields = dict(FATHER_FIELDS)
        fields["worth_remembering"] = [{"text": f"a fact worth carrying number {index}"}]
        fields["logistics"] = [{"what": f"errand {index}", "when": "Saturday"}]
        store.update_from_record(_record(f"rec_{index}", "father", fields,
                                         when=NOW - timedelta(days=index)))

    overruns = []
    non_empty = 0
    for budget in range(150, 1400):
        brief = carry_forward_brief(cfg, "father", store, budget=budget, now=NOW)
        if len(brief) > budget:
            overruns.append((budget, len(brief)))
        if brief:
            non_empty += 1
    assert overruns == []
    assert non_empty > 1000, "almost every budget bought nothing, so nothing was tested"


# =========================================================================
# Rendering for a person
# =========================================================================
def test_render_ledger_folds_older_sightings_and_shows_who_closed_a_promise(tmp_path, monkeypatch):
    cfg, store = _store(tmp_path, monkeypatch)
    for index in range(5):
        store.update_from_record(_fact(f"rec_{index}", "the same fact every week",
                                       when=NOW - timedelta(days=7 * (4 - index))))
    store.update_from_record(_record("rec_promise", "father", FATHER_FIELDS,
                                     when=NOW - timedelta(days=1)))
    store.update_from_record(_record("rec_closer", "father",
                                     {"completed": ["sign the permission slip"]}))

    out = render_ledger(store.ledger("father"), cfg=cfg, now=NOW, sightings=3)

    assert "... and 2 earlier" in out
    assert out.count("rec_4") == 1 and "rec_0" not in out, "the fold hid the wrong sightings"
    assert "[closed] sign the permission slip" in out
    assert f"closed by rec_closer on {NOW.date().isoformat()}: sign the permission slip" in out
    # The note overlaps next_action's wording completely too, so that closed as well.
    assert "[closed] Sign the permission slip tonight" in out
    assert "[open]" not in out


def test_a_ledger_of_only_closed_promises_renders_them_as_history(tmp_path, monkeypatch):
    cfg, store = _store(tmp_path, monkeypatch)
    store.update_from_record(_record("rec_promise", "father",
                                     {"promises_i_made": [{"what": "book the dentist"}]},
                                     when=NOW - timedelta(days=1)))
    store.close_commitment("father", "book the dentist", "rec_manual")

    out = render_ledger(store.ledger("father"), cfg=cfg, now=NOW)
    assert "[closed] book the dentist" in out
    assert "closed by rec_manual on" in out
    assert carry_forward_brief(cfg, "father", store, now=NOW) == "", (
        "a closed promise is history, not context"
    )


def test_os_replace_is_restored_after_the_disk_failure_test():
    """Guard against the monkeypatch above leaking into the rest of the suite."""
    assert memory.os.replace is os.replace
