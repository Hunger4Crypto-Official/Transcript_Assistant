"""
People edges: what the roster refuses to claim when it cannot check.

A voiceprint store that will not open verifies nobody; quotes are read only
from analyses that succeeded, from profiles that exist, and never from a
personal analysis riding on a work recording; a follow-up worklist that will
not open fails the roster by name; a lookup that could mean two people
refuses to guess. The dossier renderings for the unidentified bucket and for
aged or dated commitments are pinned on their exact wording.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pytest

from _fixtures import CLIENT_CALL, build_sandbox, drop
from plaud_bridge.archive import Archive
from plaud_bridge.cli import main
from plaud_bridge.db import Database
from plaud_bridge.followups import FollowUp, FollowUpError
from plaud_bridge.people import (
    UNIDENTIFIED,
    Appearance,
    PeopleError,
    Person,
    _commitment_line,
    _enrolled_names,
    _kept_quotes,
    collect_people,
    person_detail,
    render_person,
    render_roster,
)
from plaud_bridge.storage import Vault


@pytest.fixture
def processed(tmp_path, monkeypatch):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    drop(cfg, "client.txt", CLIENT_CALL)
    assert main(["--config", str(tmp_path / "config"), "run"]) == 0
    db = Database(cfg.path("database"))
    try:
        yield cfg, db
    finally:
        db.close()


# =========================================================================
# The voiceprint store
# =========================================================================
def test_a_voiceprint_store_that_will_not_open_verifies_nobody(processed, caplog):
    cfg, db = processed
    vault = Vault(cfg.path("vault"))
    (vault.root / "voiceprints.enc").write_bytes(b"this is not a vault file")

    with caplog.at_level(logging.WARNING, logger="plaud_bridge.people"):
        assert _enrolled_names(vault) == set()
        people = collect_people(cfg, db, Archive(cfg, db))

    assert any("voiceprint store could not be opened" in r.getMessage() for r in caplog.records)
    assert people, "the roster is still built; only verification is withheld"
    assert not any(p.voice_verified for p in people)
    assert "voice-verified" not in render_roster(people).split("<sub>")[0], (
        "a row claimed verification the store could not confirm"
    )


# =========================================================================
# Which quotes are read
# =========================================================================
def test_quotes_are_taken_only_from_successful_analyses_of_known_profiles(processed):
    cfg, db = processed
    quote = {"speaker": "Marcus", "text": "kept"}
    record = {"analyses": [
        {"profile_id": "ghost", "fields": {"stated_needs": [dict(quote, text="unknown profile")]}},
        {"profile_id": "insurance_agent", "error": "boom",
         "fields": {"stated_needs": [dict(quote, text="errored analysis")]}},
        {"profile_id": "insurance_agent", "fields": {
            "stated_needs": [quote, "a bare string", 42,
                             {"speaker": "Speaker 2", "text": "placeholder speaker"},
                             {"speaker": "", "text": "no speaker"},
                             {"speaker": "Marcus", "text": "   "}],
            # Typed as a list of quotes but not a list: nothing to read.
            "commitments_by_client": {"speaker": "Marcus", "text": "not a list"},
            # Sensitive and suppressed fields stay in the vault.
            "health_disclosures": [dict(quote, text="a diagnosis")],
            "financial_disclosures": [dict(quote, text="the mortgage")],
            # Not a quote field, whatever shape it has.
            "participants": [dict(quote, text="participants list")],
        }},
    ]}
    assert _kept_quotes(cfg, record, personal=set(), include_personal=False) == [("marcus", "kept")]


def test_a_personal_analysis_on_a_work_recording_contributes_no_quotes_by_default(processed):
    """The recording-level rule alone would let a co-routed father analysis through."""
    cfg, db = processed
    record = {"analyses": [
        {"profile_id": "father", "fields": {
            "worth_remembering": [{"speaker": "Kid", "text": "starting Saturday"}]}},
        {"profile_id": "insurance_agent", "fields": {
            "stated_needs": [{"speaker": "Marcus", "text": "second kid"}]}},
    ]}
    personal = {"father"}
    assert _kept_quotes(cfg, record, personal, include_personal=False) == [("marcus", "second kid")]
    assert _kept_quotes(cfg, record, personal, include_personal=True) == [
        ("kid", "starting Saturday"), ("marcus", "second kid")]


# =========================================================================
# The follow-up worklist
# =========================================================================
def test_an_unreadable_follow_up_worklist_fails_the_roster_by_name(processed, monkeypatch):
    cfg, db = processed

    def locked(*_a, **_k):
        raise FollowUpError("state file is locked (test)")

    monkeypatch.setattr("plaud_bridge.people.collect_followups", locked)
    with pytest.raises(PeopleError) as excinfo:
        collect_people(cfg, db, Archive(cfg, db))
    assert str(excinfo.value) == "could not read the follow-up worklist: state file is locked (test)"


# =========================================================================
# Lookup
# =========================================================================
def test_an_empty_name_is_refused():
    with pytest.raises(PeopleError, match="no name given"):
        person_detail([Person(label="Marcus", display_name="Marcus")], "   ")


def test_an_ambiguous_prefix_is_refused_with_the_candidates():
    people = [Person(label="Marcus", display_name="Marcus"),
              Person(label="Marcia", display_name="Marcia"),
              Person(label="Dana", display_name="Dana")]
    with pytest.raises(PeopleError) as excinfo:
        person_detail(people, "mar")
    assert str(excinfo.value) == "'mar' matches 2 people: Marcia, Marcus. Use more of the name."
    # An exact match wins over a prefix that would otherwise be ambiguous.
    people.append(Person(label="Marc", display_name="Marc"))
    assert person_detail(people, "marc").display_name == "Marc"


# =========================================================================
# Rendering
# =========================================================================
def test_an_unknown_format_is_refused_for_both_pages():
    person = Person(label="Dana", display_name="Dana")
    with pytest.raises(PeopleError, match="unknown format 'pdf'. Use markdown or html."):
        render_roster([person], fmt="pdf")
    with pytest.raises(PeopleError, match="unknown format 'pdf'. Use markdown or html."):
        render_person(person, fmt="pdf")
    assert "<h1>" in render_person(person, fmt="html")


def test_a_dossier_shows_last_heard_only_when_it_differs_from_first_heard():
    dana = Person(label="Dana", display_name="Dana", appearances=[
        Appearance("rec_1", "2026-03-01", "one.txt", 2.0, "sales_trainer"),
        Appearance("rec_2", "2026-04-09", "two.txt", 3.5, "sales_trainer"),
    ])
    out = render_person(dana)
    assert "first heard 2026-03-01" in out and "last heard 2026-04-09" in out
    assert "profiles: sales_trainer" in out
    assert "- **2026-04-09** — two.txt (`rec_2`), 3.5 min, sales_trainer" in out

    once = Person(label="Dana", display_name="Dana", appearances=[dana.appearances[0]])
    assert "last heard" not in render_person(once)


def test_the_bucket_dossier_explains_itself_rather_than_posing_as_a_person():
    bucket = Person(label=UNIDENTIFIED, display_name=UNIDENTIFIED, is_bucket=True)
    out = render_person(bucket)
    assert "placeholder labels, grouped here rather than presented as a person" in out
    assert "run.py speakers enroll" in out
    assert "This name is a speaker label" not in out
    assert "Never, in this window." in out


def test_a_commitment_line_carries_its_age_and_due_date():
    old = (datetime.now(timezone.utc) - timedelta(days=12)).date().isoformat()
    item = FollowUp(id="fu_1", text="send | the quote", profile_id="insurance_agent",
                    recording_id="rec_1", first_seen=old, due="Friday")
    assert _commitment_line(item) == "- **open/12d/due Friday** — send / the quote  (`rec_1`)"

    fresh = FollowUp(id="fu_2", text="call", profile_id="insurance_agent", recording_id="rec_2")
    assert _commitment_line(fresh) == "- **open** — call  (`rec_2`)"

    done = FollowUp(id="fu_3", text="call", profile_id="insurance_agent",
                    recording_id="rec_3", first_seen=old, status="done")
    assert _commitment_line(done) == "- **done** — call  (`rec_3`)", "a closed item has no age"
