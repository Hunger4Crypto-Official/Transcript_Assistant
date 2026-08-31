"""
People: the roster, and what it refuses to claim.

Half of these tests check the aggregation -- that a speaker heard in two
recordings is one person with two appearances, that commitments land on the
right counterparty, that the window is respected. The other half pin the
honesty rules the module docstring makes in plain language: a placeholder
label is never presented as a person, a name is only marked voice-verified
when a voiceprint was actually enrolled, personal recordings stay out unless
asked for, suppressed fields never surface, and a recording that will not
open is reported rather than guessed at.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pytest

from _fixtures import CLIENT_CALL, FAMILY_DINNER, build_sandbox, drop
from plaud_bridge.archive import Archive
from plaud_bridge.cli import main
from plaud_bridge.db import Database
from plaud_bridge.models import (
    ProfileAnalysis,
    Recording,
    RouteMatch,
    Segment,
    Transcript,
)
from plaud_bridge.people import (
    UNIDENTIFIED,
    PeopleError,
    Person,
    collect_people,
    person_detail,
    render_person,
    render_roster,
)
from plaud_bridge.storage import Vault

# CLIENT_CALL with a line appended, so it hashes as a second recording while
# every verified quote the stub returns is still verbatim inside it.
SECOND_CLIENT_CALL = CLIENT_CALL + "Sasson: Thanks again Marcus, talk soon.\n"


@pytest.fixture
def processed(tmp_path, monkeypatch):
    """A sandbox with the two fixture conversations run through the real CLI."""
    cfg, _stub = build_sandbox(tmp_path, monkeypatch)
    drop(cfg, "client.txt", CLIENT_CALL)
    drop(cfg, "dinner.txt", FAMILY_DINNER)
    assert main(["--config", str(tmp_path / "config"), "run"]) == 0
    db = Database(cfg.path("database"))
    yield cfg, db
    db.close()


def people_for(cfg, db, **kw):
    return collect_people(cfg, db, Archive(cfg, db), **kw)


def by_name(people, name):
    return {p.display_name: p for p in people}[name]


def cli(cfg, *argv) -> int:
    return main(["--config", str(cfg.root / "config"), *argv])


def add_row(cfg, db, vault, recording_id, *, speakers, profile_id="insurance_agent",
            days_ago=0, fields=None, encrypt=False) -> Recording:
    """
    Write a recording straight into the index, the way test_followups does.

    Direct rows are the only way to control the dates the window test needs
    and to manufacture a recording whose vault artifact has gone missing.
    """
    rec = Recording(
        id=recording_id,
        source_name=f"{recording_id}.txt",
        source_path=f"/inbox/{recording_id}.txt",
        content_hash=f"hash-{recording_id}",
        kind="text",
        recorded_at=datetime.now(timezone.utc) - timedelta(days=days_ago),
    )
    rec.transcript = Transcript(segments=[
        Segment(i * 10.0, i * 10.0 + 6.0, f"talking about premiums and quotes {i}", spk)
        for i, spk in enumerate(speakers)
    ])
    rec.routes = [RouteMatch(profile_id=profile_id, confidence=0.9)]
    rec.compliance.governing_profile = profile_id
    rec.compliance.encrypt_at_rest = encrypt
    rec.analyses = [ProfileAnalysis(profile_id=profile_id, fields=fields or {})]
    if encrypt:
        rec.artifact_paths["analysis"] = str(
            vault.write(f"{recording_id}.analysis.json", rec.to_json(), recording_id)
        )
    db.upsert(rec)
    return rec


# =========================================================================
# Aggregation
# =========================================================================
def test_a_speaker_heard_in_two_recordings_is_one_person_with_two_appearances(
        tmp_path, monkeypatch):
    cfg, _stub = build_sandbox(tmp_path, monkeypatch)
    drop(cfg, "client.txt", CLIENT_CALL)
    drop(cfg, "client2.txt", SECOND_CLIENT_CALL)
    assert main(["--config", str(tmp_path / "config"), "run"]) == 0

    db = Database(cfg.path("database"))
    try:
        people = people_for(cfg, db)
        marcus = by_name(people, "Marcus")
        assert marcus.conversations == 2
        assert len({a.recording_id for a in marcus.appearances}) == 2
        assert marcus.minutes_heard > 0
        assert marcus.first_heard and marcus.last_heard
        assert marcus.profiles == ["insurance_agent"]
        # Appearances are the chronological dossier spine: sorted, dated, and
        # each carrying the minutes this person actually spoke.
        assert [a.when for a in marcus.appearances] == sorted(
            a.when for a in marcus.appearances
        )
        # The same sentence kept from both recordings is one thing they said.
        assert len(marcus.things_they_said) == len(set(marcus.things_they_said))
        # Topics come from their own words, not the stoplist's.
        assert marcus.topics
        assert "the" not in marcus.topics
    finally:
        db.close()


def test_personal_recordings_stay_out_of_the_roster_unless_asked_for(processed):
    cfg, db = processed
    names = {p.display_name for p in people_for(cfg, db)}
    assert "Marcus" in names
    assert "Kid" not in names, "a father-governed recording leaked into the default roster"

    with_flag = {p.display_name for p in people_for(cfg, db, include_personal=True)}
    assert "Kid" in with_flag


def test_the_owner_is_listed_but_marked_as_the_owner(processed):
    cfg, db = processed
    people = people_for(cfg, db)
    sasson = by_name(people, "Sasson")
    assert sasson.is_owner
    assert not sasson.voice_verified
    assert all(not p.is_owner for p in people if p.display_name != "Sasson")
    assert "you (the owner)" in render_roster(people)


# =========================================================================
# Honesty about identity
# =========================================================================
def test_placeholder_speaker_labels_are_bucketed_not_personified(processed):
    cfg, db = processed
    vault = Vault(cfg.path("vault"))
    add_row(cfg, db, vault, "rec_unlabelled",
            speakers=["Speaker 1", "SPEAKER", "speaker_2", "Marcus"])

    people = people_for(cfg, db)
    names = {p.display_name for p in people}
    assert UNIDENTIFIED in names
    assert not any(n.lower().startswith("speaker") for n in names), (
        "a diarization placeholder was presented as a person"
    )
    bucket = by_name(people, UNIDENTIFIED)
    assert bucket.is_bucket
    assert not bucket.voice_verified
    assert not bucket.commitments_from_them and not bucket.commitments_to_them
    # And Marcus in the same recording is unaffected by sharing a room with them.
    assert "rec_unlabelled" in {a.recording_id for a in by_name(people, "Marcus").appearances}


def test_voice_verified_marks_only_names_with_an_enrolled_voiceprint(processed):
    cfg, db = processed
    from plaud_bridge.diarize.voiceprint import VoiceprintStore

    store = VoiceprintStore(Vault(cfg.path("vault")))
    store.enroll("Marcus", [0.1, 0.2, 0.3], source="clip.wav", seconds=5.0)
    store.save()

    people = people_for(cfg, db)
    assert by_name(people, "Marcus").voice_verified
    assert not by_name(people, "Sasson").voice_verified

    # The rendering must show which kind of name each row is, not just know it.
    roster = render_roster(people)
    marcus_row = next(ln for ln in roster.splitlines() if ln.startswith("| Marcus"))
    assert "voice-verified" in marcus_row
    other_rows = [ln for ln in roster.splitlines()
                  if ln.startswith("|") and "Marcus" not in ln]
    assert not any("voice-verified |" in ln for ln in other_rows)
    # A dossier for an unverified name says what kind of name it is showing.
    assert "label only" in render_person(Person(label="Dana", display_name="Dana"))


def test_an_unopenable_recording_is_reported_and_skipped_not_guessed_at(
        processed, caplog):
    cfg, db = processed
    vault = Vault(cfg.path("vault"))
    rec = add_row(cfg, db, vault, "rec_lost", speakers=["Ghost"], encrypt=True)

    # The vault artifact vanishes -- a retention sweep, a moved disk -- so the
    # index knows the recording exists and nothing about who spoke in it.
    from pathlib import Path
    Path(rec.artifact_paths["analysis"]).unlink()

    with caplog.at_level(logging.WARNING, logger="plaud_bridge.people"):
        people = people_for(cfg, db)

    assert "Ghost" not in {p.display_name for p in people}, (
        "a speaker was invented from a recording that could not be opened"
    )
    assert any("rec_lost" in r.getMessage() for r in caplog.records), (
        "the skipped recording was not reported"
    )


def test_suppressed_and_sensitive_fields_never_reach_the_page(processed):
    cfg, db = processed
    marcus = by_name(people_for(cfg, db), "Marcus")
    everything = " ".join(marcus.things_they_said) + render_person(marcus)
    # The stub extracts Marcus's mortgage into financial_disclosures, which
    # insurance_agent suppresses. It stays in the vault artifact.
    assert "four hundred thousand" not in everything
    # A quote from a non-suppressed field of the same analysis does surface.
    assert any("second kid" in text for text in marcus.things_they_said)


# =========================================================================
# Commitments and the window
# =========================================================================
def test_commitments_land_on_the_right_counterparty(processed):
    cfg, db = processed
    people = people_for(cfg, db, include_personal=True)

    marcus = by_name(people, "Marcus")
    assert any("email you tonight" in i.text for i in marcus.commitments_from_them)
    assert any("by Thursday" in i.text for i in marcus.commitments_to_them)

    kid = by_name(people, "Kid")
    assert any("permission slip" in i.text for i in kid.commitments_to_them)
    assert any("pizza" in i.text for i in kid.commitments_from_them)

    # Nothing from the dinner table lands on the client, and vice versa.
    marcus_texts = " ".join(
        i.text for i in (*marcus.commitments_from_them, *marcus.commitments_to_them)
    )
    assert "pizza" not in marcus_texts and "permission slip" not in marcus_texts
    kid_texts = " ".join(
        i.text for i in (*kid.commitments_from_them, *kid.commitments_to_them)
    )
    assert "Thursday" not in kid_texts

    assert marcus.open_items > 0


def test_the_days_window_limits_who_is_on_the_roster(tmp_path, monkeypatch):
    cfg, _stub = build_sandbox(tmp_path, monkeypatch)
    db = Database(cfg.path("database"))
    try:
        vault = Vault(cfg.path("vault"))
        add_row(cfg, db, vault, "rec_old", speakers=["Old Contact"], days_ago=40)
        add_row(cfg, db, vault, "rec_new", speakers=["New Contact"], days_ago=1)

        recent = {p.display_name for p in people_for(cfg, db, days=7)}
        assert recent == {"New Contact"}
        everyone = {p.display_name for p in people_for(cfg, db)}
        assert everyone == {"Old Contact", "New Contact"}
    finally:
        db.close()


# =========================================================================
# Lookup and the CLI route
# =========================================================================
def test_an_unknown_name_is_refused_with_the_names_that_exist(processed):
    cfg, db = processed
    people = people_for(cfg, db)
    assert person_detail(people, "marcus").display_name == "Marcus"
    assert person_detail(people, "Marc").display_name == "Marcus"
    with pytest.raises(PeopleError) as excinfo:
        person_detail(people, "Zorp")
    assert "Marcus" in str(excinfo.value), "the refusal should name who does exist"


def test_the_cli_renders_the_roster_and_the_dossier(processed, capsys):
    cfg, db = processed

    assert cli(cfg, "people") == 0
    roster = capsys.readouterr().out
    assert "Marcus" in roster and "Sasson" in roster
    assert "Kid" not in roster

    assert cli(cfg, "people", "--name", "Marcus") == 0
    dossier = capsys.readouterr().out
    assert "Every time they were heard" in dossier
    assert "client.txt" in dossier

    assert cli(cfg, "people", "--days", "30", "--include-personal") == 0
    assert "Kid" in capsys.readouterr().out

    assert cli(cfg, "people", "--name", "Zorp") == 1

    out = cfg.root / "people.html"
    assert cli(cfg, "people", "--format", "html", "--out", str(out)) == 0
    assert out.read_text(encoding="utf-8").strip()


def test_an_empty_index_renders_a_roster_that_says_so(tmp_path, monkeypatch, capsys):
    cfg, _stub = build_sandbox(tmp_path, monkeypatch)
    assert cli(cfg, "people") == 0
    assert "Nobody has been heard" in capsys.readouterr().out
