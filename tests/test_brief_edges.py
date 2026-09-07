"""
Brief edges: the honesty paths test_brief.py does not reach.

Each of these is a way the memo could quietly say less than it knows, or
more: an analysis that errored or was withheld, a follow-up state file that
will not open, PII that must be scrubbed before a model sees it, note blocks
that did not fit, a spend or audit row that could not be written, a model
that returned nothing usable, and personal content in a bundle that would
otherwise be allowed to reach a cloud provider. Each is pinned on the Brief
that comes back and on what the model was actually shown.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

import pytest

from _fixtures import CLIENT_CALL, build_sandbox, drop
from plaud_bridge.archive import Archive
from plaud_bridge.brief import (
    Brief,
    _bundle,
    _template_sections,
    _user_prompt,
    _validate_receipts,
    build_brief,
    render,
)
from plaud_bridge.cli import main
from plaud_bridge.db import Database
from plaud_bridge.followups import FollowUpError
from plaud_bridge.llm.base import LLMResponse
from plaud_bridge.models import (
    ProfileAnalysis,
    Recording,
    RouteMatch,
    Segment,
    Transcript,
)

SSN = "123-45-6789"


class FakeLLM:
    def __init__(self, payload=None, cost_usd=0.0):
        self.calls: list[dict] = []
        self.payload = payload

    def __call__(self, cfg, system, user, local_only=False, max_tokens=None):
        self.calls.append({"user": user, "local_only": local_only})
        body = self.payload if self.payload is not None else {
            "the_week": "w", "aging": "a", "people": "p", "next": "n", "receipts": []}
        return body, LLMResponse(provider="stub", model="stub-1", cost_usd=0.002)


def _install(monkeypatch, fake: FakeLLM) -> FakeLLM:
    monkeypatch.setattr("plaud_bridge.brief.complete_json", fake)
    return fake


def add_row(db, recording_id, *, profile_id="insurance_agent", fields=None,
            attention=False, error="", days_ago=1) -> Recording:
    """A plaintext recording straight into the index, the way test_people does."""
    rec = Recording(
        id=recording_id, source_name=f"{recording_id}.txt",
        source_path=f"/inbox/{recording_id}.txt", content_hash=f"hash-{recording_id}",
        kind="text", recorded_at=datetime.now(timezone.utc) - timedelta(days=days_ago),
        duration_seconds=120.0,
    )
    rec.transcript = Transcript(segments=[Segment(0, 60, "policy talk", "Sasson"),
                                          Segment(60, 120, "sure", "Marcus")])
    rec.routes = [RouteMatch(profile_id=profile_id, confidence=0.9)]
    rec.compliance.governing_profile = profile_id
    rec.compliance.encrypt_at_rest = False
    rec.analyses = [ProfileAnalysis(profile_id=profile_id, fields=fields or {},
                                    requires_human_attention=attention, error=error)]
    db.upsert(rec)
    return rec


@pytest.fixture
def bench(tmp_path, monkeypatch):
    """One processed client call plus an open database and archive."""
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    drop(cfg, "client.txt", CLIENT_CALL)
    assert main(["--config", str(tmp_path / "config"), "run"]) == 0
    db = Database(cfg.path("database"))
    archive = Archive(cfg, db)
    try:
        yield cfg, db, archive
    finally:
        db.close()


def _build(cfg, db, archive, **kw) -> Brief:
    return build_brief(cfg, db, archive, vault=archive.vault, **kw)


# =========================================================================
# The skeleton carries attention and errors
# =========================================================================
def test_flagged_and_errored_analyses_are_listed_for_attention_and_shown_to_the_model_as_such(
        bench, monkeypatch):
    cfg, db, archive = bench
    # The extractor blanks every field to its type default when it flags a
    # recording or fails, so these rows carry the shape real ones do: the
    # stated needs are a list the model would otherwise be shown.
    blank = {"next_action": "", "stated_needs": [{"speaker": "Marcus", "text": "kept"}]}
    add_row(db, "rec_flagged", attention=True, fields=dict(blank))
    add_row(db, "rec_broken", error="provider exploded mid-call", fields=dict(blank))
    fake = _install(monkeypatch, FakeLLM())

    brief = _build(cfg, db, archive)

    why = {a["recording_id"]: a["why"] for a in brief.attention}
    assert why["rec_flagged"] == "flagged for human attention"
    assert why["rec_broken"] == "provider exploded mid-call"
    assert {"rec_flagged", "rec_broken"} <= set(brief.recording_ids)
    assert not any(a["source_name"] in ("rec_flagged.txt", "rec_broken.txt")
                   for a in brief.next_actions)

    # The model is told WHY each is missing and shown none of its fields.
    material = fake.calls[0]["user"]
    assert "flagged for human attention; the analysis was withheld on purpose" in material
    assert "analysis unavailable: provider exploded mid-call" in material
    assert "Stated Needs: " not in material.split("RECORDING rec_flagged")[1].split("RECORDING")[0]
    assert "kept" not in material

    # The template's Next line and the numbers footer both count them.
    assert "review 2 recording(s) flagged for attention" in _template_sections(brief)["next"]
    assert "2 recording(s) flagged for attention" in render(brief)


# =========================================================================
# Follow-ups that cannot be read
# =========================================================================
def test_an_unreadable_follow_up_state_is_reported_not_fatal(bench, monkeypatch):
    cfg, db, archive = bench

    def locked(*_a, **_k):
        raise FollowUpError("state file is locked (test)")

    monkeypatch.setattr("plaud_bridge.brief.collect", locked)
    _install(monkeypatch, FakeLLM())
    brief = _build(cfg, db, archive)

    assert brief.followups == []
    assert "Open follow-ups could not be read: state file is locked (test)" in brief.note
    assert "Open follow-ups could not be read" in render(brief)


# =========================================================================
# What the model is shown: redaction and the context budget
# =========================================================================
def test_pii_is_scrubbed_from_the_material_and_the_scrub_is_noted(bench, monkeypatch):
    cfg, db, archive = bench
    add_row(db, "rec_pii", fields={"next_action": f"call back about ssn {SSN}"})
    fake = _install(monkeypatch, FakeLLM())

    brief = _build(cfg, db, archive)

    assert SSN not in fake.calls[0]["user"], "the SSN reached the model"
    assert "[SSN_REDACTED]" in fake.calls[0]["user"]
    assert brief.redactions.get("ssn", 0) >= 1
    assert "Redacted before the model saw it: ssn=" in brief.note
    # The copy a person reads is the original: the next action still names it.
    assert any(SSN in a["action"] for a in brief.next_actions)


def test_blocks_past_the_context_budget_are_left_out_and_counted(tmp_path, monkeypatch):
    cfg, _ = build_sandbox(tmp_path, monkeypatch,
                           overrides={"brief": {"max_context_chars": 120}})
    drop(cfg, "client.txt", CLIENT_CALL)
    assert main(["--config", str(tmp_path / "config"), "run"]) == 0
    db = Database(cfg.path("database"))
    try:
        archive = Archive(cfg, db)
        add_row(db, "rec_second", fields={"next_action": "send the illustration"})
        fake = _install(monkeypatch, FakeLLM())
        brief = _build(cfg, db, archive)
    finally:
        db.close()

    assert brief.left_out >= 1
    assert f"{brief.left_out} note block(s) did not fit the context budget" in brief.note
    user = fake.calls[0]["user"]
    assert f"{brief.left_out} note block(s) were left out" in user
    assert "Do not guess at their contents" in user
    # Exactly one recording block was sent: the first one always goes.
    assert user.count("RECORDING rec_") == 1


def test_bundle_redacts_each_block_and_stops_at_the_budget():
    blocks = [
        ("rec_a", f"RECORDING rec_a\n  Next Action: call {SSN}"),
        ("", "OPEN FOLLOW-UPS (oldest first):"),
        ("rec_b", "  - (3d) send the quote [insurance_agent, from rec_b]"),
        ("rec_c", "RECORDING rec_c\n  Topic: something long enough to overflow"),
    ]
    patterns = {"ssn": r"\b\d{3}-\d{2}-\d{4}\b"}
    # After redaction the first three blocks total 134 characters; the fourth
    # would push past 160 and is the one left out.
    text, haystacks, counts, left_out = _bundle(blocks, patterns, redact=True, max_chars=160)

    assert counts == {"ssn": 1}
    assert SSN not in text and "[SSN_REDACTED]" in text
    assert left_out == 1 and "rec_c" not in haystacks
    # A block with no recording id is sent but owns no haystack.
    assert set(haystacks) == {"rec_a", "rec_b"}
    assert " redacted " in haystacks["rec_a"], "the haystack is the redacted copy the model saw"

    # Redaction off: the text goes through verbatim and nothing is counted.
    text, _, counts, _ = _bundle(blocks[:1], patterns, redact=False, max_chars=10_000)
    assert SSN in text and counts == {}


def test_the_prompt_only_mentions_left_out_blocks_when_there_are_any():
    assert "left out" not in _user_prompt(7, "notes", 0)
    assert "3 note block(s) were left out" in _user_prompt(7, "notes", 3)


# =========================================================================
# Receipts
# =========================================================================
def test_receipts_that_are_not_objects_are_ignored_without_being_counted():
    haystacks = {"rec_a": " the quote is here "}
    kept, bad_quotes, bad_recordings = _validate_receipts(
        ["a string", 42, None, {"recording_id": "rec_a", "quote": "quote is"}], haystacks)
    assert kept == [{"recording_id": "rec_a", "quote": "quote is"}]
    assert (bad_quotes, bad_recordings) == (0, 0)
    assert _validate_receipts("not a list", haystacks) == ([], 0, 0)
    assert _validate_receipts([{"recording_id": "rec_a", "text": "quote is"}], haystacks)[0] == [
        {"recording_id": "rec_a", "quote": "quote is"}]


# =========================================================================
# Spend and audit rows that cannot be written
# =========================================================================
def test_a_spend_row_that_cannot_be_written_does_not_lose_the_brief(bench, monkeypatch, caplog):
    cfg, db, archive = bench
    _install(monkeypatch, FakeLLM())

    def broken(*_a, **_k):
        raise RuntimeError("disk full (test)")

    monkeypatch.setattr(db, "record_spend", broken)
    with caplog.at_level(logging.WARNING, logger="plaud_bridge.brief"):
        brief = _build(cfg, db, archive)

    assert brief.narrated and brief.cost_usd == pytest.approx(0.002)
    assert any("could not record what this brief cost" in r.getMessage() for r in caplog.records)
    assert "brief" not in db.stats()["by_source"]


def test_an_audit_row_that_cannot_be_written_does_not_lose_the_brief(bench, monkeypatch, caplog):
    cfg, db, archive = bench
    _install(monkeypatch, FakeLLM())
    monkeypatch.setattr(db, "audit", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("locked")))
    with caplog.at_level(logging.WARNING, logger="plaud_bridge.brief"):
        brief = _build(cfg, db, archive)
    assert brief.narrated
    assert any("could not write the audit entry" in r.getMessage() for r in caplog.records)
    assert not db.audit_log(action="brief")


# =========================================================================
# A model that returns nothing usable
# =========================================================================
def test_a_reply_with_every_section_blank_is_the_labelled_template(bench, monkeypatch):
    cfg, db, archive = bench
    _install(monkeypatch, FakeLLM({"the_week": "", "aging": None, "receipts": []}))
    brief = _build(cfg, db, archive)

    assert not brief.narrated
    assert brief.sections == _template_sections(brief)
    assert "The model returned no usable narrative" in brief.note
    assert "returned nothing for" not in brief.note, "the partial-gap note must not also fire"
    assert "Assembled, not narrated" in render(brief)
    # The call still cost money and that is still recorded.
    assert db.stats()["by_source"]["brief"]["cost_usd"] == pytest.approx(0.002)


# =========================================================================
# Personal content forces locality even when every profile allows cloud
# =========================================================================
def test_personal_content_forces_local_only_even_for_a_cloud_permitting_profile(
        tmp_path, monkeypatch):
    """
    The shipped personal profiles are hard-local, so this guard only matters
    for a custom profile marked personal without being locked. sales_trainer
    allows cloud; marking it personal must still keep the brief local.
    """
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    trainer = cfg.profile("sales_trainer")
    assert trainer.allow_cloud_llm and not trainer.hard_local_only, "fixture premise"
    trainer.exclude_from_combined_export = True

    db = Database(cfg.path("database"))
    try:
        archive = Archive(cfg, db)
        add_row(db, "rec_coach", profile_id="sales_trainer",
                fields={"next_action": "practice the close"})
        fake = _install(monkeypatch, FakeLLM())
        brief = _build(cfg, db, archive, include_personal=True)
    finally:
        db.close()

    assert [p["profile_id"] for p in brief.profiles] == ["sales_trainer"]
    assert brief.local_only is True
    assert all(c["local_only"] for c in fake.calls)
    assert "personal content never reaches a cloud model" in brief.note


# =========================================================================
# Serialisation and rendering
# =========================================================================
def test_a_brief_serialises_to_plain_json_with_every_field(bench, monkeypatch):
    cfg, db, archive = bench
    _install(monkeypatch, FakeLLM())
    brief = _build(cfg, db, archive)
    payload = json.loads(json.dumps(brief.to_dict()))
    for key in ("days", "generated_at", "profiles", "followups", "attention", "next_actions",
                "quarantined", "spend", "recording_ids", "narrated", "sections", "receipts",
                "dropped_quotes", "dropped_recordings", "local_only", "provider", "cost_usd",
                "redactions", "left_out", "note"):
        assert key in payload
    assert payload["narrated"] is True and payload["provider"] == "stub"
    assert payload["recording_ids"] == brief.recording_ids


def test_a_section_with_no_text_is_not_rendered_as_an_empty_heading():
    brief = Brief(sections={"the_week": "Quiet.", "aging": "", "next": "Nothing."})
    out = render(brief)
    assert "## The week" in out and "## Next" in out
    assert "## Aging" not in out and "## People waiting on you" not in out
    assert "No recordings in this window." in out


def test_the_template_names_quarantine_and_leaves_spend_off_when_unknown():
    brief = Brief(quarantined=3)
    sections = _template_sections(brief)
    assert sections["next"] == "triage 3 recording(s) in quarantine (run.py quarantine)."
    assert sections["aging"] == "Nothing is outstanding."
    out = render(brief)
    assert "3 in quarantine" in out and "API spend all-time" not in out
