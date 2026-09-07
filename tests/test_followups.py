"""
Follow-ups: the promises, and what happens to them.

Two kinds of test live here. The ordinary ones check that a commitment said
three times is one item, that an old one sorts above a new one, and that
marking something done sticks.

The rest exist because this module writes a document intended to leave the
machine, which makes it the riskiest thing in the tool after `export`. Those
tests pin the guarantees the module docstring makes in plain language: the
strictest profile governs a whole draft, redaction happens before a model and
before the file, personal profiles stay out, and nothing in here opens a
socket. Each of them fails if the corresponding line of code is removed.
"""

from __future__ import annotations

import json
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from _fixtures import build_sandbox
from plaud_bridge.archive import Archive
from plaud_bridge.db import Database
from plaud_bridge.followups import (
    FollowUp,
    FollowUpError,
    collect,
    commitment_fields,
    draft,
    draft_local_only,
    render,
    set_status,
    stable_id,
    state_path,
)
from plaud_bridge.llm.base import LLMError, LLMResponse
from plaud_bridge.models import (
    ProfileAnalysis,
    Recording,
    RouteMatch,
    Segment,
    Transcript,
)
from plaud_bridge.storage import Vault

PRODUCER_PROMISE = "I'll have the two quote options over to you by Thursday."


def quote(text: str, stamp: str = "00:45", speaker: str = "Sasson") -> dict:
    """A quote-shaped extraction item, the way the extractor returns them."""
    return {"timestamp": stamp, "speaker": speaker, "text": text}


class Bench:
    """
    A sandbox with an index, a vault, and an archive over both.

    Recordings are written straight into the index rather than run through the
    pipeline: these tests are about what happens to analyses that already
    exist, and building them by hand is the only way to control the recording
    dates that ageing depends on.
    """

    def __init__(self, tmp_path, monkeypatch, overrides=None):
        self.cfg, self.stub = build_sandbox(tmp_path, monkeypatch, overrides=overrides)
        self.db = Database(self.cfg.path("database"))
        self.vault = Vault(self.cfg.path("vault"))
        self.archive = Archive(self.cfg, self.db, self.vault)

    def close(self):
        self.db.close()

    def add(self, recording_id: str, profile_id: str, fields: dict, *,
            days_ago: int = 0, name: str = "", encrypt: bool = False) -> Recording:
        rec = Recording(
            id=recording_id,
            source_name=name or f"{recording_id}.txt",
            source_path=f"/inbox/{recording_id}.txt",
            content_hash=f"hash-{recording_id}",
            kind="text",
            recorded_at=datetime.now(timezone.utc) - timedelta(days=days_ago),
        )
        rec.transcript = Transcript(segments=[Segment(0.0, 2.0, "spoken words", "Sasson")])
        rec.routes = [RouteMatch(profile_id=profile_id, confidence=0.9)]
        rec.compliance.governing_profile = profile_id
        rec.compliance.encrypt_at_rest = encrypt
        rec.analyses = [ProfileAnalysis(profile_id=profile_id, fields=fields)]

        if encrypt:
            # Same shape the pipeline persists: the index holds no fields and
            # the vault holds the whole record.
            rec.artifact_paths["analysis"] = str(
                self.vault.write(f"{recording_id}.analysis.json", rec.to_json(), recording_id)
            )
        self.db.upsert(rec)
        return rec

    def collect(self, **kw) -> list[FollowUp]:
        return collect(self.cfg, self.db, self.archive, vault=self.vault, **kw)


@pytest.fixture
def bench(tmp_path, monkeypatch):
    b = Bench(tmp_path, monkeypatch)
    yield b
    b.close()


class DraftLLM:
    """A stand-in for `complete_json` that records how it was called."""

    def __init__(self, subject: str = "Following up", body: str = "Hi,\n\nAs promised."):
        self.subject = subject
        self.body = body
        self.calls: list[dict] = []

    def __call__(self, cfg, system, user, local_only=False, max_tokens=None):
        self.calls.append({"local_only": local_only, "system": system, "user": user})
        return (
            {"subject": self.subject, "body": self.body},
            LLMResponse(provider="stub", model="stub-1"),
        )


# =========================================================================
# One commitment, however many times it was said
# =========================================================================
def test_the_same_promise_in_three_recordings_collapses_to_one_item(bench):
    """
    The point of the stable id. The producer said the same thing on Monday,
    again on Wednesday, and again on Friday; that is one outstanding promise,
    not three, and a worklist that shows it three times is a worklist nobody
    reads twice.
    """
    bench.add("rec_a", "insurance_agent",
              {"commitments_by_producer": [quote(PRODUCER_PROMISE, "00:45")]}, days_ago=11)
    bench.add("rec_b", "insurance_agent",
              {"commitments_by_producer": [quote(PRODUCER_PROMISE.upper(), "12:03")]}, days_ago=6)
    bench.add("rec_c", "insurance_agent",
              {"commitments_by_producer": [quote(PRODUCER_PROMISE.rstrip("."), "03:11")]},
              days_ago=2)

    items = bench.collect()
    assert len(items) == 1, [i.text for i in items]

    only = items[0]
    assert only.mentions == 3
    assert sorted(only.recording_ids) == ["rec_a", "rec_b", "rec_c"]
    # It traces to where the promise was MADE, not to the last time it came up.
    assert only.recording_id == "rec_a"
    assert only.age_days == 11


def test_the_id_is_content_derived_and_not_time_derived():
    """Timestamps differ between recordings of the same promise; ids must not."""
    assert stable_id("I'll call the plumber", "husband") == stable_id(
        "  I'll CALL the plumber!  ", "husband"
    )
    assert stable_id("I'll call the plumber", "husband") != stable_id(
        "I'll call the plumber", "father"
    )


def test_different_commitments_stay_separate(bench):
    bench.add("rec_a", "insurance_agent", {
        "commitments_by_producer": [quote(PRODUCER_PROMISE)],
        "commitments_by_client": [quote("I'll email you tonight.", speaker="Marcus")],
    })
    assert len(bench.collect()) == 2


# =========================================================================
# Ageing
# =========================================================================
def test_the_oldest_open_promise_sorts_to_the_top(bench):
    bench.add("rec_new", "insurance_agent", {"next_action": "Send the illustration"}, days_ago=1)
    bench.add("rec_old", "insurance_agent", {"next_action": "Call Marcus back"}, days_ago=11)
    bench.add("rec_mid", "insurance_agent", {"next_action": "Order the paramed"}, days_ago=4)

    items = bench.collect()
    assert [i.age_days for i in items] == [11, 4, 1]
    assert items[0].text == "Call Marcus back"


def test_closed_items_sort_below_every_open_one(bench):
    bench.add("rec_old", "insurance_agent", {"next_action": "Call Marcus back"}, days_ago=30)
    bench.add("rec_new", "insurance_agent", {"next_action": "Send the illustration"}, days_ago=1)

    items = bench.collect()
    oldest = next(i for i in items if i.text == "Call Marcus back")
    set_status(bench.cfg, bench.vault, oldest.id, "done", items=items)

    after = bench.collect()
    assert [i.status for i in after] == ["open", "done"]
    assert after[0].text == "Send the illustration"


# =========================================================================
# Status survives the next run
# =========================================================================
def test_marking_one_done_is_not_undone_by_the_next_collection(bench):
    bench.add("rec_a", "insurance_agent", {"next_action": "Send two quote options"}, days_ago=3)

    first = bench.collect()
    assert first[0].status == "open"
    marked = set_status(bench.cfg, bench.vault, first[0].id, "done", items=first)
    assert marked.status == "done"

    again = bench.collect()
    assert [i.status for i in again] == ["done"]
    assert bench.collect(status="open") == []
    assert len(bench.collect(status="done")) == 1


def test_a_short_id_prefix_is_enough_to_mark_something(bench):
    bench.add("rec_a", "insurance_agent", {"next_action": "Send two quote options"})
    items = bench.collect()
    set_status(bench.cfg, bench.vault, items[0].short_id, "dropped", items=items)
    assert bench.collect()[0].status == "dropped"


def test_an_unknown_id_is_refused_rather_than_recorded(bench):
    """A status against something no recording produced is an invented item."""
    bench.add("rec_a", "insurance_agent", {"next_action": "Send two quote options"})
    with pytest.raises(FollowUpError, match="no follow-up with id"):
        set_status(bench.cfg, bench.vault, "fu_deadbeefdead", "done", items=bench.collect())
    assert not state_path(bench.cfg).exists()


def test_every_follow_up_names_a_recording_that_is_really_in_the_index(bench):
    bench.add("rec_a", "insurance_agent", {"next_action": "Send two quote options"})
    bench.add("rec_b", "sales_trainer", {"next_action": "Rewrite the discovery question"})

    known = {row["id"] for row in bench.db.query()}
    for item in bench.collect():
        assert item.recording_id in known
        assert set(item.recording_ids) <= known


# =========================================================================
# Nothing sensitive in plaintext
# =========================================================================
def test_the_status_file_is_encrypted_when_a_passphrase_is_available(bench):
    bench.add("rec_a", "insurance_agent", {"next_action": "Call Marcus about the biopsy result"})
    items = bench.collect()
    set_status(bench.cfg, bench.vault, items[0].id, "done", items=items)

    raw = state_path(bench.cfg).read_bytes()
    assert raw.startswith(b"PBV1"), "the status file was written in the clear"
    assert b"biopsy" not in raw
    assert b"rec_a" not in raw


def test_without_a_passphrase_the_status_file_still_holds_no_wording(
    tmp_path, monkeypatch
):
    """
    The fallback for someone running with no vault at all. It degrades to
    plaintext rather than refusing to remember anything, and what it writes is
    hashed ids and recording ids -- never what was said.
    """
    b = Bench(tmp_path, monkeypatch)
    try:
        b.add("rec_a", "sales_trainer", {"next_action": "Call Marcus about the biopsy result"})
        items = b.collect()
        monkeypatch.delenv("PLAUD_BRIDGE_PASSPHRASE")
        set_status(b.cfg, b.vault, items[0].id, "done", items=items)

        raw = state_path(b.cfg).read_text(encoding="utf-8")
        assert "biopsy" not in raw
        assert json.loads(raw)["items"][items[0].id]["status"] == "done"
    finally:
        b.close()


def test_a_status_file_that_will_not_open_stops_the_run(bench, monkeypatch):
    """
    Reading an unopenable state file as "no statuses" marks every finished item
    open again. Refusing is the honest answer, and it is the same stance
    `verify` takes about an artifact it could not decrypt.
    """
    bench.add("rec_a", "insurance_agent", {"next_action": "Send two quote options"})
    items = bench.collect()
    set_status(bench.cfg, bench.vault, items[0].id, "done", items=items)

    monkeypatch.delenv("PLAUD_BRIDGE_PASSPHRASE")
    with pytest.raises(FollowUpError, match="cannot be read"):
        bench.collect()
    # And it must not answer a write by overwriting statuses it could not read.
    with pytest.raises(FollowUpError):
        set_status(bench.cfg, bench.vault, items[0].id, "open", items=items)


# =========================================================================
# Which fields are commitments
# =========================================================================
def test_commitment_fields_are_discovered_across_every_shipped_profile(bench):
    found = {pid: commitment_fields(bench.cfg, p) for pid, p in bench.cfg.profiles.items()}

    assert "promises_i_made" in found["father"]
    assert "asks_from_them" in found["father"]
    assert "commitments_i_made" in found["husband"]
    assert "she_asked_for" in found["husband"]
    assert "commitments_by_producer" in found["insurance_agent"]
    assert "commitments_by_client" in found["insurance_agent"]
    assert "action_items" in found["unfiled"]
    for fields in found.values():
        assert "next_action" in fields

    # Memory is not a commitment. Pulling these in would make the worklist a
    # second digest, which is what it exists not to be.
    assert "worth_remembering" not in found["father"]
    assert "logistics" not in found["father"]
    assert "milestones" not in found["father"]
    assert "health_disclosures" not in found["insurance_agent"]
    assert "content_hooks" not in found["sales_trainer"]


def test_the_field_list_is_configurable(tmp_path, monkeypatch):
    b = Bench(tmp_path, monkeypatch, overrides={"followups": {"field_tokens": ["milestone"]}})
    try:
        assert commitment_fields(b.cfg, b.cfg.profile("father")) == ["milestones"]
    finally:
        b.close()


def test_an_explicit_field_list_wins_and_bad_names_are_dropped(tmp_path, monkeypatch):
    b = Bench(tmp_path, monkeypatch, overrides={
        "followups": {"fields": {"father": ["logistics", "not_a_field"]}},
    })
    try:
        assert commitment_fields(b.cfg, b.cfg.profile("father")) == ["logistics"]
    finally:
        b.close()


def test_placeholders_and_flags_do_not_become_follow_ups(bench):
    bench.add("rec_a", "father", {
        "requires_human_attention": False,
        "next_action": "none",
        "promises_i_made": [],
        "asks_from_them": [quote("can we get pizza after the game?", speaker="Kid")],
    })
    items = bench.collect(include_personal=True)
    assert [i.text for i in items] == ["can we get pizza after the game?"]


def test_an_object_shaped_commitment_keeps_its_due_date(bench):
    bench.add("rec_a", "father",
              {"promises_i_made": [{"what": "sign the permission slip", "when": "tonight"}]})
    item = bench.collect(include_personal=True)[0]
    assert item.text == "sign the permission slip"
    assert item.due == "tonight"


def test_a_failed_analysis_contributes_nothing(bench):
    """Its fields are schema defaults, so anything read out is an artifact."""
    rec = bench.add("rec_a", "insurance_agent", {"next_action": "Send two quote options"})
    rec.analyses[0].error = "all LLM providers failed"
    bench.db.upsert(rec)
    assert bench.collect() == []


def test_forgetting_one_of_several_recordings_keeps_a_done_status(bench):
    """
    A commitment mentioned in two recordings collapses to one item whose state
    now stores both. Forgetting the earliest -- merely its primary mention --
    must not un-complete it: the commitment still exists via the other recording,
    so its done status is preserved rather than resurrecting as open.
    """
    from plaud_bridge.followups import forget_recording

    fields = {"commitments_by_producer": [quote(PRODUCER_PROMISE)]}
    bench.add("rec_a", "insurance_agent", fields, days_ago=10)
    bench.add("rec_b", "insurance_agent", fields, days_ago=1)

    items = bench.collect()
    assert len(items) == 1 and set(items[0].recording_ids) == {"rec_a", "rec_b"}
    assert items[0].recording_id == "rec_a", "the earliest mention should be the primary"
    set_status(bench.cfg, bench.vault, items[0].id, "done", items=items)

    # Forget the earliest recording: drop it from the index and the state.
    forget_recording(bench.cfg, bench.vault, "rec_a")
    bench.db.delete_recording("rec_a")

    again = bench.collect()
    assert len(again) == 1, "the commitment should still exist via rec_b"
    assert again[0].status == "done", "a completed commitment resurrected as open after forget"


def test_a_scoped_query_does_not_leak_a_co_routed_personal_commitment(bench):
    """
    `db.query(profile_id=X)` joins on routes, so a recording co-routed to a work
    profile but governed by a locked personal one comes back carrying the
    personal analysis too. `followups --profile insurance_agent` must not
    surface the Husband commitment riding along on that record.
    """
    rec = Recording(
        id="rec_corouted", source_name="rec_corouted.txt",
        source_path="/inbox/rec_corouted.txt", content_hash="hash-co", kind="text",
        recorded_at=datetime.now(timezone.utc),
    )
    rec.transcript = Transcript(segments=[Segment(0.0, 2.0, "spoken words", "Sasson")])
    rec.routes = [RouteMatch(profile_id="husband", confidence=0.95),
                  RouteMatch(profile_id="insurance_agent", confidence=0.95)]
    rec.compliance.governing_profile = "husband"
    rec.compliance.encrypt_at_rest = True
    rec.analyses = [
        ProfileAnalysis(profile_id="husband",
                        fields={"commitments_i_made": [quote("book the anniversary dinner")]}),
        ProfileAnalysis(profile_id="insurance_agent",
                        fields={"next_action": "Send two quote options"}),
    ]
    # Husband governs, so this is encrypted at rest: the whole record lives in
    # the vault and the index holds withheld fields, exactly as the pipeline
    # persists it.
    rec.artifact_paths["analysis"] = str(
        bench.vault.write("rec_corouted.analysis.json", rec.to_json(), rec.id)
    )
    bench.db.upsert(rec)

    scoped = bench.collect(profile="insurance_agent")
    texts = [i.text for i in scoped]
    assert "Send two quote options" in texts, "the asked-for profile's own commitment is missing"
    assert all("anniversary" not in t for t in texts), (
        "a scoped work-profile query surfaced a locked personal commitment"
    )
    assert all(i.profile_id == "insurance_agent" for i in scoped)


# =========================================================================
# Encrypted analyses, and ones that will not open
# =========================================================================
def test_follow_ups_are_read_out_of_the_vault_when_that_is_where_they_live(bench):
    bench.add("rec_locked", "father",
              {"promises_i_made": [{"what": "pizza after the game", "when": "Saturday"}]},
              encrypt=True)

    payload = json.loads(bench.db.query()[0]["payload_json"])
    assert payload["analyses"][0]["fields_withheld"] is True, "the index kept the fields"

    items = bench.collect(include_personal=True)
    assert [i.text for i in items] == ["pizza after the game"]


def test_a_recording_that_will_not_open_is_skipped_not_guessed(bench):
    """The rest of the window still renders; the unreadable one contributes nothing."""
    bench.add("rec_locked", "father", {"promises_i_made": [{"what": "pizza", "when": "Sat"}]},
              encrypt=True)
    bench.add("rec_plain", "insurance_agent", {"next_action": "Send two quote options"})

    Path(bench.db.load("rec_locked")["artifact_paths"]["analysis"]).unlink()

    items = bench.collect(include_personal=True)
    assert [i.text for i in items] == ["Send two quote options"]


# =========================================================================
# Personal profiles
# =========================================================================
def test_personal_follow_ups_stay_out_of_the_worklist_by_default(bench):
    bench.add("rec_home", "husband",
              {"commitments_i_made": [{"what": "call the plumber", "when": "Monday"}]})
    bench.add("rec_work", "insurance_agent", {"next_action": "Send two quote options"})

    assert [i.profile_id for i in bench.collect()] == ["insurance_agent"]
    assert len(bench.collect(include_personal=True)) == 2
    # Asking for the profile by name is asking for it.
    assert len(bench.collect(profile="husband")) == 1


def test_a_personal_follow_up_is_refused_for_drafting(bench):
    bench.add("rec_home", "husband",
              {"commitments_i_made": [{"what": "call the plumber", "when": "Monday"}]})
    items = bench.collect(include_personal=True)

    with pytest.raises(FollowUpError, match="personal profile"):
        draft(items, bench.cfg, use_llm=False)

    path = draft(items, bench.cfg, use_llm=False, include_personal=True)
    assert "call the plumber" in path.read_text(encoding="utf-8")


def test_a_mixed_draft_drops_the_personal_half(bench):
    bench.add("rec_home", "husband", {"commitments_i_made": [{"what": "call the plumber"}]})
    bench.add("rec_work", "insurance_agent", {"next_action": "Send two quote options"})

    body = draft(bench.collect(include_personal=True), bench.cfg,
                 use_llm=False).read_text(encoding="utf-8")
    assert "Send two quote options" in body
    assert "plumber" not in body


# =========================================================================
# Drafting: the template path
# =========================================================================
def test_the_template_path_writes_a_usable_draft_with_no_model(bench, monkeypatch):
    def explode(*a, **k):
        raise AssertionError("a model was consulted on the no-LLM path")

    monkeypatch.setattr("plaud_bridge.followups.complete_json", explode)
    bench.add("rec_a", "insurance_agent", {"next_action": "Send two quote options"}, days_ago=3)

    path = draft(bench.collect(), bench.cfg, db=bench.db, use_llm=False)
    body = path.read_text(encoding="utf-8")

    assert "DRAFT" in path.name and path.name.endswith(".draft.md")
    assert "Send two quote options" in body
    assert "Nothing has been sent" in body
    assert "rec_a" in body, "the draft did not say which recording it came from"
    assert "template (no model involved)" in body


def test_an_unreachable_model_falls_back_to_the_template(bench, monkeypatch):
    def unavailable(*a, **k):
        raise LLMError("no LLM provider available")

    monkeypatch.setattr("plaud_bridge.followups.complete_json", unavailable)
    bench.add("rec_a", "insurance_agent", {"next_action": "Send two quote options"})

    body = draft(bench.collect(), bench.cfg, use_llm=True).read_text(encoding="utf-8")
    assert "Send two quote options" in body
    assert "template (no model involved)" in body


def test_a_plain_text_draft_carries_no_markdown(bench):
    bench.add("rec_a", "insurance_agent", {"next_action": "Send two quote options"})
    path = draft(bench.collect(), bench.cfg, use_llm=False, fmt="text")
    body = path.read_text(encoding="utf-8")

    assert path.name.endswith(".draft.txt")
    assert "DRAFT - NOT SENT" in body
    assert "<sub>" not in body and not body.startswith("#")


def test_a_second_draft_does_not_overwrite_the_first(bench):
    """The one on disk may already have been edited by the person sending it."""
    bench.add("rec_a", "insurance_agent", {"next_action": "Send two quote options"})
    items = bench.collect()
    first = draft(items, bench.cfg, use_llm=False)
    first.write_text("edited by a human", encoding="utf-8")

    second = draft(items, bench.cfg, use_llm=False)
    assert second != first
    assert first.read_text(encoding="utf-8") == "edited by a human"


def test_drafting_a_whole_recording_by_id(bench):
    bench.add("rec_a", "insurance_agent", {
        "next_action": "Send two quote options",
        "commitments_by_producer": [quote(PRODUCER_PROMISE)],
    })
    bench.add("rec_b", "insurance_agent", {"next_action": "Order the paramed"})

    body = draft("rec_a", bench.cfg, db=bench.db, archive=bench.archive,
                 use_llm=False).read_text(encoding="utf-8")
    assert "Send two quote options" in body
    assert PRODUCER_PROMISE in body
    assert "Order the paramed" not in body


def test_drafting_an_unknown_recording_id_refuses(bench):
    with pytest.raises(FollowUpError, match="has to trace to a recording"):
        draft("rec_nope", bench.cfg, db=bench.db, archive=bench.archive, use_llm=False)


# =========================================================================
# Drafting: the model path
# =========================================================================
def test_the_model_phrases_the_draft_when_one_is_reachable(bench, monkeypatch):
    llm = DraftLLM(subject="Two quotes by Thursday", body="Hi Marcus,\n\nAs promised.")
    monkeypatch.setattr("plaud_bridge.followups.complete_json", llm)
    bench.add("rec_a", "sales_trainer", {"next_action": "Send the roleplay recap"})

    body = draft(bench.collect(), bench.cfg, use_llm=True).read_text(encoding="utf-8")

    assert len(llm.calls) == 1
    assert "Two quotes by Thursday" in body
    assert "As promised." in body
    assert "stub/stub-1" in body, "the draft did not say what phrased it"
    assert "Send the roleplay recap" in llm.calls[0]["user"]


def test_a_model_that_returns_nothing_usable_falls_back(bench, monkeypatch):
    monkeypatch.setattr("plaud_bridge.followups.complete_json", DraftLLM(body="   "))
    bench.add("rec_a", "sales_trainer", {"next_action": "Send the roleplay recap"})

    body = draft(bench.collect(), bench.cfg, use_llm=True).read_text(encoding="utf-8")
    assert "template (no model involved)" in body
    assert "Send the roleplay recap" in body


# =========================================================================
# The strictest profile governs the whole draft
# =========================================================================
def test_one_locked_follow_up_forces_the_whole_draft_local(bench, monkeypatch):
    """
    ADR-002, applied to a draft. A set mixing Sales Trainer -- which permits a
    cloud model -- with Father, which is locked in code, is one document. There
    is no honest way to send half of it to a third party.
    """
    llm = DraftLLM()
    monkeypatch.setattr("plaud_bridge.followups.complete_json", llm)
    bench.add("rec_work", "sales_trainer", {"next_action": "Send the roleplay recap"})
    bench.add("rec_home", "father", {"promises_i_made": [{"what": "pizza after the game"}]})

    draft(bench.collect(include_personal=True), bench.cfg,
          use_llm=True, include_personal=True)

    assert [c["local_only"] for c in llm.calls] == [True]


def test_a_cloud_permitting_set_is_not_forced_local(bench, monkeypatch):
    """The control for the test above. Without it, always-true would also pass."""
    llm = DraftLLM()
    monkeypatch.setattr("plaud_bridge.followups.complete_json", llm)
    bench.add("rec_work", "sales_trainer", {"next_action": "Send the roleplay recap"})

    draft(bench.collect(), bench.cfg, use_llm=True)
    assert [c["local_only"] for c in llm.calls] == [False]


def test_locality_is_decided_per_profile_policy_not_per_item(bench):
    trainer = FollowUp(id="fu_1", text="x", profile_id="sales_trainer", recording_id="rec_a")
    agent = FollowUp(id="fu_2", text="y", profile_id="insurance_agent", recording_id="rec_b")
    father = FollowUp(id="fu_3", text="z", profile_id="father", recording_id="rec_c")
    stranger = FollowUp(id="fu_4", text="w", profile_id="gone_missing", recording_id="rec_d")

    assert draft_local_only(bench.cfg, [trainer]) is False
    assert draft_local_only(bench.cfg, [agent]) is True      # no cloud llm
    assert draft_local_only(bench.cfg, [trainer, father]) is True
    assert draft_local_only(bench.cfg, [stranger]) is True   # unknown is not permission


# =========================================================================
# Redaction happens before the model and before the file
# =========================================================================
DISCLOSING = "Email the illustration to marcus@example.com and call 702-555-0143"


def test_a_draft_is_redacted_before_it_is_written(bench):
    bench.add("rec_a", "insurance_agent", {"next_action": DISCLOSING})

    body = draft(bench.collect(), bench.cfg, use_llm=False).read_text(encoding="utf-8")
    assert "marcus@example.com" not in body
    assert "702-555-0143" not in body
    assert "[EMAIL_REDACTED]" in body
    assert "Redacted before this draft was written" in body


def test_the_model_never_sees_the_raw_wording(bench, monkeypatch):
    llm = DraftLLM()
    monkeypatch.setattr("plaud_bridge.followups.complete_json", llm)
    bench.add("rec_a", "sales_trainer", {"next_action": DISCLOSING})

    draft(bench.collect(), bench.cfg, use_llm=True)
    sent = llm.calls[0]["user"]
    assert "marcus@example.com" not in sent
    assert "[EMAIL_REDACTED]" in sent


def test_the_redaction_note_prints_even_when_nothing_matched(bench):
    """"No pattern fired" is not "nothing sensitive is in here"."""
    bench.add("rec_a", "insurance_agent", {"next_action": "Send two quote options"})
    body = draft(bench.collect(), bench.cfg, use_llm=False).read_text(encoding="utf-8")
    assert "No redaction pattern matched anything in this draft" in body
    assert "not a guarantee" in body


def test_the_worklist_itself_is_not_redacted(bench):
    """
    It is the local document, like the digest. Redacting the address out of
    your own worklist would remove the thing you need in order to act on it.
    """
    bench.add("rec_a", "insurance_agent", {"next_action": DISCLOSING})
    assert "marcus@example.com" in render(bench.collect())


# =========================================================================
# Rendering
# =========================================================================
def test_markdown_and_html_carry_the_same_facts(bench):
    bench.add("rec_a", "insurance_agent", {"next_action": "Send two quote options"}, days_ago=11)
    items = bench.collect()

    markdown = render(items, title="This week")
    html = render(items, fmt="html", title="This week")

    assert markdown.startswith("# This week")
    assert "## Still open" in markdown
    assert "| Follow-up | Open for | Profile |" in markdown
    assert "11d" in markdown

    assert "<title>This week</title>" in html
    assert "<table>" in html
    assert "Send two quote options" in html
    assert "cannot send it" in html
    assert "<script" not in html


def test_a_pipe_in_a_commitment_does_not_break_the_table(bench):
    """In either format: the HTML converter splits on pipes and has no escape."""
    bench.add("rec_a", "insurance_agent", {"next_action": "Compare term | whole life"})
    items = bench.collect()

    row = next(
        line for line in render(items).splitlines() if line.startswith("| Compare")
    )
    assert row.count("|") == 4, row

    html_row = next(
        line for line in render(items, fmt="html").splitlines()
        if line.startswith("<tr><td>Compare")
    )
    assert html_row.count("<td>") == 3, html_row


def test_an_empty_worklist_says_so_rather_than_rendering_a_skeleton():
    out = render([])
    assert "Nothing outstanding" in out
    assert "## Still open" not in out


def test_render_rejects_a_format_it_cannot_produce():
    with pytest.raises(FollowUpError, match="unknown format"):
        render([], fmt="pdf")


# =========================================================================
# Offline means offline
# =========================================================================
def test_nothing_in_this_module_opens_a_socket(bench, monkeypatch):
    """
    The whole no-model path, with the network physically unavailable. This is
    the test that fails if somebody later decides that mailing the draft would
    be convenient.
    """
    def refuse(*a, **k):
        raise AssertionError("followups opened a network socket")

    bench.add("rec_a", "insurance_agent", {"next_action": DISCLOSING}, days_ago=5)
    bench.add("rec_b", "insurance_agent", {"commitments_by_client": [quote("I'll email you")]})

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)

    items = bench.collect()
    render(items)
    render(items, fmt="html")
    set_status(bench.cfg, bench.vault, items[0].id, "done", items=items)
    path = draft(bench.collect(status="open"), bench.cfg, db=bench.db, use_llm=False)
    assert path.exists()
