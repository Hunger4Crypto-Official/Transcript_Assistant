"""
Follow-ups at the edges.

test_followups.py pins the promises. These pin the branches: the inputs that
are refused, the shapes a model can return that must not become a follow-up,
the state file when it is plaintext, unreadable, wrongly keyed, or not JSON,
resolving an id from the state file alone, the worklist's closed section and
mention history, and the draft caps and fallbacks.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from _fixtures import build_sandbox
from plaud_bridge.archive import Archive
from plaud_bridge.db import Database
from plaud_bridge.followups import (
    FollowUp,
    FollowUpError,
    _merge,
    _unclaimed,
    collect,
    draft,
    forget_recording,
    render,
    set_status,
    stable_id,
    state_path,
)
from plaud_bridge.llm.base import LLMResponse
from plaud_bridge.models import ProfileAnalysis, Recording, RouteMatch, Segment, Transcript
from plaud_bridge.storage import Vault


class Bench:
    def __init__(self, tmp_path, monkeypatch, overrides=None):
        self.cfg, self.stub = build_sandbox(tmp_path, monkeypatch, overrides=overrides)
        self.db = Database(self.cfg.path("database"))
        self.vault = Vault(self.cfg.path("vault"))
        self.archive = Archive(self.cfg, self.db, self.vault)

    def close(self):
        self.db.close()

    def add(self, recording_id, profile_id, fields, *, days_ago=0, analysis_profile=None):
        rec = Recording(
            id=recording_id, source_name=f"{recording_id}.txt",
            source_path=f"/inbox/{recording_id}.txt", content_hash=f"hash-{recording_id}",
            kind="text", recorded_at=datetime.now(timezone.utc) - timedelta(days=days_ago),
        )
        rec.transcript = Transcript(segments=[Segment(0.0, 2.0, "spoken words", "Sasson")])
        rec.routes = [RouteMatch(profile_id=profile_id, confidence=0.9)]
        rec.compliance.governing_profile = profile_id
        rec.compliance.encrypt_at_rest = False
        rec.analyses = [ProfileAnalysis(profile_id=analysis_profile or profile_id, fields=fields)]
        self.db.upsert(rec)
        return rec

    def collect(self, **kw):
        return collect(self.cfg, self.db, self.archive, vault=self.vault, **kw)


@pytest.fixture
def bench(tmp_path, monkeypatch):
    b = Bench(tmp_path, monkeypatch)
    yield b
    b.close()


def _item(text, *, rid="rec_a", profile="insurance_agent", **kw) -> FollowUp:
    return FollowUp(id=stable_id(text, profile), text=text, profile_id=profile,
                    recording_id=rid, recording_ids=[rid], **kw)


# =========================================================================
# Inputs that are refused outright
# =========================================================================
def test_collect_refuses_a_status_it_does_not_know(bench):
    with pytest.raises(FollowUpError, match="unknown status 'pending'"):
        bench.collect(status="pending")


def test_set_status_refuses_a_status_it_does_not_know(bench):
    bench.add("rec_a", "insurance_agent", {"next_action": "Send two quote options"})
    items = bench.collect()
    with pytest.raises(FollowUpError, match="unknown status 'later'"):
        set_status(bench.cfg, bench.vault, items[0].id, "later", items=items)
    assert not state_path(bench.cfg).exists()


def test_an_empty_id_is_refused(bench):
    bench.add("rec_a", "insurance_agent", {"next_action": "Send two quote options"})
    with pytest.raises(FollowUpError, match="no follow-up id given"):
        set_status(bench.cfg, bench.vault, "   ", "done", items=bench.collect())


def test_a_prefix_matching_several_collected_items_is_refused(bench):
    bench.add("rec_a", "insurance_agent", {"next_action": "Send two quote options"})
    bench.add("rec_b", "insurance_agent", {"next_action": "Order the paramed"})
    items = bench.collect()
    with pytest.raises(FollowUpError, match="matches 2 follow-ups") as exc:
        set_status(bench.cfg, bench.vault, "fu_", "done", items=items)
    assert "Use more of the id" in str(exc.value)
    assert not state_path(bench.cfg).exists()


def test_render_and_draft_refuse_formats_they_cannot_produce(bench):
    with pytest.raises(FollowUpError, match="unknown format 'docx'"):
        draft([_item("x")], bench.cfg, fmt="docx", use_llm=False)
    with pytest.raises(FollowUpError, match="unknown format 'pdf'"):
        render([], fmt="pdf")


# =========================================================================
# Shapes that must not become a follow-up
# =========================================================================
def test_an_analysis_from_a_profile_that_no_longer_exists_contributes_nothing(bench):
    bench.add("rec_a", "insurance_agent", {"next_action": "Call the retired client"},
              analysis_profile="retired_profile")
    assert bench.collect() == []


def test_a_boolean_in_a_commitment_field_is_a_flag_not_a_promise(bench):
    bench.add("rec_a", "insurance_agent", {"next_action": True})
    assert bench.collect() == []


def test_an_object_with_no_body_key_is_rendered_rather_than_dropped(bench):
    """An ugly line beats a silently lost one, the digest's rule."""
    bench.add("rec_a", "father", {"promises_i_made": [{"when": "tonight"}]})
    items = bench.collect(include_personal=True)
    assert len(items) == 1
    assert "tonight" in items[0].text


# =========================================================================
# Merging and ageing
# =========================================================================
def test_a_later_mention_moves_last_seen_forward_but_not_first_seen():
    merged = {}
    first = _item("send the quotes", first_seen="2026-01-01", last_seen="2026-01-01")
    later = _item("send the quotes", rid="rec_b", first_seen="2026-01-09", last_seen="2026-01-09")
    _merge(merged, first)
    _merge(merged, later)

    only = merged[first.id]
    assert only.mentions == 2
    assert only.first_seen == "2026-01-01" and only.recording_id == "rec_a"
    assert only.last_seen == "2026-01-09"
    assert only.recording_ids == ["rec_a", "rec_b"]


def test_age_is_zero_when_the_date_is_missing_or_malformed():
    assert _item("x", first_seen="").age_days == 0
    assert _item("x", first_seen="not-a-date").age_days == 0
    assert _item("x", first_seen="2026-13-45").age_days == 0


# =========================================================================
# The state file in every condition
# =========================================================================
def test_a_state_file_that_cannot_be_read_stops_the_run(bench):
    bench.add("rec_a", "insurance_agent", {"next_action": "Send two quote options"})
    state_path(bench.cfg).mkdir()      # exists, but is not something read_bytes can open
    with pytest.raises(FollowUpError, match="cannot read"):
        bench.collect()


def test_a_state_file_keyed_under_another_passphrase_will_not_decrypt(bench, monkeypatch):
    bench.add("rec_a", "insurance_agent", {"next_action": "Send two quote options"})
    items = bench.collect()
    set_status(bench.cfg, bench.vault, items[0].id, "done", items=items)

    monkeypatch.setenv("PLAUD_BRIDGE_PASSPHRASE", "a-completely-different-passphrase")
    with pytest.raises(FollowUpError, match="will not decrypt"):
        bench.collect()


def test_a_plaintext_state_file_is_honoured(bench):
    """The no-vault fallback has to be readable by the same code that wrote it."""
    bench.add("rec_a", "insurance_agent", {"next_action": "Send two quote options"})
    fid = bench.collect()[0].id
    state_path(bench.cfg).write_text(json.dumps(
        {"version": 1, "items": {fid: {"status": "dropped", "recording_id": "rec_a"}}}
    ), encoding="utf-8")
    assert [i.status for i in bench.collect()] == ["dropped"]


def test_a_state_file_that_is_not_json_is_refused_with_advice(bench):
    bench.add("rec_a", "insurance_agent", {"next_action": "Send two quote options"})
    state_path(bench.cfg).write_text("{not json", encoding="utf-8")
    with pytest.raises(FollowUpError, match="is not valid JSON") as exc:
        bench.collect()
    assert "Move it aside" in str(exc.value)


def test_a_state_file_holding_the_wrong_shape_reads_as_no_statuses(bench):
    bench.add("rec_a", "insurance_agent", {"next_action": "Send two quote options"})
    state_path(bench.cfg).write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")
    assert [i.status for i in bench.collect()] == ["open"]


def test_a_chmod_failure_does_not_lose_the_status(bench, monkeypatch):
    def refuse(*_a, **_k):
        raise OSError("chmod is not supported here")

    monkeypatch.setattr(os, "chmod", refuse)
    bench.add("rec_a", "insurance_agent", {"next_action": "Send two quote options"})
    items = bench.collect()
    set_status(bench.cfg, bench.vault, items[0].id, "done", items=items)
    assert state_path(bench.cfg).exists()
    assert [i.status for i in bench.collect()] == ["done"]


# =========================================================================
# Resolving from the state file alone
# =========================================================================
def test_a_status_can_be_changed_from_the_state_file_without_recollecting(bench):
    bench.add("rec_a", "insurance_agent", {"next_action": "Send two quote options"})
    items = bench.collect()
    set_status(bench.cfg, bench.vault, items[0].id, "done", items=items)

    reopened = set_status(bench.cfg, bench.vault, items[0].short_id, "open")
    assert reopened.id == items[0].id
    assert reopened.text == "", "the state file deliberately holds no wording"
    assert reopened.profile_id == "insurance_agent"
    assert reopened.recording_id == "rec_a"
    assert reopened.status == "open"
    assert [i.status for i in bench.collect()] == ["open"]


def test_a_prefix_matching_several_saved_entries_is_refused(bench):
    bench.add("rec_a", "insurance_agent", {"next_action": "Send two quote options"})
    bench.add("rec_b", "insurance_agent", {"next_action": "Order the paramed"})
    items = bench.collect()
    for item in items:
        set_status(bench.cfg, bench.vault, item.id, "done", items=items)

    with pytest.raises(FollowUpError, match="matches 2 known follow-ups"):
        set_status(bench.cfg, bench.vault, "fu_", "open")


# =========================================================================
# forget_recording
# =========================================================================
def test_forgetting_one_recording_leaves_statuses_from_other_recordings_alone(bench):
    bench.add("rec_a", "insurance_agent", {"next_action": "Send two quote options"})
    bench.add("rec_b", "insurance_agent", {"next_action": "Order the paramed"})
    items = bench.collect()
    for item in items:
        set_status(bench.cfg, bench.vault, item.id, "done", items=items)
    a = next(i for i in items if i.recording_id == "rec_a")
    b = next(i for i in items if i.recording_id == "rec_b")

    assert forget_recording(bench.cfg, bench.vault, "rec_a") == [a.id]
    bench.db.delete_recording("rec_a")
    remaining = bench.collect()
    assert [(i.id, i.status) for i in remaining] == [(b.id, "done")]


def test_forgetting_a_recording_no_status_names_leaves_the_file_byte_for_byte(bench):
    bench.add("rec_a", "insurance_agent", {"next_action": "Send two quote options"})
    items = bench.collect()
    set_status(bench.cfg, bench.vault, items[0].id, "done", items=items)
    before = state_path(bench.cfg).read_bytes()

    assert forget_recording(bench.cfg, bench.vault, "rec_never_seen") == []
    assert state_path(bench.cfg).read_bytes() == before


def test_forgetting_with_no_state_file_is_a_no_op(bench):
    assert forget_recording(bench.cfg, bench.vault, "rec_a") == []
    assert not state_path(bench.cfg).exists()


# =========================================================================
# The worklist
# =========================================================================
def test_the_worklist_shows_the_mention_history_and_a_closed_section():
    open_item = _item("Send two quote options", first_seen="2026-01-01", last_seen="2026-01-09",
                      mentions=3, due="Thursday", counterparty="Sasson")
    closed_item = _item("Order the paramed", status="done")

    out = render([open_item, closed_item])
    assert "1 open" in out and "1 closed" in out
    assert "last mentioned 2026-01-09" in out
    assert "3 mentions" in out
    assert "- Due: Thursday" in out
    assert "- Said by: Sasson" in out
    assert "## Closed" in out
    assert f"- **done** — Order the paramed  (`{closed_item.short_id}`)" in out
    assert out.index("## Still open") < out.index("## Closed")


def test_a_worklist_of_only_closed_items_has_no_open_section():
    out = render([_item("Order the paramed", status="dropped")])
    assert "0 open" in out and "1 closed" in out
    assert "oldest" not in out
    assert "## Still open" not in out
    assert "**dropped**" in out


# =========================================================================
# Drafting
# =========================================================================
def test_a_single_follow_up_can_be_drafted_on_its_own(bench):
    path = draft(_item("Send two quote options"), bench.cfg, use_llm=False)
    body = path.read_text(encoding="utf-8")
    assert "Send two quote options" in body
    assert "Following up: Send two quote options" in body
    assert path.name.startswith("DRAFT-") and "insurance-agent-send-two-quote-options" in path.name


def test_drafting_from_a_recording_id_needs_the_index_and_the_archive(bench):
    with pytest.raises(FollowUpError, match="pass db= and archive="):
        draft("rec_a", bench.cfg, use_llm=False)
    with pytest.raises(FollowUpError, match="pass db= and archive="):
        draft("rec_a", bench.cfg, db=bench.db, use_llm=False)


def test_drafting_nothing_is_refused_rather_than_writing_an_empty_message(bench):
    with pytest.raises(FollowUpError, match="nothing to draft"):
        draft([], bench.cfg, use_llm=False)
    assert not (bench.cfg.path("outbox") / "drafts").exists()


def test_a_draft_is_capped_at_the_oldest_follow_ups(tmp_path, monkeypatch):
    b = Bench(tmp_path, monkeypatch, overrides={"followups": {"max_per_draft": 1}})
    try:
        b.add("rec_old", "insurance_agent", {"next_action": "Call Marcus back"}, days_ago=10)
        b.add("rec_new", "insurance_agent", {"next_action": "Send the illustration"}, days_ago=1)
        items = b.collect()
        assert len(items) == 2

        body = draft(items, b.cfg, use_llm=False).read_text(encoding="utf-8")
        assert "Call Marcus back" in body
        assert "Send the illustration" not in body
        assert "Built from 1 follow-up(s), traced to: rec_old." in body
    finally:
        b.close()


def test_a_spend_record_that_fails_does_not_lose_the_model_phrased_draft(bench, monkeypatch):
    def phrase(cfg, system, user, local_only=False, max_tokens=None):
        return ({"subject": "Two quotes", "body": "Hi Marcus, as promised."},
                LLMResponse(provider="stub", model="stub-1", cost_usd=0.01))

    def refuse(*_a, **_k):
        raise RuntimeError("spend table is locked")

    monkeypatch.setattr("plaud_bridge.followups.complete_json", phrase)
    monkeypatch.setattr(bench.db, "record_spend", refuse)
    bench.add("rec_a", "sales_trainer", {"next_action": "Send the roleplay recap"})

    path = draft(bench.collect(), bench.cfg, db=bench.db, use_llm=True)
    body = path.read_text(encoding="utf-8")
    assert "Hi Marcus, as promised." in body
    assert "stub/stub-1" in body
    assert bench.db.audit_log(action="followup_draft", limit=5)


def test_after_ninety_nine_drafts_of_one_name_the_next_is_refused(tmp_path):
    base = tmp_path / "DRAFT-2026-01-01-x.draft.md"
    base.write_text("1")
    for n in range(2, 100):
        (tmp_path / f"DRAFT-2026-01-01-x-{n}.draft.md").write_text(str(n))
    with pytest.raises(FollowUpError, match="already 99 drafts"):
        _unclaimed(base)
