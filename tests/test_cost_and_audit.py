"""
Spend accounting and the audit trail.

Two guarantees are pinned here:

1. LLM spend is counted. It used to be invisible, which meant
   `cost.halt_usd_per_run` was really an ASR-only ceiling and `status` under
   reported. A guardrail that cannot see the largest charge is decoration.
2. The audit log is readable. COMPLIANCE.md section 8 commits to recording
   every ingest, route, compliance decision, quarantine, release, and deletion.
"""

import json
from pathlib import Path

import pytest

from _fixtures import CLIENT_CALL, StubLLM, build_sandbox, drop
from plaud_bridge.db import Database
from plaud_bridge.llm.anthropic_provider import AnthropicLLM
from plaud_bridge.llm.openai_compat_provider import OpenAICompatLLM
from plaud_bridge.pipeline import Pipeline

CALL_COST = 0.01


# =========================================================================
# Pricing
# =========================================================================
def test_price_uses_the_configured_token_rates(sandbox):
    cfg, _ = sandbox
    provider = AnthropicLLM(cfg)

    # config ships 3.00 in / 15.00 out per million
    assert provider.price(1_000_000, 1_000_000) == pytest.approx(18.00)
    assert provider.price(500_000, 0) == pytest.approx(1.50)
    assert provider.price(0, 0) == 0.0


def test_a_provider_with_no_configured_rate_reports_zero_rather_than_guessing(sandbox):
    cfg, _ = sandbox
    provider = OpenAICompatLLM(cfg, "a_provider_nobody_configured", is_cloud=True)
    assert provider.price(10_000_000, 10_000_000) == 0.0


def test_local_provider_is_free(sandbox):
    cfg, _ = sandbox
    assert OpenAICompatLLM(cfg, "local", is_cloud=False).price(9_000_000, 9_000_000) == 0.0


def test_anthropic_bills_cache_tokens_rather_than_dropping_them(sandbox, monkeypatch):
    cfg, _ = sandbox
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    monkeypatch.setattr(
        "plaud_bridge.llm.anthropic_provider.post_json",
        lambda *a, **k: {
            "content": [{"type": "text", "text": '{"ok": true}'}],
            "usage": {
                "input_tokens": 1_000_000,
                "cache_creation_input_tokens": 500_000,
                "cache_read_input_tokens": 500_000,
                "output_tokens": 1_000_000,
            },
        },
    )

    response = AnthropicLLM(cfg).complete("sys", "user")
    assert response.input_tokens == 2_000_000
    # 2M in at $3 + 1M out at $15
    assert response.cost_usd == pytest.approx(21.00)


# =========================================================================
# Spend reaches the guardrail
# =========================================================================
def test_routing_and_analysis_spend_both_land_on_the_recording(tmp_path, monkeypatch):
    cfg, _ = build_sandbox(tmp_path, monkeypatch, stub=StubLLM(cost_usd=CALL_COST))
    drop(cfg, "client-marcus.txt", CLIENT_CALL)

    pipe = Pipeline(cfg)
    try:
        stats = pipe.run()
        assert stats.processed == 1

        payload = json.loads(pipe.db.query()[0]["payload_json"])
        # One routing call plus one call per profile the recording matched.
        expected = CALL_COST * (1 + len(payload["analyses"]))
        assert payload["total_cost_usd"] == pytest.approx(expected)
        assert stats.cost_usd == pytest.approx(expected)
    finally:
        pipe.close()


def test_asking_a_question_is_spend_that_status_can_see(tmp_path, monkeypatch):
    """
    ADR-014 says spend is counted wherever it is incurred. `ask` had a cost and
    no recording to hang it on, so it was spent without appearing anywhere a
    person would look -- fifty questions against a cloud model, and `status`
    still reported only what ingestion had cost.
    """
    cfg, _ = build_sandbox(tmp_path, monkeypatch, stub=StubLLM(cost_usd=CALL_COST))
    drop(cfg, "client-marcus.txt", CLIENT_CALL)
    pipe = Pipeline(cfg)
    try:
        pipe.run()
        pipeline_cost = pipe.db.stats()["total_cost_usd"]
    finally:
        pipe.close()

    from plaud_bridge import ask as ask_module
    from plaud_bridge.archive import Archive
    from plaud_bridge.llm.base import LLMResponse

    def fake(cfg_, system, user, local_only=False, max_tokens=None):
        return ({"answer": "Two quote options by Thursday.", "citations": [],
                 "confidence": "high", "unanswered": ""},
                LLMResponse(text="", provider="stub", model="stub-1",
                            cost_usd=CALL_COST))

    monkeypatch.setattr(ask_module, "complete_json", fake)

    db = Database(cfg.path("database"))
    try:
        ask_module.ask("what did I promise Marcus?", cfg, db, Archive(cfg, db))
        stats = db.stats()
        assert stats["by_source"]["ask"]["calls"] == 1
        assert stats["by_source"]["ask"]["cost_usd"] == pytest.approx(CALL_COST)
        assert stats["pipeline_cost_usd"] == pytest.approx(pipeline_cost)
        assert stats["total_cost_usd"] == pytest.approx(pipeline_cost + CALL_COST)
    finally:
        db.close()


def test_a_quarantined_recording_still_counts_toward_run_spend(tmp_path, monkeypatch):
    """It routed before it was stopped. That call was billed."""
    cfg, _ = build_sandbox(tmp_path, monkeypatch, stub=StubLLM(cost_usd=CALL_COST))
    drop(cfg, "no-consent.txt", CLIENT_CALL.split("\n", 2)[2])

    pipe = Pipeline(cfg)
    try:
        stats = pipe.run()
        assert stats.quarantined == 1
        assert stats.cost_usd == pytest.approx(CALL_COST), (
            "a quarantined recording's routing call vanished from the run total"
        )
    finally:
        pipe.close()


def test_llm_spend_can_trip_the_halt_threshold(tmp_path, monkeypatch):
    cfg, _ = build_sandbox(
        tmp_path, monkeypatch,
        stub=StubLLM(cost_usd=CALL_COST),
        overrides={"cost": {"warn_usd_per_run": 0.001, "halt_usd_per_run": 0.005}},
    )
    drop(cfg, "client-marcus.txt", CLIENT_CALL)

    pipe = Pipeline(cfg)
    try:
        stats = pipe.run()
        assert stats.failed == 1
        assert stats.processed == 0

        payload = json.loads(pipe.db.query()[0]["payload_json"])
        assert any("halt_usd_per_run" in e for e in payload["errors"]), payload["errors"]
    finally:
        pipe.close()


# =========================================================================
# Audit trail
# =========================================================================
def test_the_pipeline_writes_a_readable_audit_trail(sandbox):
    cfg, _ = sandbox
    drop(cfg, "client-marcus.txt", CLIENT_CALL)

    pipe = Pipeline(cfg)
    try:
        pipe.run()

        actions = {row["action"] for row in pipe.db.audit_log(limit=200)}
        for expected in ("ingest", "route", "compliance", "run_complete"):
            assert expected in actions, f"'{expected}' never reached the audit log"

        rec_id = pipe.db.query()[0]["id"]
        scoped = pipe.db.audit_log(recording_id=rec_id)
        assert scoped
        assert all(row["recording_id"] == rec_id for row in scoped)

        only_routes = pipe.db.audit_log(action="route")
        assert only_routes and all(r["action"] == "route" for r in only_routes)
    finally:
        pipe.close()


def test_human_actions_are_distinguishable_from_pipeline_actions(sandbox):
    cfg, _ = sandbox
    drop(cfg, "no-consent.txt", CLIENT_CALL.split("\n", 2)[2])

    pipe = Pipeline(cfg)
    try:
        pipe.run()
        rec_id = pipe.db.query()[0]["id"]
        pipe.db.audit("quarantine_release", "released after human review", rec_id, actor="human")

        human = pipe.db.audit_log(actor="human")
        assert len(human) == 1
        assert human[0]["action"] == "quarantine_release"

        assert all(r["actor"] == "pipeline" for r in pipe.db.audit_log(actor="pipeline"))
    finally:
        pipe.close()


def test_audit_log_is_newest_first_and_respects_the_limit(sandbox):
    cfg, _ = sandbox
    pipe = Pipeline(cfg)
    try:
        for i in range(5):
            pipe.db.audit("marker", f"entry {i}")
        rows = pipe.db.audit_log(action="marker", limit=3)
        assert len(rows) == 3
        assert [r["detail"] for r in rows] == ["entry 4", "entry 3", "entry 2"]
    finally:
        pipe.close()


# =========================================================================
# The review cadence
# =========================================================================
def test_review_reports_standing_consent_as_due_and_then_satisfied(sandbox, capsys):
    """
    `reaffirm_every_days` was parsed and never read. father.yaml promises a
    prompt "on first run each month"; this is where that promise is kept.
    """

    from plaud_bridge.cli import build_parser, cmd_review

    cfg, _ = sandbox
    args = build_parser().parse_args(["--config", str(cfg.root / "config"), "review"])

    assert cmd_review(args) == 0
    first = capsys.readouterr().out
    assert "[DUE] Father" in first
    assert "never reaffirmed" in first
    assert "review --reaffirm father" in first

    db = Database(cfg.path("database"))
    try:
        db.audit("consent_reaffirm", "father", actor="human")
    finally:
        db.close()

    assert cmd_review(args) == 0
    second = capsys.readouterr().out
    assert "[ ok] Father" in second
    assert "[DUE] Husband" in second, "reaffirming one profile satisfied another"


def test_review_uses_the_most_recent_reaffirmation(sandbox, capsys):
    from datetime import datetime, timedelta, timezone

    from plaud_bridge.cli import build_parser, cmd_review

    cfg, _ = sandbox
    db = Database(cfg.path("database"))
    try:
        stale = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
        with db.tx() as cur:
            cur.execute(
                "INSERT INTO audit(at,recording_id,action,detail,actor) VALUES (?,?,?,?,?)",
                (stale, None, "consent_reaffirm", "father", "human"),
            )
        db.audit("consent_reaffirm", "father", actor="human")
    finally:
        db.close()

    args = build_parser().parse_args(["--config", str(cfg.root / "config"), "review"])
    assert cmd_review(args) == 0
    out = capsys.readouterr().out
    assert "[ ok] Father" in out, "an old reaffirmation masked a current one"


def test_review_surfaces_expired_artifacts_without_deleting_them(sandbox, capsys):
    from datetime import datetime, timedelta, timezone

    from plaud_bridge.cli import build_parser, cmd_review

    cfg, _ = sandbox
    drop(cfg, "client-marcus.txt", CLIENT_CALL)
    pipe = Pipeline(cfg)
    try:
        pipe.run()
        rec_id = pipe.db.query()[0]["id"]
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        with pipe.db.tx() as cur:
            cur.execute("UPDATE artifacts SET expires_at=? WHERE recording_id=?", (past, rec_id))
        paths = [Path(r["path"]) for r in pipe.db.expired_artifacts()]
    finally:
        pipe.close()

    args = build_parser().parse_args(["--config", str(cfg.root / "config"), "review"])
    assert cmd_review(args) == 0
    out = capsys.readouterr().out
    assert "past their expiry" in out
    assert "retention --execute" in out
    assert all(p.exists() for p in paths), "review deleted something; it only reports"
