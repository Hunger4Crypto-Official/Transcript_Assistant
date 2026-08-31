"""
The brief: a synthesis across the archive that is not allowed to invent.

The deterministic half is tested against recordings processed through the real
pipeline, so the counts pin down what a person would actually see. The
narrative half is tested through a fake `complete_json`, because the interesting
claims are about what happens AROUND the model: a fabricated quote must be
dropped and counted, a citation of an unsent recording must be dropped, a
missing model must produce the labelled template rather than an error, and
personal content must force locality. Several of these are mutation-style on
purpose -- remove the validation and the fabrication renders, and the test
goes red.
"""

from __future__ import annotations

import re

import pytest

from _fixtures import CLIENT_CALL, FAMILY_DINNER, build_sandbox, drop
from plaud_bridge.archive import Archive
from plaud_bridge.brief import Brief, BriefError, build_brief, render
from plaud_bridge.cli import main
from plaud_bridge.db import Database
from plaud_bridge.llm.base import LLMError, LLMResponse

# A sentence the stub extractor attributes to the insurance recording, so a
# receipt quoting it is genuine, and one nobody ever said, so a receipt quoting
# it is a fabrication. The test for the difference is the point of the feature.
REAL_QUOTE = "I'll have them to you by Thursday"
FAKE_QUOTE = "I guarantee the premium will never rise"


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Two recordings processed through the real CLI: one work, one personal."""
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    drop(cfg, "client.txt", CLIENT_CALL)
    drop(cfg, "dinner.txt", FAMILY_DINNER)
    assert main(["--config", str(tmp_path / "config"), "run"]) == 0
    return cfg


class Bench:
    """cfg + db + archive, closed together so no test leaks a connection."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.db = Database(cfg.path("database"))
        self.archive = Archive(cfg, self.db)

    def build(self, **kw) -> Brief:
        return build_brief(self.cfg, self.db, self.archive,
                           vault=self.archive.vault, **kw)

    def close(self):
        self.db.close()


@pytest.fixture
def bench(sandbox):
    b = Bench(sandbox)
    try:
        yield b
    finally:
        b.close()


class BriefLLM:
    """
    Stand-in for `complete_json` that keeps everything it was given.

    The recorded `user` string is the actual subject of half these tests: it is
    the only place you can see what material left this module for a model.
    """

    def __init__(self, payload=None, cost_usd=0.0):
        self.calls: list[dict] = []
        self.payload = payload
        self.cost_usd = cost_usd

    def __call__(self, cfg, system, user, local_only=False, max_tokens=None):
        self.calls.append({"system": system, "user": user, "local_only": local_only})
        body = self.payload(user) if callable(self.payload) else self.payload
        if body is None:
            body = {"the_week": "A steady week.", "aging": "One promise is aging.",
                    "people": "Marcus is waiting on quotes.", "next": "Send them.",
                    "receipts": []}
        return body, LLMResponse(provider="stub", model="stub-1", cost_usd=self.cost_usd)


def _install(monkeypatch, fake: BriefLLM) -> BriefLLM:
    monkeypatch.setattr("plaud_bridge.brief.complete_json", fake)
    return fake


def _no_model(monkeypatch):
    def down(*_a, **_k):
        raise LLMError("no LLM provider available (test)")
    monkeypatch.setattr("plaud_bridge.brief.complete_json", down)


def _sent_recording(user: str) -> str:
    """A recording id that really was in the material sent to the model."""
    found = re.search(r"RECORDING (\S+)", user)
    assert found, f"no recording block in the material:\n{user}"
    return found.group(1)


# =========================================================================
# The skeleton is the brief
# =========================================================================
def test_the_skeleton_is_correct_with_no_model_at_all(bench, monkeypatch):
    """Counts, follow-ups, and receipts-free honesty, from real processed data."""
    _no_model(monkeypatch)
    brief = bench.build()

    profiles = {p["profile_id"]: p for p in brief.profiles}
    assert "insurance_agent" in profiles
    assert profiles["insurance_agent"]["recordings"] == 1
    assert brief.recordings >= 1
    assert brief.recording_ids, "the skeleton lost track of which recordings it read"

    # The open follow-ups are present and ordered worst-first, which is what
    # makes the Aging section mean something.
    assert brief.followups, "no open follow-up was collected from the analyses"
    ages = [i["age_days"] for i in brief.followups]
    assert ages == sorted(ages, reverse=True)
    assert any("Thursday" in i["text"] for i in brief.followups)

    assert brief.quarantined == 0
    assert "total_cost_usd" in brief.spend
    assert not brief.narrated
    assert not brief.receipts


def test_no_model_renders_the_labelled_template_not_an_error(bench, monkeypatch):
    """A missing model is a labelled outcome, never a traceback or a blank."""
    _no_model(monkeypatch)
    brief = bench.build()
    out = render(brief)

    assert "Assembled, not narrated" in out
    for heading in ("The week", "Aging", "People waiting on you", "Next"):
        assert heading in out
    # The template really carries the skeleton: the oldest promise appears.
    assert "Thursday" in out
    # And nothing was spent or recorded as spent.
    assert "brief" not in bench.db.stats()["by_source"]


def test_an_empty_archive_is_a_calm_brief_and_no_model_is_asked(tmp_path, monkeypatch):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    fake = _install(monkeypatch, BriefLLM())
    bench = Bench(cfg)
    try:
        brief = bench.build()
    finally:
        bench.close()

    assert not fake.calls, "an empty window was still sent to a model"
    assert not brief.narrated
    assert "Nothing was recorded" in render(brief)


# =========================================================================
# Honesty: quotes are validated against what was actually sent
# =========================================================================
def test_a_fabricated_quote_is_dropped_counted_and_never_rendered(bench, monkeypatch):
    """
    Mutation-style: if `_validate_receipts` stopped checking quotes, the
    fabricated sentence would be kept, rendered, and this test would fail on
    all three assertions about it.
    """
    def payload(user):
        rid = _sent_recording(user)
        return {
            "the_week": "w", "aging": "a", "people": "p", "next": "n",
            "receipts": [
                {"recording_id": rid, "quote": REAL_QUOTE},
                {"recording_id": rid, "quote": FAKE_QUOTE},
            ],
        }

    _install(monkeypatch, BriefLLM(payload))
    brief = bench.build()

    assert brief.dropped_quotes == 1
    kept = [r["quote"] for r in brief.receipts]
    assert kept == [REAL_QUOTE], "the genuine quote must survive validation"

    out = render(brief)
    assert FAKE_QUOTE not in out
    assert REAL_QUOTE in out
    assert "were dropped" in out, "the drop has to be reported, not silent"


def test_a_receipt_citing_an_unsent_recording_is_dropped(bench, monkeypatch):
    def payload(_user):
        return {
            "the_week": "w", "aging": "a", "people": "p", "next": "n",
            "receipts": [{"recording_id": "rec_never_sent", "quote": REAL_QUOTE}],
        }

    _install(monkeypatch, BriefLLM(payload))
    brief = bench.build()

    assert brief.dropped_recordings == 1
    assert brief.receipts == []
    assert "rec_never_sent" not in render(brief)


def test_a_missing_key_falls_back_to_the_template_line(bench, monkeypatch):
    """The output contract is enforced by filling gaps, never by raising."""
    _install(monkeypatch, BriefLLM({"the_week": "Only this arrived."}))
    brief = bench.build()

    assert brief.narrated
    assert brief.sections["the_week"] == "Only this arrived."
    assert brief.sections["aging"], "a missing section must be filled by the template"
    assert "returned nothing for" in brief.note


# =========================================================================
# Personal content
# =========================================================================
def test_personal_profiles_stay_out_by_default(bench, monkeypatch):
    fake = _install(monkeypatch, BriefLLM())
    brief = bench.build()

    assert "father" not in [p["profile_id"] for p in brief.profiles]
    assert fake.calls and "permission slip" not in fake.calls[0]["user"], (
        "a personal analysis reached the material sent to the model"
    )
    assert "permission slip" not in render(brief)


def test_personal_appears_with_the_flag(bench, monkeypatch):
    fake = _install(monkeypatch, BriefLLM())
    brief = bench.build(include_personal=True)

    assert "father" in [p["profile_id"] for p in brief.profiles]
    assert "permission slip" in fake.calls[0]["user"]


def test_a_personal_brief_only_ever_sees_local_only_true(bench, monkeypatch):
    """
    The privacy guarantee, in the shape test_privacy_guarantees uses: with a
    personal profile in the gathered material, no call that leaves this module
    may permit a cloud provider.
    """
    fake = _install(monkeypatch, BriefLLM())
    brief = bench.build(include_personal=True)

    assert fake.calls, "no model call happened, so nothing was proven"
    leaked = [c for c in fake.calls if not c["local_only"]]
    assert not leaked, (
        f"a brief holding personal content reached a call that permitted cloud: {leaked}"
    )
    assert brief.local_only


# =========================================================================
# Spend and rendering
# =========================================================================
def test_narrating_a_brief_records_what_it_cost(bench, monkeypatch):
    """ADR-014: spend is counted where it is incurred, under its own source."""
    _install(monkeypatch, BriefLLM(cost_usd=0.0123))
    brief = bench.build()

    by_source = bench.db.stats()["by_source"]
    assert "brief" in by_source
    assert by_source["brief"]["cost_usd"] == pytest.approx(0.0123)
    assert brief.cost_usd == pytest.approx(0.0123)
    assert brief.provider == "stub"


def test_html_is_rendered_from_the_same_markdown(bench, monkeypatch):
    _install(monkeypatch, BriefLLM())
    brief = bench.build()

    markdown = render(brief)
    page = render(brief, fmt="html")
    assert "A steady week." in markdown
    assert "A steady week." in page
    assert "<h2>" in page

    with pytest.raises(BriefError):
        render(brief, fmt="xml")


# =========================================================================
# The CLI route
# =========================================================================
def test_the_cli_route_prints_and_writes(sandbox, tmp_path, capsys):
    # No stub is installed, so the real provider chain runs and (with no usable
    # provider in the sandbox) the assembled template is the brief. Exit 0: a
    # memo built entirely from real data is a success, not a failure.
    assert main(["--config", str(sandbox.root / "config"), "brief"]) == 0
    out = capsys.readouterr().out
    assert "# Brief" in out
    assert "Assembled, not narrated" in out

    dest = tmp_path / "brief.html"
    assert main(["--config", str(sandbox.root / "config"), "brief",
                 "--format", "html", "--out", str(dest)]) == 0
    assert "<h2>" in dest.read_text(encoding="utf-8")
