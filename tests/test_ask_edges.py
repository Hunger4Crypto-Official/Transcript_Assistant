"""
Ask at the edges.

test_ask.py pins retrieval, locality, redaction and citation checking. These
pin the branches around them: what the rendered answer says about cost and
what it could not answer, how excerpts are cut and labelled, the quoted-phrase
pass and what it reports about files it could not open, the caveats the model
is given when the scan was cut short, and the housekeeping failures -- spend
and audit writes -- that must never cost the person their answer.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

import pytest

from _fixtures import build_sandbox
from plaud_bridge.archive import Archive
from plaud_bridge.ask import (
    Answer,
    _age_days,
    _analysis_excerpts,
    _overlap,
    _redaction_required,
    _render_field,
    ask,
    retrieve,
)
from plaud_bridge.db import Database
from plaud_bridge.llm.base import LLMResponse
from plaud_bridge.models import (
    ComplianceVerdict,
    ProfileAnalysis,
    Recording,
    RouteMatch,
    Segment,
    Stage,
    Transcript,
    utc_now,
)
from plaud_bridge.storage import Vault


def _index(cfg, db, name, segments, profiles, *, days_ago=1, encrypt=False, analyses=None):
    """One finished recording in the index, from explicit segments."""
    rec = Recording(
        source_name=name, source_path=str(cfg.path("inbox") / name),
        content_hash=hashlib.sha256(name.encode()).hexdigest(), kind="text",
        stage=Stage.COMPLETE, recorded_at=utc_now() - timedelta(days=days_ago),
    )
    rec.transcript = Transcript(segments=segments, duration_seconds=segments[-1].end)
    rec.routes = [RouteMatch(profile_id=pid, confidence=0.9) for pid in profiles]
    governing = cfg.strictest(list(profiles))
    rec.compliance = ComplianceVerdict(
        governing_profile=governing.id, governing_sensitivity=governing.sensitivity,
        encrypt_at_rest=encrypt, force_local_processing=not governing.allow_cloud_llm,
    )
    for pid, fields in (analyses or {}).items():
        rec.analyses.append(ProfileAnalysis(profile_id=pid, fields=fields))
    if encrypt:
        path = Vault(cfg.path("vault")).write(f"{rec.id}/analysis", rec.to_json(), rec.id)
        rec.artifact_paths["analysis"] = str(path)
    db.upsert(rec)
    return rec.id


def _lines(*pairs) -> list[Segment]:
    out, cursor = [], 0.0
    for speaker, text in pairs:
        out.append(Segment(cursor, cursor + 6.0, text, speaker))
        cursor += 6.0
    return out


class Bridge:
    def __init__(self, cfg):
        self.cfg = cfg
        self.db = Database(cfg.path("database"))
        self.archive = Archive(cfg, self.db)

    def ask(self, question, **kw):
        return ask(question, self.cfg, self.db, self.archive, **kw)

    def retrieve(self, question, **kw):
        return retrieve(question, self.cfg, self.db, self.archive, **kw)

    def index(self, name, segments, profiles, **kw):
        return _index(self.cfg, self.db, name, segments, profiles, **kw)

    def close(self):
        self.db.close()


@pytest.fixture
def bridge(tmp_path, monkeypatch):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    b = Bridge(cfg)
    try:
        yield b
    finally:
        b.close()


class FakeLLM:
    def __init__(self, payload=None, cost_usd=0.0):
        self.calls = []
        self.payload = payload
        self.cost_usd = cost_usd

    def __call__(self, cfg, system, user, local_only=False, max_tokens=None):
        self.calls.append({"system": system, "user": user, "local_only": local_only})
        body = self.payload(user) if callable(self.payload) else self.payload
        if body is None:
            body = {"answer": "A plausible answer.", "citations": [],
                    "confidence": "medium", "unanswered": ""}
        return body, LLMResponse(provider="fake", model="fake", cost_usd=self.cost_usd)


def _install(monkeypatch, fake):
    monkeypatch.setattr("plaud_bridge.ask.complete_json", fake)
    return fake


def _first_header(user: str) -> tuple[str, str]:
    for line in user.splitlines():
        line = line.strip()
        if line.startswith("[rec_") and "@" in line:
            inside = line[1:line.index("]")]
            rec_id, _, stamp = inside.partition(" @ ")
            return rec_id.strip(), stamp.strip()
    raise AssertionError(f"no excerpt header in the bundle:\n{user}")


HENDERSON = _lines(
    ("Sasson", "Good to see you both. Let's talk about the Henderson term policy."),
    ("Wife", "We were told the premium is locked for twenty years."),
    ("Sasson", "I promised you I would confirm the conversion rider in writing."),
    ("Henderson", "That was the part we could not remember."),
    ("Sasson", "I will email the conversion language on Monday. That is a promise."),
)


# =========================================================================
# What the answer says about itself
# =========================================================================
def test_retrieval_calls_itself_complete_only_when_every_recording_was_opened(
    bridge, monkeypatch
):
    bridge.index("henderson.txt", HENDERSON, ["insurance_agent"], encrypt=True, days_ago=0)
    bridge.index("henderson-2.txt", HENDERSON, ["insurance_agent"], days_ago=1)
    assert bridge.retrieve("conversion rider").complete is True

    bridge.cfg._d["ask"] = {"scan_limit": 1}
    assert bridge.retrieve("conversion rider").complete is False, "a cut-short scan is not complete"

    bridge.cfg._d["ask"] = {}
    monkeypatch.setenv("PLAUD_BRIDGE_PASSPHRASE", "a-completely-different-passphrase")
    found = bridge.retrieve("conversion rider")
    assert found.unopened and found.complete is False


def test_the_rendered_answer_shows_what_was_not_answered_and_what_it_cost():
    answer = Answer(text="You promised the rider in writing.", unanswered="the premium amount",
                    provider="fake/model", cost_usd=0.0123)
    out = answer.render()
    assert "Not answered: the premium amount" in out
    assert "fake/model" in out
    assert "$0.0123" in out
    assert "$" not in Answer(text="free").render(), "a zero cost printed a price"


def test_a_malformed_date_ages_to_zero_and_a_naive_one_is_read_as_utc():
    now = datetime(2026, 1, 2, tzinfo=timezone.utc)
    assert _age_days({"recorded_at": "garbage"}, now) == 0.0
    assert _age_days({"recorded_at": None, "ingested_at": "2026-01-01T00:00:00"}, now) == 1.0
    assert _age_days({"recorded_at": "2026-01-03T00:00:00"}, now) == 0.0, "the future is not negative"


# =========================================================================
# How excerpts are cut
# =========================================================================
def test_a_blank_analysis_field_and_a_flag_render_as_expected(bridge):
    record = {"analyses": [{"profile_id": "insurance_agent",
                            "fields": {"next_action": "   ", "open_questions": []}}]}
    row = {"id": "rec_x", "source_name": "x.txt", "recorded_at": None, "ingested_at": ""}
    assert _analysis_excerpts(bridge.cfg, record, row, ["anything"]) == []
    assert _render_field(True) == [("", "yes")]
    assert _render_field(False) == []


def test_excerpts_per_recording_caps_how_many_stretches_come_back(bridge):
    bridge.cfg._d["ask"] = {"excerpts_per_recording": 1, "excerpt_context": 0}
    bridge.index("policy.txt", _lines(
        ("Sasson", "the term policy first"),
        ("Marcus", "then the whole policy"),
        ("Sasson", "and the policy rider last"),
    ), ["sales_trainer"])

    found = bridge.retrieve("policy")
    transcript_excerpts = [e for e in found.candidates[0].excerpts if e.kind == "transcript"]
    assert len(transcript_excerpts) == 1
    assert found.candidates[0].matched_terms == ["polic"]


def test_empty_segments_beside_a_hit_do_not_pad_the_excerpt(bridge):
    bridge.index("gappy.txt", [
        Segment(0.0, 1.0, "", "Sasson"),
        Segment(1.0, 2.0, "we discussed the conversion rider", "Sasson"),
        Segment(2.0, 3.0, "   ", "Marcus"),
    ], ["sales_trainer"])
    excerpt = bridge.retrieve("conversion rider").candidates[0].excerpts[0]
    assert excerpt.text == "Sasson: we discussed the conversion rider"
    assert excerpt.speaker == "Sasson", "a blank line from another speaker unlabelled the window"


def test_consecutive_lines_from_one_speaker_are_labelled_once(bridge):
    bridge.index("run-on.txt", _lines(
        ("Sasson", "about the conversion rider"),
        ("Sasson", "and the premium after that"),
    ), ["sales_trainer"])
    excerpt = bridge.retrieve("conversion rider").candidates[0].excerpts[0]
    assert excerpt.text == "Sasson: about the conversion rider and the premium after that"
    assert excerpt.speaker == "Sasson"


def test_a_question_made_only_of_stopwords_retrieves_nothing_and_opens_nothing(bridge):
    bridge.index("henderson.txt", HENDERSON, ["insurance_agent"])
    found = bridge.retrieve("what did the")
    assert found.terms == [] and found.phrases == []
    assert found.candidates == [] and found.considered == 0 and found.total == 0


# =========================================================================
# Quoted phrases
# =========================================================================
def test_a_quoted_phrase_outranks_term_overlap_and_stays_out_of_personal_recordings(bridge):
    exact = bridge.index("exact.txt", _lines(
        ("Sasson", "we talked about the conversion rider today"),
    ), ["sales_trainer"], days_ago=5)
    scattered = bridge.index("scattered.txt", _lines(
        ("Sasson", "the rider was fine and the conversion took a while"),
        ("Marcus", "the rider again, and that conversion again"),
    ), ["sales_trainer"], days_ago=0)
    home = bridge.index("dinner.txt", _lines(
        ("Kid", "we talked about the conversion rider at dinner"),
    ), ["father"], days_ago=0)

    found = bridge.retrieve('what about the "conversion rider"?')
    assert found.phrases == ["conversion rider"]
    ids = [c.recording_id for c in found.candidates]
    assert ids[0] == exact, f"the exact phrase lost to scattered hits: {ids}"
    assert scattered in ids
    assert home not in ids and found.personal_skipped == 1


def test_a_quoted_phrase_search_reports_each_unopenable_recording_once(bridge, monkeypatch):
    bridge.index("henderson.txt", HENDERSON, ["insurance_agent"], encrypt=True)
    monkeypatch.setenv("PLAUD_BRIDGE_PASSPHRASE", "a-completely-different-passphrase")

    found = bridge.retrieve('what did I say about the "conversion rider"?')
    assert len(found.unopened) == 1, "the phrase pass and the main pass reported it twice"
    assert found.candidates == []
    assert found.complete is False


# =========================================================================
# Locality and redaction defaults
# =========================================================================
def test_a_candidate_whose_every_excerpt_is_withheld_contributes_nothing_to_a_cloud_bundle(
    bridge, monkeypatch
):
    bridge.cfg.profiles["sales_trainer"].suppress_fields = ["skill_gaps"]
    bridge.index("debrief.txt", _lines(("Sasson", "your discovery questions were weak"),),
                 ["sales_trainer"],
                 analyses={"sales_trainer": {"skill_gaps": ["quoted the mortgage figure early"]}})
    fake = _install(monkeypatch, FakeLLM())

    answer = bridge.ask("what did I say about the mortgage?")
    assert answer.local_only is False, "this test needs a cloud-permitted bundle"
    assert answer.excerpts == []
    assert "mortgage figure" not in fake.calls[0]["user"]
    assert "1 suppressed field(s) were withheld" in answer.note


def test_asking_for_local_only_is_honoured_and_explained(bridge, monkeypatch):
    bridge.index("debrief.txt", _lines(("Sasson", "your discovery questions were weak"),),
                 ["sales_trainer"])
    fake = _install(monkeypatch, FakeLLM())

    answer = bridge.ask("discovery questions", local_only=True)
    assert answer.local_only is True
    assert fake.calls[0]["local_only"] is True
    assert "local-only was requested." in answer.note


def test_redaction_is_required_when_no_known_profile_is_involved(bridge):
    assert _redaction_required(bridge.cfg, set()) is True
    assert _redaction_required(bridge.cfg, {"a_profile_that_was_deleted"}) is True
    bridge.cfg.profiles["sales_trainer"].redact_before_llm = False
    assert _redaction_required(bridge.cfg, {"sales_trainer"}) is False


# =========================================================================
# What the model is told it is not seeing
# =========================================================================
def test_the_model_is_told_when_the_scan_was_cut_short_and_when_files_would_not_open(
    bridge, monkeypatch
):
    bridge.index("locked.txt", HENDERSON, ["insurance_agent"], encrypt=True, days_ago=0)
    bridge.index("open.txt", HENDERSON, ["sales_trainer"], days_ago=1)
    bridge.index("unscanned.txt", HENDERSON, ["sales_trainer"], days_ago=2)
    bridge.cfg._d["ask"] = {"scan_limit": 2}
    monkeypatch.setenv("PLAUD_BRIDGE_PASSPHRASE", "a-completely-different-passphrase")
    fake = _install(monkeypatch, FakeLLM())

    answer = bridge.ask("conversion rider")
    user = fake.calls[0]["user"]
    assert "WHAT YOU ARE NOT SEEING" in user
    assert "Not every recording in the window was searched." in user
    assert "1 recording(s) could not be opened and were not searched." in user
    assert "Only 1 of 3 recording(s)" in answer.note and "ask.scan_limit" in answer.note
    assert "could not be opened and were NOT searched" in answer.note
    assert answer.recordings_considered == 1


# =========================================================================
# Citations that are not even objects
# =========================================================================
def test_a_citation_that_is_not_an_object_is_ignored_not_counted(bridge, monkeypatch):
    real_line = HENDERSON[4].text
    bridge.index("henderson.txt", HENDERSON, ["insurance_agent"])
    _install(monkeypatch, FakeLLM(lambda user: {
        "answer": "Noted.",
        "citations": ["not a citation", 42, None,
                      {"recording_id": _first_header(user)[0], "stamp": _first_header(user)[1],
                       "quote": real_line}],
        "confidence": "high", "unanswered": "",
    }))

    answer = bridge.ask("what did I promise about the conversion language?")
    assert len(answer.citations) == 1 and answer.citations[0].quote == real_line
    assert answer.dropped_citations == []


def test_overlap_counts_shared_stems_once_each():
    assert _overlap("I promised the rider", "the rider promise, the promise") == 3
    assert _overlap("nothing", "shared") == 0


# =========================================================================
# Housekeeping that must not cost the person their answer
# =========================================================================
def test_a_spend_record_that_fails_does_not_lose_the_answer(bridge, monkeypatch):
    bridge.index("henderson.txt", HENDERSON, ["insurance_agent"])
    _install(monkeypatch, FakeLLM(cost_usd=0.02))

    def refuse(*_a, **_k):
        raise RuntimeError("spend table is locked")

    monkeypatch.setattr(bridge.db, "record_spend", refuse)
    answer = bridge.ask("conversion rider")
    assert answer.degraded is False
    assert answer.text == "A plausible answer."
    assert answer.cost_usd == 0.02
    assert bridge.db.audit_log(action="ask", limit=5)


def test_a_model_that_returns_no_answer_text_says_so_rather_than_printing_nothing(
    bridge, monkeypatch
):
    bridge.index("henderson.txt", HENDERSON, ["insurance_agent"])
    _install(monkeypatch, FakeLLM({"answer": "  ", "citations": [], "confidence": "weird",
                                   "unanswered": ""}))

    answer = bridge.ask("conversion rider")
    assert answer.text == "The model returned no answer text."
    assert "did not contain the answer" in answer.note
    assert answer.confidence == "low", "an unknown confidence label was not rounded down"


def test_an_audit_write_that_fails_does_not_eat_the_answer(bridge, monkeypatch):
    bridge.index("henderson.txt", HENDERSON, ["insurance_agent"])
    _install(monkeypatch, FakeLLM())

    def refuse(*_a, **_k):
        raise RuntimeError("audit table is locked")

    monkeypatch.setattr(bridge.db, "audit", refuse)
    answer = bridge.ask("conversion rider")
    assert answer.text == "A plausible answer."
    assert answer.degraded is False
