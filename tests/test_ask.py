"""
Conversational question answering over the archive.

Two halves, tested apart. Retrieval is deterministic and never touches a model,
so most of what follows builds an index by hand and asserts on rankings and
excerpts. Generation is tested through a fake `complete_json` that records
exactly what it was handed, because the interesting questions are all about what
reached the model rather than what came back from it: whether a family recording
was in the bundle, whether an SSN was still in the text, whether the call was
allowed to leave the machine at all.

The citation tests are the ones that matter most. An answer about your own
archive that cites a recording which does not exist is indistinguishable from a
correct one until you go looking for it, and by then you have acted on it.
"""

import hashlib
import json
from datetime import timedelta
from pathlib import Path

import pytest

from _fixtures import build_sandbox
from plaud_bridge.archive import Archive
from plaud_bridge.ask import Answer, ask, question_terms, retrieve, save_answer
from plaud_bridge.db import Database
from plaud_bridge.llm.base import LLMError, LLMResponse
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
from plaud_bridge.storage import Vault, VaultError

CLIENT_LINES = [
    ("Sasson", "Hey Marcus, before we get started I record these calls for my notes."),
    ("Marcus", "Yeah that's fine."),
    ("Sasson", "So you have a term policy through work and nothing else."),
    ("Marcus", "Right, about two hundred thousand."),
    ("Sasson", "I promised you two quote options and I'll have them by Thursday."),
    ("Marcus", "Perfect, send them over."),
]

HENDERSON_LINES = [
    ("Sasson", "Good to see you both. Let's talk about the Henderson term policy."),
    ("Wife", "We were told the premium is locked for twenty years."),
    ("Sasson", "I promised you I would confirm the conversion rider in writing."),
    ("Henderson", "That was the part we could not remember."),
    ("Sasson", "I will email the conversion language on Monday. That is a promise."),
]

ROLEPLAY_LINES = [
    ("Sasson", "Let's debrief that roleplay. Your discovery questions were weak."),
    ("Producer", "I jumped to the term policy pitch too early."),
    ("Sasson", "Slow down. Ask about the mortgage before you quote anything."),
]

DINNER_LINES = [
    ("Sasson", "How was practice today buddy?"),
    ("Kid", "Coach said I'm starting on Saturday, and I need the permission slip signed."),
    ("Sasson", "I promised you pizza after the game and I meant it."),
]


# =========================================================================
# Building an index by hand
# =========================================================================
def _index(cfg, db, name, lines, profiles, *, days_ago=1, encrypt=False,
           analyses=None, vault=None):
    """
    Put one finished recording into the index, the way the pipeline would.

    Hand-built rather than run through the Pipeline on purpose: these tests are
    about ranking and locality, and both need recordings placed in specific
    profiles on specific dates, which the router would not cooperate with.
    """
    rec = Recording(
        source_name=name,
        source_path=str(cfg.path("inbox") / name),
        content_hash=hashlib.sha256(name.encode()).hexdigest(),
        kind="text",
        stage=Stage.COMPLETE,
        recorded_at=utc_now() - timedelta(days=days_ago),
    )
    segments, cursor = [], 0.0
    for speaker, text in lines:
        segments.append(Segment(cursor, cursor + 6.0, text, speaker))
        cursor += 6.0
    rec.transcript = Transcript(segments=segments, duration_seconds=cursor)
    rec.duration_seconds = cursor
    rec.routes = [RouteMatch(profile_id=pid, confidence=0.9) for pid in profiles]

    governing = cfg.strictest(list(profiles))
    rec.compliance = ComplianceVerdict(
        governing_profile=governing.id,
        governing_sensitivity=governing.sensitivity,
        encrypt_at_rest=encrypt,
        force_local_processing=not governing.allow_cloud_llm,
    )
    for pid, fields in (analyses or {}).items():
        rec.analyses.append(ProfileAnalysis(profile_id=pid, fields=fields))

    if encrypt:
        vault = vault or Vault(cfg.path("vault"))
        path = vault.write(f"{rec.id}/analysis", rec.to_json(), rec.id)
        rec.artifact_paths["analysis"] = str(path)

    db.upsert(rec)
    return rec.id


class Bridge:
    """cfg + db + archive, closed together so no test leaks a connection."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.db = Database(cfg.path("database"))
        self.archive = Archive(cfg, self.db)

    def ask(self, question, **kw):
        return ask(question, self.cfg, self.db, self.archive, **kw)

    def retrieve(self, question, **kw):
        return retrieve(question, self.cfg, self.db, self.archive, **kw)

    def index(self, name, lines, profiles, **kw):
        return _index(self.cfg, self.db, name, lines, profiles, **kw)

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
    """
    Stand-in for `complete_json` that keeps everything it was given.

    The recorded `user` string is the actual subject of most of these tests: it
    is the only place you can see what text left this module for a model.
    """

    def __init__(self, payload=None, cost_usd=0.0, provider="fake"):
        self.calls: list[dict] = []
        self.payload = payload
        self.cost_usd = cost_usd
        self.provider = provider

    def __call__(self, cfg, system, user, local_only=False, max_tokens=None):
        self.calls.append({"system": system, "user": user, "local_only": local_only})
        body = self.payload(user) if callable(self.payload) else self.payload
        if body is None:
            body = {"answer": "A plausible answer.", "citations": [],
                    "confidence": "medium", "unanswered": ""}
        return body, LLMResponse(provider=self.provider, model="fake", cost_usd=self.cost_usd)


def _install(monkeypatch, fake: FakeLLM) -> FakeLLM:
    monkeypatch.setattr("plaud_bridge.ask.complete_json", fake)
    return fake


def _cite_first(user: str) -> dict:
    """Cite whatever the first excerpt header in the bundle actually says."""
    rec_id, stamp = _first_header(user)
    return {
        "answer": "You said you would send it.",
        "citations": [{"recording_id": rec_id, "stamp": stamp, "quote": "I promised"}],
        "confidence": "high",
        "unanswered": "",
    }


def _first_header(user: str) -> tuple[str, str]:
    for line in user.splitlines():
        line = line.strip()
        if line.startswith("[rec_") and "@" in line:
            inside = line[1:line.index("]")]
            rec_id, _, stamp = inside.partition(" @ ")
            return rec_id.strip(), stamp.strip()
    raise AssertionError(f"no excerpt header in the bundle:\n{user}")


# =========================================================================
# Terms and stemming
# =========================================================================
def test_question_terms_drop_the_words_that_carry_nothing():
    terms = question_terms("What did I promise the Hendersons about their term policy?")
    assert "promis" in terms, terms
    assert "henderson" in terms, terms
    assert "polic" in terms, terms
    assert not {"what", "did", "the", "about", "their"} & set(terms)


def test_a_question_matches_the_tense_it_was_not_asked_in():
    """"what did I promise" against "I promised" is the ordinary case."""
    assert question_terms("what did I promise") == question_terms("what I promised")
    assert question_terms("the policy") == question_terms("their policies")


# =========================================================================
# Retrieval, with no model anywhere near it
# =========================================================================
def test_retrieval_ranks_term_overlap_above_recency(bridge):
    """
    A client conversation from six weeks ago must beat yesterday's coffee.
    Recency is a tie-break, not the ranking.
    """
    old = bridge.index("henderson-review.txt", HENDERSON_LINES, ["insurance_agent"],
                       days_ago=42)
    new = bridge.index("roleplay-debrief.txt", ROLEPLAY_LINES, ["sales_trainer"], days_ago=0)

    found = bridge.retrieve("what did I promise the Hendersons about their term policy?")
    ids = [c.recording_id for c in found.candidates]
    assert ids[0] == old, f"the newest recording outranked the relevant one: {ids}"
    assert new in ids, "the weaker match should still be a candidate, just lower"


def test_recency_orders_two_equally_relevant_recordings(bridge):
    """
    Identical content, different dates. The newer one has to come first --
    recency is a weak signal but it is not a decorative one.
    """
    older = bridge.index("a-term-policy.txt", CLIENT_LINES, ["sales_trainer"], days_ago=30)
    newer = bridge.index("b-term-policy.txt", CLIENT_LINES, ["sales_trainer"], days_ago=1)

    found = bridge.retrieve("term policy quote options")
    ids = [c.recording_id for c in found.candidates]
    assert ids[:2] == [newer, older], ids


def test_retrieval_is_deterministic_and_needs_no_llm(bridge, monkeypatch):
    """
    Nothing in retrieval may reach a provider. If it did, this feature would be
    unusable on the machines it was written for.
    """
    def explode(*a, **kw):
        raise AssertionError("retrieval called an LLM")

    monkeypatch.setattr("plaud_bridge.ask.complete_json", explode)
    bridge.index("henderson-review.txt", HENDERSON_LINES, ["insurance_agent"])

    first = bridge.retrieve("Henderson conversion rider")
    second = bridge.retrieve("Henderson conversion rider")
    assert [c.recording_id for c in first.candidates] == [c.recording_id for c in second.candidates]
    assert first.candidates[0].excerpts


def test_excerpts_carry_a_timestamp_that_was_really_in_the_transcript(bridge):
    rec_id = bridge.index("henderson-review.txt", HENDERSON_LINES, ["insurance_agent"])
    found = bridge.retrieve("conversion rider")
    excerpt = found.candidates[0].excerpts[0]

    assert excerpt.recording_id == rec_id
    real = {f"{int(s.start) // 60:02d}:{int(s.start) % 60:02d}"
            for s in Transcript(segments=[Segment(i * 6.0, i * 6.0 + 6.0, t, sp)
                                          for i, (sp, t) in enumerate(HENDERSON_LINES)]).segments}
    assert excerpt.stamp in real, f"{excerpt.stamp} is not a real segment start"


def test_retrieval_reads_the_stored_analyses_too(bridge, monkeypatch):
    """A commitment lives in the analysis, not only in the words around it."""
    bridge.index(
        "henderson-review.txt", HENDERSON_LINES, ["insurance_agent"],
        analyses={"insurance_agent": {
            "next_action": "Email the Henderson conversion language on Monday",
            "commitments_by_producer": [
                {"timestamp": "00:24", "speaker": "Sasson",
                 "text": "I will email the conversion language on Monday."},
            ],
        }},
    )
    fake = _install(monkeypatch, FakeLLM())
    bridge.ask("what did I promise about the conversion language?")

    user = fake.calls[0]["user"]
    assert "Email the Henderson conversion language on Monday" in user


def test_retrieval_finds_a_recording_whose_transcript_is_encrypted(bridge):
    """
    insurance_agent encrypts at rest, so the words are only in the vault. If
    this fails, ask can never answer a question about a client call.
    """
    rec_id = bridge.index("henderson-review.txt", HENDERSON_LINES, ["insurance_agent"],
                          encrypt=True)
    found = bridge.retrieve("conversion rider")
    assert [c.recording_id for c in found.candidates] == [rec_id]
    assert any("conversion" in e.text.lower() for e in found.candidates[0].excerpts)


def test_a_recording_that_would_not_open_is_reported_not_ignored(bridge, monkeypatch):
    """ADR-021, restated for answers: what did not open must not read as absent."""
    bridge.index("henderson-review.txt", HENDERSON_LINES, ["insurance_agent"], encrypt=True)
    monkeypatch.setenv("PLAUD_BRIDGE_PASSPHRASE", "a-completely-different-passphrase")

    answer = bridge.ask("what did I promise about the conversion rider?")
    assert answer.unopened, "a locked recording was silently treated as empty"
    assert "could not be opened" in answer.note
    assert "NOT" in answer.note


# =========================================================================
# The degraded path: no model at all
# =========================================================================
def test_with_no_provider_you_get_excerpts_and_are_told_they_are_not_an_answer(bridge):
    """
    No monkeypatching here on purpose. The sandbox has no API keys, so this is
    the real provider chain failing the way it would on a fresh machine.
    """
    bridge.index("henderson-review.txt", HENDERSON_LINES, ["insurance_agent"])
    answer = bridge.ask("what did I promise the Hendersons?")

    assert answer.degraded is True
    assert answer.excerpts, "the degraded path returned nothing at all"
    assert "not an answer" in answer.text
    assert "search output" in answer.text
    assert "llm.local" in answer.note
    assert "[rec_" in answer.render()


def test_the_degraded_path_works_with_zero_providers_configured(bridge):
    bridge.cfg._d["llm"]["providers"] = []
    bridge.index("henderson-review.txt", HENDERSON_LINES, ["insurance_agent"])

    answer = bridge.ask("conversion rider")
    assert answer.degraded is True
    assert answer.excerpts


def test_an_empty_archive_says_so_rather_than_inventing_something(bridge, monkeypatch):
    fake = _install(monkeypatch, FakeLLM())
    answer = bridge.ask("what did I promise the Hendersons?")

    assert answer.degraded is True
    assert not fake.calls, "a model was asked a question with no material at all"
    assert "Nothing in the archive matched" in answer.text


def test_an_llm_failure_mid_flight_still_returns_the_excerpts(bridge, monkeypatch):
    def fail(*a, **kw):
        raise LLMError("all LLM providers failed: connection refused")

    monkeypatch.setattr("plaud_bridge.ask.complete_json", fail)
    bridge.index("henderson-review.txt", HENDERSON_LINES, ["insurance_agent"])

    answer = bridge.ask("conversion rider")
    assert answer.degraded is True
    assert "connection refused" in answer.note


# =========================================================================
# Citations
# =========================================================================
def test_a_fabricated_recording_id_is_dropped_and_said_out_loud(bridge, monkeypatch):
    """The failure this feature exists to not have."""
    real = bridge.index("henderson-review.txt", HENDERSON_LINES, ["insurance_agent"])
    fake = _install(monkeypatch, FakeLLM(lambda user: {
        "answer": "You promised to email the conversion language.",
        "citations": [
            {"recording_id": _first_header(user)[0], "stamp": _first_header(user)[1],
             "quote": "I will email the conversion language on Monday."},
            {"recording_id": "rec_totallymadeup00", "stamp": "01:02:03",
             "quote": "and I said the premium was locked for life"},
        ],
        "confidence": "high",
        "unanswered": "",
    }))

    answer = bridge.ask("what did I promise the Hendersons?")
    assert fake.calls
    assert [c.recording_id for c in answer.citations] == [real]
    assert answer.dropped_citations == ["rec_totallymadeup00"]
    assert "rec_totallymadeup00" in answer.note
    assert "dropped" in answer.note


def test_every_surviving_citation_names_a_recording_that_was_actually_sent(
    bridge, monkeypatch
):
    bridge.index("henderson-review.txt", HENDERSON_LINES, ["insurance_agent"])
    bridge.index("client-marcus.txt", CLIENT_LINES, ["insurance_agent"])
    _install(monkeypatch, FakeLLM(lambda user: {
        "answer": "Two things.",
        "citations": [
            {"recording_id": "rec_nope", "stamp": "00:00", "quote": "x"},
            {"recording_id": _first_header(user)[0], "stamp": "00:00", "quote": "y"},
        ],
        "confidence": "low", "unanswered": "",
    }))

    answer = bridge.ask("what did I promise about the term policy?")
    sent = set(answer.recordings_used)
    assert sent
    assert all(c.recording_id in sent for c in answer.citations)


def test_a_timestamp_that_was_never_sent_is_repaired_from_real_data(bridge, monkeypatch):
    """
    A wrong stamp on a real recording is repaired to a stamp that was in the
    bundle, never to one the model chose. Inventing a plausible timestamp is the
    same failure as inventing an id, only harder to notice.
    """
    bridge.index("henderson-review.txt", HENDERSON_LINES, ["insurance_agent"])
    _install(monkeypatch, FakeLLM(lambda user: {
        "answer": "You promised to email the conversion language.",
        "citations": [{"recording_id": _first_header(user)[0], "stamp": "09:99:99",
                       "quote": "I will email the conversion language on Monday."}],
        "confidence": "high", "unanswered": "",
    }))

    answer = bridge.ask("what did I promise the Hendersons?")
    assert answer.repaired_citations == 1
    stamps = {e.stamp for e in answer.excerpts}
    assert answer.citations[0].stamp in stamps
    assert answer.citations[0].stamp != "09:99:99"


def test_render_prints_citations_the_way_the_terminal_wants_them(bridge, monkeypatch):
    rec_id = bridge.index("henderson-review.txt", HENDERSON_LINES, ["insurance_agent"])
    _install(monkeypatch, FakeLLM(_cite_first))

    out = bridge.ask("what did I promise the Hendersons?").render()
    assert f"[{rec_id} @ " in out
    assert "Sources" in out
    assert "recording(s) searched" in out


# =========================================================================
# Locality: the strictest profile in the bundle governs all of it
# =========================================================================
def test_a_cloud_permitting_bundle_is_allowed_to_use_cloud(bridge, monkeypatch):
    """The control for the test below. Without it, that one proves nothing."""
    bridge.index("roleplay-debrief.txt", ROLEPLAY_LINES, ["sales_trainer"])
    fake = _install(monkeypatch, FakeLLM())

    answer = bridge.ask("what did I say about discovery questions?")
    assert answer.local_only is False
    assert fake.calls[0]["local_only"] is False


def test_one_cloud_forbidding_recording_forces_the_whole_call_local(bridge, monkeypatch):
    bridge.index("roleplay-debrief.txt", ROLEPLAY_LINES, ["sales_trainer"])
    bridge.index("client-marcus.txt", CLIENT_LINES, ["insurance_agent"])
    fake = _install(monkeypatch, FakeLLM())

    answer = bridge.ask("what did I say about the term policy?")
    assert "insurance_agent" in answer.profiles
    assert answer.local_only is True
    assert fake.calls[0]["local_only"] is True, (
        "a client fact-find shared a prompt with a training debrief and the call "
        "was still allowed to reach a cloud provider"
    )
    assert "insurance_agent" in answer.note


def test_asking_for_cloud_cannot_unlock_a_profile_that_forbids_it(bridge, monkeypatch):
    bridge.index("client-marcus.txt", CLIENT_LINES, ["insurance_agent"])
    fake = _install(monkeypatch, FakeLLM())

    answer = bridge.ask("what did I promise Marcus?", local_only=False)
    assert answer.local_only is True
    assert fake.calls[0]["local_only"] is True


def test_a_code_locked_profile_stays_local_even_when_asked_for(bridge, monkeypatch):
    """father is locked in code, not config. Nothing here may loosen it."""
    bridge.index("dinner-with-kid.txt", DINNER_LINES, ["father"])
    fake = _install(monkeypatch, FakeLLM())

    answer = bridge.ask("what did I promise about pizza?", include_personal=True,
                        local_only=False)
    assert answer.local_only is True
    assert fake.calls[0]["local_only"] is True


def test_offline_forces_local_even_for_a_cloud_permitting_profile(bridge, monkeypatch):
    """Offline is an assertion. Anything that could reach the network honours it."""
    bridge.index("roleplay-debrief.txt", ROLEPLAY_LINES, ["sales_trainer"])
    fake = _install(monkeypatch, FakeLLM())

    before = bridge.ask("discovery questions")
    assert before.local_only is False

    bridge.cfg._d["runtime"]["offline"] = True
    after = bridge.ask("discovery questions")
    assert after.local_only is True
    assert fake.calls[-1]["local_only"] is True
    assert "runtime.offline" in after.note


def test_with_nothing_retrieved_locality_defaults_to_the_careful_answer(bridge):
    """Before you know what a question is about, assume it is about the worst."""
    answer = bridge.ask("something nobody ever recorded")
    assert answer.local_only is True


# =========================================================================
# Personal profiles
# =========================================================================
def test_personal_recordings_stay_out_of_the_bundle_by_default(bridge, monkeypatch):
    bridge.index("dinner-with-kid.txt", DINNER_LINES, ["father"])
    bridge.index("client-marcus.txt", CLIENT_LINES, ["insurance_agent"])
    fake = _install(monkeypatch, FakeLLM())

    answer = bridge.ask("what did I promise?")
    user = fake.calls[0]["user"]
    assert "pizza" not in user, "a father recording reached a model prompt by default"
    assert "father" not in answer.profiles
    assert "personal recording(s) were left out" in answer.note


def test_include_personal_actually_includes_them(bridge, monkeypatch):
    bridge.index("dinner-with-kid.txt", DINNER_LINES, ["father"])
    fake = _install(monkeypatch, FakeLLM())

    answer = bridge.ask("what did I promise about pizza?", include_personal=True)
    assert "pizza" in fake.calls[0]["user"]
    assert "father" in answer.profiles


def test_naming_a_personal_profile_without_the_flag_explains_itself(bridge):
    bridge.index("dinner-with-kid.txt", DINNER_LINES, ["father"])
    answer = bridge.ask("what did I promise about pizza?", profile="father")
    assert "--include-personal" in answer.note


def test_an_unknown_profile_is_a_sentence_not_a_traceback(bridge):
    answer = bridge.ask("anything", profile="not_a_profile")
    assert answer.degraded is True
    assert "no profile called" in answer.text
    assert "father" in answer.note


# =========================================================================
# Redaction
# =========================================================================
SSN_LINES = [
    ("Sasson", "Before I quote the term policy I need your details."),
    ("Marcus", "Sure, my social is 123-45-6789 and you can reach me at marcus@example.com."),
    ("Sasson", "Got it. I promised you two options by Thursday."),
]


def test_redaction_happens_before_the_text_reaches_the_model(bridge, monkeypatch):
    bridge.index("client-ssn.txt", SSN_LINES, ["sales_trainer"])
    fake = _install(monkeypatch, FakeLLM())

    answer = bridge.ask("what did I promise about the term policy?")
    user = fake.calls[0]["user"]
    assert "123-45-6789" not in user, "an SSN was handed to a model verbatim"
    assert "SSN_REDACTED" in user
    # Counted per excerpt rather than per utterance: excerpts overlap by a
    # segment of context, so one spoken SSN can be scrubbed more than once.
    assert answer.redactions.get("ssn", 0) >= 1
    assert "Redacted before the model" in answer.note


def test_the_copy_you_read_is_not_the_redacted_one(bridge, monkeypatch):
    """
    Redaction is a travel document, not the record. The excerpts handed back to
    the owner on their own machine keep the words that were said.
    """
    bridge.index("client-ssn.txt", SSN_LINES, ["sales_trainer"])
    _install(monkeypatch, FakeLLM())

    answer = bridge.ask("what did I promise about the term policy?")
    assert any("123-45-6789" in e.text for e in answer.excerpts)


def test_redaction_is_driven_by_the_profile_and_not_hardcoded(bridge, monkeypatch):
    """
    The control for the test above. With every shipped profile setting
    redact_before_llm, a test that only asserts redaction happens would pass
    just as well if the flag were ignored entirely.
    """
    bridge.cfg.profiles["sales_trainer"].redact_before_llm = False
    bridge.index("client-ssn.txt", SSN_LINES, ["sales_trainer"])
    fake = _install(monkeypatch, FakeLLM())

    bridge.ask("what did I promise about the term policy?")
    assert "123-45-6789" in fake.calls[0]["user"]


def test_one_profile_wanting_redaction_redacts_the_whole_bundle(bridge, monkeypatch):
    bridge.cfg.profiles["sales_trainer"].redact_before_llm = False
    bridge.index("client-ssn.txt", SSN_LINES, ["sales_trainer"])
    bridge.index("client-marcus.txt", CLIENT_LINES, ["insurance_agent"])
    fake = _install(monkeypatch, FakeLLM())

    bridge.ask("what did I promise about the term policy?")
    assert "123-45-6789" not in fake.calls[0]["user"]


# =========================================================================
# The context budget
# =========================================================================
def test_a_bundle_that_did_not_fit_says_so_instead_of_truncating_quietly(
    bridge, monkeypatch
):
    bridge.cfg._d["ask"] = {"max_context_chars": 200}
    for n in range(4):
        bridge.index(f"term-policy-{n}.txt", HENDERSON_LINES, ["sales_trainer"], days_ago=n)
    fake = _install(monkeypatch, FakeLLM())

    answer = bridge.ask("what did I promise about the term policy conversion rider?")
    assert answer.truncated is True
    assert answer.left_out > 0
    assert answer.bundle_chars <= 400, answer.bundle_chars
    assert "context budget" in answer.note
    assert "left out" in fake.calls[0]["user"], (
        "the model was given a partial bundle and not told it was partial"
    )


def test_a_bundle_that_fits_is_not_reported_as_truncated(bridge, monkeypatch):
    bridge.index("henderson-review.txt", HENDERSON_LINES, ["sales_trainer"])
    _install(monkeypatch, FakeLLM())

    answer = bridge.ask("conversion rider")
    assert answer.truncated is False
    assert answer.left_out == 0


def test_the_first_excerpt_survives_an_absurdly_small_budget(bridge, monkeypatch):
    """A budget too small for one excerpt must still produce one, not zero."""
    bridge.cfg._d["ask"] = {"max_context_chars": 1}
    bridge.index("henderson-review.txt", HENDERSON_LINES, ["sales_trainer"])
    fake = _install(monkeypatch, FakeLLM())

    answer = bridge.ask("conversion rider")
    assert answer.excerpts
    assert fake.calls[0]["user"].count("[rec_") == 1


# =========================================================================
# Suppressed fields
# =========================================================================
FINANCIAL = {"insurance_agent": {
    "financial_disclosures": [
        {"timestamp": "00:27", "speaker": "Marcus",
         "text": "the mortgage is about four hundred thousand"},
    ],
    "next_action": "Send two mortgage protection options by Thursday",
}}

COACHING = {"sales_trainer": {
    "skill_gaps": ["kept quoting the mortgage figure back before it was confirmed"],
    "next_action": "Rehearse the mortgage discovery question",
}}


def test_a_suppressed_field_never_reaches_a_cloud_permitted_call(bridge, monkeypatch):
    """
    A field the profile said never renders into a shareable document does not
    become shareable because it is a prompt. Sales Trainer is the only shipped
    profile that permits cloud at all, so it is the only place this can be seen.
    """
    bridge.cfg.profiles["sales_trainer"].suppress_fields = ["skill_gaps"]
    bridge.index("roleplay-debrief.txt", ROLEPLAY_LINES, ["sales_trainer"], analyses=COACHING)
    fake = _install(monkeypatch, FakeLLM())

    answer = bridge.ask("what did we say about the mortgage question?")
    assert answer.local_only is False, "this test needs a cloud-permitted bundle"
    assert "kept quoting the mortgage figure" not in fake.calls[0]["user"]
    assert "withheld" in answer.note


def test_a_suppressed_field_is_available_when_nothing_leaves_the_machine(
    bridge, monkeypatch
):
    bridge.index("client-marcus.txt", CLIENT_LINES, ["insurance_agent"], analyses=FINANCIAL)
    fake = _install(monkeypatch, FakeLLM())

    answer = bridge.ask("what did Marcus say about the mortgage?")
    assert answer.local_only is True
    assert "four hundred thousand" in fake.calls[0]["user"]


# =========================================================================
# Persisting an answer
# =========================================================================
def test_a_saved_answer_is_encrypted_or_not_written(bridge, monkeypatch):
    bridge.index("henderson-review.txt", HENDERSON_LINES, ["insurance_agent"])
    _install(monkeypatch, FakeLLM(_cite_first))
    answer = bridge.ask("what did I promise the Hendersons?")

    path = save_answer(answer, bridge.cfg)
    assert path.suffix == ".enc"
    blob = path.read_bytes()
    assert b"I promised" not in blob
    assert b"Henderson" not in blob

    restored = json.loads(Vault(bridge.cfg.path("vault")).read_text(path))
    assert restored["question"] == "what did I promise the Hendersons?"
    assert restored["citations"]


def test_saving_refuses_rather_than_writing_plaintext(bridge, monkeypatch):
    bridge.index("henderson-review.txt", HENDERSON_LINES, ["insurance_agent"])
    _install(monkeypatch, FakeLLM())
    answer = bridge.ask("what did I promise the Hendersons?")

    monkeypatch.delenv("PLAUD_BRIDGE_PASSPHRASE", raising=False)
    with pytest.raises(VaultError) as exc:
        save_answer(answer, bridge.cfg)
    assert "PLAUD_BRIDGE_PASSPHRASE" in str(exc.value)
    assert not list((bridge.cfg.path("vault") / "ask").glob("*")) or all(
        p.suffix == ".enc" for p in (bridge.cfg.path("vault") / "ask").glob("*")
    )


# =========================================================================
# Housekeeping
# =========================================================================
def test_the_audit_entry_records_the_shape_of_the_question_not_the_question(
    bridge, monkeypatch
):
    """
    The audit table is a plain file (ADR-013). "what did I promise the
    Hendersons about the biopsy" is as identifying as the recording it names.
    """
    bridge.index("henderson-review.txt", HENDERSON_LINES, ["insurance_agent"])
    _install(monkeypatch, FakeLLM())
    bridge.ask("what did I promise the Hendersons about the biopsy?")

    trail = bridge.db.audit_log(action="ask")
    assert trail, "asking a question left no audit entry at all"
    assert "biopsy" not in trail[0]["detail"]
    assert "local_only=True" in trail[0]["detail"]
    assert b"biopsy" not in Path(bridge.cfg.path("database")).read_bytes()


def test_an_empty_question_is_a_sentence_not_a_crash(bridge):
    answer = bridge.ask("   ")
    assert answer.degraded is True
    assert "No question was asked" in answer.text


def test_the_answer_dataclass_renders_without_anything_in_it():
    assert Answer().render()
