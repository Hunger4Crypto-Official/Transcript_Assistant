"""
The guarantees the README and COMPLIANCE.md make in plain language.

Every test here corresponds to a sentence a reader would take as a promise:
"nothing about a family conversation ever leaves it", "the strictest matched
profile governs the whole file", "Father and Husband are locked to local
processing in code". Unit tests elsewhere check that the pieces work. These
check that the promises are true end to end.

If one of these fails, the failure is a privacy regression, not a bug.
"""

import json

import pytest

from _fixtures import CLIENT_CALL, FAMILY_DINNER, StubLLM, build_sandbox, drop
from plaud_bridge.models import Segment, Transcript
from plaud_bridge.pipeline import Pipeline


class CoRoutingStub(StubLLM):
    """Scores a recording into a locked profile and a cloud-allowed one at once."""

    def __call__(self, cfg, system, user, local_only=False, max_tokens=None):
        if '"scores"' in user:
            self.calls.append({"local_only": local_only, "kind": "route"})
            return {
                "scores": [
                    {"profile_id": "husband", "score": 0.95, "evidence": ["marital"]},
                    {"profile_id": "sales_trainer", "score": 0.95, "evidence": ["debrief"]},
                    {"profile_id": "insurance_agent", "score": 0.05, "evidence": []},
                    {"profile_id": "father", "score": 0.05, "evidence": []},
                ],
            }, self._response()

        self.calls.append({"local_only": local_only, "kind": "extract"})
        return {"requires_human_attention": False}, self._response()


MARITAL_AND_WORK = """\
Sasson: I'm worried about the biopsy results, I haven't told my mother yet.
Wife: We'll get through it. Don't carry that on your own.
Sasson: Anyway, let me debrief that call. My discovery questions were weak.
Wife: You always say that after a call.
"""


# =========================================================================
# ADR-002: the strictest matched profile governs the WHOLE file
# =========================================================================
def test_a_locked_profile_forces_every_analysis_local_even_when_co_routed(tmp_path, monkeypatch):
    """
    A recording that is Husband and Sales Trainer at once must not hand the
    transcript to a cloud provider for the Sales Trainer half. The whole file
    is governed by the strictest match, and Husband is locked.
    """
    cfg, stub = build_sandbox(tmp_path, monkeypatch, stub=CoRoutingStub())
    drop(cfg, "REC0042.txt", MARITAL_AND_WORK)

    pipe = Pipeline(cfg)
    try:
        pipe.run()
        payload = json.loads(pipe.db.query()[0]["payload_json"])

        assert payload["compliance"]["governing_profile"] == "husband"
        assert payload["compliance"]["force_local_processing"] is True
        assert len(payload["analyses"]) > 1, "the co-routing this test exists for did not happen"

        leaked = [c for c in stub.calls if not c["local_only"]]
        assert not leaked, (
            "a maximum-sensitivity recording reached a call that permitted cloud "
            f"providers: {leaked}"
        )
    finally:
        pipe.close()


def test_the_gates_locality_verdict_is_actually_obeyed(tmp_path, monkeypatch):
    """`force_local_processing` must drive behaviour, not just get recorded."""
    cfg, stub = build_sandbox(tmp_path, monkeypatch, stub=CoRoutingStub())
    drop(cfg, "REC0042.txt", MARITAL_AND_WORK)

    pipe = Pipeline(cfg)
    try:
        pipe.run()
        payload = json.loads(pipe.db.query()[0]["payload_json"])
        if payload["compliance"]["force_local_processing"]:
            assert all(c["local_only"] for c in stub.calls)
    finally:
        pipe.close()


# =========================================================================
# Routing happens before the gate, so it has to be conservative
# =========================================================================
def test_routing_stays_local_when_any_profile_forbids_cloud(tmp_path, monkeypatch):
    """
    The routing call sees the entire transcript before anything is known about
    it. In the shipped config only Sales Trainer permits a cloud LLM, so the
    routing call has to stay local regardless of what the filename looks like.
    """
    cfg, stub = build_sandbox(tmp_path, monkeypatch)
    drop(cfg, "client-marcus.txt", CLIENT_CALL)

    pipe = Pipeline(cfg)
    try:
        pipe.run()
        routing = [c for c in stub.calls if "scores" in c.get("system", "") or True]
        assert stub.calls
        assert all(c["local_only"] for c in stub.calls), (
            "a client fact-find containing health and financial disclosures was "
            "routed via a provider chain that permitted cloud"
        )
        assert routing
    finally:
        pipe.close()


# =========================================================================
# ASR runs before anything is known about the content
# =========================================================================
def test_cloud_asr_is_not_used_for_a_default_named_export(tmp_path, monkeypatch):
    """
    Plaud exports are named things like REC0042.wav. Nothing in that filename
    says whether the recording is a client call or a conversation with a child,
    so cloud ASR must not be the default for it.
    """
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    pipe = Pipeline(cfg)
    try:
        for name in ("REC0042.wav", "20260726-143022.mp3", "New Recording 4.m4a",
                     "IMG_4021.wav", "dinner-with-kid.mp3"):
            assert pipe._asr_local_only(tmp_path / name)[0], (
                f"{name} would have been uploaded to a cloud transcription service"
            )
    finally:
        pipe.close()


def test_cloud_asr_stays_reachable_when_the_file_names_a_cloud_allowed_profile(
    tmp_path, monkeypatch
):
    """The cheap path still has to exist, or the safe default is unusable."""
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    pipe = Pipeline(cfg)
    try:
        local_only, why = pipe._asr_local_only(tmp_path / "sales_trainer-roleplay-debrief.mp3")
        assert not local_only, why
    finally:
        pipe.close()


# =========================================================================
# COMPLIANCE section 7: encrypted at rest means encrypted everywhere
# =========================================================================
def test_the_index_does_not_hold_a_plaintext_copy_of_an_encrypted_transcript(
    tmp_path, monkeypatch
):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    drop(cfg, "dinner-with-kid.txt", FAMILY_DINNER)

    pipe = Pipeline(cfg)
    try:
        pipe.run()
        payload = json.loads(pipe.db.query()[0]["payload_json"])
        assert payload["compliance"]["governing_profile"] == "father"
    finally:
        pipe.close()

    raw = cfg.path("database").read_bytes()
    assert b"permission slip" not in raw, (
        "the vault encrypted the transcript and the SQLite index kept a "
        "plaintext copy of the same words"
    )


def test_an_unencrypted_profile_keeps_its_transcript_in_the_index(tmp_path, monkeypatch):
    """Stripping the transcript is a response to encryption, not a blanket rule."""
    from plaud_bridge.models import Recording

    rec = Recording(source_name="x.txt")
    rec.transcript = Transcript(segments=[Segment(0.0, 1.0, "hello there")])
    assert "hello there" in rec.to_json()
    assert "hello there" not in rec.to_json(include_transcript=False)


def test_the_withheld_index_leaks_no_transcript_derived_string():
    """
    A red-team pass read a client's own words out of the plain SQLite index of
    an ENCRYPTED recording. Withholding the transcript segments was not enough:
    the consent quote and the router's evidence phrases are lifted verbatim from
    the transcript and were riding into payload_json in the clear. The old test
    greps one phrase that happens to appear in neither, so it missed this.
    """
    from plaud_bridge.models import (ComplianceVerdict, Recording, RouteMatch,
                                     Segment, Transcript)

    rec = Recording(source_name="client.txt", kind="text")
    rec.transcript = Transcript(segments=[
        Segment(0.0, 5.0, "my elimination period worries me", "Client"),
        Segment(5.0, 9.0, "the biopsy results came back positive", "Client"),
    ])
    rec.compliance = ComplianceVerdict()
    rec.compliance.consent_quote = "Sure. my elimination period worries me."
    rec.routes = [RouteMatch(profile_id="insurance_agent", confidence=0.9,
                             evidence=["biopsy results", "elimination period"])]

    withheld = rec.to_json(include_transcript=False, include_analysis_fields=False)
    for phrase in ("elimination period", "biopsy results", "worries me"):
        assert phrase not in withheld, f"withheld index still leaks {phrase!r}"

    # And the clear path is unchanged -- withholding is a response to encryption.
    clear = rec.to_json(include_transcript=True, include_analysis_fields=True)
    assert "biopsy results" in clear and "elimination period" in clear


class WorkOnlyRouterStub(StubLLM):
    """Insists every recording is low-sensitivity work, whatever it contains."""

    def __call__(self, cfg, system, user, local_only=False, max_tokens=None):
        if '"scores"' in user:
            self.calls.append({"local_only": local_only, "kind": "route"})
            return {"scores": [
                {"profile_id": "father", "score": 0.03, "evidence": []},
                {"profile_id": "husband", "score": 0.03, "evidence": []},
                {"profile_id": "sales_trainer", "score": 0.40, "evidence": ["debrief"]},
                {"profile_id": "insurance_agent", "score": 0.03, "evidence": []},
            ]}, self._response()
        self.calls.append({"local_only": local_only, "kind": "extract"})
        return {"requires_human_attention": False}, self._response()


def test_a_strong_family_keyword_match_cannot_be_scored_away(tmp_path, monkeypatch):
    """
    Missing a family recording is the asymmetric failure the tool exists to
    prevent: it could hand that conversation to a cloud model. A transcript dense
    with Father keywords, where the model insists it is work, must still route to
    Father -- the locked-profile keyword floor holds the line the model tried to
    talk past.
    """
    from plaud_bridge.profiles.router import route

    cfg, _ = build_sandbox(tmp_path, monkeypatch, stub=WorkOnlyRouterStub())
    segments = [Segment(i * 4.0, i * 4.0 + 4.0, line, "Sasson")
                for i, line in enumerate(FAMILY_DINNER.strip().splitlines())]
    result = route(Transcript(segments=segments), cfg)

    routed = {m.profile_id for m in result.matches}
    assert "father" in routed, (
        "a family recording the keywords matched was dropped because the model "
        "scored it as work; the locked-profile keyword floor did not hold"
    )


class CoRedactStub(StubLLM):
    """Leads with a cloud-permitting profile and co-routes a redacting one."""

    def __init__(self):
        super().__init__()
        self.extract_bodies: list[str] = []

    def __call__(self, cfg, system, user, local_only=False, max_tokens=None):
        if '"scores"' in user:
            return {"scores": [
                {"profile_id": "sales_trainer", "score": 0.95, "evidence": ["debrief"]},
                {"profile_id": "insurance_agent", "score": 0.90, "evidence": ["fact find"]},
                {"profile_id": "father", "score": 0.02, "evidence": []},
                {"profile_id": "husband", "score": 0.02, "evidence": []},
            ]}, self._response()
        self.extract_bodies.append(user)
        return {"requires_human_attention": False}, self._response()


def test_redaction_is_a_floor_when_the_lead_profile_does_not_redact(tmp_path, monkeypatch):
    """
    Redaction before a cloud model is a floor, like locality and encryption. With
    strictest_profile_governs off and a lead profile that does not redact, a
    co-routed profile that does must still force redaction, or PII reaches the
    model unredacted. (No shipped profile skips redaction; this pins the floor
    for a custom one that would.)
    """
    cfg, stub = build_sandbox(tmp_path, monkeypatch, stub=CoRedactStub(),
                              overrides={"compliance": {"strictest_profile_governs": False}})
    cfg.profile("sales_trainer").redact_before_llm = False
    drop(cfg, "sales_trainer-debrief.txt",
         "Sasson: My SSN is 123-45-6789 and I record these, is that okay?\n"
         "Marcus: Yeah that's fine.\n")

    pipe = Pipeline(cfg)
    try:
        pipe.run()
        assert stub.extract_bodies, "no extraction happened"
        for body in stub.extract_bodies:
            assert "123-45-6789" not in body, (
                "PII reached the model because the lead profile skipped redaction"
            )
    finally:
        pipe.close()


def test_encryption_at_rest_is_a_floor_not_a_routing_preference(tmp_path, monkeypatch):
    """
    strictest_profile_governs is a routing preference; encryption at rest is a
    floor. With the flag off and a work profile leading the routing, a recording
    co-routed to a locked personal profile must still be encrypted -- otherwise
    a Husband conversation lands in the plaintext index because a sales debrief
    happened to sort first. The locality floor already ignores this flag.
    """
    from plaud_bridge.compliance.gate import ComplianceGate
    from plaud_bridge.models import Recording, RouteMatch

    cfg, _ = build_sandbox(tmp_path, monkeypatch,
                           overrides={"compliance": {"strictest_profile_governs": False}})
    rec = Recording(source_name="x.txt", kind="text")
    rec.transcript = Transcript(segments=[Segment(0.0, 2.0, "hi", "Sasson")])
    rec.routes = [RouteMatch(profile_id="sales_trainer", confidence=0.95),
                  RouteMatch(profile_id="husband", confidence=0.90)]

    verdict = ComplianceGate(cfg).evaluate(rec)
    assert verdict.governing_profile == "sales_trainer", "the flag-off preference did not take effect"
    assert verdict.encrypt_at_rest is True, (
        "a Husband-co-routed recording was left unencrypted because a work "
        "profile led the routing"
    )


# =========================================================================
# Consent
# =========================================================================
REFUSAL = """\
Sasson: Hey Marcus, before we get started I record these calls for my notes. Is that okay?
Marcus: Hold on, I really don't want this being recorded.
Sasson: Understood, no problem.
Marcus: So about that term policy through work, the elimination period question.
Sasson: Your income is the asset here, not the house. Disability matters.
"""


def test_a_refusal_is_not_treated_as_consent(tmp_path, monkeypatch):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    drop(cfg, "client-refused.txt", REFUSAL)

    pipe = Pipeline(cfg)
    try:
        stats = pipe.run()
        assert stats.quarantined == 1, (
            "the other party explicitly refused to be recorded and the gate let it through"
        )
        assert stats.processed == 0
    finally:
        pipe.close()


def test_a_refusal_quarantines_even_when_missing_consent_is_only_flagged(tmp_path, monkeypatch):
    """
    compliance.on_missing_consent governs SILENCE -- nobody said either way. An
    explicit objection is not silence: a refusal quarantines unconditionally,
    even with the gate set only to flag missing consent. A config flag must not
    wave a refusal through.
    """
    cfg, _ = build_sandbox(tmp_path, monkeypatch,
                           overrides={"compliance": {"on_missing_consent": "flag"}})
    drop(cfg, "client-refused.txt", REFUSAL)

    pipe = Pipeline(cfg)
    try:
        stats = pipe.run()
        assert stats.quarantined == 1, "a refusal was flagged instead of quarantined"
        assert stats.processed == 0
    finally:
        pipe.close()


def test_a_refusal_does_not_write_the_refusers_words_to_the_plaintext_index(tmp_path, monkeypatch):
    """
    The refusal reason is written to the plaintext audit table and the quarantine
    WHY.md. It must carry none of the verbatim objection -- that would put
    transcript speech in the clear. The words live only in the withheld
    consent_quote.
    """
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    drop(cfg, "client-refused.txt", REFUSAL)

    pipe = Pipeline(cfg)
    try:
        pipe.run()
    finally:
        pipe.close()

    needle = b"really don't want this being recorded"
    assert needle not in cfg.path("database").read_bytes(), (
        "the refuser's verbatim words reached the plaintext index"
    )
    why = cfg.path("quarantine")
    leaked = [p for p in why.rglob("WHY.md") if needle in p.read_bytes()]
    assert not leaked, "the refuser's verbatim words reached a plaintext WHY.md"


def test_consent_notes_never_carry_the_verbatim_speech():
    """
    The notes flow into verdict.reasons, which are written to the plaintext audit
    trail and WHY.md. Neither a refusal note nor an announced-by-other note may
    carry the words; those live in the dedicated (withheld) quote fields.
    """
    from plaud_bridge.compliance.consent import detect_consent

    refusal = [Segment(0.0, 4.0, "Honestly I really don't want this being recorded.", "Marcus")]
    r1 = detect_consent(Transcript(segments=refusal), 90.0, owner_label="Sasson")
    assert r1.refused and r1.refusal_quote, "the refusal and its quote must still be captured"
    assert "really don't want" not in " ".join(r1.notes), "a note leaked the verbatim refusal"

    other = [
        Segment(0.0, 4.0, "Morning.", "Sasson"),
        Segment(4.0, 8.0, "Just so you know, I am recording this call on my end.", "Marcus"),
    ]
    r2 = detect_consent(Transcript(segments=other), 90.0, owner_label="Sasson")
    assert "recording this call on my end" not in " ".join(r2.notes), (
        "a note leaked the other party's verbatim announcement"
    )


@pytest.mark.parametrize("body,expected", [
    ("Sasson: I record these calls for my notes, is that okay?\n"
     "Marcus: Yeah that's fine.\n", True),
    ("Sasson: I record these calls for my notes, is that okay?\n"
     "Marcus: No, please don't record this.\n", False),
    ("Marcus: Just so you know, this call is being recorded on my end.\n"
     "Sasson: Sure.\n", False),
])
def test_consent_detection_cases(body, expected):
    from plaud_bridge.compliance.consent import detect_consent

    segments = []
    cursor = 0.0
    for line in body.strip().splitlines():
        speaker, _, text = line.partition(": ")
        segments.append(Segment(cursor, cursor + 4.0, text, speaker))
        cursor += 4.0

    result = detect_consent(Transcript(segments=segments), 90.0, owner_label="Sasson")
    assert result.complete is expected, result.notes


def test_owner_identity_is_an_exact_match_not_a_substring():
    """
    Whose announcement counts as YOURS is decided by the speaker label, and a
    substring test (`owner_label in speaker`) accepted "Not Sasson" and "Sasson's
    assistant" as the owner. Speaker labels come from diarization or, worse, from
    an untrusted imported transcript, so a lookalike must not pass.
    """
    from plaud_bridge.compliance.consent import _is_owner

    assert _is_owner("Sasson", "Sasson")
    assert _is_owner(" sasson ", "Sasson"), "case and surrounding space must still match"
    for impostor in ("Not Sasson", "Sasson's assistant", "Sassonx", "The Sasson"):
        assert not _is_owner(impostor, "Sasson"), f"{impostor!r} was accepted as the owner"


def test_a_lookalike_cannot_announce_the_recording_for_the_owner():
    """
    End to end: the real owner is in the room, but the 'I record these' line is
    spoken by a differently-labelled speaker whose label merely contains the
    owner's name. That is not the owner announcing, so consent is not announced.
    """
    from plaud_bridge.compliance.consent import detect_consent

    segments = [
        Segment(0.0, 4.0, "Morning, thanks for coming in.", "Sasson"),
        Segment(4.0, 8.0, "Just so you know, I record these for my notes, okay?",
                "Sasson's assistant"),
        Segment(8.0, 12.0, "Yeah, that's fine.", "Marcus"),
    ]
    result = detect_consent(Transcript(segments=segments), 90.0, owner_label="Sasson")
    assert not result.announced, (
        "an announcement by a lookalike label was accepted as the owner's"
    )
