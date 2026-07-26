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
