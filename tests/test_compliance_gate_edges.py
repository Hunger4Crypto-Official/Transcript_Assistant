"""
The compliance gate's refusal paths -- the branches that decide a recording
is NOT analysed.

The happy path (consent found, work profile, cloud allowed) runs in every
end-to-end test. These are the other doors: the gate switched off, the static
family/spousal lock, a recording with no words to check, and the softer
"flag" policy. Each was a branch that had never executed under test until this
file existed, which for the code that decides whether a private recording is
processed at all is not an acceptable state.
"""

from __future__ import annotations

import logging

import pytest
import yaml

from _fixtures import build_sandbox
from plaud_bridge.compliance import gate as gate_module
from plaud_bridge.compliance.gate import ComplianceGate
from plaud_bridge.config import Config
from plaud_bridge.models import ConsentStatus, Recording, RouteMatch, Segment, Transcript

# A transcript with an unmistakable consent exchange in it, so a test can prove
# that a lock which must ignore spoken consent actually ignores it.
CONSENTED = Transcript(segments=[
    Segment(0.0, 4.0, "Before we start, I record these calls for my notes. Is that okay?", "Sasson"),
    Segment(4.0, 6.0, "Yeah that's fine, no problem at all.", "Marcus"),
    Segment(6.0, 9.0, "Great. So walk me through what you have in place.", "Sasson"),
])


def _routed(profile_id: str, transcript: Transcript | None = CONSENTED) -> Recording:
    rec = Recording(source_name="call.txt", kind="text")
    rec.transcript = transcript
    rec.routes = [RouteMatch(profile_id=profile_id, confidence=0.95)]
    return rec


# =========================================================================
# The gate switched off
# =========================================================================
def test_a_disabled_gate_says_so_loudly_rather_than_passing_silently(tmp_path, monkeypatch, caplog):
    """
    `compliance.enabled: false` is a legitimate config, but it must never be a
    quiet one: the verdict carries a warning naming the file, and the log
    records it, so a digest built with the gate off cannot look like one built
    with the gate on.
    """
    cfg, _ = build_sandbox(tmp_path, monkeypatch, overrides={"compliance": {"enabled": False}})
    with caplog.at_level(logging.WARNING, logger="plaud_bridge.compliance"):
        verdict = ComplianceGate(cfg).evaluate(_routed("insurance_agent"))

    assert verdict.allow is True
    assert any("DISABLED" in w and "pipeline.yaml" in w for w in verdict.warnings), verdict.warnings
    assert "disabled by config" in caplog.text
    # Nothing downstream was decided: no governing profile was even chosen.
    assert verdict.governing_profile == ""


# =========================================================================
# The static family / spousal lock
# =========================================================================
def _lock_family_consent(cfg: Config, tmp_path) -> Config:
    """Flip father.yaml's house rule to false and reload, as a person would."""
    path = tmp_path / "config" / "profiles" / "father.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["family_consent"]["everyone_knows_device_records"] = False
    path.write_text(yaml.safe_dump(raw))
    return Config.load(tmp_path / "config", root=tmp_path)


def test_a_profile_whose_consent_flag_is_false_refuses_outright(tmp_path, monkeypatch):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    cfg = _lock_family_consent(cfg, tmp_path)
    profile = cfg.profile("father")
    assert profile.consent_gate_key == "family_consent" and profile.consent_gate_value is False

    verdict = ComplianceGate(cfg).evaluate(_routed("father"))

    assert verdict.allow is False
    assert verdict.consent is ConsentStatus.NOT_DETECTED
    reason = " ".join(verdict.reasons)
    assert "family_consent" in reason and "father" in reason
    # The reason has to say the unlock is not a config edit, because the
    # person reading it is about to go looking for one.
    assert "not a config change" in reason


def test_the_static_lock_is_decided_before_spoken_consent_is_even_looked_at(tmp_path, monkeypatch):
    """
    Mutation-style. The static lock must come BEFORE spoken-consent detection:
    "everyone in this house knows the device records" is a standing fact the
    owner attests to, and no sentence inside one recording can substitute for
    it. If the two checks were reordered, the consent exchange in CONSENTED
    would reach the detector -- and this test replaces the detector with a
    tripwire, so that reordering fails here rather than in someone's kitchen.
    """
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    cfg = _lock_family_consent(cfg, tmp_path)

    def tripwire(*_args, **_kwargs):
        raise AssertionError("spoken-consent detection ran despite the static lock")

    monkeypatch.setattr(gate_module, "detect_consent", tripwire)
    verdict = ComplianceGate(cfg).evaluate(_routed("father", CONSENTED))
    assert verdict.allow is False and verdict.consent is ConsentStatus.NOT_DETECTED


def test_the_static_lock_leaves_other_profiles_untouched(tmp_path, monkeypatch):
    """Locking the family profile must not start refusing client calls."""
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    cfg = _lock_family_consent(cfg, tmp_path)
    verdict = ComplianceGate(cfg).evaluate(_routed("insurance_agent"))
    assert verdict.allow is True
    assert verdict.consent is ConsentStatus.DETECTED


# =========================================================================
# Nothing to check
# =========================================================================
def test_a_consent_profile_with_no_transcript_is_not_detected_not_assumed(tmp_path, monkeypatch):
    """
    No words means no consent could be found. The verdict must say NOT_DETECTED
    with the reason spelled out -- never NOT_REQUIRED, never a pass by default.
    """
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    assert cfg.profile("insurance_agent").require_consent is True

    verdict = ComplianceGate(cfg).evaluate(_routed("insurance_agent", transcript=None))

    assert verdict.consent is ConsentStatus.NOT_DETECTED
    assert any("no transcript" in r for r in verdict.reasons), verdict.reasons


# =========================================================================
# The softer policy
# =========================================================================
def test_flag_policy_flags_missing_consent_instead_of_blocking(tmp_path, monkeypatch):
    """
    `on_missing_consent: flag` is the operator saying "warn me, do not stop".
    The verdict then carries a warning that names the policy -- so nobody
    reading a digest can mistake a flagged recording for a consented one --
    and processing is allowed to continue.
    """
    cfg, _ = build_sandbox(tmp_path, monkeypatch,
                           overrides={"compliance": {"on_missing_consent": "flag"}})
    silent = Transcript(segments=[
        Segment(0.0, 3.0, "So walk me through what you have in place.", "Sasson"),
        Segment(3.0, 6.0, "A term policy through work, about two hundred thousand.", "Marcus"),
    ])

    verdict = ComplianceGate(cfg).evaluate(_routed("insurance_agent", silent))

    assert verdict.consent is ConsentStatus.NOT_DETECTED
    assert verdict.allow is True, "flag policy blocked the recording anyway"
    assert any("'flag'" in w and "flagged rather than blocked" in w for w in verdict.warnings), (
        verdict.warnings)


def test_quarantine_policy_is_the_default_and_blocks(tmp_path, monkeypatch):
    """The other side of the same fork, so the two policies are pinned against each other."""
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    silent = Transcript(segments=[
        Segment(0.0, 3.0, "So walk me through what you have in place.", "Sasson"),
    ])
    verdict = ComplianceGate(cfg).evaluate(_routed("insurance_agent", silent))
    assert verdict.consent is ConsentStatus.NOT_DETECTED
    assert verdict.allow is False


# =========================================================================
# The statute note
# =========================================================================
def test_the_all_party_note_lists_the_configured_states_and_defers_to_counsel(tmp_path, monkeypatch):
    cfg, _ = build_sandbox(tmp_path, monkeypatch,
                           overrides={"compliance": {"all_party_consent_states": ["NV", "CA"]}})
    note = ComplianceGate(cfg).all_party_state_note()
    assert "NV, CA" in note
    # It is an operational list, not legal advice, and it has to say so.
    assert "counsel" in note


@pytest.mark.parametrize("policy", ["quarantine", "flag"])
def test_both_policies_are_accepted_by_config_validation(tmp_path, monkeypatch, policy):
    cfg, _ = build_sandbox(tmp_path, monkeypatch,
                           overrides={"compliance": {"on_missing_consent": policy}})
    assert ComplianceGate(cfg).on_missing == policy
