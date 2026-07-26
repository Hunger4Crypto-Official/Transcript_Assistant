"""
The compliance gate.

Runs after routing and before analysis, and it can stop the pipeline. Its job
is to answer three questions before any content reaches a model:

  1. Which profile governs this recording? When several match, the strictest
     one wins and its policy applies to the WHOLE file. One private sentence in
     a business meeting locks the entire recording down. That is deliberate.
     The alternative is deciding, mid-file, that half a conversation is safe to
     ship to a third party, which is not a decision software should make.

  2. Was consent obtained, if this profile requires it? If not, the recording
     is quarantined rather than analysed.

  3. Does processing have to stay local? If so, cloud providers are removed
     from the chain entirely rather than deprioritised.
"""

from __future__ import annotations

from ..logging_setup import get
from ..models import ComplianceVerdict, ConsentStatus, Recording
from .consent import detect_consent
from .redact import redact_text

log = get("compliance")


class ComplianceGate:
    def __init__(self, cfg):
        self.cfg = cfg
        self.enabled = bool(cfg.get("compliance.enabled", True))
        self.window = float(cfg.get("compliance.consent_window_seconds", 90))
        self.on_missing = cfg.get("compliance.on_missing_consent", "quarantine")
        self.patterns = cfg.get("compliance.redact_patterns") or {}
        self.strictest_governs = bool(cfg.get("compliance.strictest_profile_governs", True))

    def evaluate(self, rec: Recording) -> ComplianceVerdict:
        verdict = ComplianceVerdict()

        if not self.enabled:
            verdict.warnings.append(
                "compliance gate is DISABLED in pipeline.yaml. Nothing was checked."
            )
            log.warning("compliance gate disabled by config")
            return verdict

        profile_ids = rec.profile_ids or [self.cfg.get("routing.fallback_profile", "unfiled")]

        governing = (
            self.cfg.strictest(profile_ids)
            if self.strictest_governs
            else self.cfg.profile(profile_ids[0])
        )
        verdict.governing_profile = governing.id
        verdict.governing_sensitivity = governing.sensitivity

        if len(profile_ids) > 1 and self.strictest_governs:
            verdict.reasons.append(
                f"{len(profile_ids)} profiles matched; '{governing.id}' "
                f"({governing.sensitivity.value}) governs the whole recording"
            )

        # --- processing locality -----------------------------------------
        any_local_required = any(
            self.cfg.profile(pid).hard_local_only or not self.cfg.profile(pid).allow_cloud_llm
            for pid in profile_ids
            if pid in self.cfg.profiles
        )
        verdict.force_local_processing = any_local_required
        if any_local_required:
            locked = [
                pid for pid in profile_ids
                if pid in self.cfg.profiles and self.cfg.profile(pid).hard_local_only
            ]
            if locked:
                verdict.reasons.append(
                    f"local-only processing enforced by profile(s): {', '.join(locked)}"
                )
            else:
                verdict.reasons.append("local-only processing required by profile policy")

        # --- static consent gates (family / spousal) ----------------------
        for pid in profile_ids:
            if pid not in self.cfg.profiles:
                continue
            profile = self.cfg.profile(pid)
            if profile.consent_gate_key and not profile.consent_gate_value:
                verdict.allow = False
                verdict.consent = ConsentStatus.NOT_DETECTED
                verdict.reasons.append(
                    f"profile '{pid}' has {profile.consent_gate_key} set to false. "
                    "Processing refused. If the people on this recording have not "
                    "agreed to be recorded, the answer is not a config change."
                )
                log.error("blocked by static consent gate on profile %s", pid)
                return verdict

        # --- spoken consent ------------------------------------------------
        needs_consent = any(
            self.cfg.profile(pid).require_consent
            for pid in profile_ids
            if pid in self.cfg.profiles
        )

        if not needs_consent:
            verdict.consent = ConsentStatus.NOT_REQUIRED
        elif rec.transcript is None:
            verdict.consent = ConsentStatus.NOT_DETECTED
            verdict.reasons.append("no transcript available to check for consent")
        else:
            result = detect_consent(
                rec.transcript,
                self.window,
                owner_label=self.cfg.get("diarization.owner_label"),
            )
            if result.complete:
                verdict.consent = ConsentStatus.DETECTED
                verdict.consent_quote = result.announce_quote
                verdict.consent_timestamp = result.timestamp
                verdict.reasons.append("consent announcement and agreement detected")
            else:
                verdict.consent = ConsentStatus.NOT_DETECTED
                verdict.reasons.extend(result.notes)
                if result.announced and not result.agreed:
                    verdict.reasons.append(
                        "you announced the recording but no agreement was captured "
                        "from the other party"
                    )
                if self.on_missing == "quarantine":
                    verdict.allow = False
                    verdict.reasons.append(
                        "QUARANTINED. compliance.on_missing_consent is 'quarantine'. "
                        "Review the recording, confirm consent was actually obtained, "
                        "then release it with: run.py release <recording_id>"
                    )
                    log.warning("quarantined for missing consent: %s", rec.source_name)
                else:
                    verdict.warnings.append(
                        "consent not detected; output is flagged rather than blocked "
                        "because compliance.on_missing_consent is 'flag'"
                    )

        return verdict

    def redact_for_llm(self, text: str, profile) -> tuple[str, dict[str, int]]:
        """Return the copy of the transcript that is safe to hand to a model."""
        redacted, report = redact_text(text, self.patterns, enabled=profile.redact_before_llm)
        return redacted, report.counts

    def all_party_state_note(self) -> str:
        states = self.cfg.get("compliance.all_party_consent_states", []) or []
        return (
            "All-party consent states configured: " + ", ".join(states) + ". "
            "Verify current statutes with counsel; this list is an operational "
            "default, not legal advice."
        )
