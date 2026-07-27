"""
Shared test scaffolding.

A sandbox is a full copy of the real `config/` pointed at a temp directory, with
diarization off and the LLM replaced by a deterministic stub. Copying the real
config rather than writing a minimal fake is deliberate: it means the tests
exercise the profiles and thresholds actually shipped, so a careless YAML edit
shows up as a red test rather than as a surprise six months later.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from plaud_bridge.config import Config
from plaud_bridge.llm.base import LLMResponse

ROOT = Path(__file__).resolve().parents[1]

CLIENT_CALL = """\
Sasson: Hey Marcus, before we get started I record these calls for my notes. Is that okay with you?
Marcus: Yeah that's fine, no problem at all.
Sasson: Appreciate it. So walk me through what you've got in place right now.
Marcus: I've got a term policy through work, about two hundred thousand I think. That's it.
Sasson: And your wife, does she have anything separate?
Marcus: No, nothing. That's honestly why I called. We just had our second kid.
Sasson: Congratulations. So the concern is if something happened to you, she's covering the mortgage alone.
Marcus: Right. And the mortgage is about four hundred thousand still.
Sasson: Okay. The other thing worth looking at is disability. Your income is the asset here, not the house.
Marcus: I hadn't thought about that. What does that run?
Sasson: Depends on the elimination. Period and the own occupation definition. I'd need to quote it properly.
Marcus: The price is my worry honestly. Money's tight with the new baby.
Sasson: That's fair, and I'd rather right-size it than oversell you. Let me put two options together.
Marcus: Okay, send them over and I'll look with my wife this weekend.
Sasson: I'll have them to you by Thursday. Can you get me your date of birth and current income before then?
Marcus: Yeah I'll email you tonight.
"""

FAMILY_DINNER = """\
Sasson: How was practice today buddy?
Kid: Coach said I'm starting on Saturday!
Sasson: That's awesome. What time is the game?
Kid: Ten in the morning. And I need the permission slip signed for the field trip.
Sasson: I'll sign it tonight. Remind me at bedtime.
Kid: Also can we get pizza after the game? You said we could.
Sasson: I did say that. Pizza after the game, deal.
"""


class StubLLM:
    """
    Deterministic stand-in for `complete_json`.

    Returns plausible shapes for both the router and the extractor, and records
    every call so a test can assert on `local_only`. `cost_usd` is settable so
    the spend-accounting tests have a known number to follow through the
    pipeline.
    """

    def __init__(self, cost_usd: float = 0.0):
        self.calls: list[dict] = []
        self.cost_usd = cost_usd

    def _response(self) -> LLMResponse:
        return LLMResponse(provider="stub", model="stub", cost_usd=self.cost_usd)

    def __call__(self, cfg, system, user, local_only=False, max_tokens=None):
        self.calls.append({"local_only": local_only, "system": system[:80]})

        if '"scores"' in user:
            work = any(k in user.lower() for k in ("elimination period", "term policy", "disability"))
            scores = [
                {"profile_id": "insurance_agent", "score": 0.92 if work else 0.02,
                 "evidence": ["client fact find"]},
                {"profile_id": "sales_trainer", "score": 0.1, "evidence": []},
                {"profile_id": "father", "score": 0.05 if work else 0.93,
                 "evidence": ["talking with his son"]},
                {"profile_id": "husband", "score": 0.02, "evidence": []},
            ]
            return {"scores": scores}, self._response()

        if "meeting_type" in system or "meeting_type" in user:
            payload = {
                "meeting_type": "fact_find",
                "participants": ["Sasson", "Marcus"],
                "stated_needs": [{"timestamp": "00:21", "speaker": "Marcus",
                                  "text": "We just had our second kid."}],
                "financial_disclosures": [{"timestamp": "00:27", "speaker": "Marcus",
                                           "text": "the mortgage is about four hundred thousand"}],
                "health_disclosures": [],
                "objections": [{"type": "price", "quote": "The price is my worry honestly."}],
                "commitments_by_client": [{"timestamp": "00:48", "speaker": "Marcus",
                                           "text": "I'll email you tonight."}],
                "commitments_by_producer": [{"timestamp": "00:45", "speaker": "Sasson",
                                             "text": "I'll have them to you by Thursday."}],
                "statements_needing_review": [],
                "open_questions": ["What disability benefit amount fits the budget?"],
                "next_action": "Send two quote options by Thursday",
            }
        else:
            payload = {
                "requires_human_attention": False,
                "worth_remembering": [{"timestamp": "00:03", "speaker": "Kid",
                                       "text": "Coach said I'm starting on Saturday!"}],
                "promises_i_made": [{"what": "sign the permission slip", "when": "tonight"},
                                    {"what": "pizza after the game", "when": "Saturday"}],
                "logistics": [{"what": "game", "when": "Saturday 10:00 AM"}],
                "asks_from_them": [{"timestamp": "00:12", "speaker": "Kid",
                                    "text": "can we get pizza after the game?"}],
                "milestones": ["First time starting"],
                "next_action": "Sign the permission slip tonight",
            }
        return payload, self._response()


def build_sandbox(tmp_path, monkeypatch, stub: StubLLM | None = None,
                  overrides: dict | None = None) -> tuple[Config, StubLLM]:
    """
    Copy the shipped config into `tmp_path`, redirect every path at it, and
    swap in the stub LLM. `overrides` is merged one level deep into the
    pipeline config so a test can change a single block without rewriting it.
    """
    cfg_dir = tmp_path / "config"
    shutil.copytree(ROOT / "config", cfg_dir)

    data = yaml.safe_load((cfg_dir / "pipeline.yaml").read_text())
    for key in ("inbox", "work", "outbox", "vault", "quarantine", "logs"):
        data["paths"][key] = str(tmp_path / "data" / key)
    data["paths"]["database"] = str(tmp_path / "data" / "bridge.db")
    data["diarization"]["enabled"] = False
    # Tests write a file and process it in the same millisecond. The settle
    # window exists for real inboxes where a large file may still be copying;
    # here it would just skip everything.
    data["ingest"]["settle_seconds"] = 0
    for block, values in (overrides or {}).items():
        data.setdefault(block, {}).update(values)
    (cfg_dir / "pipeline.yaml").write_text(yaml.safe_dump(data))

    monkeypatch.setenv("PLAUD_BRIDGE_PASSPHRASE", "a-long-enough-test-passphrase")

    stub = stub or StubLLM()
    for module in ("plaud_bridge.profiles.router", "plaud_bridge.profiles.extractor"):
        monkeypatch.setattr(f"{module}.complete_json", stub)

    cfg = Config.load(cfg_dir, root=tmp_path)
    cfg.ensure_dirs()
    return cfg, stub


def drop(cfg, name: str, body: str) -> Path:
    """Put a text transcript in the inbox."""
    path = cfg.path("inbox") / name
    path.write_text(body, encoding="utf-8")
    return path
