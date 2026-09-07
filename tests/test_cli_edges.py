"""
The edges of the command line: every branch a route takes when something is
missing, locked, refused, or over a limit.

tests/test_cli_routes.py proves each subcommand runs. This file pins what each
one DOES on the paths a happy run never takes -- the exit code it chooses, the
sentence it prints, the file it writes or refuses to write, and the index it
leaves behind. Each test drives `main(argv)` the way a shell would, and each
asserts an effect rather than merely reaching a line.
"""

from __future__ import annotations

import copy
import csv
import runpy
import shutil
import sys
import wave
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from _fixtures import CLIENT_CALL, FAMILY_DINNER, StubLLM, build_sandbox, drop
from plaud_bridge.archive import Archive, SearchResult
from plaud_bridge.cli import _classify_verdict, cmd_speakers, main
from plaud_bridge.db import Database
from plaud_bridge.diarize.voiceprint import Embedder, VoiceprintStore, normalise
from plaud_bridge.models import Segment
from plaud_bridge.storage import Vault

# The consent exchange stripped from the client call: nobody asked on tape.
NO_ANNOUNCEMENT = CLIENT_CALL.split("\n", 2)[2]

# A coaching session. Its vocabulary belongs to sales_trainer -- the one
# shipped profile that keeps its artifacts in plaintext -- and to nothing else.
SALES_SESSION = """\
Sasson: Hey Dana, before we get started I record these calls for my notes. Is that okay with you?
Dana: Yeah that's fine, no problem at all.
Sasson: Let's run the role play again. This time practise the objection handling from the script.
Dana: Okay. When they say it costs too much, I go back to the value framework and build rapport first.
Sasson: Good. Then ask the closing question and stop talking.
"""

# Nothing in here belongs to any profile, so the router files it under unfiled.
GARDEN_CHAT = """\
Sasson: Hey Bob, before we get started I record these calls for my notes. Is that okay with you?
Bob: Yeah that's fine, no problem at all.
Sasson: So the tomatoes need staking before the weekend.
Bob: And the compost bin is full again. I will turn it on Sunday.
"""


class ScriptedStub(StubLLM):
    """
    The shared stub, with the router's scores and extra extraction fields
    chosen by the test instead of guessed from the transcript.
    """

    def __init__(self, scores: dict[str, float] | None = None,
                 extra_fields: dict | None = None, unfiled_fields: dict | None = None):
        super().__init__()
        self.scores = scores
        self.extra_fields = extra_fields or {}
        self.unfiled_fields = unfiled_fields or {}

    def __call__(self, cfg, system, user, local_only=False, max_tokens=None):
        payload, response = super().__call__(cfg, system, user, local_only, max_tokens)
        if "scores" in payload:
            if self.scores is not None:
                payload["scores"] = [
                    {"profile_id": pid, "score": score, "evidence": []}
                    for pid, score in self.scores.items()
                ]
        elif "suggested_keywords" in user:
            payload = {"topic": "gardening", "suggested_profile": "none",
                       "action_items": [], "next_action": "none", **self.unfiled_fields}
        else:
            payload = {**payload, **self.extra_fields}
        return payload, response


def cli(cfg, *argv) -> int:
    return main(["--config", str(cfg.root / "config"), *argv])


def rows(cfg, **kw) -> list[dict]:
    db = Database(cfg.path("database"))
    try:
        return db.query(limit=50, **kw)
    finally:
        db.close()


def one_id(cfg, **kw) -> str:
    found = rows(cfg, **kw)
    assert len(found) == 1, f"expected exactly one recording, found {len(found)}"
    return found[0]["id"]


def load(cfg, rid) -> dict | None:
    db = Database(cfg.path("database"))
    try:
        return db.load(rid)
    finally:
        db.close()


def audit(cfg, **kw) -> list[dict]:
    db = Database(cfg.path("database"))
    try:
        return db.audit_log(limit=100, **kw)
    finally:
        db.close()


def vec(*values: float) -> list[float]:
    return normalise(list(values))


def _no_stdin(monkeypatch):
    def refuse(*_args, **_kw):
        raise EOFError("EOF when reading a line")
    monkeypatch.setattr("builtins.input", refuse)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Two recordings processed: one work call, one family dinner."""
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    drop(cfg, "client.txt", CLIENT_CALL)
    drop(cfg, "dinner.txt", FAMILY_DINNER)
    assert cli(cfg, "run") == 0
    return cfg


@pytest.fixture
def held(tmp_path, monkeypatch):
    """One processed work call and one quarantined for a missing announcement."""
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    drop(cfg, "client.txt", CLIENT_CALL)
    drop(cfg, "unannounced.txt", NO_ANNOUNCEMENT)
    assert cli(cfg, "run") == 0
    quarantined = one_id(cfg, stage="quarantined")
    return cfg, quarantined


# =========================================================================
# doctor
# =========================================================================
def test_doctor_reports_a_missing_ffmpeg_as_fatal(sandbox, monkeypatch, capsys):
    from plaud_bridge.audio import AudioError, AudioPreparer

    def gone(self):
        raise AudioError("'ffmpeg' is not on PATH. Install ffmpeg.")
    monkeypatch.setattr(AudioPreparer, "check_tools", gone)

    assert cli(sandbox, "doctor") == 1
    out = capsys.readouterr().out
    assert "[ FAIL ] ffmpeg" in out
    assert "not on PATH" in out
    assert "NOT READY" in out


def test_doctor_stops_demanding_local_asr_once_a_local_provider_is_ready(sandbox, monkeypatch,
                                                                         capsys):
    from plaud_bridge.asr.local_provider import LocalWhisperASR

    monkeypatch.setattr(LocalWhisperASR, "available", lambda self: (True, "ready"))
    cli(sandbox, "doctor")
    out = capsys.readouterr().out
    assert "[  ok  ] asr:local" in out
    assert "required for father/husband profiles" not in out


def test_doctor_reports_an_unreadable_voiceprint_store_as_fatal(sandbox, capsys):
    """A corrupt store is biometric data that can no longer be checked; that is a FAIL."""
    Vault(sandbox.path("vault")).write("voiceprints", "this is not json", "voiceprints")
    assert cli(sandbox, "doctor") == 1
    out = capsys.readouterr().out
    assert "[ FAIL ] speakers:enrolled" in out
    assert "unreadable" in out


def test_doctor_reports_a_locked_vault_as_fatal(sandbox, monkeypatch, capsys):
    monkeypatch.delenv("PLAUD_BRIDGE_PASSPHRASE", raising=False)
    assert cli(sandbox, "doctor") == 1
    out = capsys.readouterr().out
    assert "[ FAIL ] vault" in out
    assert "PLAUD_BRIDGE_PASSPHRASE" in out


_OFFLINE = {
    "runtime": {"offline": True},
    "asr": {"providers": ["local"], "groq": {"enabled": False}},
    "llm": {"providers": ["local"], "anthropic": {"enabled": False},
            "groq": {"enabled": False},
            "local": {"enabled": True, "is_cloud": False,
                      "base_url": "http://localhost:11434/v1", "model": "llama3.3:70b"}},
}


def test_doctor_offline_is_fatal_without_the_whisper_weights_and_skips_an_unset_model(
        tmp_path, monkeypatch, capsys):
    """
    With runtime.offline on, a missing ASR model is a FAIL rather than a WARN:
    the machine cannot fetch it later. A model left unconfigured is not audited
    at all, because there is nothing to look for.
    """
    overrides = dict(_OFFLINE)
    overrides["diarization"] = {"pyannote": {"model": ""}}
    cfg, _ = build_sandbox(tmp_path, monkeypatch, overrides=overrides)

    assert cli(cfg, "doctor", "--offline") == 1
    out = capsys.readouterr().out
    assert "[  ok  ] runtime.offline" in out
    assert "[ FAIL ] offline:asr" in out
    assert "fetch_models.py --whisper large-v3" in out
    assert "offline:diarization" not in out


# =========================================================================
# run
# =========================================================================
def test_run_names_five_unsupported_files_and_counts_the_rest(tmp_path, monkeypatch, capsys):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    for i in range(7):
        (cfg.path("inbox") / f"photo{i}.jpg").write_bytes(b"not a recording")

    assert cli(cfg, "run") == 0
    out = capsys.readouterr().out
    assert "7 file(s) in the inbox are not a kind this reads" in out
    assert "...and 2 more" in out
    assert out.count(".jpg") == 5


# =========================================================================
# status
# =========================================================================
def test_status_itemises_spend_outside_the_pipeline(sandbox, capsys):
    """Money spent by `ask` used to vanish from the one place a person looks."""
    db = Database(sandbox.path("database"))
    try:
        db.record_spend("ask", 0.0123, provider="stub", model="stub")
    finally:
        db.close()

    assert cli(sandbox, "status") == 0
    out = capsys.readouterr().out
    assert "pipeline   $" in out
    assert "ask        $0.0123  (1 call(s))" in out


# =========================================================================
# search --content
# =========================================================================
def test_search_explains_quarantined_recordings_hold_nothing_to_search(held, capsys):
    cfg, quarantined = held
    assert cli(cfg, "search", "zzz-never-said", "--content") == 0
    out = capsys.readouterr().out
    assert "nothing matching" in out
    assert "1 recording(s) are quarantined and hold no searchable content" in out
    assert quarantined in out
    assert "release <id>" in out


def test_search_with_a_bounded_scan_and_hits_is_marked_incomplete(sandbox, capsys):
    assert cli(sandbox, "search", "the", "--content", "--scan-limit", "1") == 2
    out = capsys.readouterr().out
    assert "hit(s) across 1 recording(s)" in out
    assert "INCOMPLETE: only 1 of 2 recording(s) were searched" in out


def test_search_truncates_long_unopened_and_quarantined_lists_but_counts_them(sandbox,
                                                                           monkeypatch,
                                                                           capsys):
    """Twenty-five of each: the first twenty are named, the rest are counted, never dropped."""
    from plaud_bridge.archive import Match

    crafted = SearchResult(
        matches=[Match("rec_a", "a.txt", "2026-01-01 10:00", "insurance_agent", False,
                       "00:01", "Marcus", "we said the word")],
        unopened=[f"rec_u{i}  locked{i}.txt" for i in range(25)],
        quarantined=[f"rec_q{i}  held{i}.txt" for i in range(25)],
        scanned=51, total=51,
    )
    monkeypatch.setattr(Archive, "search_content", lambda self, *a, **k: crafted)

    assert cli(sandbox, "search", "word", "--content") == 2
    out = capsys.readouterr().out
    assert "25 recording(s) could not be opened and were NOT searched" in out
    assert "rec_u19" in out and "rec_u20" not in out
    assert "... and 5 more" in out
    assert "25 recording(s) are quarantined" in out
    assert "rec_q19" in out and "rec_q20" not in out


# =========================================================================
# ask
# =========================================================================
def test_ask_save_writes_an_encrypted_answer_into_the_vault(sandbox, capsys):
    ask_dir = sandbox.path("vault") / "ask"
    assert not ask_dir.exists()

    assert cli(sandbox, "ask", "what did I promise Marcus?", "--save") in (0, 2)
    assert "saved, encrypted:" in capsys.readouterr().out
    saved = list(ask_dir.glob("*.enc"))
    assert len(saved) == 1
    assert b"Marcus" not in saved[0].read_bytes(), "the saved answer is not ciphertext"


def test_ask_save_refuses_rather_than_writing_plaintext_when_the_vault_is_locked(
        tmp_path, monkeypatch, capsys):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    monkeypatch.delenv("PLAUD_BRIDGE_PASSPHRASE", raising=False)

    assert cli(cfg, "ask", "anything", "--save") == 1
    out = capsys.readouterr().out
    assert "NOT saved:" in out
    assert "PLAUD_BRIDGE_PASSPHRASE" in out
    assert not list(cfg.path("vault").rglob("ask/*")), "something was written anyway"


def test_ask_with_no_question_is_a_usage_error(sandbox, capsys):
    assert cli(sandbox, "ask", "") == 1
    assert "No question was asked" in capsys.readouterr().out


# =========================================================================
# forget
# =========================================================================
def test_forget_removes_an_index_entry_whose_files_are_already_gone(sandbox, capsys):
    rid = one_id(sandbox, profile_id="insurance_agent")
    db = Database(sandbox.path("database"))
    try:
        for path in Archive(sandbox, db).plan_forget(rid):
            path.unlink()
    finally:
        db.close()

    assert cli(sandbox, "forget", rid, "--yes") == 0
    out = capsys.readouterr().out
    assert "(no files on disk; the index entry will be removed)" in out
    assert "deleted 0 file(s) and the index entry" in out
    assert load(sandbox, rid) is None


# =========================================================================
# export
# =========================================================================
def test_export_rejects_an_unknown_profile_by_name(sandbox, capsys):
    assert cli(sandbox, "export", "--profile", "ghost") == 1
    out = capsys.readouterr().out
    assert "unknown profile 'ghost'" in out
    assert "insurance_agent" in out


def test_export_omits_a_recording_that_will_not_decrypt_and_says_so(tmp_path, monkeypatch,
                                                                    capsys):
    """Exit 2: a document was produced, but it is not the whole window."""
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    drop(cfg, "first.txt", CLIENT_CALL)
    drop(cfg, "second.txt", CLIENT_CALL + "Sasson: One more thing, thanks again.\n")
    assert cli(cfg, "run") == 0
    victim = rows(cfg, profile_id="insurance_agent")[0]
    Path(load(cfg, victim["id"])["artifact_paths"]["analysis"]).write_bytes(b"garbage")

    assert cli(cfg, "export") == 2
    captured = capsys.readouterr()
    assert "1 recording(s) could not be decrypted and were omitted" in captured.err
    assert "1 recording(s). Redacted for sharing." in captured.out
    assert victim["source_name"] not in captured.out


def test_transcript_export_counts_what_redaction_removed(tmp_path, monkeypatch, capsys):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    drop(cfg, "call.txt", CLIENT_CALL + "Marcus: My number is 555-123-4567 if you need it.\n")
    assert cli(cfg, "run") == 0

    assert cli(cfg, "export", "--transcripts") == 0
    out = capsys.readouterr().out
    assert "555-123-4567" not in out
    assert "Redacted before export: phone (1)" in out


def test_export_skips_an_analysis_from_a_profile_that_no_longer_exists_and_redacts_fields(
        sandbox, monkeypatch, capsys):
    """
    What the CLI does with the record it is handed: a section for a deleted
    profile is dropped rather than crashing on the lookup, and a field carrying
    a phone number is redacted and counted like a transcript line would be.
    """
    real = Archive.full_record

    def doctored(self, row):
        record = real(self, row)
        if record is None:
            return None
        record = copy.deepcopy(record)
        record.setdefault("analyses", []).append(
            {"profile_id": "ghost", "fields": {"topic": "a profile that was deleted"}})
        for analysis in record["analyses"]:
            if analysis.get("profile_id") == "insurance_agent":
                analysis["fields"]["open_questions"] = ["Call back on 555-123-4567"]
        return record
    monkeypatch.setattr(Archive, "full_record", doctored)

    assert cli(sandbox, "export") == 0
    out = capsys.readouterr().out
    assert "a profile that was deleted" not in out
    assert "555-123-4567" not in out
    assert "Redacted before export: phone (1)" in out


# =========================================================================
# watch
# =========================================================================
def test_watch_stops_cleanly_on_ctrl_c_between_passes(sandbox, monkeypatch, capsys):
    def interrupted(_seconds):
        raise KeyboardInterrupt
    monkeypatch.setattr("time.sleep", interrupted)

    assert cli(sandbox, "watch", "--interval", "1") == 0
    assert "stopped" in capsys.readouterr().out


# =========================================================================
# open
# =========================================================================
def test_open_names_the_artifacts_that_exist_when_asked_for_one_that_does_not(sandbox, capsys):
    rid = one_id(sandbox, profile_id="insurance_agent")
    assert cli(sandbox, "open", rid, "--kind", "audio") == 1
    out = capsys.readouterr().out
    assert "no 'audio' artifact" in out
    assert "transcript" in out and "analysis" in out


def test_open_reports_an_artifact_missing_from_disk(sandbox, capsys):
    rid = one_id(sandbox, profile_id="insurance_agent")
    Path(load(sandbox, rid)["artifact_paths"]["transcript"]).unlink()
    assert cli(sandbox, "open", rid) == 1
    assert "artifact missing from disk" in capsys.readouterr().out


def test_open_refuses_to_print_an_original_to_the_terminal(sandbox, capsys):
    rid = one_id(sandbox, profile_id="insurance_agent")
    assert cli(sandbox, "open", rid, "--kind", "source") == 1
    out = capsys.readouterr().out
    assert "Pass --out" in out
    assert f"run.py open {rid} --kind audio --out" in out


def test_open_cannot_stream_an_original_out_of_a_locked_vault(sandbox, monkeypatch, tmp_path,
                                                             capsys):
    rid = one_id(sandbox, profile_id="insurance_agent")
    monkeypatch.delenv("PLAUD_BRIDGE_PASSPHRASE", raising=False)
    dest = tmp_path / "copy.txt"
    assert cli(sandbox, "open", rid, "--kind", "source", "--out", str(dest)) == 1
    assert "could not decrypt" in capsys.readouterr().out
    assert not dest.exists()
    assert not dest.with_name("copy.txt.part").exists(), "a partial plaintext file was left"


@pytest.fixture
def plaintext(tmp_path, monkeypatch):
    """A recording governed by sales_trainer alone, the profile that does not encrypt."""
    stub = ScriptedStub(scores={"sales_trainer": 0.9, "insurance_agent": 0.0,
                                "father": 0.0, "husband": 0.0})
    cfg, _ = build_sandbox(tmp_path, monkeypatch, stub=stub)
    drop(cfg, "coaching.txt", SALES_SESSION)
    assert cli(cfg, "run") == 0
    row = rows(cfg)[0]
    assert row["governing_profile"] == "sales_trainer", row["governing_profile"]
    assert not row["encrypted"]
    return cfg, row["id"]


def test_open_reads_a_plaintext_transcript_without_touching_the_vault(plaintext, monkeypatch,
                                                                     capsys):
    cfg, rid = plaintext
    monkeypatch.delenv("PLAUD_BRIDGE_PASSPHRASE", raising=False)
    assert cli(cfg, "open", rid) == 0
    assert "role play" in capsys.readouterr().out


def test_open_copies_a_plaintext_original_and_still_says_to_delete_it(plaintext, tmp_path,
                                                                     capsys):
    cfg, rid = plaintext
    dest = tmp_path / "original.txt"
    assert cli(cfg, "open", rid, "--kind", "source", "--out", str(dest)) == 0
    out = capsys.readouterr().out
    assert dest.read_text(encoding="utf-8") == SALES_SESSION
    assert "Delete it when you are done with it" in out


# =========================================================================
# audit
# =========================================================================
def test_audit_writes_csv_with_one_row_per_entry(sandbox, tmp_path, capsys):
    dest = tmp_path / "trail" / "audit.csv"
    assert cli(sandbox, "audit", "--action", "ingest", "--out", str(dest)) == 0
    assert "wrote 2 entry(ies)" in capsys.readouterr().out

    with dest.open(encoding="utf-8", newline="") as fh:
        entries = list(csv.DictReader(fh))
    assert len(entries) == 2
    assert set(entries[0]) == {"at", "actor", "action", "recording_id", "detail"}
    assert {e["action"] for e in entries} == {"ingest"}
    assert all(e["recording_id"].startswith("rec_") for e in entries)


# =========================================================================
# release and the quarantine listing
# =========================================================================
def test_release_refuses_a_folder_holding_only_the_explanation(held, capsys):
    cfg, quarantined = held
    (cfg.path("quarantine") / quarantined / "unannounced.txt").unlink()
    assert cli(cfg, "release", quarantined, "--yes") == 1
    assert "no media to release" in capsys.readouterr().out
    assert not audit(cfg, action="quarantine_release")


def test_the_listing_marks_a_recording_that_was_already_released(held, capsys):
    cfg, quarantined = held
    assert cli(cfg, "release", quarantined, "--yes") == 0
    capsys.readouterr()
    assert cli(cfg, "quarantine") == 0
    assert "(already released; re-run `run.py run --force` to process it)" in capsys.readouterr().out


def test_the_listing_keeps_an_index_row_whose_folder_has_vanished(held, capsys):
    cfg, quarantined = held
    shutil.rmtree(cfg.path("quarantine") / quarantined)
    assert cli(cfg, "quarantine") == 0
    out = capsys.readouterr().out
    assert quarantined in out, "an index row whose folder vanished was silently dropped"
    assert "(folder has no media; releasable never, forgettable always)" in out


def test_the_listing_classifies_a_standing_consent_gate_that_is_off(tmp_path, monkeypatch,
                                                                    capsys):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    profile = tmp_path / "config" / "profiles" / "father.yaml"
    profile.write_text(profile.read_text().replace(
        "everyone_knows_device_records: true", "everyone_knows_device_records: false"))
    drop(cfg, "dinner.txt", FAMILY_DINNER)
    assert cli(cfg, "run") == 0
    assert rows(cfg, stage="quarantined"), "the static gate did not fire"

    assert cli(cfg, "quarantine") == 0
    out = capsys.readouterr().out
    assert "Standing consent gate is off (1):" in out
    assert "profile 'father' has family_consent set to false" in out


def test_the_listing_reads_why_md_when_the_index_row_is_gone(held, capsys):
    cfg, quarantined = held
    db = Database(cfg.path("database"))
    try:
        db.delete_recording(quarantined)
    finally:
        db.close()

    assert cli(cfg, "quarantine") == 0
    out = capsys.readouterr().out
    assert quarantined in out
    assert "No consent announcement detected (1):" in out
    why = (cfg.path("quarantine") / quarantined / "WHY.md").read_text(encoding="utf-8")
    reasons = [line[2:] for line in why.split("## Reasons", 1)[1].split("## ", 1)[0].splitlines()
               if line.startswith("- ")]
    assert reasons, "WHY.md carries no reasons to distil"
    distilled = next(r for r in reasons if "QUARANTINED" not in r and "local-only" not in r)
    assert distilled.split(". ")[0][:110] in out


def test_a_folder_with_no_why_md_and_no_index_row_is_still_listed(tmp_path, monkeypatch,
                                                                  capsys):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    stray = cfg.path("quarantine") / "rec_stray"
    stray.mkdir(parents=True)
    (stray / "mystery.mp3").write_bytes(b"\x00")

    assert cli(cfg, "quarantine") == 0
    out = capsys.readouterr().out
    assert "rec_stray" in out
    assert "mystery.mp3" in out
    assert "no consent announcement detected" in out


def test_classify_verdict_falls_back_when_every_reason_is_scaffolding():
    boilerplate = ["QUARANTINED. Something.", "local-only processing enforced",
                   "profile x governs the whole recording"]
    assert _classify_verdict("", boilerplate) == ("no-announcement",
                                                  "no consent announcement detected")
    assert _classify_verdict("", []) == ("no-announcement", "no consent announcement detected")
    assert _classify_verdict("refused", boilerplate)[0] == "refusal"


def test_forget_all_on_an_empty_quarantine_is_a_sentence_not_an_error(tmp_path, monkeypatch,
                                                                      capsys):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    assert cli(cfg, "quarantine", "--forget-all", "--yes") == 0
    assert "quarantine is empty; nothing to forget" in capsys.readouterr().out


def test_forget_all_reports_a_refused_deletion_and_leaves_the_index_alone(held, monkeypatch,
                                                                          capsys):
    """
    The processed call left encrypted ledgers behind. With the vault locked,
    Archive.forget refuses the whole operation; the bulk verb has to surface
    that refusal as a failure and an exit 1 rather than claim a clean sweep.
    """
    cfg, quarantined = held
    monkeypatch.delenv("PLAUD_BRIDGE_PASSPHRASE", raising=False)

    assert cli(cfg, "quarantine", "--forget-all", "--yes") == 1
    out = capsys.readouterr().out
    assert "deleted 0 file(s) across 1 recording(s)" in out
    assert "the vault is locked" in out
    assert load(cfg, quarantined) is not None
    assert (cfg.path("quarantine") / quarantined).is_dir()


# =========================================================================
# retention --execute
# =========================================================================
def _expire_everything(cfg) -> list[Path]:
    past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    db = Database(cfg.path("database"))
    try:
        with db.tx() as cur:
            cur.execute("UPDATE artifacts SET expires_at=?", (past,))
        return [Path(r["path"]) for r in db.expired_artifacts()]
    finally:
        db.close()


def test_retention_execute_deletes_only_after_the_typed_word(sandbox, monkeypatch, capsys):
    expired = _expire_everything(sandbox)
    assert expired and all(p.exists() for p in expired)

    monkeypatch.setattr("builtins.input", lambda *_: "yes please")
    assert cli(sandbox, "retention", "--execute") == 1
    assert all(p.exists() for p in expired), "a wrong confirmation deleted something"

    monkeypatch.setattr("builtins.input", lambda *_: "DELETE")
    assert cli(sandbox, "retention", "--execute") == 0
    assert f"deleted {len(expired)} artifact(s)" in capsys.readouterr().out
    assert not any(p.exists() for p in expired)
    assert len(audit(sandbox, action="retention_delete")) == len(expired)


# =========================================================================
# memory
# =========================================================================
def test_memory_rejects_an_unknown_profile(sandbox, capsys):
    assert cli(sandbox, "memory", "--profile", "ghost") == 1
    assert "unknown profile 'ghost'" in capsys.readouterr().out


def test_memory_forget_names_the_ledgers_it_touched(sandbox, capsys):
    rid = one_id(sandbox, profile_id="insurance_agent")
    assert cli(sandbox, "memory", "--forget", rid) == 0
    out = capsys.readouterr().out
    assert f"removed {rid} from 1 ledger(s): insurance_agent" in out

    capsys.readouterr()
    assert cli(sandbox, "memory", "--forget", rid) == 0
    assert f"removed {rid} from 0 ledger(s)" in capsys.readouterr().out


def test_memory_rebuild_replays_the_archive_and_saves(sandbox, capsys):
    assert cli(sandbox, "memory", "--rebuild") == 0
    out = capsys.readouterr().out
    assert "replayed 2" in out.lower() or "2 recording" in out


def test_memory_rebuild_refuses_to_save_a_ledger_it_could_not_fully_rebuild(sandbox, monkeypatch,
                                                                            capsys):
    from plaud_bridge.memory import MemoryStore

    ledger_dir = MemoryStore(sandbox).dir
    stamps = {p: p.stat().st_mtime_ns for p in ledger_dir.glob("*.enc")}
    assert stamps, "the run wrote no ledgers to protect"
    monkeypatch.delenv("PLAUD_BRIDGE_PASSPHRASE", raising=False)

    assert cli(sandbox, "memory", "--rebuild") == 1
    captured = capsys.readouterr()
    assert "PLAUD_BRIDGE_PASSPHRASE" in captured.err
    assert {p: p.stat().st_mtime_ns for p in ledger_dir.glob("*.enc")} == stamps, (
        "an incomplete rebuild overwrote a ledger"
    )


def test_memory_reports_ledgers_it_cannot_open_and_exits_one(sandbox, monkeypatch, capsys):
    monkeypatch.delenv("PLAUD_BRIDGE_PASSPHRASE", raising=False)
    assert cli(sandbox, "memory") == 1
    err = capsys.readouterr().err
    assert "ledger is on disk but cannot be opened" in err
    assert "PLAUD_BRIDGE_PASSPHRASE" in err


# =========================================================================
# new-profile, voices
# =========================================================================
@pytest.mark.parametrize("bad", ["not an identifier", "_private", "9lives"])
def test_new_profile_rejects_an_id_that_cannot_be_a_profile(sandbox, bad, capsys):
    assert cli(sandbox, "new-profile", bad) == 1
    assert "valid Python identifier" in capsys.readouterr().out
    assert not list((sandbox.root / "config" / "profiles").glob(f"*{bad.split()[0]}*"))


def test_voices_says_so_when_no_pack_is_installed(sandbox, capsys):
    shutil.rmtree(sandbox.root / "config" / "voice")
    assert cli(sandbox, "voices") == 0
    out = capsys.readouterr().out
    assert "no voice packs found; using the built-in defaults" in out


# =========================================================================
# speakers
# =========================================================================
def store_for(cfg) -> VoiceprintStore:
    return VoiceprintStore(Vault(cfg.path("vault")))


def _enroll(cfg, name: str, *values: float) -> None:
    store = store_for(cfg)
    store.enroll(name, vec(*values), source="clip.wav", seconds=12.0)
    store.save()


def _fake_model(monkeypatch, vector: list[float]) -> None:
    """The embedding model, replaced by a vector the test chose."""
    monkeypatch.setattr(Embedder, "available", staticmethod(lambda cfg: (True, "ready")))
    monkeypatch.setattr(Embedder, "require", lambda self: None)
    monkeypatch.setattr(Embedder, "embed", lambda self, a, s=None, e=None: list(vector))


def _no_ffmpeg_needed(monkeypatch, seconds: float = 30.0) -> None:
    monkeypatch.setattr("plaud_bridge.cli._prepared_audio", lambda cfg, src, work: src)
    monkeypatch.setattr("plaud_bridge.audio.probe_duration", lambda path, ffprobe="": seconds)


def test_speakers_list_shows_each_enrolled_voice_and_where_it_came_from(sandbox, capsys):
    _enroll(sandbox, "Marcus", 1.0, 0.0, 0.0)
    assert cli(sandbox, "speakers", "list") == 0
    out = capsys.readouterr().out
    assert "1 enrolled voice(s)" in out
    assert "Marcus" in out and "1 sample(s), 12s" in out
    assert "from clip.wav" in out


def test_speakers_forget_needs_the_name_typed_back(sandbox, monkeypatch, capsys):
    _enroll(sandbox, "Marcus", 1.0, 0.0, 0.0)

    monkeypatch.setattr("builtins.input", lambda *_: "marcus")
    assert cli(sandbox, "speakers", "forget", "Marcus") == 1
    assert store_for(sandbox).find("Marcus") is not None, "a wrong confirmation deleted a voice"

    monkeypatch.setattr("builtins.input", lambda *_: "Marcus")
    assert cli(sandbox, "speakers", "forget", "Marcus") == 0
    assert "Marcus is no longer recognised" in capsys.readouterr().out
    assert store_for(sandbox).find("Marcus") is None


def test_speakers_forget_an_unknown_name_lists_who_is_known(sandbox, capsys):
    _enroll(sandbox, "Marcus", 1.0, 0.0, 0.0)
    assert cli(sandbox, "speakers", "forget", "Dana", "--yes") == 1
    assert "nobody enrolled under 'Dana'. Known: Marcus" in capsys.readouterr().out
    assert store_for(sandbox).find("Marcus") is not None


def test_speakers_enroll_refuses_a_missing_clip(sandbox, tmp_path, capsys):
    assert cli(sandbox, "speakers", "enroll", "Marcus", "--audio", str(tmp_path / "no.wav")) == 1
    assert "no such file" in capsys.readouterr().out
    assert store_for(sandbox).is_empty()


def test_speakers_enroll_explains_when_the_model_is_unavailable(sandbox, tmp_path, monkeypatch,
                                                                capsys):
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"RIFF")
    monkeypatch.setattr(Embedder, "available",
                        staticmethod(lambda cfg: (False, "pyannote.audio is not installed")))
    assert cli(sandbox, "speakers", "enroll", "Marcus", "--audio", str(clip)) == 1
    assert "cannot enroll: pyannote.audio is not installed" in capsys.readouterr().out
    assert store_for(sandbox).is_empty()


def test_speakers_enroll_stores_the_vector_and_clears_the_scratch_copy(sandbox, tmp_path,
                                                                       monkeypatch, capsys):
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"RIFF")
    _fake_model(monkeypatch, vec(1.0, 0.0, 0.0))
    _no_ffmpeg_needed(monkeypatch, seconds=18.0)

    assert cli(sandbox, "speakers", "enroll", "Marcus", "--audio", str(clip)) == 0
    out = capsys.readouterr().out
    assert "Marcus enrolled from 18s of speech (1 sample(s) total)" in out
    person = store_for(sandbox).find("Marcus")
    assert person is not None and person.samples[0].source == "clip.wav"
    assert person.samples[0].seconds == 18.0
    assert not (sandbox.path("work") / "enroll").exists(), "the scratch copy lingered"


def test_speakers_enroll_with_a_span_counts_only_that_span(sandbox, tmp_path, monkeypatch, capsys):
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"RIFF")
    _fake_model(monkeypatch, vec(1.0, 0.0, 0.0))
    _no_ffmpeg_needed(monkeypatch, seconds=999.0)
    _enroll(sandbox, "Marcus", 0.9, 0.1, 0.0)

    assert cli(sandbox, "speakers", "enroll", "Marcus", "--audio", str(clip),
               "--start", "5", "--end", "25", "--replace") == 0
    out = capsys.readouterr().out
    assert "[5s-25s]" in out
    assert "Marcus enrolled from 20s of speech (1 sample(s) total)" in out
    person = store_for(sandbox).find("Marcus")
    assert len(person.samples) == 1 and person.samples[0].seconds == 20.0


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                    reason="needs ffmpeg on PATH")
def test_speakers_enroll_normalises_a_real_clip_through_ffmpeg(sandbox, tmp_path, monkeypatch,
                                                              capsys):
    clip = tmp_path / "clip.wav"
    with wave.open(str(clip), "wb") as fh:
        fh.setnchannels(2)
        fh.setsampwidth(2)
        fh.setframerate(44100)
        frames = bytes((i * 7) % 256 for i in range(44100 * 2 * 2 * 2))
        fh.writeframes(frames)
    _fake_model(monkeypatch, vec(1.0, 0.0, 0.0))

    assert cli(sandbox, "speakers", "enroll", "Marcus", "--audio", str(clip)) == 0
    assert "Marcus enrolled from 2s of speech" in capsys.readouterr().out
    assert not (sandbox.path("work") / "enroll").exists()


def test_speakers_identify_refuses_a_missing_file_and_an_empty_store(sandbox, tmp_path, capsys):
    assert cli(sandbox, "speakers", "identify", str(tmp_path / "no.wav")) == 1
    assert "no such file" in capsys.readouterr().out

    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"RIFF")
    assert cli(sandbox, "speakers", "identify", str(clip)) == 1
    assert "Nobody is enrolled" in capsys.readouterr().out


def test_speakers_identify_scores_the_whole_file_when_diarization_is_unavailable(
        sandbox, tmp_path, monkeypatch, capsys):
    from plaud_bridge.diarize import engine

    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"RIFF")
    _enroll(sandbox, "Marcus", 1.0, 0.0, 0.0)
    _enroll(sandbox, "Dana", 0.0, 1.0, 0.0)
    _fake_model(monkeypatch, vec(0.99, 0.14, 0.0))
    _no_ffmpeg_needed(monkeypatch, seconds=30.0)

    def unavailable(path, cfg):
        raise engine.DiarizationError("pyannote is not installed")
    monkeypatch.setattr(engine, "speaker_turns", unavailable)

    assert cli(sandbox, "speakers", "identify", str(clip)) == 0
    out = capsys.readouterr().out
    assert "cannot separate speakers: pyannote is not installed" in out
    assert "1 cluster(s)" in out
    assert "WHOLE FILE" in out and "-> Marcus" in out
    assert "* Marcus" in out and "  Dana" in out
    assert "Nothing was written" in out
    assert not (sandbox.path("work") / "identify").exists()
    assert len(store_for(sandbox).people()) == 2


def test_speakers_identify_reports_a_stranger_with_the_reason(sandbox, tmp_path, monkeypatch,
                                                              capsys):
    from plaud_bridge.diarize import engine

    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"RIFF")
    _enroll(sandbox, "Marcus", 1.0, 0.0, 0.0)
    _fake_model(monkeypatch, vec(0.0, 1.0, 0.0))
    _no_ffmpeg_needed(monkeypatch)
    monkeypatch.setattr(engine, "speaker_turns",
                        lambda path, cfg: [Segment(start=0.0, end=30.0, text="...",
                                                   speaker="SPEAKER_00")])

    assert cli(sandbox, "speakers", "identify", str(clip)) == 0
    out = capsys.readouterr().out
    assert "SPEAKER_00" in out and "-> unnamed" in out
    assert "below the" in out


def test_speakers_reports_a_locked_vault_instead_of_a_traceback(sandbox, monkeypatch, capsys):
    _enroll(sandbox, "Marcus", 1.0, 0.0, 0.0)
    monkeypatch.delenv("PLAUD_BRIDGE_PASSPHRASE", raising=False)
    assert cli(sandbox, "speakers", "list") == 1
    assert "PLAUD_BRIDGE_PASSPHRASE" in capsys.readouterr().err


def test_an_unknown_speakers_verb_is_exit_one_not_none(sandbox):
    args = SimpleNamespace(config=str(sandbox.root / "config"), log_level=None,
                           speakers_action="dance")
    assert cmd_speakers(args) == 1


# =========================================================================
# review
# =========================================================================
def test_review_reads_a_naive_timestamp_as_utc_and_flags_it_overdue(sandbox, capsys):
    db = Database(sandbox.path("database"))
    try:
        with db.tx() as cur:
            cur.execute(
                "INSERT INTO audit(at,recording_id,action,detail,actor) VALUES (?,?,?,?,?)",
                ("2020-01-01T00:00:00", None, "consent_reaffirm", "father", "human"),
            )
    finally:
        db.close()

    assert cli(sandbox, "review") == 0
    out = capsys.readouterr().out
    assert "never reaffirmed" not in out.split("Husband")[0]
    assert "[DUE] Father" in out
    assert "(every 30d)" in out
    assert "run.py review --reaffirm father" in out


def test_review_says_when_no_profile_carries_a_standing_consent_block(tmp_path, monkeypatch,
                                                                       capsys):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    for name in ("father", "husband"):
        profile = tmp_path / "config" / "profiles" / f"{name}.yaml"
        profile.write_text(profile.read_text().replace("reaffirm_every_days: 30",
                                                       "reaffirm_every_days: 0"))
    assert cli(cfg, "review") == 0
    out = capsys.readouterr().out
    assert "no profiles carry a standing consent block" in out
    assert "--reaffirm" not in out


def test_review_lists_statements_needing_review_and_skips_other_profiles_analyses(
        tmp_path, monkeypatch, capsys):
    """
    A plaintext work profile carries its fields in the index, so the review
    can quote the flagged statement itself. The recording also matched
    sales_trainer, whose analysis sits in the same payload and has to be
    skipped rather than searched for a field it does not have.
    """
    stub = ScriptedStub(
        scores={"insurance_agent": 0.92, "sales_trainer": 0.9, "father": 0.0, "husband": 0.0},
        extra_fields={"statements_needing_review": [
            {"timestamp": "00:39", "speaker": "Marcus", "text": "The price is my worry honestly."},
        ]},
    )
    cfg, _ = build_sandbox(tmp_path, monkeypatch, stub=stub)
    profile = tmp_path / "config" / "profiles" / "insurance_agent.yaml"
    profile.write_text(profile.read_text().replace("encrypt_at_rest: true",
                                                   "encrypt_at_rest: false"))
    drop(cfg, "client.txt", CLIENT_CALL)
    assert cli(cfg, "run") == 0
    payload = load(cfg, rows(cfg)[0]["id"])
    assert {a["profile_id"] for a in payload["analyses"]} >= {"insurance_agent", "sales_trainer"}

    assert cli(cfg, "review") == 0
    out = capsys.readouterr().out
    assert out.count("[!!] client.txt") == 1
    assert "The price is my worry honestly." in out
    assert "nothing flagged" not in out


def _unfiled_sandbox(tmp_path, monkeypatch, keywords: list[str]):
    stub = ScriptedStub(scores={"insurance_agent": 0.0, "sales_trainer": 0.0,
                                "father": 0.0, "husband": 0.0},
                        unfiled_fields={"suggested_keywords": keywords})
    cfg, _ = build_sandbox(tmp_path, monkeypatch, stub=stub)
    drop(cfg, "garden.txt", GARDEN_CHAT)
    assert cli(cfg, "run") == 0
    row = rows(cfg)[0]
    assert row["governing_profile"] == "unfiled" and row["encrypted"]
    return cfg


def test_review_harvests_keywords_from_encrypted_unfiled_recordings(tmp_path, monkeypatch,
                                                                    capsys):
    """
    The bug this pins: unfiled encrypts at rest, so the index withholds its
    fields, and the harvest read the index alone. In the shipped config it
    answered "no keyword suggestions" every single month.
    """
    cfg = _unfiled_sandbox(tmp_path, monkeypatch, ["Tomatoes", "compost", "tomatoes"])

    assert cli(cfg, "review") == 0
    out = capsys.readouterr().out
    assert "1 recording(s) the router could not place" in out
    assert "tomatoes (2), compost (1)" in out
    assert "add the keywords above to the right profile's routing.keywords" in out
    assert "no keyword suggestions" not in out


def test_review_says_when_unfiled_recordings_could_not_be_opened(tmp_path, monkeypatch, capsys):
    """Locked out is not the same as empty, and the review must not say it is."""
    cfg = _unfiled_sandbox(tmp_path, monkeypatch, ["tomatoes"])
    monkeypatch.delenv("PLAUD_BRIDGE_PASSPHRASE", raising=False)

    assert cli(cfg, "review") == 0
    out = capsys.readouterr().out
    assert "1 of them could not be opened" in out
    assert "PLAUD_BRIDGE_PASSPHRASE" in out
    assert "no keyword suggestions" not in out
    assert "tomatoes" not in out


def test_review_points_at_search_when_unfiled_recordings_suggest_nothing(tmp_path, monkeypatch,
                                                                         capsys):
    cfg = _unfiled_sandbox(tmp_path, monkeypatch, [])

    assert cli(cfg, "review") == 0
    out = capsys.readouterr().out
    assert "1 recording(s) the router could not place" in out
    assert "no keyword suggestions; read them with `run.py search`" in out
    assert "keywords worth adding" not in out


# =========================================================================
# app
# =========================================================================
def test_app_serves_until_ctrl_c_and_prints_the_phone_link_in_phone_mode(sandbox, tmp_path,
                                                                        monkeypatch, capsys):
    from plaud_bridge.desktop import launch

    real_build = launch.build
    seen = {}

    def build(base_dir=None, host="127.0.0.1", port=0, phone=False):
        # Loopback bind only; phone mode is simulated by naming an address,
        # so the test never opens this machine to the network.
        app, httpd, url = real_build(base_dir=base_dir, host=host, port=port)
        if phone:
            app.enable_phone("192.0.2.9", httpd.server_address[1])

        def stop():
            seen["served"] = True
            raise KeyboardInterrupt
        httpd.serve_forever = stop
        return app, httpd, url
    monkeypatch.setattr(launch, "build", build)

    assert cli(sandbox, "app", "--home", str(tmp_path / "home"), "--phone") == 0
    out = capsys.readouterr().out
    assert seen.get("served"), "serve_forever was never entered"
    assert "Plaud Bridge is running." in out
    assert "http://127.0.0.1:" in out
    assert "http://192.0.2.9:" in out and "?token=" in out
    assert "Home network only" in out
    assert "Stopping." in out


# =========================================================================
# main
# =========================================================================
def test_ctrl_c_inside_a_command_is_exit_130(sandbox, monkeypatch, capsys):
    def interrupted(_args):
        raise KeyboardInterrupt
    monkeypatch.setattr("plaud_bridge.cli.cmd_status", interrupted)
    assert cli(sandbox, "status") == 130
    assert "interrupted" in capsys.readouterr().err


def test_running_the_module_as_a_script_exits_with_the_parsers_code(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["plaud-bridge", "--version"])
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("plaud_bridge.cli", run_name="__main__", alter_sys=True)
    assert excinfo.value.code == 0


# =========================================================================
# followups
# =========================================================================
def _open_followups(cfg) -> list:
    from plaud_bridge.followups import collect

    db = Database(cfg.path("database"))
    try:
        archive = Archive(cfg, db)
        return collect(cfg, db, archive, status="open", vault=archive.vault)
    finally:
        db.close()


def test_followups_done_records_the_status_and_the_wording(sandbox, capsys):
    items = _open_followups(sandbox)
    assert items, "the client call produced no follow-ups to mark"
    target = items[0]

    assert cli(sandbox, "followups", "--done", target.id) == 0
    assert f"{target.id} is now done: {target.text}" in capsys.readouterr().out
    assert target.id not in {i.id for i in _open_followups(sandbox)}

    assert cli(sandbox, "followups", "--reopen", target.id[:8]) == 0
    assert target.id in {i.id for i in _open_followups(sandbox)}


def test_followups_refuse_a_status_for_an_id_nobody_produced(sandbox, capsys):
    assert cli(sandbox, "followups", "--drop", "fu_nobody_said_this") == 1
    assert "no follow-up with id" in capsys.readouterr().err


def test_followups_draft_by_prefix_and_by_recording_write_drafts_nothing_sends(sandbox, tmp_path,
                                                                             capsys):
    items = _open_followups(sandbox)
    rid = one_id(sandbox, profile_id="insurance_agent")
    drafts = sandbox.path("outbox") / "drafts"

    assert cli(sandbox, "followups", "--draft", items[0].id[:10]) == 0
    out = capsys.readouterr().out
    assert "wrote " in out and "Nothing has been sent" in out
    written = list(drafts.glob("*"))
    assert len(written) == 1 and "DRAFT" in written[0].name

    dest = tmp_path / "chase.txt"
    assert cli(sandbox, "followups", "--draft", rid, "--format", "text",
               "--out", str(dest)) == 0
    assert dest.exists() and items[0].text in dest.read_text(encoding="utf-8")

    assert cli(sandbox, "followups", "--draft", "open") == 0
    assert len(list(drafts.glob("*"))) == 2
    assert len(audit(sandbox, action="followup_draft")) == 3


def test_followups_draft_with_no_such_prefix_writes_nothing(sandbox, capsys):
    assert cli(sandbox, "followups", "--draft", "fu_zzz") == 1
    assert "no follow-up here starts with 'fu_zzz'" in capsys.readouterr().out
    assert not list((sandbox.path("outbox") / "drafts").glob("*")) if (
        sandbox.path("outbox") / "drafts").exists() else True


def test_followups_draft_refuses_when_nothing_is_open(tmp_path, monkeypatch, capsys):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    assert cli(cfg, "followups", "--draft", "open") == 1
    assert "nothing to draft" in capsys.readouterr().err


def test_followups_worklist_can_be_written_to_a_file(sandbox, tmp_path, capsys):
    dest = tmp_path / "out" / "worklist.md"
    assert cli(sandbox, "followups", "--out", str(dest)) == 0
    assert f"wrote {dest}" in capsys.readouterr().out
    body = dest.read_text(encoding="utf-8")
    assert _open_followups(sandbox)[0].text in body
