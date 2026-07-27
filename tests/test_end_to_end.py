"""
End to end spine test.

Feeds a text transcript through the whole pipeline with a stubbed LLM, then
renders a digest. Text input skips ASR, which keeps the test hermetic while
still exercising ingest, correction, routing, the compliance gate, extraction,
encrypted persistence, and digest rendering.

The sandbox fixture and the sample transcripts live in `_fixtures.py`.
"""

import json
from pathlib import Path

from _fixtures import CLIENT_CALL, FAMILY_DINNER
from _fixtures import drop as _drop
from plaud_bridge.db import Database
from plaud_bridge.digest import DigestBuilder, DigestOptions
from plaud_bridge.models import Stage
from plaud_bridge.pipeline import Pipeline


def test_client_call_routes_analyses_and_encrypts(sandbox):
    cfg, stub = sandbox
    _drop(cfg, "client-marcus.txt", CLIENT_CALL)

    pipe = Pipeline(cfg)
    try:
        stats = pipe.run()
        assert stats.processed == 1, f"expected 1 processed, got {stats.summary()}"

        rows = pipe.db.query(profile_id="insurance_agent")
        assert len(rows) == 1
        payload = json.loads(rows[0]["payload_json"])

        assert payload["stage"] == Stage.COMPLETE.value
        assert payload["compliance"]["consent"] == "detected"
        assert payload["compliance"]["governing_profile"] == "insurance_agent"

        # insurance_agent is encrypt_at_rest, so the index holds the shape of
        # the analysis but not its contents.
        analysis = payload["analyses"][0]
        assert analysis["fields_withheld"] is True
        assert analysis["fields"] == {}

        # insurance_agent is encrypt_at_rest, so artifacts must be .enc
        for path in payload["artifact_paths"].values():
            assert path.endswith(".enc"), f"{path} was written in plaintext"
            assert Path(path).exists()

        # and must decrypt back to readable content
        from plaud_bridge.storage import Vault

        vault = Vault(cfg.path("vault"))
        stored = json.loads(
            vault.read_text(Path(payload["artifact_paths"]["analysis"]), payload["id"])
        )
        fields = stored["analyses"][0]["fields"]
        assert fields["next_action"].startswith("Send two quote")
        assert fields["objections"][0]["type"] == "price"

        text = vault.read_text(
            Path(payload["artifact_paths"]["transcript"]), payload["id"]
        )
        assert "Marcus" in text
        # glossary correction survived into the stored transcript
        assert "elimination period" in text
    finally:
        pipe.close()


def test_family_recording_never_touches_a_cloud_provider(sandbox):
    cfg, stub = sandbox
    _drop(cfg, "dinner-with-kid.txt", FAMILY_DINNER)

    pipe = Pipeline(cfg)
    try:
        stats = pipe.run()
        assert stats.processed == 1

        rows = pipe.db.query(profile_id="father")
        assert len(rows) == 1

        # Every LLM call made for this recording had to be local-only.
        assert stub.calls, "no LLM calls recorded"
        assert all(c["local_only"] for c in stub.calls), (
            "a family recording reached a call that permitted cloud providers: "
            f"{[c for c in stub.calls if not c['local_only']]}"
        )

        payload = json.loads(rows[0]["payload_json"])
        assert payload["compliance"]["force_local_processing"] is True
        assert payload["compliance"]["governing_profile"] == "father"
        for path in payload["artifact_paths"].values():
            assert path.endswith(".enc")
    finally:
        pipe.close()


def test_missing_consent_quarantines(sandbox):
    cfg, stub = sandbox
    no_consent = CLIENT_CALL.split("\n", 2)[2]  # strip the consent exchange
    _drop(cfg, "client-no-consent.txt", no_consent)

    pipe = Pipeline(cfg)
    try:
        stats = pipe.run()
        assert stats.quarantined == 1
        assert stats.processed == 0

        qdirs = list(cfg.path("quarantine").iterdir())
        assert len(qdirs) == 1
        why = (qdirs[0] / "WHY.md").read_text()
        assert "consent" in why.lower()
        assert "run.py release" in why
    finally:
        pipe.close()


def test_forcing_a_quarantined_file_again_does_not_orphan_a_second_folder(sandbox):
    """
    `--force` means process this file again, not pretend it is a different file.

    It used to mint a fresh recording id, which wrote a second quarantine folder
    and then failed to index it because content_hash is UNIQUE -- leaving a
    folder on disk under an id that no index knew about, so `status`, `audit`,
    and `review` never mentioned it while `release` would still have put it back
    in the inbox.
    """
    cfg, _ = sandbox
    no_consent = CLIENT_CALL.split("\n", 2)[2]
    _drop(cfg, "client-no-consent.txt", no_consent)

    pipe = Pipeline(cfg)
    try:
        assert pipe.run().quarantined == 1
        first = {p.name for p in cfg.path("quarantine").iterdir()}
        assert len(first) == 1

        # The file is still in the inbox, because quarantine does not archive it.
        assert pipe.run(force=True).quarantined == 1
        assert {p.name for p in cfg.path("quarantine").iterdir()} == first

        indexed = {row["id"] for row in Database(cfg.path("database")).query(limit=50)}
        assert first <= indexed, "a quarantine folder exists that the index does not know about"
    finally:
        pipe.close()


def test_dedupe_skips_identical_file(sandbox):
    cfg, _ = sandbox
    _drop(cfg, "client-marcus.txt", CLIENT_CALL)
    pipe = Pipeline(cfg)
    try:
        assert pipe.run().processed == 1
        _drop(cfg, "client-marcus-copy.txt", CLIENT_CALL)
        stats = pipe.run()
        assert stats.skipped == 1 and stats.processed == 0
    finally:
        pipe.close()


def test_digest_excludes_personal_by_default_and_suppresses_sensitive_fields(sandbox):
    cfg, _ = sandbox
    _drop(cfg, "client-marcus.txt", CLIENT_CALL)
    _drop(cfg, "dinner-with-kid.txt", FAMILY_DINNER)

    pipe = Pipeline(cfg)
    try:
        pipe.run()
        builder = DigestBuilder(cfg, pipe.db)

        combined = builder.render_markdown(DigestOptions(days=30))
        assert "Production" in combined
        assert "Family" not in combined, "family content leaked into the combined digest"
        # suppressed fields must not print their contents
        assert "four hundred thousand" not in combined
        assert "withheld from this view" in combined
        assert "Send two quote options by Thursday" in combined

        with_personal = builder.render_markdown(DigestOptions(days=30, include_personal=True))
        assert "Family" in with_personal
        assert "permission slip" in with_personal

        only_dad = builder.render_markdown(DigestOptions(profile_id="father", days=30))
        assert "Family" in only_dad
        assert "Production" not in only_dad
    finally:
        pipe.close()


def test_digest_surfaces_next_actions_at_the_top(sandbox):
    cfg, _ = sandbox
    _drop(cfg, "client-marcus.txt", CLIENT_CALL)
    pipe = Pipeline(cfg)
    try:
        pipe.run()
        md = DigestBuilder(cfg, pipe.db).render_markdown(DigestOptions(days=30))
        needs = md.split("## At a Glance")[0]
        assert "Needs You" in needs
        assert "Send two quote options by Thursday" in needs
    finally:
        pipe.close()
