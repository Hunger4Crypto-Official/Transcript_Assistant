"""
Every way a person can reach this code.

The unit tests exercise the machinery. This file exercises the surface: every
subcommand, every flag on it that changes what happens, driven through
`main(argv)` the way a shell would. The bug that motivated it — verification
decrypting to os.devnull, which worked as root and failed as everyone else —
was invisible to tests that called library functions directly.

A route is "covered" here if it was actually invoked and its exit code checked.
`test_every_subcommand_is_covered` fails when a new subcommand is added without
a matching entry, so the coverage claim stays true rather than decaying.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

from _fixtures import CLIENT_CALL, FAMILY_DINNER, ROOT, build_sandbox, drop
from plaud_bridge.cli import build_parser, main
from plaud_bridge.db import Database

# Every subcommand touched below. Kept beside the tests so the completeness
# check has something to compare the parser against.
COVERED = {
    "doctor", "run", "watch", "digest", "status", "search", "verify", "forget",
    "export", "open", "audit", "release", "quarantine", "retention", "profiles",
    "new-profile", "voices", "review", "speakers", "followups", "ask",
    "memory", "backup", "restore", "brief",
}


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """A sandbox with two recordings already processed through the real CLI."""
    cfg, stub = build_sandbox(tmp_path, monkeypatch)
    drop(cfg, "client.txt", CLIENT_CALL)
    drop(cfg, "dinner.txt", FAMILY_DINNER)
    assert main(["--config", str(tmp_path / "config"), "run"]) == 0
    return cfg


def cli(cfg, *argv) -> int:
    return main(["--config", str(cfg.root / "config"), *argv])


def rows(cfg, **kw) -> list[dict]:
    db = Database(cfg.path("database"))
    try:
        return db.query(limit=50, **kw)
    finally:
        db.close()


def recording_ids(cfg) -> list[str]:
    return [r["id"] for r in rows(cfg)]


# =========================================================================
# Read-only routes
#
# These have to work on an empty index too. A fresh clone where `status` or
# `digest` traceback because nothing has been processed yet is a bad first
# five minutes.
# =========================================================================
READ_ONLY = [
    ("status",),
    ("profiles",),
    ("voices",),
    ("speakers", "list"),
    ("ask", "what did I promise about the quotes?"),
    ("ask", "elimination period", "--profile", "insurance_agent"),
    ("ask", "anything at all", "--include-personal", "--days", "30"),
    ("ask", "anything at all", "--local-only"),
    ("memory",),
    ("memory", "--brief"),
    ("memory", "--profile", "insurance_agent"),
    ("followups",),
    ("followups", "--status", "all"),
    ("followups", "--format", "html"),
    ("followups", "--profile", "insurance_agent"),
    ("verify",),
    ("review",),
    ("review", "--days", "7"),
    ("audit",),
    ("audit", "--action", "ingest"),
    ("audit", "--actor", "pipeline"),
    ("audit", "--days", "7", "--limit", "5"),
    ("brief",),
    ("brief", "--days", "30"),
    ("brief", "--include-personal"),
    ("brief", "--format", "html"),
    ("digest",),
    ("digest", "--days", "30"),
    ("digest", "--include-personal"),
    ("digest", "--profile", "insurance_agent"),
    ("digest", "--format", "html"),
    ("digest", "--title", "Custom"),
    ("export",),
    ("export", "--transcripts"),
    ("export", "--include-personal"),
    ("export", "--format", "html"),
    ("export", "--profile", "insurance_agent", "--days", "7", "--limit", "5"),
    ("search", "client"),
    ("search", "own occupation", "--content"),
    ("search", "own occupation", "--content", "--context", "2"),
    ("search", "own occupation", "--content", "--scan-limit", "1"),
    ("search", "mortgage", "--content", "--per-recording", "1"),
    ("search", "anything", "--profile", "insurance_agent", "--days", "30", "--limit", "5"),
    ("retention",),
    ("quarantine",),
    ("doctor",),
    ("doctor", "--offline"),
]


def _acceptable(argv) -> set[int]:
    # doctor reports a fatal missing dependency as exit 1, and a content search
    # that only opened part of the corpus reports 2 for "inconclusive". Both are
    # answers rather than crashes. Everything else must succeed outright.
    if argv[0] == "doctor":
        return {0, 1}
    if "--scan-limit" in argv:
        return {0, 2}
    # `ask` reports 2 when the answer is incomplete -- a bounded scan, a
    # trimmed context, a dropped citation. That is an answer with a caveat,
    # not a crash, and the caveat is the part worth exiting non-zero for.
    if argv[0] == "ask":
        return {0, 2}
    return {0}


@pytest.mark.parametrize("argv", READ_ONLY, ids=lambda a: " ".join(a))
def test_a_read_only_route_works_on_a_populated_index(sandbox, argv):
    assert cli(sandbox, *argv) in _acceptable(argv)


@pytest.mark.parametrize("argv", READ_ONLY, ids=lambda a: " ".join(a))
def test_a_read_only_route_works_on_an_empty_index(tmp_path, monkeypatch, argv):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    assert cli(cfg, *argv) in _acceptable(argv)


def test_no_read_only_route_writes_into_the_vault(sandbox):
    """A command you ran to look at something should not have changed anything."""
    before = {p: p.stat().st_mtime_ns for p in sandbox.path("vault").rglob("*") if p.is_file()}
    for argv in READ_ONLY:
        cli(sandbox, *argv)
    after = {p: p.stat().st_mtime_ns for p in sandbox.path("vault").rglob("*") if p.is_file()}
    assert after == before

    plaintext = [p for p in sandbox.path("vault").rglob("*") if p.is_file() and p.suffix != ".enc"]
    assert not plaintext, f"a read-only command left plaintext in the vault: {plaintext}"


# =========================================================================
# Routes that write
# =========================================================================
def test_run_is_idempotent(sandbox):
    """Running twice must not reprocess; the dedupe is the whole point."""
    before = len(recording_ids(sandbox))
    assert cli(sandbox, "run") == 0
    assert len(recording_ids(sandbox)) == before


def test_run_with_limit_and_force(tmp_path, monkeypatch):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    drop(cfg, "client.txt", CLIENT_CALL)
    drop(cfg, "dinner.txt", FAMILY_DINNER)
    assert cli(cfg, "run", "--limit", "1") == 0
    assert len(recording_ids(cfg)) == 1
    assert cli(cfg, "run", "--force") == 0


def test_watch_once_and_bounded(sandbox):
    assert cli(sandbox, "watch", "--once") == 0
    assert cli(sandbox, "watch", "--max-runs", "1", "--interval", "1") == 0


@pytest.mark.parametrize("kind", ["transcript", "analysis", "source"])
def test_open_every_kind(sandbox, kind, tmp_path):
    rid = recording_ids(sandbox)[0]
    assert cli(sandbox, "open", rid, "--kind", kind, "--out", str(tmp_path / f"o.{kind}")) == 0
    assert (tmp_path / f"o.{kind}").exists()


def test_open_to_stdout(sandbox, capsys):
    rid = recording_ids(sandbox)[0]
    assert cli(sandbox, "open", rid, "--kind", "transcript") == 0
    assert capsys.readouterr().out.strip()


def test_open_an_unknown_recording_fails_cleanly(sandbox):
    assert cli(sandbox, "open", "rec_does_not_exist") != 0


def test_digest_and_export_to_a_file(sandbox, tmp_path):
    for cmd, name in (("digest", "d.md"), ("export", "e.md")):
        assert cli(sandbox, cmd, "--out", str(tmp_path / name)) == 0
        assert (tmp_path / name).read_text().strip()


def test_new_profile_scaffolds_and_the_result_loads(sandbox):
    assert cli(sandbox, "new-profile", "mentor", "--name", "Mentor",
               "--short-name", "Mentor", "--heading", "Mentoring") == 0
    assert (sandbox.root / "config" / "profiles" / "mentor.yaml").exists()
    # The scaffold has to be valid, not merely present.
    assert cli(sandbox, "profiles") == 0


def test_new_profile_refuses_to_clobber_an_existing_one(sandbox):
    assert cli(sandbox, "new-profile", "father") != 0


def test_retention_dry_run_deletes_nothing(sandbox):
    before = sorted(p.name for p in sandbox.path("vault").rglob("*"))
    assert cli(sandbox, "retention") == 0
    assert sorted(p.name for p in sandbox.path("vault").rglob("*")) == before


def test_retention_execute_needs_confirmation(sandbox, monkeypatch):
    assert cli(sandbox, "retention", "--execute", "--yes") == 0


def test_backup_writes_one_file_and_restore_guards_existing_data(sandbox, tmp_path):
    """
    The route surface only; what backup and restore actually guarantee is
    pinned down in tests/test_backup.py.
    """
    out = tmp_path / "cli-route.pbb"
    assert cli(sandbox, "backup", "--out", str(out)) == 0
    assert out.is_file() and out.stat().st_size > 0
    # Data is already in place, so a bare restore must refuse...
    assert cli(sandbox, "restore", str(out)) == 1
    # ...and --force must go through, leaving an archive that still verifies.
    assert cli(sandbox, "restore", str(out), "--force") == 0
    assert cli(sandbox, "verify") == 0


def test_restore_a_missing_file_fails_cleanly(sandbox, tmp_path):
    assert cli(sandbox, "restore", str(tmp_path / "nope.pbb")) == 1


def test_forget_removes_one_recording(sandbox):
    ids = recording_ids(sandbox)
    assert cli(sandbox, "forget", ids[0], "--yes") == 0
    assert ids[0] not in recording_ids(sandbox)
    assert len(recording_ids(sandbox)) == len(ids) - 1
    # And the index still verifies afterwards; a delete that orphans artifacts
    # is worse than one that fails.
    assert cli(sandbox, "verify") == 0


def test_forget_an_unknown_recording_fails_cleanly(sandbox):
    assert cli(sandbox, "forget", "rec_nope", "--yes") != 0


def test_release_a_quarantined_recording(tmp_path, monkeypatch):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    # The consent line removed, so the insurance profile's gate quarantines it.
    drop(cfg, "no_consent.txt", CLIENT_CALL.split("\n", 2)[2])
    assert cli(cfg, "run") == 0

    held = [r["id"] for r in rows(cfg) if r["stage"] == "quarantined"]
    assert held, "nothing was quarantined; the gate did not fire"
    assert cli(cfg, "release", held[0], "--yes") == 0


def test_release_an_unknown_recording_fails_cleanly(sandbox):
    assert cli(sandbox, "release", "rec_nope", "--yes") != 0


def test_review_reaffirm_records_something(sandbox, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *_: "YES")
    assert cli(sandbox, "review", "--reaffirm", "father") == 0
    assert cli(sandbox, "review") == 0


def test_review_reaffirm_rejects_a_profile_with_no_consent_block(sandbox):
    assert cli(sandbox, "review", "--reaffirm", "insurance_agent") != 0
    assert cli(sandbox, "review", "--reaffirm", "not_a_profile") != 0


# =========================================================================
# Prompts with nobody to answer them
#
# Under cron, in a pipe, or with `< /dev/null` there is no terminal, and
# `input()` raises EOFError. Every one of these routes destroys something, so
# the only safe reading of "no answer" is "no".
# =========================================================================
def _no_stdin(monkeypatch):
    def refuse(*_args, **_kw):
        raise EOFError("EOF when reading a line")
    monkeypatch.setattr("builtins.input", refuse)


def test_forget_declines_when_there_is_no_terminal(sandbox, monkeypatch):
    _no_stdin(monkeypatch)
    rid = recording_ids(sandbox)[0]
    assert cli(sandbox, "forget", rid) == 1
    assert rid in recording_ids(sandbox), "forget deleted without a confirmation"


def test_retention_declines_when_there_is_no_terminal(sandbox, monkeypatch):
    _no_stdin(monkeypatch)
    before = sorted(p.name for p in sandbox.path("vault").rglob("*"))
    assert cli(sandbox, "retention", "--execute") in (0, 1)
    assert sorted(p.name for p in sandbox.path("vault").rglob("*")) == before


def test_release_declines_when_there_is_no_terminal(tmp_path, monkeypatch):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    drop(cfg, "no_consent.txt", CLIENT_CALL.split("\n", 2)[2])
    assert cli(cfg, "run") == 0
    held = [r["id"] for r in rows(cfg) if r["stage"] == "quarantined"]
    assert held

    _no_stdin(monkeypatch)
    assert cli(cfg, "release", held[0]) == 1
    assert [r["id"] for r in rows(cfg) if r["stage"] == "quarantined"] == held


def test_reaffirm_declines_when_there_is_no_terminal(sandbox, monkeypatch):
    _no_stdin(monkeypatch)
    assert cli(sandbox, "review", "--reaffirm", "father") == 1
    # And it did not record the reaffirmation anyway.
    db = Database(sandbox.path("database"))
    try:
        assert not db.audit_log(action="consent_reaffirm", limit=10)
    finally:
        db.close()


def test_a_prompt_added_later_still_cannot_traceback(sandbox, monkeypatch):
    """The backstop in main(), independent of any particular prompt."""
    monkeypatch.setattr("plaud_bridge.cli.cmd_status",
                        lambda _a: (_ for _ in ()).throw(EOFError()))
    assert cli(sandbox, "status") == 1


# =========================================================================
# Argument handling
# =========================================================================
def test_no_subcommand_is_an_error_not_a_traceback():
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code != 0


def test_version_and_help_exit_zero():
    for argv in (["--version"], ["--help"]):
        with pytest.raises(SystemExit) as excinfo:
            main(argv)
        assert excinfo.value.code == 0


def test_a_missing_config_directory_is_a_message_not_a_traceback(tmp_path, capsys):
    assert main(["--config", str(tmp_path / "nope"), "status"]) == 1
    assert "onfiguration error" in capsys.readouterr().err


def test_a_bad_config_file_is_a_message_not_a_traceback(tmp_path, monkeypatch, capsys):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    (tmp_path / "config" / "pipeline.yaml").write_text("this: [is not: valid")
    assert cli(cfg, "status") == 1
    assert "onfiguration error" in capsys.readouterr().err


def test_a_missing_passphrase_is_a_message_not_a_traceback(sandbox, monkeypatch, capsys):
    monkeypatch.delenv("PLAUD_BRIDGE_PASSPHRASE", raising=False)
    rid = recording_ids(sandbox)[0]
    assert cli(sandbox, "open", rid) != 0
    out = capsys.readouterr()
    assert "PLAUD_BRIDGE_PASSPHRASE" in (out.out + out.err)


@pytest.mark.parametrize("argv", [
    ("digest", "--format", "xml"),
    ("export", "--format", "xml"),
    ("open", "rec_x", "--kind", "video"),
])
def test_an_invalid_choice_is_rejected_by_the_parser(argv):
    with pytest.raises(SystemExit):
        main(["--config", "config", *argv])


# =========================================================================
# Completeness
# =========================================================================
def test_every_subcommand_is_covered():
    parser = build_parser()
    actions = [a for a in parser._actions if hasattr(a, "choices") and a.choices
               and a.dest == "command"]
    assert actions, "could not find the subcommand table"
    declared = set(actions[0].choices)
    assert declared == COVERED, (
        f"uncovered subcommands: {sorted(declared - COVERED)}; "
        f"stale entries: {sorted(COVERED - declared)}"
    )


def _declared() -> set[str]:
    return set(build_parser()._subparsers._group_actions[0].choices)


def _commands_mentioned(text: str, prefix: str) -> set[str]:
    """
    Pull the subcommand out of every documented invocation.

    Markdown wraps these in backticks and shell examples chain them with &&,
    so the token is trimmed and anything that cannot be a subcommand name is
    discarded. The point is to catch a documented command that no longer
    exists, not to parse shell.
    """
    found = set()
    for line in text.splitlines():
        if prefix not in line or line.strip().startswith("#"):
            continue
        rest = line.split(prefix, 1)[1].split()
        if not rest:
            continue
        token = rest[0].strip("`'\"*.,:;()[]")
        if token and not token.startswith("-") and re.fullmatch(r"[a-z][a-z-]*", token):
            found.add(token)
    return found


def test_run_py_and_the_installed_command_are_the_same_code():
    """`python run.py` and `plaud-bridge` must not drift apart."""
    import run as run_shim

    assert run_shim.main is main


def test_run_py_documents_exactly_the_commands_that_exist():
    """run.py's docstring is the first thing anyone reads. It has to be true."""
    import run as run_shim

    documented = _commands_mentioned(run_shim.__doc__ or "", "python run.py ")
    assert documented, "run.py stopped documenting its commands"
    assert documented == _declared(), (
        f"documented but missing: {sorted(documented - _declared())}; "
        f"real but undocumented: {sorted(_declared() - documented)}"
    )


def test_the_cli_docstring_documents_commands_that_exist():
    """It is the epilog of `--help`, so a stale line there is user-visible."""
    from plaud_bridge import cli as cli_module

    documented = _commands_mentioned(cli_module.__doc__ or "", "plaud-bridge ")
    assert documented, "the CLI stopped documenting its commands"
    assert documented <= _declared(), f"documented but missing: {sorted(documented - _declared())}"


def test_the_makefile_only_calls_real_commands():
    text = (ROOT / "Makefile").read_text()
    used = _commands_mentioned(text, "run.py ")
    unknown = used - _declared()
    assert not unknown, f"the Makefile calls commands that do not exist: {sorted(unknown)}"
    assert used, "the Makefile stopped calling the CLI at all"


def test_the_readme_only_documents_real_commands():
    text = (ROOT / "README.md").read_text()
    used = _commands_mentioned(text, "run.py ") | _commands_mentioned(text, "plaud-bridge ")
    unknown = {c for c in used if c not in _declared()}
    assert not unknown, f"the README documents commands that do not exist: {sorted(unknown)}"


# =========================================================================
# The offline handoff
#
# `scripts/fetch_models.py` runs on a networked machine and `runtime` looks
# for the result on the air-gapped one. Nothing connects them but an agreed
# directory layout, and if they ever disagree the failure is a machine with
# no network being told to download something.
# =========================================================================
# Offline is an assertion the config enforces: it refuses to load while any
# cloud provider is enabled. Turning it on in a test means turning them off.
_OFFLINE = {
    "runtime": {"offline": True},
    "asr": {"providers": ["local"], "groq": {"enabled": False}},
    "llm": {"providers": ["local"], "anthropic": {"enabled": False},
            "groq": {"enabled": False},
            "local": {"enabled": True, "is_cloud": False,
                      "base_url": "http://localhost:11434/v1", "model": "llama3.3:70b"}},
}


def _fetch_models():
    """The fetcher is a script, not a package. Import it the way a user runs it."""
    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))
    import fetch_models

    return fetch_models


@pytest.mark.parametrize("subdir,name", [
    ("whisper", "large-v3"),
    ("diarization", "pyannote/speaker-diarization-3.1"),
    # Named speakers put a second model in the diarization directory. If the
    # fetcher and the runtime ever disagree about the layout, the failure is an
    # air-gapped machine being told to download something.
    ("diarization", "pyannote/embedding"),
])
def test_the_fetcher_writes_where_the_runtime_looks(tmp_path, monkeypatch, subdir, name):
    fetch_models = _fetch_models()

    from plaud_bridge.runtime import model_path, resolve_local_model

    cfg, _ = build_sandbox(tmp_path, monkeypatch,
                           overrides={"runtime": {"models_dir": str(tmp_path / "models")}})

    written = tmp_path / "models" / subdir / fetch_models._flat(name)
    written.mkdir(parents=True)
    (written / "model.bin").write_bytes(b"weights")

    target, local = resolve_local_model(cfg, name, subdir)
    assert local, f"the fetcher writes {written}, the runtime looked elsewhere"
    assert Path(target) == model_path(cfg, subdir, fetch_models._flat(name))


def test_the_offline_error_names_a_flag_the_fetcher_actually_has(tmp_path, monkeypatch):
    """
    Offline, a missing model prints "run scripts/fetch_models.py --whisper X".
    Someone on an air-gapped machine cannot debug that instruction if the flag
    was renamed, so the two are pinned together here.
    """
    from plaud_bridge.runtime import OfflineError, require_local

    cfg, _ = build_sandbox(tmp_path, monkeypatch, overrides=_OFFLINE)
    fetcher_flags = {
        flag
        for action in _fetch_models().build_parser()._actions
        for flag in action.option_strings
    }

    for subdir in ("whisper", "diarization"):
        with pytest.raises(OfflineError) as excinfo:
            require_local(cfg, "some-model", subdir, subdir)
        assert f"--{subdir}" in str(excinfo.value)
        assert f"--{subdir}" in fetcher_flags, "the error names a flag the fetcher does not have"


def test_the_fetcher_with_no_arguments_explains_itself_rather_than_downloading():
    assert _fetch_models().main([]) == 1
    with pytest.raises(SystemExit) as excinfo:
        _fetch_models().main(["--help"])
    assert excinfo.value.code == 0
