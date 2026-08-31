#!/usr/bin/env python3
"""
Plaud Bridge command line.

    plaud-bridge doctor                        preflight every dependency and key
    plaud-bridge run                           process everything in the inbox
    plaud-bridge watch                         keep processing on an interval
    plaud-bridge digest                        combined digest, last 7 days
    plaud-bridge digest --format html          self-contained page, prints cleanly
    plaud-bridge review                        what the review cadence says is due
    plaud-bridge followups                     commitments still open, oldest first
    plaud-bridge insights                      how you talk: share, pace, questions
    plaud-bridge status                        index summary
    plaud-bridge search "own occupation" --content    search what was actually said
    plaud-bridge ask "what did I promise Marcus?"      answer it, with citations
    plaud-bridge open <recording_id>           decrypt and print an artifact
    plaud-bridge verify                        confirm every artifact still opens
    plaud-bridge export                        redacted document for someone else
    plaud-bridge forget <recording_id>         delete one recording, permanently
    plaud-bridge memory                        what it has learned across recordings
    plaud-bridge audit                         read the compliance audit log
    plaud-bridge release <recording_id>        release a quarantined recording
    plaud-bridge quarantine                    triage everything in quarantine at once
    plaud-bridge retention --execute           delete expired artifacts
    plaud-bridge profiles                      show the routing table
    plaud-bridge new-profile <id>              scaffold a profile from the template
    plaud-bridge voices                        show installed voice packs
    plaud-bridge speakers enroll "Marcus" --audio clip.wav   teach it a voice
    plaud-bridge speakers identify <audio>     score a recording without writing anything

`python run.py <command>` runs the same code without installing anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .archive import Archive
from .ask import ask, save_answer
from .compliance import RetentionSweeper
from .config import Config, ConfigError
from .db import Database
from .digest import DigestBuilder, DigestOptions, fmt_value, to_html
from .followups import FollowUpError, collect, draft, render, set_status
from .insights import InsightsError, recording_metrics, render_recording, render_trend
from .insights import trend as insights_trend
from .logging_setup import setup
from .memory import MemoryStore, carry_forward_brief, render_ledger
from .models import format_stamp
from .pipeline import Pipeline
from .storage import Vault, VaultError
from .voice import Voice

OK, WARN, BAD = "  ok  ", " warn ", " FAIL "


def _confirm(prompt: str, expected: str) -> bool:
    """
    Ask for a typed confirmation before something irreversible.

    Under cron, in a pipe, or with stdin closed there is nobody to ask, and
    `input()` raises EOFError. Treating that as "not confirmed" is the only
    safe reading: a delete that proceeds because nobody was there to say no
    is the exact failure these prompts exist to prevent.
    """
    try:
        answer = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print("\nno answer (stdin is not a terminal); nothing was changed. "
              "Pass --yes to skip this prompt.")
        return False
    if answer != expected:
        print("aborted")
        return False
    return True


def _load(args) -> Config:
    cfg = Config.load(args.config)
    setup(
        cfg.path("logs"),
        level=args.log_level or cfg.get("logging.level", "INFO"),
        redact_content=bool(cfg.get("logging.redact_content", True)),
        rotate_mb=int(cfg.get("logging.rotate_mb", 20)),
        backups=int(cfg.get("logging.backups", 5)),
        redact_patterns=cfg.get("compliance.redact_patterns") or {},
    )
    return cfg


# =========================================================================
def cmd_doctor(args) -> int:
    cfg = _load(args)
    cfg.ensure_dirs()
    rows: list[tuple[str, str, str]] = []
    fatal = False

    # tools
    try:
        from .audio import AudioPreparer

        AudioPreparer(cfg).check_tools()
        rows.append((OK, "ffmpeg", "found"))
    except Exception as exc:  # noqa: BLE001
        rows.append((BAD, "ffmpeg", str(exc)[:90]))
        fatal = True

    # ASR
    from .asr.registry import build_asr_chain

    local_ok = False
    for provider in build_asr_chain(cfg, cfg.glossary):
        ok, why = provider.available()
        rows.append((OK if ok else WARN, f"asr:{provider.name}", why))
        if ok and not provider.is_cloud:
            local_ok = True
    if not local_ok:
        rows.append((BAD, "asr:local",
                     "required for father/husband profiles. pip install faster-whisper"))
        fatal = True

    # diarization
    from .diarize.engine import _available as diar_available

    ok, why = diar_available(cfg)
    rows.append((OK if ok else WARN, "diarization", why if ok else why + " (speaker labels off)"))

    # named speakers
    from .diarize.voiceprint import Embedder, VoiceprintError, VoiceprintStore

    ok, why = Embedder.available(cfg)
    rows.append((OK if ok else WARN, "speakers:model",
                 why if ok else why + " (speakers stay numbered)"))
    try:
        enrolled = VoiceprintStore(Vault(cfg.path("vault"))).people()
        rows.append((
            OK if enrolled else WARN, "speakers:enrolled",
            ", ".join(p.name for p in enrolled) if enrolled
            else 'nobody enrolled yet (run.py speakers enroll "Name" --audio clip.wav)',
        ))
    except (VoiceprintError, VaultError) as exc:
        rows.append((BAD, "speakers:enrolled", str(exc).splitlines()[0]))
        fatal = True

    # LLM
    from .llm.registry import build_llm_chain

    any_llm = False
    for provider in build_llm_chain(cfg):
        ok, why = provider.available()
        rows.append((OK if ok else WARN, f"llm:{provider.name}", why))
        any_llm = any_llm or ok
    if not any_llm:
        rows.append((BAD, "llm", "no usable LLM provider; analysis will fail"))
        fatal = True

    local_llm = list(build_llm_chain(cfg, local_only=True))
    local_llm_ok = any(p.available()[0] for p in local_llm)
    rows.append((
        OK if local_llm_ok else WARN,
        "llm:local",
        "ready" if local_llm_ok
        else "not configured. father/husband analysis will fail by design, not by accident. "
             "Enable llm.local in pipeline.yaml.",
    ))
    rows.append((
        OK if any_llm else WARN, "ask",
        "ready" if any_llm else
        "no usable LLM: `ask` returns ranked excerpts instead of answers, which "
        "still beats `search --content` but is not an answer",
    ))

    # memory
    memory_ok, memory_why = MemoryStore(cfg).ready()
    rows.append((
        OK if memory_ok else WARN, "memory",
        "ready" if memory_ok
        else f"{memory_why} Nothing will be carried forward between recordings.",
    ))

    # offline readiness
    from .runtime import cloud_providers_enabled, is_offline, model_path, resolve_local_model

    offline = is_offline(cfg)
    if args.offline or offline:
        rows.append((OK if offline else WARN, "runtime.offline",
                     "on" if offline else "OFF - set runtime.offline: true to enforce it"))

        offenders = cloud_providers_enabled(cfg)
        rows.append((
            OK if not offenders else BAD, "offline:providers",
            "no cloud provider is enabled" if not offenders
            else "these would reach the network: " + ", ".join(offenders),
        ))
        fatal = fatal or (bool(offenders) and offline)

        for label, configured, subdir in (
            ("asr", cfg.get("asr.local.model", "large-v3"), "whisper"),
            ("diarization", cfg.get("diarization.pyannote.model", ""), "diarization"),
            ("speakers", cfg.get("diarization.identify.model", ""), "diarization"),
        ):
            if not configured:
                continue
            _target, local = resolve_local_model(cfg, configured, subdir)
            rows.append((
                OK if local else (BAD if offline and label == "asr" else WARN),
                f"offline:{label}",
                f"'{configured}' on disk" if local
                else f"'{configured}' NOT in {model_path(cfg, subdir)} "
                     f"(python scripts/fetch_models.py --{subdir} {configured})",
            ))
            if offline and not local and label == "asr":
                fatal = True

    # vault
    ok, why = Vault(cfg.path("vault")).ready()
    rows.append((OK if ok else BAD, "vault", why))
    if not ok:
        fatal = True

    # profiles
    for pid, profile in sorted(cfg.profiles.items()):
        lock = "local-only" if profile.hard_local_only else (
            "cloud ok" if profile.allow_cloud_llm else "no cloud llm")
        rows.append((OK, f"profile:{pid}", f"{profile.sensitivity.value}, {lock}, "
                                           f"{len(profile.fields)} fields"))

    print(f"\nPlaud Bridge {__version__} preflight\n")
    for status, name, detail in rows:
        print(f"[{status}] {name:22s} {detail}")

    print("\n" + ComplianceNote(cfg).text())
    print("\nRESULT:", "NOT READY (fix the FAIL rows above)" if fatal else "ready")
    return 1 if fatal else 0


class ComplianceNote:
    def __init__(self, cfg):
        self.cfg = cfg

    def text(self) -> str:
        states = self.cfg.get("compliance.all_party_consent_states", []) or []
        on_missing = self.cfg.get("compliance.on_missing_consent", "quarantine")
        return (
            "Consent policy\n"
            f"  Missing consent -> {on_missing}\n"
            f"  All-party states configured: {', '.join(states)}\n"
            "  Operational rule: announce every time, capture a verbal yes on tape,\n"
            "  every call, every state. Verify statutes with counsel; this tool is\n"
            "  not legal advice."
        )


# =========================================================================
def cmd_run(args) -> int:
    cfg = _load(args)
    pipe = Pipeline(cfg)
    try:
        stats = pipe.run(force=args.force, limit=args.limit)
        print("\n" + stats.summary())
        if stats.quarantined:
            print(f"\n{stats.quarantined} recording(s) quarantined. See "
                  f"{cfg.path('quarantine')} for why.")
            print("Triage them together with `python run.py quarantine` "
                  "(then --release-all / --forget-all).")
        if stats.failed:
            print(f"{stats.failed} recording(s) failed. Check the log at "
                  f"{cfg.path('logs') / 'bridge.log'}")
        # A file in the inbox that was not processed needs a reason, and the
        # reason has to reach the person rather than only the log. Otherwise
        # this looks like the tool ignored the file you just dropped in, which
        # is exactly what it looks like.
        if pipe.unsettled:
            print(f"\n{len(pipe.unsettled)} file(s) were written moments ago and were "
                  f"skipped in case they are still copying:")
            for path in pipe.unsettled[:5]:
                print(f"  {path.name}")
            print(f"Run again in {cfg.get('ingest.settle_seconds', 5)}s, or set "
                  f"ingest.settle_seconds: 0 if your files always arrive complete.")

        if pipe.unsupported:
            audio = ", ".join(cfg.get("ingest.audio_extensions", []))
            text = ", ".join(cfg.get("ingest.text_extensions", []))
            print(f"\n{len(pipe.unsupported)} file(s) in the inbox are not a kind this "
                  f"reads and were left alone:")
            for path in pipe.unsupported[:5]:
                print(f"  {path.name}")
            if len(pipe.unsupported) > 5:
                print(f"  ...and {len(pipe.unsupported) - 5} more")
            print(f"Audio: {audio}\nText:  {text}")
            print("Rename the file if it is really one of these, or add the extension "
                  "to ingest.audio_extensions / ingest.text_extensions.")

        if pipe.oversize:
            limit_mb = float(cfg.get("ingest.max_file_mb", 512))
            print(f"\n{len(pipe.oversize)} file(s) are larger than ingest.max_file_mb "
                  f"({limit_mb:.0f}MB) and were skipped:")
            for path, size_mb in pipe.oversize[:5]:
                print(f"  {path.name}  {size_mb:.0f}MB")
            print("Raise ingest.max_file_mb if you meant to process them. Originals are "
                  "encrypted a chunk at a time, so this is a disk budget, not a memory one.")

        return 0 if stats.failed == 0 else 2
    finally:
        pipe.close()


def cmd_digest(args) -> int:
    cfg = _load(args)
    db = Database(cfg.path("database"))
    try:
        opts = DigestOptions(
            profile_id=args.profile,
            days=args.days or int(cfg.get("digest.default_window_days", 7)),
            include_personal=args.include_personal,
            include_costs=bool(cfg.get("digest.include_cost_summary", True)),
            include_links=bool(cfg.get("digest.include_transcript_links", True)),
            max_items=int(cfg.get("digest.max_items_per_section", 40)),
            title=args.title or "",
        )
        builder = DigestBuilder(cfg, db)

        # HTML is still rendered from the markdown — with charts layered on
        # top — so the two formats cannot drift into saying different things.
        if args.format == "html":
            title = opts.title or (
                cfg.profile(opts.profile_id).name if opts.profile_id else "Digest"
            )
            body = builder.render_html(opts, title=title)
        else:
            body = builder.render_markdown(opts)

        if args.out:
            dest = Path(args.out)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(body, encoding="utf-8")
            print(f"wrote {dest}")
        else:
            print(body)
        return 0
    finally:
        db.close()


def cmd_followups(args) -> int:
    """
    Commitments across every recording, and drafts for the ones you chase.

    Nothing here sends anything. `--draft` writes a file into the outbox for
    you to read, edit, and send from your own mail client, which is the whole
    difference between this and the feature it replaces.
    """
    cfg = _load(args)
    db = Database(cfg.path("database"))
    try:
        archive = Archive(cfg, db)
        try:
            items = collect(
                cfg, db, archive,
                profile=args.profile,
                days=args.days,
                status=None if args.status == "all" else args.status,
                include_personal=args.include_personal,
                vault=archive.vault,
            )

            for followup_id, status in ((args.done, "done"), (args.drop, "dropped"),
                                        (args.reopen, "open")):
                if not followup_id:
                    continue
                updated = set_status(cfg, archive.vault, followup_id, status, items=items)
                print(f"{updated.id} is now {status}"
                      + (f": {updated.text}" if updated.text else ""))
                return 0

            if args.draft:
                # A recording id drafts everything that recording still owes,
                # a follow-up id drafts one thing, and 'open' drafts the lot.
                if args.draft.startswith("rec_"):
                    target = args.draft
                elif args.draft == "open":
                    target = [i for i in items if i.is_open]
                else:
                    target = [i for i in items if i.id.startswith(args.draft)]
                    if not target:
                        print(f"no follow-up here starts with '{args.draft}'")
                        return 1
                path = draft(
                    target, cfg, db=db, archive=archive, vault=archive.vault,
                    out=args.out, fmt="text" if args.format == "text" else "markdown",
                    include_personal=args.include_personal,
                )
                print(f"\nwrote {path}")
                print("That is a draft. Nothing has been sent, and this tool has no "
                      "way to send it. Read it, fix it, and send it yourself.\n")
                return 0
        except FollowUpError as exc:
            print(f"\n{exc}\n", file=sys.stderr)
            return 1

        body = render(items, fmt="html" if args.format == "html" else "markdown",
                      title=args.title or None)
        if args.out:
            dest = Path(args.out)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(body, encoding="utf-8")
            print(f"wrote {dest}")
        else:
            print(body)
        return 0
    finally:
        db.close()


def cmd_insights(args) -> int:
    """
    Conversation metrics, derived on demand from the stored segments.

    No model and no storage: everything printed here is arithmetic the reader
    can check against `open <id>`, and there is no cache for `forget` to miss.
    Exit 2 means the numbers are honest but incomplete -- some recordings
    would not open -- which is the same convention `search --content` uses.
    """
    cfg = _load(args)
    db = Database(cfg.path("database"))
    try:
        archive = Archive(cfg, db)
        try:
            if args.recording:
                print(render_recording(recording_metrics(cfg, db, archive, args.recording)))
                return 0
            report = insights_trend(
                cfg, db, archive,
                profile=args.profile, days=args.days,
                include_personal=args.include_personal,
            )
        except InsightsError as exc:
            print(f"\n{exc}\n", file=sys.stderr)
            return 2 if exc.unopened else 1
        print(render_trend(report))
        return 2 if report.unopened else 0
    finally:
        db.close()


def cmd_status(args) -> int:
    cfg = _load(args)
    db = Database(cfg.path("database"))
    try:
        stats = db.stats()
        print(f"\nPlaud Bridge index :: {cfg.path('database')}\n")
        print(f"  recordings   {stats['recordings']}")
        print(f"  audio hours  {stats['audio_hours']}")
        print(f"  api spend    ${stats['total_cost_usd']}")
        if stats["by_source"]:
            # Asking questions and phrasing drafts cost money too, and it used
            # to be spent without appearing anywhere a person would look.
            print(f"    pipeline   ${stats['pipeline_cost_usd']}")
            for source, entry in sorted(stats["by_source"].items()):
                print(f"    {source:10s} ${entry['cost_usd']}  ({entry['calls']} call(s))")
        print("\n  by profile")
        for pid, count in sorted(stats["by_profile"].items(), key=lambda kv: -kv[1]):
            name = cfg.profiles[pid].name if pid in cfg.profiles else pid
            print(f"    {name:20s} {count}")
        print("\n  by stage")
        for stage, count in sorted(stats["by_stage"].items()):
            print(f"    {stage:20s} {count}")
        print()
        return 0
    finally:
        db.close()


def cmd_search(args) -> int:
    cfg = _load(args)
    db = Database(cfg.path("database"))
    try:
        if args.content:
            return _search_content(cfg, db, args)

        rows = db.query(profile_id=args.profile, since_days=args.days,
                        search=args.query, limit=args.limit)
        if not rows:
            print("no filename matches. Add --content to search what was said.")
            return 0
        print(f"\n{len(rows)} match(es)\n")
        for row in rows:
            when = (row["recorded_at"] or row["ingested_at"] or "")[:16].replace("T", " ")
            lock = " [encrypted]" if row["encrypted"] else ""
            print(f"  {row['id']}  {when}  {row['duration_seconds'] / 60:5.1f}m  "
                  f"{row['governing_profile'] or '-':16s} {row['source_name']}{lock}")
        print("\nThat searched filenames. Add --content to search the transcripts.\n")
        return 0
    finally:
        db.close()


def _search_content(cfg, db, args) -> int:
    """Search what was said, decrypting the vault where it has to."""
    result = Archive(cfg, db).search_content(
        args.query, profile_id=args.profile, since_days=args.days,
        scan_limit=args.scan_limit, context=args.context,
    )
    matches, unopened = result.matches, result.unopened

    def _say_quarantined() -> None:
        if not result.quarantined:
            return
        print(f"{len(result.quarantined)} recording(s) are quarantined and hold no "
              f"searchable content:")
        for entry in result.quarantined[:20]:
            print(f"  {entry}")
        if len(result.quarantined) > 20:
            print(f"  ... and {len(result.quarantined) - 20} more")
        print("  The gate stopped these before anything was stored. Review them and, "
              "if they are fine,\n  run.py release <id> to put them back in the inbox.\n")

    if not matches and not unopened:
        print(f'\nnothing matching "{args.query}" was said in the '
              f'{result.scanned} recording(s) searched')
        if result.truncated:
            print(f'  ...but only {result.scanned} of {result.total} were opened. '
                  f'Raise --scan-limit or pass 0 to search everything.')
            print()
            return 2
        print()
        _say_quarantined()
        return 0

    by_recording: dict[str, list] = {}
    for match in matches:
        by_recording.setdefault(match.recording_id, []).append(match)

    print(f'\n{len(matches)} hit(s) across {len(by_recording)} recording(s) '
          f'for "{args.query}", from {result.scanned} searched\n')
    for recording_id, hits in by_recording.items():
        head = hits[0]
        tag = "  [personal]" if head.personal else ""
        print(f"  {head.source_name}  ({head.when}, {head.profile_id or '-'}){tag}")
        print(f"  {recording_id}")
        for hit in hits[:args.per_recording]:
            speaker = f"{hit.speaker}: " if hit.speaker and hit.speaker != "SPEAKER" else ""
            print(f"      [{hit.stamp}] {speaker}{hit.text[:220]}")
        if len(hits) > args.per_recording:
            print(f"      ... {len(hits) - args.per_recording} more in this recording")
        print()

    # Never let a search quietly under-report. Concluding a phrase was never
    # said, when really the file would not open or was never looked at, is the
    # worst outcome this command has.
    if result.truncated:
        print(f"INCOMPLETE: only {result.scanned} of {result.total} recording(s) were "
              f"searched (--scan-limit). Pass --scan-limit 0 to search everything.\n")
    if unopened:
        print(f"{len(unopened)} recording(s) could not be opened and were NOT searched:")
        for entry in unopened[:20]:
            print(f"  {entry}")
        if len(unopened) > 20:
            print(f"  ... and {len(unopened) - 20} more")
        print("  Set PLAUD_BRIDGE_PASSPHRASE if these are encrypted.\n")
    _say_quarantined()
    return 0 if result.complete else 2


def cmd_ask(args) -> int:
    """
    Answer a question from what was actually said, with citations.

    Exits 2 when the answer is incomplete -- a recording would not open, the
    scan was bounded, the context budget cut material, or a citation had to be
    dropped. Same reasoning as `search --content`: an answer you would read
    differently having seen the caveat should not look like a clean run to
    whatever called it.
    """
    cfg = _load(args)
    db = Database(cfg.path("database"))
    try:
        answer = ask(
            args.question, cfg, db, Archive(cfg, db),
            profile=args.profile,
            days=args.days,
            limit=args.limit,
            include_personal=args.include_personal,
            local_only=True if args.local_only else None,
        )
        print()
        print(answer.render())
        print()

        if args.save:
            try:
                print(f"saved, encrypted: {save_answer(answer, cfg)}\n")
            except VaultError as exc:
                # The answer has already been printed, so nothing is lost by
                # refusing to write it. Saying so and failing is the point: a
                # silent non-write is how you discover months later that
                # nothing was ever kept.
                print(f"NOT saved: {exc}\n")
                return 1

        if answer.usage_error:
            return 1

        # "Nothing in the archive matched" is a complete answer that happens to
        # be nothing, and `search --content` returns 0 for exactly that. What
        # earns a 2 is an answer the archive could not fully support: excerpts
        # returned because no model was reachable, a recording that would not
        # open, a citation dropped, or context trimmed to fit.
        incomplete = bool(
            (answer.degraded and answer.bundle_chars)
            or answer.unopened or answer.dropped_citations or answer.truncated
        )
        return 2 if incomplete else 0
    finally:
        db.close()


def cmd_verify(args) -> int:
    """
    Check that everything the index points at still exists and still opens.

    An encrypted archive you have never tried to decrypt is an archive you might
    already have lost. This is the command that tells you before it matters.
    """
    cfg = _load(args)
    db = Database(cfg.path("database"))
    try:
        report = Archive(cfg, db).verify()
        print("\n" + report.render() + "\n")
        return 0 if report.healthy else 1
    finally:
        db.close()


def cmd_forget(args) -> int:
    """Delete one recording and everything belonging to it."""
    cfg = _load(args)
    db = Database(cfg.path("database"))
    try:
        payload = db.load(args.recording_id)
        targets = Archive(cfg, db).plan_forget(args.recording_id)
        if payload is None and not targets:
            print(f"no recording with id {args.recording_id}")
            return 1

        name = (payload or {}).get("source_name", "(not in the index)")
        print(f"\nAbout to permanently delete {args.recording_id}")
        print(f"  {name}\n")
        for path in targets:
            print(f"  {path}")
        if not targets:
            print("  (no files on disk; the index entry will be removed)")
        print(
            "\nThis removes the files above, the index entry, and the memory, "
            "follow-up, and saved-answer traces. The audit log keeps a record "
            "that the deletion happened. It cannot reach a digest or export you "
            "saved elsewhere yourself -- delete those where you put them."
        )

        if not args.yes and not _confirm("\nType FORGET to confirm: ", "FORGET"):
            return 1

        # Archive.forget is the single destructive path: it removes the files,
        # the index row, and every derived store that carried the recording's
        # words -- the memory ledgers, saved answers, and follow-up state. It
        # refuses the whole operation when the vault is locked, rather than
        # deleting the plaintext half and stranding the rest.
        removed, failures = Archive(cfg, db).forget(args.recording_id)
        if db.load(args.recording_id) is not None:
            # The index row is still there, so nothing was deleted: the vault
            # was locked and the operation was refused whole.
            print("\nnothing was deleted")
        else:
            print(f"\ndeleted {removed} file(s) and the index entry")
        for failure in failures:
            print(f"  {failure}")
        return 1 if failures else 0
    finally:
        db.close()


def cmd_export(args) -> int:
    """
    Build something you can hand to another person.

    The digest is written for you and assumes you are the only reader. This is
    the opposite: redaction is applied, personal profiles are refused unless you
    force them, and the result is a plain file with no links back into the vault.
    """
    cfg = _load(args)
    db = Database(cfg.path("database"))
    try:
        from .compliance import ComplianceGate

        if args.profile and args.profile not in cfg.profiles:
            print(f"unknown profile '{args.profile}'. Known: {sorted(cfg.profiles)}")
            return 1

        personal = {p.id for p in cfg.profiles.values() if p.exclude_from_combined_export}
        if args.profile in personal and not args.include_personal:
            print(
                f"\n'{args.profile}' is a personal profile and is excluded from exports "
                "by default.\nAn export is a document meant to leave this machine. "
                "Pass --include-personal if\nthat is genuinely what you want.\n"
            )
            return 1

        archive = Archive(cfg, db)
        gate = ComplianceGate(cfg)
        rows = db.query(profile_id=args.profile, since_days=args.days, limit=args.limit)

        if args.transcripts:
            # `suppress_fields` keeps a client's health and financial
            # disclosures out of rendered output. A raw transcript contains the
            # sentences those fields were extracted FROM, so exporting
            # transcripts necessarily includes them. Say so before writing, not
            # in a footnote afterwards.
            at_risk = sorted({
                (p.field_by_key(f).label if p.field_by_key(f) else f)
                for p in cfg.profiles.values() for f in p.suppress_fields
            })
            if at_risk:
                print(
                    "WARNING: --transcripts exports the raw conversation, which "
                    "contains the material these fields exist to withhold:\n"
                    f"  {', '.join(at_risk)}\n"
                    "Redaction is pattern matching and will not catch a spoken "
                    "diagnosis. Read the output before you send it.\n",
                    file=sys.stderr,
                )

        included, skipped, unopened = [], [], []
        for row in rows:
            governing = row["governing_profile"] or ""
            if governing in personal and not args.include_personal:
                skipped.append(row["source_name"])
                continue
            record = archive.full_record(row)
            if record is None:
                unopened.append(row["source_name"])
                continue
            included.append((row, record))

        if not included:
            print("nothing to export in that window")
            return 0

        out: list[str] = [
            f"# {args.title or 'Export'}",
            "",
            f"{len(included)} recording(s). Redacted for sharing. "
            "Generated by plaud-bridge.",
            "",
        ]
        redaction_total: dict[str, int] = {}

        for row, record in included:
            profile = cfg.profile(row["governing_profile"]) if row["governing_profile"] in cfg.profiles else None
            when = (row["recorded_at"] or row["ingested_at"] or "")[:16].replace("T", " ")
            out += [f"## {row['source_name']}", "", f"`{when}`", ""]

            if args.transcripts:
                segments = (record.get("transcript") or {}).get("segments") or []
                body = "\n".join(
                    f"[{format_stamp(float(s.get('start', 0)))}] {s.get('speaker', '')}: {s.get('text', '')}"
                    for s in segments
                )
                redacted, counts = gate.redact_for_llm(body, profile) if profile else (body, {})
                for key, value in counts.items():
                    redaction_total[key] = redaction_total.get(key, 0) + value
                out += ["```", redacted, "```", ""]
                continue

            for analysis in record.get("analyses", []):
                pid = analysis.get("profile_id", "")
                if pid not in cfg.profiles:
                    continue
                section = cfg.profile(pid)
                out += [f"### {section.name}", ""]
                for spec in section.fields:
                    # Suppressed fields never leave, and an export is the last
                    # place to make an exception.
                    if spec.key in section.suppress_fields:
                        continue
                    # Same renderer the digest uses, so an export reads like a
                    # document rather than a JSON dump.
                    lines = fmt_value(analysis.get("fields", {}).get(spec.key))
                    if not lines:
                        continue
                    redacted, counts = gate.redact_for_llm("\n".join(lines), section)
                    for key, value in counts.items():
                        redaction_total[key] = redaction_total.get(key, 0) + value
                    out += [f"**{spec.label}**", ""]
                    out += [f"- {line}" for line in redacted.splitlines()]
                    out += [""]

        # The caveat prints whether or not anything matched. Zero matches means
        # no pattern fired, which is not the same as nothing sensitive being
        # present, and an export with no note reads as an export that was
        # cleared by something.
        if redaction_total or args.transcripts:
            found = (
                "Redacted before export: "
                + ", ".join(f"{k} ({v})" for k, v in sorted(redaction_total.items()))
                if redaction_total
                else "No redaction pattern matched anything in this export"
            )
            out += ["---", "",
                    f"<sub>{found}. Redaction is pattern matching, not a guarantee. "
                    "Read this before you send it.</sub>", ""]
        if skipped:
            out += [f"<sub>{len(skipped)} personal recording(s) omitted.</sub>", ""]

        body = "\n".join(out)
        if args.format == "html":
            body = to_html(body, title=args.title or "Export")

        if args.out:
            dest = Path(args.out)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(body, encoding="utf-8")
            print(f"wrote {dest}")
        else:
            print(body)

        if unopened:
            print(f"\n{len(unopened)} recording(s) could not be decrypted and were omitted.",
                  file=sys.stderr)
            return 2
        return 0
    finally:
        db.close()


def cmd_watch(args) -> int:
    """Process the inbox, then keep doing it. Ctrl-C to stop."""
    import time

    cfg = _load(args)
    print(f"\nwatching {cfg.path('inbox')} every {args.interval}s. Ctrl-C to stop.\n")
    runs = 0
    while True:
        pipe = Pipeline(cfg)
        try:
            stats = pipe.run()
            runs += 1
            if stats.processed or stats.quarantined or stats.failed:
                print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {stats.summary()}")
            if args.once or (args.max_runs and runs >= args.max_runs):
                return 0 if stats.failed == 0 else 2
        finally:
            pipe.close()
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nstopped")
            return 0


def cmd_open(args) -> int:
    cfg = _load(args)
    db = Database(cfg.path("database"))
    try:
        payload = db.load(args.recording_id)
        if payload is None:
            print(f"no recording with id {args.recording_id}")
            return 1
        paths = payload.get("artifact_paths", {})
        key = args.kind
        if key not in paths:
            print(f"no '{key}' artifact. Available: {', '.join(paths) or 'none'}")
            return 1
        path = Path(paths[key])
        if not path.exists():
            print(f"artifact missing from disk (retention sweep?): {path}")
            return 1

        # Audio is bytes, and printing it to a terminal helps nobody.
        if key in ("audio", "source") and not args.out:
            print(f"'{path.name}' is audio. Pass --out to write it somewhere:\n"
                  f"  run.py open {args.recording_id} --kind audio --out recording{path.suffixes[0] if path.suffixes else ''}")
            return 1

        vault = Vault(cfg.path("vault"))

        # A streamed artifact is by definition too big to hold in memory, so
        # writing it out goes chunk by chunk too.
        if args.out and path.suffix == ".enc" and Vault.is_streamed(path):
            try:
                dest = vault.read_stream(path, Path(args.out), args.recording_id)
            except VaultError as exc:
                print(f"could not decrypt: {exc}")
                return 1
            print(f"wrote {dest} ({dest.stat().st_size} bytes)")
            print("That is a decrypted copy of the original recording. "
                  "Delete it when you are done with it.")
            return 0

        try:
            if path.suffix == ".enc":
                blob = vault.read(path, args.recording_id)
            else:
                blob = path.read_bytes()
        except VaultError as exc:
            print(f"could not decrypt: {exc}")
            return 1

        if args.out:
            dest = Path(args.out)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(blob)
            print(f"wrote {dest} ({len(blob)} bytes)")
            if key in ("audio", "source"):
                print("That is a decrypted copy of the original recording. "
                      "Delete it when you are done with it.")
            return 0

        print(blob.decode("utf-8", "replace"))
        return 0
    finally:
        db.close()


def cmd_audit(args) -> int:
    """
    Print the audit trail.

    COMPLIANCE.md section 8 commits to every ingest, route, compliance decision,
    quarantine, release, and retention deletion being recorded. An audit trail
    nobody can read is not an audit trail, so this is the way to read it.
    """
    cfg = _load(args)
    db = Database(cfg.path("database"))
    try:
        rows = db.audit_log(
            recording_id=args.recording_id,
            action=args.action,
            actor=args.actor,
            since_days=args.days,
            limit=args.limit,
        )
        if not rows:
            print("no audit entries match")
            return 0

        if args.out:
            import csv

            dest = Path(args.out)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(
                    fh, fieldnames=["at", "actor", "action", "recording_id", "detail"]
                )
                writer.writeheader()
                for row in rows:
                    writer.writerow({k: row.get(k) or "" for k in writer.fieldnames})
            print(f"wrote {len(rows)} entry(ies) to {dest}")
            return 0

        print(f"\n{len(rows)} audit entry(ies)\n")
        for row in rows:
            when = (row["at"] or "")[:19].replace("T", " ")
            actor = row["actor"] or "pipeline"
            marker = " <- HUMAN" if actor != "pipeline" else ""
            print(f"  {when}  {row['action']:20s} {row['recording_id'] or '-':24s} {marker}")
            if row["detail"]:
                print(f"      {row['detail'][:160]}")
        print()
        return 0
    finally:
        db.close()


def _quarantine_media(cfg, recording_id: str) -> list[Path]:
    """The held files a release would move. WHY.md stays; it is ours, not media."""
    qdir = cfg.path("quarantine") / recording_id
    if not qdir.is_dir():
        return []
    return [p for p in qdir.iterdir() if p.name != "WHY.md"]


def _release_media(cfg, db, recording_id: str,
                   detail: str = "released after human review") -> list[Path]:
    """
    The one release path: copy the held media back into the inbox and audit it.

    Both `release <id>` and `quarantine --release-all` end here, so a bulk
    release cannot quietly come to mean something different from a single one
    -- same destination, same audit action, same actor.
    """
    inbox = cfg.path("inbox")
    released: list[Path] = []
    for path in _quarantine_media(cfg, recording_id):
        dest = inbox / path.name
        dest.write_bytes(path.read_bytes())
        released.append(dest)
    db.audit("quarantine_release", detail, recording_id, actor="human")
    return released


def cmd_release(args) -> int:
    """Move a quarantined file back to the inbox after human review."""
    cfg = _load(args)
    db = Database(cfg.path("database"))
    try:
        qdir = cfg.path("quarantine") / args.recording_id
        if not qdir.is_dir():
            print(f"no quarantine folder for {args.recording_id}")
            return 1
        if not _quarantine_media(cfg, args.recording_id):
            print("quarantine folder has no media to release")
            return 1

        if not args.yes:
            print(f"\nAbout to release "
                  f"{len(_quarantine_media(cfg, args.recording_id))} file(s) back to the inbox.\n")
            print((qdir / "WHY.md").read_text(encoding="utf-8"))
            if not _confirm("Type RELEASE to confirm you verified consent: ", "RELEASE"):
                return 1

        for dest in _release_media(cfg, db, args.recording_id):
            print(f"released -> {dest}")

        print("\nRe-run `python run.py run --force` to process it.")
        print("The release is recorded in the audit log.")
        return 0
    finally:
        db.close()


# ---- quarantine triage --------------------------------------------------
#
# A backlog run can quarantine dozens of recordings at once, correctly, and
# the per-recording tools (`release <id>`, WHY.md files read one at a time) do
# not scale to that without turning review into a rubber stamp. This surface
# lists everything in quarantine with the reason distilled, and offers bulk
# release and bulk forget behind the same typed-confirmation discipline that
# `forget` uses. The one thing it refuses to scale is a refusal: a recording
# whose verdict shows a party objecting is never part of --release-all, and
# releasing it stays a one-at-a-time act on purpose.

_RELEASE_ALL_PHRASE = "RELEASE ALL"
_FORGET_ALL_PHRASE = "FORGET ALL"

# Reason classes, in the order the listing shows them. Refusals come first
# because they are the rows a person must not skim past.
_CLASS_REFUSAL = "refusal"
_CLASS_STATIC_GATE = "static-gate"
_CLASS_NO_ANNOUNCEMENT = "no-announcement"

_CLASS_TITLES = {
    _CLASS_STATIC_GATE: "Standing consent gate is off",
    _CLASS_NO_ANNOUNCEMENT: "No consent announcement detected",
}


def _classify_verdict(consent_status: str, reasons: list[str]) -> tuple[str, str]:
    """
    (class, one-line reason) distilled from a stored compliance verdict.

    Refusal is checked first because it can co-occur with everything else and
    must win: it is the one class policy forbids releasing in bulk. The text
    matches are against sentences this codebase writes itself (gate.py and
    consent.py), which is what makes matching on them safe; the consent_status
    column is still consulted first so an index row alone is enough.
    """
    joined = " ".join(reasons)
    if consent_status == "refused" or "objected to being recorded" in joined:
        return _CLASS_REFUSAL, "a party objected to being recorded"
    if "set to false" in joined:
        line = next((r for r in reasons if "set to false" in r), "")
        return _CLASS_STATIC_GATE, (line.split(". ")[0] or "standing consent gate is off")
    # Whatever the consent detector noted is more specific than the boilerplate
    # around it, so keep the first reason that is not scaffolding.
    boilerplate = ("QUARANTINED", "local-only processing", "governs the whole recording")
    for reason in reasons:
        if not any(marker in reason for marker in boilerplate):
            return _CLASS_NO_ANNOUNCEMENT, reason.split(". ")[0][:110]
    return _CLASS_NO_ANNOUNCEMENT, "no consent announcement detected"


def _reasons_from_why(qdir: Path) -> list[str]:
    """The WHY.md bullets, for a folder whose index row is gone."""
    try:
        text = (qdir / "WHY.md").read_text(encoding="utf-8")
    except OSError:
        return []
    reasons, in_reasons = [], False
    for line in text.splitlines():
        if line.startswith("## "):
            in_reasons = line.strip() == "## Reasons"
            continue
        if in_reasons and line.startswith("- "):
            reasons.append(line[2:].strip())
    return reasons


def _quarantine_entries(cfg, db) -> list[dict]:
    """
    Everything currently held in quarantine, with the reason distilled.

    The quarantine directory is the ground truth for what can be released --
    `release` moves files, not index rows -- but the index and the audit log
    know things the folder does not: which verdict put it there, when, and
    whether a human already released it. So this reads all three, and also
    reports index rows whose folder has gone missing rather than pretending
    the index is wrong.
    """
    qroot = cfg.path("quarantine")
    ids: list[str] = []
    if qroot.is_dir():
        ids = sorted(p.name for p in qroot.iterdir() if p.is_dir())
    for row in db.query(stage="quarantined", limit=10_000):
        if row["id"] not in ids:
            ids.append(row["id"])

    entries: list[dict] = []
    for rid in ids:
        payload = db.load(rid) or {}
        verdict = payload.get("compliance") or {}
        reasons = list(verdict.get("reasons") or []) or _reasons_from_why(qroot / rid)
        klass, reason = _classify_verdict(str(verdict.get("consent", "")), reasons)

        held = db.audit_log(recording_id=rid, action="quarantine", limit=1)
        when = (held[0]["at"] if held else payload.get("ingested_at") or "")[:16].replace("T", " ")
        entries.append({
            "id": rid,
            "source": payload.get("source_name")
                      or next((p.name for p in _quarantine_media(cfg, rid)), "(unknown source)"),
            "when": when or "(unknown time)",
            "klass": klass,
            "reason": reason,
            "media": _quarantine_media(cfg, rid),
            "released": bool(db.audit_log(recording_id=rid, action="quarantine_release", limit=1)),
        })
    entries.sort(key=lambda e: e["when"])
    return entries


def _print_entry(entry: dict) -> None:
    note = ""
    if entry["released"]:
        note = "  (already released; re-run `run.py run --force` to process it)"
    elif not entry["media"]:
        note = "  (folder has no media; releasable never, forgettable always)"
    print(f"  {entry['id']}  {entry['when']}  {entry['source']}{note}")
    print(f"      {entry['reason']}")


def _quarantine_list(cfg, entries: list[dict]) -> int:
    if not entries:
        print("\nquarantine is empty\n")
        return 0

    refusals = [e for e in entries if e["klass"] == _CLASS_REFUSAL]
    print(f"\n{len(entries)} recording(s) in quarantine :: {cfg.path('quarantine')}\n")

    if refusals:
        print(f"REFUSED -- a party said no ({len(refusals)}). "
              "Never included in --release-all:")
        for entry in refusals:
            _print_entry(entry)
        print("  A refusal is releasable only one at a time, `run.py release <id>`, after")
        print("  you have listened and are certain the objection is not what it reads as.")
        print("  Deleting these is the expected outcome, not an inconvenience.\n")

    for klass, title in _CLASS_TITLES.items():
        group = [e for e in entries if e["klass"] == klass]
        if not group:
            continue
        print(f"{title} ({len(group)}):")
        for entry in group:
            _print_entry(entry)
        print()

    print("Full stored verdicts are in each folder's WHY.md.")
    print("One at a time:  run.py release <id>   /  run.py forget <id>")
    print("In bulk:        run.py quarantine --release-all   (refusals stay put)")
    print("                run.py quarantine --forget-all    (deletes every trace)\n")
    return 0


def _quarantine_release_all(cfg, db, entries: list[dict], yes: bool) -> int:
    refusals = [e for e in entries if e["klass"] == _CLASS_REFUSAL]
    eligible = [e for e in entries
                if e["klass"] != _CLASS_REFUSAL and e["media"] and not e["released"]]

    if refusals:
        print(f"\n{len(refusals)} recording(s) hold an explicit refusal and are "
              "EXCLUDED from this bulk release:")
        for entry in refusals:
            _print_entry(entry)
        print("  A refusal means the other party answered the consent question, and the")
        print("  answer was no. Bulk release exists for recordings where nobody asked on")
        print("  tape; it must never wave through one where somebody said no. If you are")
        print("  certain a refusal was misread, release it alone: run.py release <id>.")
        print("  That friction is the point.")

    if not eligible:
        already = [e for e in entries if e["released"]]
        print("\nnothing eligible for bulk release"
              + (f" ({len(already)} already released)" if already else "")
              + ("" if entries else "; quarantine is empty"))
        print()
        return 0

    print(f"\nAbout to release {len(eligible)} recording(s) back to the inbox:\n")
    for entry in eligible:
        _print_entry(entry)
    print("\nReleasing is an affirmation, for EVERY recording above, that you verified")
    print("consent was actually obtained even though the gate could not hear it --")
    print("not a way to empty a folder. Each release is written to the audit log.")

    if not yes and not _confirm(
        f"\nType {_RELEASE_ALL_PHRASE} to confirm you verified consent for all of them: ",
        _RELEASE_ALL_PHRASE,
    ):
        return 1

    print()
    for entry in eligible:
        for dest in _release_media(
            cfg, db, entry["id"],
            detail="released after human review (quarantine --release-all)",
        ):
            print(f"released -> {dest}")

    print(f"\nreleased {len(eligible)} recording(s)"
          + (f"; {len(refusals)} refusal(s) stay quarantined" if refusals else ""))
    print("Re-run `python run.py run --force` to process them.")
    print("Every release is recorded in the audit log.")
    return 0


def _quarantine_forget_all(cfg, db, entries: list[dict], yes: bool) -> int:
    if not entries:
        print("\nquarantine is empty; nothing to forget\n")
        return 0

    archive = Archive(cfg, db)
    print(f"\nAbout to permanently delete {len(entries)} quarantined recording(s), "
          "their index entries, and every derived trace:\n")
    for entry in entries:
        targets = archive.plan_forget(entry["id"])
        print(f"  {entry['id']}  {entry['source']}  ({len(targets)} file(s))")
    print("\nThe audit log keeps a record that each deletion happened. Nothing else survives.")

    if not yes and not _confirm(
        f"\nType {_FORGET_ALL_PHRASE} to confirm: ", _FORGET_ALL_PHRASE,
    ):
        return 1

    removed_total, failures_total = 0, []
    for entry in entries:
        removed, failures = archive.forget(entry["id"])
        removed_total += removed
        failures_total.extend(failures)

    print(f"\ndeleted {removed_total} file(s) across {len(entries)} recording(s)")
    for failure in failures_total:
        print(f"  {failure}")
    return 1 if failures_total else 0


def cmd_quarantine(args) -> int:
    """
    Triage the quarantine folder as a whole.

    The default is the read-only listing; the two bulk verbs sit behind the
    same typed-confirmation discipline as `forget`, and --release-all excludes
    explicit refusals unconditionally.
    """
    cfg = _load(args)
    db = Database(cfg.path("database"))
    try:
        entries = _quarantine_entries(cfg, db)
        if args.release_all:
            return _quarantine_release_all(cfg, db, entries, yes=args.yes)
        if args.forget_all:
            return _quarantine_forget_all(cfg, db, entries, yes=args.yes)
        return _quarantine_list(cfg, entries)
    finally:
        db.close()


def cmd_retention(args) -> int:
    cfg = _load(args)
    db = Database(cfg.path("database"))
    try:
        sweeper = RetentionSweeper(cfg, db)
        dry = not args.execute
        plan = sweeper.plan(dry_run=dry)
        print("\n" + plan.render() + "\n")
        if not dry and not plan.empty:
            if not args.yes and not _confirm(
                f"Delete {len(plan.items)} artifact(s)"
                + (f" and {plan.audit_rows} audit row(s)" if plan.audit_rows else "")
                + "? Type DELETE: ", "DELETE"):
                return 1
            removed = sweeper.execute(plan)
            print(f"deleted {removed} artifact(s)")
        return 0
    finally:
        db.close()


def cmd_memory(args) -> int:
    """
    What this tool has learned across recordings, per profile.

    Read-only unless you ask otherwise. `--rebuild` throws the ledgers away and
    replays the archive: slower, and the answer to believe whenever the ledger
    and the archive disagree.
    """
    cfg = _load(args)
    if args.profile and args.profile not in cfg.profiles:
        print(f"unknown profile '{args.profile}'. Known: {sorted(cfg.profiles)}")
        return 1

    db = Database(cfg.path("database"))
    try:
        store = MemoryStore(cfg)

        if args.forget:
            changed = store.forget_recording(args.forget)
            print(f"\nremoved {args.forget} from {len(changed)} ledger(s)"
                  + (f": {', '.join(changed)}" if changed else ""))
        elif args.rebuild:
            report = store.rebuild(db, Archive(cfg, db), force=args.force)
            print("\n" + report.render())
            if not report.saved:
                for problem in store.problems:
                    print(f"\n  {problem}", file=sys.stderr)
                return 1

        for pid in ([args.profile] if args.profile else sorted(cfg.profiles)):
            if args.brief:
                brief = carry_forward_brief(cfg, pid, store)
                if brief:
                    print(f"\n----- {pid} -----\n{brief}")
            else:
                print("\n" + render_ledger(store.ledger(pid), cfg=cfg))

        for problem in store.problems:
            print(f"\n  {problem}", file=sys.stderr)
        print()
        return 1 if store.problems else 0
    finally:
        db.close()


def cmd_new_profile(args) -> int:
    """Scaffold a profile from the documented template."""
    cfg_dir = Path(args.config)
    template = cfg_dir / "profiles" / "_TEMPLATE.yaml"
    if not template.exists():
        print(f"template not found: {template}")
        return 1

    pid = args.profile_id.strip()
    if not pid.isidentifier() or pid.startswith("_"):
        print(f"'{pid}' must be a valid Python identifier and not start with an underscore")
        return 1

    dest = cfg_dir / "profiles" / f"{pid}.yaml"
    if dest.exists():
        print(f"{dest} already exists. Pick another id or edit that file.")
        return 1

    name = args.name or pid.replace("_", " ").title()
    # These come from the command line and land inside quoted YAML scalars, so a
    # value containing a quote, a colon, or a newline would break out and inject
    # structure -- or, more likely for a value the user typed themselves, just
    # write a file that no longer parses. json.dumps produces a correctly escaped
    # double-quoted scalar that is also valid YAML, which keeps the templated
    # values safe without discarding the template's comments the way re-dumping
    # the whole thing would. `pid` is already validated as an identifier.
    body = (
        template.read_text(encoding="utf-8")
        .replace('id: "PROFILE_ID"', f'id: {pid}')
        .replace('name: "Profile Name"', f'name: {json.dumps(name)}')
        .replace('short_name: "Short"', f'short_name: {json.dumps(args.short_name or name)}')
        .replace('heading: "Section Heading"', f'heading: {json.dumps(args.heading or name)}')
    )
    dest.write_text(body, encoding="utf-8")

    print(f"\nwrote {dest}\n")
    print("Next:")
    print("  1. Edit it. The routing keywords and llm_hint are what make it work;")
    print("     an empty keyword list means the router has nothing to go on.")
    print("  2. Read the sensitivity and processing blocks before your first run.")
    print("  3. python run.py profiles      confirm it loaded")
    print("  4. python run.py doctor        confirm nothing else broke")
    print("\nIt will not route anything until you give it keywords.\n")
    return 0


def cmd_voices(args) -> int:
    """Show the installed voice packs and which one is active."""
    cfg = _load(args)
    active = cfg.voice.id
    packs = Voice.available(Path(args.config) / "voice")

    print(f"\nactive voice: {cfg.voice.name} ({active})\n")
    if not packs:
        print("  no voice packs found; using the built-in defaults\n")
        return 0
    for pid, name, description in packs:
        marker = "*" if pid == active else " "
        print(f" {marker} {pid:10s} {name}")
        if description:
            print(f"              {description}")
    print("\nSet voice.preset in pipeline.yaml to switch. Override individual")
    print("strings with voice.overrides without copying a whole pack.\n")
    return 0


def _speakers_store(cfg):
    from .diarize.voiceprint import VoiceprintStore

    return VoiceprintStore(Vault(cfg.path("vault")))


def _prepared_audio(cfg, src: Path, work_dir: Path) -> Path:
    """
    Enrollment and identification want the same 16k mono wav the pipeline uses.

    An enrollment clip recorded on a phone and a recording exported from the
    device should produce comparable vectors, and they will not if one of them
    is a 44.1k stereo mp3.
    """
    from .audio import AudioPreparer

    normalised, _duration = AudioPreparer(cfg).normalise(src, work_dir)
    return normalised


def cmd_speakers(args) -> int:
    """
    Enroll, list, test, and forget the voices this archive can name.

    Split into sub-commands because these are four genuinely different verbs
    with different consequences, and `speakers forget` deleting a voiceprint
    should not share a flag namespace with `speakers enroll` creating one.
    """
    cfg = _load(args)
    cfg.ensure_dirs()
    from .audio import AudioError
    from .diarize.voiceprint import Embedder, VoiceprintError, identify

    action = args.speakers_action
    store = _speakers_store(cfg)

    try:
        # ---- list --------------------------------------------------------
        if action == "list":
            people = store.people()
            if not people:
                print("\nNobody is enrolled, so every speaker stays numbered.\n")
                print('  run.py speakers enroll "Marcus" --audio clips/marcus.wav\n')
                return 0
            print(f"\n{len(people)} enrolled voice(s), encrypted in {store.path}\n")
            for person in people:
                sources = ", ".join(s.source for s in person.samples if s.source)
                print(f"  {person.name:24s} {len(person.samples)} sample(s), "
                      f"{person.seconds:.0f}s, updated {person.updated_at[:10]}")
                if sources:
                    print(f"  {'':24s} from {sources}")
            print("\nThreshold and margin live under diarization.identify in pipeline.yaml.")
            print("Run `speakers identify <audio>` to see the actual scores before changing them.\n")
            return 0

        # ---- forget ------------------------------------------------------
        if action == "forget":
            person = store.find(args.name)
            if person is None:
                print(f"nobody enrolled under '{args.name}'. Known: "
                      f"{', '.join(p.name for p in store.people()) or 'nobody'}")
                return 1
            if not args.yes and not _confirm(
                f"\nDelete the voiceprint for {person.name} "
                f"({len(person.samples)} sample(s))?\nType the name to confirm: ",
                person.name,
            ):
                return 1
            store.forget(person.id)
            store.save()
            print(f"\n{person.name} is no longer recognised. Transcripts already written keep "
                  "the name they were given; this only affects future recordings.\n")
            return 0

        # ---- enroll ------------------------------------------------------
        if action == "enroll":
            src = Path(args.audio)
            if not src.exists():
                print(f"no such file: {src}")
                return 1
            ok, why = Embedder.available(cfg)
            if not ok:
                print(f"\ncannot enroll: {why}\n")
                return 1

            work = cfg.path("work") / "enroll"
            prepared = _prepared_audio(cfg, src, work)
            span = ""
            if args.start is not None or args.end is not None:
                span = f" [{args.start or 0:.0f}s-{args.end:.0f}s]" if args.end else ""
            print(f"\nembedding {src.name}{span} ...")

            vector = Embedder(cfg).embed(prepared, args.start, args.end)
            seconds = (args.end - (args.start or 0.0)) if args.end else 0.0
            if not seconds:
                from .audio import probe_duration

                seconds = probe_duration(prepared, cfg.get("audio.ffprobe_binary", "ffprobe"))

            person = store.enroll(args.name, vector, source=src.name, seconds=seconds,
                                  replace=args.replace)
            store.save()
            _discard_scratch(work)
            print(f"\n{person.name} enrolled from {seconds:.0f}s of speech "
                  f"({len(person.samples)} sample(s) total).")
            print("Add a second clip from a different room to make matching more reliable.\n")
            return 0

        # ---- identify ----------------------------------------------------
        if action == "identify":
            src = Path(args.audio)
            if not src.exists():
                print(f"no such file: {src}")
                return 1
            if store.is_empty():
                print("\nNobody is enrolled, so there is nothing to compare against.\n")
                return 1

            from .diarize.engine import DiarizationError, speaker_turns

            work = cfg.path("work") / "identify"
            prepared = _prepared_audio(cfg, src, work)
            try:
                segments = speaker_turns(prepared, cfg)
            except DiarizationError as exc:
                print(f"\ncannot separate speakers: {exc}")
                print("Without diarization this can only be scored as a single voice.\n")
                segments = []
            if not segments:
                from .audio import probe_duration
                from .models import Segment as _Segment

                total = probe_duration(prepared, cfg.get("audio.ffprobe_binary", "ffprobe"))
                segments = [_Segment(start=0.0, end=total, text="", speaker="WHOLE FILE")]

            matches = identify(prepared, segments, cfg, store)
            threshold = float(cfg.get("diarization.identify.threshold", 0.55))
            print(f"\n{src.name}: {len(matches)} cluster(s), threshold {threshold:.2f}\n")
            for match in matches:
                verdict = match.matched or "unnamed"
                print(f"  {match.cluster:14s} {match.seconds:6.1f}s  -> {verdict}")
                for name, score in match.scores[:4]:
                    marker = "*" if name == match.matched else " "
                    print(f"  {'':14s} {marker} {name:22s} {score:.3f}")
                if not match.matched:
                    print(f"  {'':14s}   ({match.reason})")
            _discard_scratch(work)
            print("\nNothing was written. Adjust diarization.identify.threshold and margin "
                  "in pipeline.yaml if these scores disagree with your ears.\n")
            return 0

    except (VoiceprintError, VaultError, AudioError) as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 1

    return 1


def _discard_scratch(work_dir: Path) -> None:
    """An enrollment clip is somebody's voice; the scratch copy does not linger."""
    from .audio import AudioPreparer

    AudioPreparer.cleanup(work_dir)


def cmd_review(args) -> int:
    """
    The review cadence from COMPLIANCE.md section 9, as a command.

    That section asks you to read certain things weekly, monthly, quarterly, and
    annually. Asking a person to remember a four-tier schedule is asking them to
    stop doing it by March, so this assembles all four and says which are due.
    """
    cfg = _load(args)
    db = Database(cfg.path("database"))
    try:
        if args.reaffirm:
            if args.reaffirm not in cfg.profiles:
                print(f"unknown profile '{args.reaffirm}'. Known: {sorted(cfg.profiles)}")
                return 1
            profile = cfg.profile(args.reaffirm)
            if not profile.consent_gate_key:
                print(f"profile '{profile.id}' has no standing consent block to reaffirm")
                return 1
            print(f"\n{profile.name}: {profile.consent_gate_key}")
            print("\n  This is not a formality. Confirm the people on these recordings")
            print("  still know the device records and are still fine with it.\n")
            if not _confirm("Type YES to record the reaffirmation: ", "YES"):
                return 1
            db.audit("consent_reaffirm", profile.id, actor="human")
            print(f"recorded. Next due in {profile.reaffirm_every_days} days.\n")
            return 0

        now = datetime.now(timezone.utc)
        due: list[str] = []

        print(f"\nReview :: {now:%Y-%m-%d}\n")

        # --- standing consent, per profile -------------------------------
        print("Standing consent")
        # audit_log is newest first, so setdefault keeps the most recent
        # reaffirmation per profile. A dict comprehension would keep the oldest
        # and tell you something was overdue forever.
        reaffirmations: dict[str, str] = {}
        for row in db.audit_log(action="consent_reaffirm", limit=500):
            reaffirmations.setdefault(row["detail"], row["at"])
        any_gate = False
        for pid in sorted(cfg.profiles):
            profile = cfg.profile(pid)
            if not profile.consent_gate_key or profile.reaffirm_every_days <= 0:
                continue
            any_gate = True
            last = reaffirmations.get(pid)
            if last is None:
                print(f"  [DUE] {profile.name:18s} never reaffirmed")
                due.append(f"run.py review --reaffirm {pid}")
                continue
            stamped = datetime.fromisoformat(last)
            if stamped.tzinfo is None:
                # Everything we write is timezone-aware, but a hand-edited row
                # or an older database need not be. Assume UTC rather than
                # raising and taking the whole review down with it.
                stamped = stamped.replace(tzinfo=timezone.utc)
            age = (now - stamped).days
            if age >= profile.reaffirm_every_days:
                print(f"  [DUE] {profile.name:18s} last {age}d ago "
                      f"(every {profile.reaffirm_every_days}d)")
                due.append(f"run.py review --reaffirm {pid}")
            else:
                remaining = profile.reaffirm_every_days - age
                print(f"  [ ok] {profile.name:18s} last {age}d ago, next in {remaining}d")
        if not any_gate:
            print("  no profiles carry a standing consent block")

        # --- weekly: statements needing review ---------------------------
        print(f"\nStatements needing review (last {args.days} days)")
        flagged = 0
        for pid in sorted(cfg.profiles):
            if "statements_needing_review" not in cfg.profile(pid).field_keys:
                continue
            for row in db.query(profile_id=pid, since_days=args.days, limit=200):
                payload = json.loads(row["payload_json"])
                for analysis in payload.get("analyses", []):
                    if analysis.get("profile_id") != pid:
                        continue
                    if analysis.get("fields_withheld"):
                        print(f"  [enc] {row['source_name']}  "
                              f"run.py open {row['id']} --kind analysis")
                        flagged += 1
                        continue
                    items = analysis.get("fields", {}).get("statements_needing_review") or []
                    for item in items:
                        flagged += 1
                        print(f"  [!!] {row['source_name']}: "
                              f"{str(item)[:110]}")
        if not flagged:
            print("  nothing flagged")

        # --- monthly: unfiled keyword harvest ----------------------------
        fallback = cfg.get("routing.fallback_profile", "unfiled")
        print(f"\nUnfiled recordings (last {args.days} days)")
        suggestions: dict[str, int] = {}
        unfiled_rows = db.query(profile_id=fallback, since_days=args.days, limit=200)
        for row in unfiled_rows:
            payload = json.loads(row["payload_json"])
            for analysis in payload.get("analyses", []):
                if analysis.get("profile_id") != fallback:
                    continue
                for word in analysis.get("fields", {}).get("suggested_keywords") or []:
                    key = str(word).strip().lower()
                    if key:
                        suggestions[key] = suggestions.get(key, 0) + 1
        print(f"  {len(unfiled_rows)} recording(s) the router could not place")
        if suggestions:
            top = sorted(suggestions.items(), key=lambda kv: -kv[1])[:15]
            print("  keywords worth adding to a profile:")
            print("    " + ", ".join(f"{w} ({n})" for w, n in top))
            due.append("add the keywords above to the right profile's routing.keywords")
        elif unfiled_rows:
            print("  no keyword suggestions; read them with `run.py search`")

        # --- quarterly: retention ----------------------------------------
        print("\nRetention")
        plan = RetentionSweeper(cfg, db).plan(dry_run=True)
        if plan.items:
            print(f"  {len(plan.items)} artifact(s) past their expiry, "
                  f"{plan.total_bytes / 1_048_576:.1f} MB")
            due.append("run.py retention --execute")
        else:
            print("  nothing has expired")

        # --- what to do --------------------------------------------------
        print("\nDue now" if due else "\nNothing is due.")
        for item in due:
            print(f"  - {item}")
        print()
        return 0
    finally:
        db.close()


def cmd_profiles(args) -> int:
    cfg = _load(args)
    print(f"\n{len(cfg.profiles)} profile(s)\n")
    for pid in sorted(cfg.profiles, key=lambda p: cfg.profiles[p].digest_priority):
        p = cfg.profiles[pid]
        print(f"  {p.name}  ({pid})")
        print(f"    sensitivity   {p.sensitivity.value}")
        print(f"    processing    asr_cloud={p.allow_cloud_asr}  llm_cloud={p.allow_cloud_llm}"
              f"  hard_local={p.hard_local_only}")
        print(f"    consent       required={p.require_consent}"
              + (f"  gate={p.consent_gate_key}={p.consent_gate_value}" if p.consent_gate_key else ""))
        print(f"    retention     transcript={p.transcript_days}d  audio={p.raw_audio_days}d")
        print(f"    routing       threshold={p.min_confidence}  keywords={len(p.keywords)}")
        print(f"    fields        {', '.join(p.field_keys)}")
        if p.suppress_fields:
            print(f"    suppressed    {', '.join(p.suppress_fields)}")
        if p.exclude_from_combined_export:
            print("    combined      excluded from combined digest by default")
        print()
    return 0


def cmd_backup(args) -> int:
    """
    Everything worth keeping, as one encrypted file.

    The vault's honesty about loss cuts both ways: lose the disk and the
    archive is gone, because nothing in this tool copies anything anywhere.
    This is the copy. It is a single file precisely so it can sit on an
    external drive or in a cloud folder -- and it is encrypted with the vault's
    own cipher precisely because that folder is not this machine.
    """
    from .backup import BackupError, create_backup, default_backup_path

    cfg = _load(args)
    out = Path(args.out) if args.out else default_backup_path()
    try:
        report = create_backup(cfg, Path(args.config), out)
    except (BackupError, VaultError) as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 1

    print("\nBacked up, encrypted:")
    for name, count in report.counts.items():
        print(f"  {name:12s} {count} file(s)")
    if report.skipped:
        print(f"  nothing to include from: {', '.join(report.skipped)}")
    print("  (data/inbox and data/work are transient and are left out)")
    print(f"\nwrote {report.path} ({report.size_bytes:,} bytes)")
    print("\nRestoring needs the passphrase in PLAUD_BRIDGE_PASSPHRASE. It is not")
    print("stored in this file or anywhere else. Lose the passphrase and this")
    print("backup is unreadable noise -- keep them apart, but keep them both.")
    return 0


def cmd_restore(args) -> int:
    """
    Bring the archive back from a backup file.

    All-or-nothing on purpose: the bundle is decrypted and checked in a temp
    directory first, so a wrong passphrase or a tampered file is the vault's
    honest error and an untouched data directory, never a half-restored one.
    """
    from .backup import BackupError, restore_backup

    cfg = _load(args)
    try:
        report = restore_backup(cfg, Path(args.config), Path(args.file), force=args.force)
    except (BackupError, VaultError) as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 1

    print("\nrestored: " + ", ".join(
        f"{name} ({count} file(s))" for name, count in report.restored.items()
    ))
    if report.replaced:
        print("\nreplaced with the backup's copy (the previous versions are gone):")
        for path in report.replaced:
            print(f"  {path}")
    if report.config_skipped:
        print("\nThe config directory was left as it is; the backup's copy is only")
        print("restored with --force, so a tuned profile is never silently undone.")
    print("\nRun `python run.py verify` to confirm every artifact still opens.")
    return 0


# =========================================================================
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="plaud-bridge",
        description="Plaud Bridge: own your recordings end to end.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--config", default="config", help="config directory (default: config)")
    ap.add_argument("--log-level", default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    ap.add_argument("--version", action="version", version=f"plaud-bridge {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor", help="preflight every dependency, key, and profile")
    p.add_argument("--offline", action="store_true",
                   help="also audit whether this machine could run with no network")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("run", help="process everything in the inbox")
    p.add_argument("--force", action="store_true", help="reprocess even if already seen")
    p.add_argument("--limit", type=int, default=None, help="stop after N files")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("digest", help="render a digest")
    p.add_argument("--profile", default=None, help="filter to one profile id")
    p.add_argument("--days", type=int, default=None, help="lookback window")
    p.add_argument("--include-personal", action="store_true",
                   help="include father/husband in a combined digest")
    p.add_argument("--out", default=None, help="write to a file instead of stdout")
    p.add_argument("--format", default="markdown", choices=["markdown", "html"],
                   help="html is self-contained and prints cleanly")
    p.add_argument("--title", default=None)
    p.set_defaults(func=cmd_digest)

    p = sub.add_parser("followups",
                       help="commitments still open, and drafts you send yourself")
    p.add_argument("--profile", default=None, help="filter to one profile id")
    p.add_argument("--days", type=int, default=None,
                   help="lookback window. The default is everything: a promise does "
                        "not stop counting because it is old.")
    p.add_argument("--status", default="open",
                   choices=["open", "done", "dropped", "all"],
                   help="which follow-ups to show (default: open)")
    p.add_argument("--include-personal", action="store_true",
                   help="include father/husband, in the list and in drafts")
    p.add_argument("--done", default=None, metavar="ID", help="mark one follow-up done")
    p.add_argument("--drop", default=None, metavar="ID", help="mark one as dropped")
    p.add_argument("--reopen", default=None, metavar="ID", help="mark one open again")
    p.add_argument("--draft", default=None, metavar="ID",
                   help="write a draft into the outbox: a follow-up id, a recording "
                        "id, or 'open' for everything outstanding. Nothing is sent.")
    p.add_argument("--format", default="markdown",
                   choices=["markdown", "html", "text"],
                   help="html for the worklist, text for a plain-text draft")
    p.add_argument("--out", default=None, help="write to a file instead of stdout")
    p.add_argument("--title", default=None)
    p.set_defaults(func=cmd_followups)

    p = sub.add_parser("insights",
                       help="how you talk: share, pace, questions, monologues")
    p.add_argument("--recording", default=None, metavar="ID",
                   help="one recording's breakdown per speaker")
    p.add_argument("--days", type=int, default=30, help="lookback window (default 30)")
    p.add_argument("--profile", default=None, help="filter to one profile id")
    p.add_argument("--include-personal", action="store_true",
                   help="count father/husband recordings too; they are left out "
                        "by default, the same rule as the digest")
    p.set_defaults(func=cmd_insights)

    p = sub.add_parser("status", help="index summary")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("search", help="find recordings by filename, or by what was said")
    p.add_argument("query")
    p.add_argument("--content", action="store_true",
                   help="search inside the transcripts, decrypting where needed")
    p.add_argument("--context", type=int, default=0,
                   help="segments of surrounding speech to include (with --content)")
    p.add_argument("--per-recording", type=int, default=5,
                   help="hits to show per recording before summarising the rest")
    p.add_argument("--scan-limit", type=int, default=0,
                   help="with --content: max recordings to open (0 = all). This "
                        "bounds work, not results; a bounded search says so.")
    p.add_argument("--profile", default=None)
    p.add_argument("--days", type=int, default=None)
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("ask", help="answer a question from what was actually said")
    p.add_argument("question", help='e.g. "what did I promise the Hendersons?"')
    p.add_argument("--profile", default=None, help="restrict to one profile id")
    p.add_argument("--days", type=int, default=None,
                   help="lookback window (default: ask.days; 0 searches everything)")
    p.add_argument("--limit", type=int, default=None,
                   help="how many recordings may contribute to one answer")
    p.add_argument("--include-personal", action="store_true",
                   help="search father/husband too; they are left out by default")
    p.add_argument("--local-only", action="store_true",
                   help="force local processing even where every profile permits cloud")
    p.add_argument("--save", action="store_true",
                   help="keep the answer in the vault, encrypted")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("verify", help="check every artifact still exists and still opens")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("forget", help="permanently delete one recording and its artifacts")
    p.add_argument("recording_id")
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    p.set_defaults(func=cmd_forget)

    p = sub.add_parser("export", help="build a redacted document for someone else")
    p.add_argument("--profile", default=None)
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--transcripts", action="store_true",
                   help="export redacted transcripts instead of the analysis")
    p.add_argument("--include-personal", action="store_true",
                   help="include profiles normally excluded from anything shareable")
    p.add_argument("--format", default="markdown", choices=["markdown", "html"])
    p.add_argument("--out", default=None)
    p.add_argument("--title", default=None)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("watch", help="process the inbox on an interval")
    p.add_argument("--interval", type=int, default=300, help="seconds between runs")
    p.add_argument("--once", action="store_true", help="run a single pass and exit")
    p.add_argument("--max-runs", type=int, default=None)
    p.set_defaults(func=cmd_watch)

    p = sub.add_parser("open", help="print an artifact, decrypting if needed")
    p.add_argument("recording_id")
    p.add_argument("--kind", default="transcript",
                   choices=["transcript", "analysis", "audio", "source"])
    p.add_argument("--out", default=None,
                   help="write to a file instead of stdout (required for audio)")
    p.set_defaults(func=cmd_open)

    p = sub.add_parser("audit", help="read the compliance audit log")
    p.add_argument("--recording-id", default=None, help="filter to one recording")
    p.add_argument("--action", default=None,
                   help="filter by action (ingest, route, compliance, quarantine, "
                        "quarantine_release, retention_delete, failure, run_complete)")
    p.add_argument("--actor", default=None, help="filter by actor (pipeline, human)")
    p.add_argument("--days", type=int, default=None, help="lookback window")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--out", default=None, help="write CSV to a file instead of stdout")
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("release", help="release a quarantined recording after review")
    p.add_argument("recording_id")
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    p.set_defaults(func=cmd_release)

    p = sub.add_parser("quarantine",
                       help="triage everything in quarantine: list, bulk release, bulk forget")
    verbs = p.add_mutually_exclusive_group()
    verbs.add_argument("--release-all", action="store_true",
                       help="release everything EXCEPT explicit refusals back to the inbox")
    verbs.add_argument("--forget-all", action="store_true",
                       help="permanently delete every quarantined recording and its traces")
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    p.set_defaults(func=cmd_quarantine)

    p = sub.add_parser("retention", help="show or execute the retention sweep")
    p.add_argument("--execute", action="store_true", help="actually delete (default is dry run)")
    p.add_argument("--yes", action="store_true")
    p.set_defaults(func=cmd_retention)

    p = sub.add_parser("profiles", help="show the routing table")
    p.set_defaults(func=cmd_profiles)

    p = sub.add_parser("memory", help="what the tool has learned across recordings")
    p.add_argument("--profile", default=None, help="one profile id (default: every profile)")
    p.add_argument("--brief", action="store_true",
                   help="show the briefing that gets injected into the next analysis")
    p.add_argument("--rebuild", action="store_true",
                   help="discard the ledgers and rebuild them from the stored archive")
    p.add_argument("--force", action="store_true",
                   help="with --rebuild: accept a rebuild that could not open everything")
    p.add_argument("--forget", default=None, metavar="RECORDING_ID",
                   help="remove one recording from every ledger")
    p.set_defaults(func=cmd_memory)

    p = sub.add_parser("new-profile", help="scaffold a profile from the template")
    p.add_argument("profile_id", help="identifier and filename stem, e.g. mentor")
    p.add_argument("--name", default=None, help='display name, e.g. "Mentor"')
    p.add_argument("--short-name", default=None)
    p.add_argument("--heading", default=None, help="how it appears in the digest")
    p.set_defaults(func=cmd_new_profile)

    p = sub.add_parser("voices", help="show installed voice packs")
    p.set_defaults(func=cmd_voices)

    p = sub.add_parser("speakers", help="teach it who is who, so transcripts use names")
    p.set_defaults(func=cmd_speakers)
    speakers = p.add_subparsers(dest="speakers_action", required=True)

    sp = speakers.add_parser("list", help="who this archive can recognise")
    sp.set_defaults(func=cmd_speakers)

    sp = speakers.add_parser("enroll", help="learn a voice from a clip of one person talking")
    sp.add_argument("name", help='display name, e.g. "Marcus"')
    sp.add_argument("--audio", required=True, help="a clip of this person and nobody else")
    sp.add_argument("--start", type=float, default=None,
                    help="seconds into the file to start (use when the clip is not clean)")
    sp.add_argument("--end", type=float, default=None, help="seconds into the file to stop")
    sp.add_argument("--replace", action="store_true",
                    help="discard this person's existing samples instead of adding to them")
    sp.set_defaults(func=cmd_speakers)

    sp = speakers.add_parser("identify",
                             help="score a recording against the enrolled voices, changing nothing")
    sp.add_argument("audio")
    sp.set_defaults(func=cmd_speakers)

    sp = speakers.add_parser("forget", help="delete a voiceprint permanently")
    sp.add_argument("name")
    sp.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    sp.set_defaults(func=cmd_speakers)

    p = sub.add_parser("review", help="the COMPLIANCE.md review cadence, assembled")
    p.add_argument("--days", type=int, default=30, help="lookback window (default 30)")
    p.add_argument("--reaffirm", default=None, metavar="PROFILE",
                   help="record a standing-consent reaffirmation for a profile")
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("backup", help="everything worth keeping, as one encrypted file")
    p.add_argument("--out", default=None,
                   help="where to write the .pbb file "
                        "(default: plaud-backup-<timestamp>.pbb in your home directory)")
    p.set_defaults(func=cmd_backup)

    p = sub.add_parser("restore", help="bring the archive back from a backup file")
    p.add_argument("file", help="a .pbb file written by `backup`")
    p.add_argument("--force", action="store_true",
                   help="replace data already in place, including the config directory")
    p.set_defaults(func=cmd_restore)

    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"\nConfiguration error:\n{exc}\n", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except EOFError:
        # A prompt with nobody to answer it. `_confirm` handles the ones we
        # know about; this is the backstop so a new prompt added later cannot
        # turn a cron job into a traceback.
        print("\nstdin closed while waiting for input; nothing was changed", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
