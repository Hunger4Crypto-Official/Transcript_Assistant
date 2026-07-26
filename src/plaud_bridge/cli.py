#!/usr/bin/env python3
"""
Plaud Bridge command line.

    plaud-bridge doctor                        preflight every dependency and key
    plaud-bridge run                           process everything in the inbox
    plaud-bridge digest                        combined digest, last 7 days
    plaud-bridge digest --profile husband      one profile only
    plaud-bridge digest --format html          self-contained page, prints cleanly
    plaud-bridge review                        what the review cadence says is due
    plaud-bridge status                        index summary
    plaud-bridge search "elimination period"   find recordings
    plaud-bridge open <recording_id>           decrypt and print an artifact
    plaud-bridge audit                         read the compliance audit log
    plaud-bridge release <recording_id>        release a quarantined recording
    plaud-bridge retention --execute           delete expired artifacts
    plaud-bridge profiles                      show the routing table
    plaud-bridge new-profile <id>              scaffold a profile from the template
    plaud-bridge voices                        show installed voice packs

`python run.py <command>` runs the same code without installing anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .compliance import RetentionSweeper
from .config import Config, ConfigError
from .db import Database
from .digest import DigestBuilder, DigestOptions, to_html
from .logging_setup import setup
from .pipeline import Pipeline
from .storage import Vault, VaultError
from .voice import Voice

OK, WARN, BAD = "  ok  ", " warn ", " FAIL "


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
        if stats.failed:
            print(f"{stats.failed} recording(s) failed. Check the log at "
                  f"{cfg.path('logs') / 'bridge.log'}")
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
        markdown = DigestBuilder(cfg, db).render_markdown(opts)

        # HTML is rendered from the markdown rather than from the section data,
        # so the two formats cannot drift into saying different things.
        if args.format == "html":
            title = opts.title or (
                cfg.profile(opts.profile_id).name if opts.profile_id else "Digest"
            )
            body = to_html(markdown, title=title)
        else:
            body = markdown

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


def cmd_status(args) -> int:
    cfg = _load(args)
    db = Database(cfg.path("database"))
    try:
        stats = db.stats()
        print(f"\nPlaud Bridge index :: {cfg.path('database')}\n")
        print(f"  recordings   {stats['recordings']}")
        print(f"  audio hours  {stats['audio_hours']}")
        print(f"  api spend    ${stats['total_cost_usd']}")
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
        rows = db.query(profile_id=args.profile, since_days=args.days,
                        search=args.query, limit=args.limit)
        if not rows:
            print("no matches")
            return 0
        print(f"\n{len(rows)} match(es)\n")
        for row in rows:
            when = (row["recorded_at"] or row["ingested_at"] or "")[:16].replace("T", " ")
            lock = " [encrypted]" if row["encrypted"] else ""
            print(f"  {row['id']}  {when}  {row['duration_seconds'] / 60:5.1f}m  "
                  f"{row['governing_profile'] or '-':16s} {row['source_name']}{lock}")
        print()
        return 0
    finally:
        db.close()


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
        if path.suffix == ".enc":
            try:
                print(Vault(cfg.path("vault")).read_text(path, args.recording_id))
            except VaultError as exc:
                print(f"could not decrypt: {exc}")
                return 1
        else:
            print(path.read_text(encoding="utf-8"))
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


def cmd_release(args) -> int:
    """Move a quarantined file back to the inbox after human review."""
    cfg = _load(args)
    db = Database(cfg.path("database"))
    try:
        qdir = cfg.path("quarantine") / args.recording_id
        if not qdir.is_dir():
            print(f"no quarantine folder for {args.recording_id}")
            return 1
        media = [p for p in qdir.iterdir() if p.name != "WHY.md"]
        if not media:
            print("quarantine folder has no media to release")
            return 1

        if not args.yes:
            print(f"\nAbout to release {len(media)} file(s) back to the inbox.\n")
            print((qdir / "WHY.md").read_text(encoding="utf-8"))
            answer = input("Type RELEASE to confirm you verified consent: ").strip()
            if answer != "RELEASE":
                print("aborted")
                return 1

        inbox = cfg.path("inbox")
        for path in media:
            dest = inbox / path.name
            dest.write_bytes(path.read_bytes())
            print(f"released -> {dest}")

        db.audit("quarantine_release", "released after human review", args.recording_id, actor="human")
        print("\nRe-run `python run.py run --force` to process it.")
        print("The release is recorded in the audit log.")
        return 0
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
        if not dry and plan.items:
            if not args.yes:
                answer = input(f"Delete {len(plan.items)} artifact(s)? Type DELETE: ").strip()
                if answer != "DELETE":
                    print("aborted")
                    return 1
            removed = sweeper.execute(plan)
            print(f"deleted {removed} artifact(s)")
        return 0
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
    body = (
        template.read_text(encoding="utf-8")
        .replace('id: "PROFILE_ID"', f'id: {pid}')
        .replace('name: "Profile Name"', f'name: "{name}"')
        .replace('short_name: "Short"', f'short_name: "{args.short_name or name}"')
        .replace('heading: "Section Heading"', f'heading: "{args.heading or name}"')
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
            answer = input("Type YES to record the reaffirmation: ").strip()
            if answer != "YES":
                print("not recorded")
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
            age = (now - datetime.fromisoformat(last)).days
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

    p = sub.add_parser("status", help="index summary")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("search", help="find recordings by filename")
    p.add_argument("query")
    p.add_argument("--profile", default=None)
    p.add_argument("--days", type=int, default=None)
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("open", help="print an artifact, decrypting if needed")
    p.add_argument("recording_id")
    p.add_argument("--kind", default="transcript", choices=["transcript", "analysis"])
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

    p = sub.add_parser("retention", help="show or execute the retention sweep")
    p.add_argument("--execute", action="store_true", help="actually delete (default is dry run)")
    p.add_argument("--yes", action="store_true")
    p.set_defaults(func=cmd_retention)

    p = sub.add_parser("profiles", help="show the routing table")
    p.set_defaults(func=cmd_profiles)

    p = sub.add_parser("new-profile", help="scaffold a profile from the template")
    p.add_argument("profile_id", help="identifier and filename stem, e.g. mentor")
    p.add_argument("--name", default=None, help='display name, e.g. "Mentor"')
    p.add_argument("--short-name", default=None)
    p.add_argument("--heading", default=None, help="how it appears in the digest")
    p.set_defaults(func=cmd_new_profile)

    p = sub.add_parser("voices", help="show installed voice packs")
    p.set_defaults(func=cmd_voices)

    p = sub.add_parser("review", help="the COMPLIANCE.md review cadence, assembled")
    p.add_argument("--days", type=int, default=30, help="lookback window (default 30)")
    p.add_argument("--reaffirm", default=None, metavar="PROFILE",
                   help="record a standing-consent reaffirmation for a profile")
    p.set_defaults(func=cmd_review)

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


if __name__ == "__main__":
    raise SystemExit(main())
