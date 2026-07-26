#!/usr/bin/env python3
"""
Plaud Bridge command line.

    python run.py doctor                        preflight every dependency and key
    python run.py run                           process everything in the inbox
    python run.py digest                        combined digest, last 7 days
    python run.py digest --profile husband      one profile only
    python run.py status                        index summary
    python run.py search "elimination period"   find recordings
    python run.py open <recording_id>           decrypt and print an artifact
    python run.py release <recording_id>        release a quarantined recording
    python run.py retention --execute           delete expired artifacts
    python run.py profiles                      show the routing table
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from plaud_bridge import __version__                                   # noqa: E402
from plaud_bridge.compliance import RetentionSweeper                   # noqa: E402
from plaud_bridge.config import Config, ConfigError                    # noqa: E402
from plaud_bridge.db import Database                                   # noqa: E402
from plaud_bridge.digest import DigestBuilder, DigestOptions           # noqa: E402
from plaud_bridge.logging_setup import setup                           # noqa: E402
from plaud_bridge.pipeline import Pipeline                             # noqa: E402
from plaud_bridge.storage import Vault, VaultError                     # noqa: E402

OK, WARN, BAD = "  ok  ", " warn ", " FAIL "


def _load(args) -> Config:
    cfg = Config.load(args.config)
    setup(
        cfg.path("logs"),
        level=args.log_level or cfg.get("logging.level", "INFO"),
        redact_content=bool(cfg.get("logging.redact_content", True)),
        rotate_mb=int(cfg.get("logging.rotate_mb", 20)),
        backups=int(cfg.get("logging.backups", 5)),
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
        from plaud_bridge.audio import AudioPreparer

        AudioPreparer(cfg).check_tools()
        rows.append((OK, "ffmpeg", "found"))
    except Exception as exc:  # noqa: BLE001
        rows.append((BAD, "ffmpeg", str(exc)[:90]))
        fatal = True

    # ASR
    from plaud_bridge.asr.registry import build_asr_chain

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
    from plaud_bridge.diarize.engine import _available as diar_available

    ok, why = diar_available(cfg)
    rows.append((OK if ok else WARN, "diarization", why if ok else why + " (speaker labels off)"))

    # LLM
    from plaud_bridge.llm.registry import build_llm_chain

    any_llm = False
    for provider in build_llm_chain(cfg):
        ok, why = provider.available()
        rows.append((OK if ok else WARN, f"llm:{provider.name}", why))
        any_llm = any_llm or ok
    if not any_llm:
        rows.append((BAD, "llm", "no usable LLM provider; analysis will fail"))
        fatal = True

    local_llm = [p for p in build_llm_chain(cfg, local_only=True)]
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

        if args.out:
            dest = Path(args.out)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(markdown, encoding="utf-8")
            print(f"wrote {dest}")
        else:
            print(markdown)
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
        prog="run.py",
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
