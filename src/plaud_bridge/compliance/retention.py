"""
Retention sweeping.

Two design choices worth stating plainly:

1. Dry run is the default. A tool that deletes your recordings by surprise gets
   uninstalled, and rightly so. You see the plan, then you approve it.

2. Raw audio expires far sooner than transcripts. Audio is the liability;
   transcripts are the asset. Keeping ten years of transcripts costs you
   megabytes. Keeping ten years of audio costs you a discovery request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..logging_setup import get

log = get("retention")


@dataclass
class RetentionItem:
    recording_id: str
    kind: str
    path: str
    expires_at: str
    exists: bool
    size_bytes: int = 0


@dataclass
class RetentionPlan:
    items: list[RetentionItem] = field(default_factory=list)
    dry_run: bool = True

    @property
    def total_bytes(self) -> int:
        return sum(i.size_bytes for i in self.items)

    def render(self) -> str:
        if not self.items:
            return "Retention sweep: nothing has expired."
        lines = [
            f"Retention sweep {'(DRY RUN, nothing deleted)' if self.dry_run else '(LIVE)'}",
            f"{len(self.items)} artifact(s), {self.total_bytes / 1_048_576:.1f} MB",
            "",
        ]
        for item in self.items:
            state = "" if item.exists else "  [already gone]"
            lines.append(f"  {item.kind:12s} {item.recording_id}  expired {item.expires_at[:10]}{state}")
            lines.append(f"               {item.path}")
        if self.dry_run:
            lines.append("")
            lines.append("Re-run with --execute to delete these.")
        return "\n".join(lines)


class RetentionSweeper:
    def __init__(self, cfg, db):
        self.cfg = cfg
        self.db = db
        self.enabled = bool(cfg.get("retention.enabled", True))

    def expires_at(self, kind: str, profile, created: datetime | None = None) -> datetime | None:
        """Per-profile retention wins over the pipeline default."""
        created = created or datetime.now(timezone.utc)
        if kind == "audio":
            days = profile.raw_audio_days
        elif kind in ("transcript", "analysis", "markdown"):
            days = profile.transcript_days
        elif kind == "audit":
            days = profile.audit_log_days
        else:
            days = int(self.cfg.get("retention.default_transcript_days", 365))
        return created + timedelta(days=days) if days > 0 else None

    def plan(self, now: datetime | None = None, dry_run: bool = True) -> RetentionPlan:
        if not self.enabled:
            log.info("retention sweeping is disabled in config")
            return RetentionPlan(dry_run=dry_run)

        now = now or datetime.now(timezone.utc)
        plan = RetentionPlan(dry_run=dry_run)
        for row in self.db.expired_artifacts(now):
            path = Path(row["path"])
            exists = path.exists()
            plan.items.append(
                RetentionItem(
                    recording_id=row["recording_id"],
                    kind=row["kind"],
                    path=str(path),
                    expires_at=row["expires_at"],
                    exists=exists,
                    size_bytes=path.stat().st_size if exists else 0,
                )
            )
        return plan

    def execute(self, plan: RetentionPlan) -> int:
        # The flag on the plan is the plan's own answer to "may this delete
        # things", and it has to be honoured here rather than trusted to every
        # caller. One caller forgetting the check is a caller that silently
        # deletes a decade of transcripts while showing a preview.
        if plan.dry_run:
            raise ValueError(
                "refusing to execute a dry-run plan. Build the plan with "
                "dry_run=False if you actually mean to delete these artifacts."
            )

        removed = 0
        for item in plan.items:
            path = Path(item.path)
            if path.exists():
                try:
                    path.unlink()
                    removed += 1
                except OSError as exc:
                    log.error("could not delete %s: %s", path, exc)
                    continue
            self.db.drop_artifact(item.recording_id, item.kind)
            self.db.audit(
                "retention_delete",
                f"kind={item.kind} expired={item.expires_at}",
                item.recording_id,
            )
        log.info("retention sweep removed %d artifact(s)", removed)
        return removed
