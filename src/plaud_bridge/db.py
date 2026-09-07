"""
SQLite index.

The filesystem holds the artifacts; this holds the index that makes them
searchable and filterable. Plain sqlite3, no ORM. Schema migrations are
explicit and versioned so a five-year-old database still opens.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .logging_setup import get
from .models import Recording

log = get("db")

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS recordings (
    id                TEXT PRIMARY KEY,
    content_hash      TEXT NOT NULL UNIQUE,
    source_name       TEXT NOT NULL,
    source_path       TEXT NOT NULL,
    kind              TEXT NOT NULL,
    size_bytes        INTEGER NOT NULL DEFAULT 0,
    duration_seconds  REAL NOT NULL DEFAULT 0,
    recorded_at       TEXT,
    ingested_at       TEXT NOT NULL,
    stage             TEXT NOT NULL,
    governing_profile TEXT,
    sensitivity       TEXT,
    consent_status    TEXT,
    encrypted         INTEGER NOT NULL DEFAULT 0,
    total_cost_usd    REAL NOT NULL DEFAULT 0,
    payload_json      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_rec_ingested  ON recordings(ingested_at);
CREATE INDEX IF NOT EXISTS idx_rec_stage     ON recordings(stage);
CREATE INDEX IF NOT EXISTS idx_rec_recorded  ON recordings(recorded_at);

CREATE TABLE IF NOT EXISTS routes (
    recording_id  TEXT NOT NULL,
    profile_id    TEXT NOT NULL,
    confidence    REAL NOT NULL,
    PRIMARY KEY (recording_id, profile_id),
    FOREIGN KEY (recording_id) REFERENCES recordings(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_routes_profile ON routes(profile_id);

CREATE TABLE IF NOT EXISTS analyses (
    recording_id            TEXT NOT NULL,
    profile_id              TEXT NOT NULL,
    requires_human_attention INTEGER NOT NULL DEFAULT 0,
    llm_model               TEXT,
    cost_usd                REAL NOT NULL DEFAULT 0,
    fields_json             TEXT NOT NULL,
    error                   TEXT,
    PRIMARY KEY (recording_id, profile_id),
    FOREIGN KEY (recording_id) REFERENCES recordings(id) ON DELETE CASCADE
);

-- Spend that belongs to no recording. `ask` and an LLM-phrased draft both cost
-- money and neither has a recording to attach it to, so without this they were
-- invisible to `status` -- which contradicts ADR-014, and quietly, which is the
-- part that matters: a spend guardrail nobody can see is not a guardrail.
CREATE TABLE IF NOT EXISTS spend (
    at        TEXT NOT NULL,
    source    TEXT NOT NULL,
    provider  TEXT NOT NULL DEFAULT '',
    model     TEXT NOT NULL DEFAULT '',
    cost_usd  REAL NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_spend_at ON spend(at);

CREATE TABLE IF NOT EXISTS artifacts (
    recording_id TEXT NOT NULL,
    kind         TEXT NOT NULL,
    path         TEXT NOT NULL,
    encrypted    INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL,
    expires_at   TEXT,
    PRIMARY KEY (recording_id, kind),
    FOREIGN KEY (recording_id) REFERENCES recordings(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_artifacts_expiry ON artifacts(expires_at);

CREATE TABLE IF NOT EXISTS audit (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    at           TEXT NOT NULL,
    recording_id TEXT,
    action       TEXT NOT NULL,
    detail       TEXT,
    actor        TEXT NOT NULL DEFAULT 'pipeline'
);

CREATE INDEX IF NOT EXISTS idx_audit_at ON audit(at);
CREATE INDEX IF NOT EXISTS idx_audit_rec ON audit(recording_id);
"""


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._migrate()

    def _migrate(self) -> None:
        with self.tx() as cur:
            cur.executescript(_SCHEMA)
            cur.execute("SELECT value FROM meta WHERE key='schema_version'")
            row = cur.fetchone()
            if row is None:
                cur.execute(
                    "INSERT INTO meta(key,value) VALUES('schema_version',?)",
                    (str(SCHEMA_VERSION),),
                )
            elif int(row["value"]) > SCHEMA_VERSION:
                raise RuntimeError(
                    f"database schema v{row['value']} is newer than this code "
                    f"(v{SCHEMA_VERSION}). Upgrade plaud-bridge."
                )

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Cursor]:
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cur.close()

    def close(self) -> None:
        self._conn.close()

    # ---- writes -----------------------------------------------------------
    def hash_exists(self, content_hash: str) -> str | None:
        with self.tx() as cur:
            cur.execute("SELECT id FROM recordings WHERE content_hash=?", (content_hash,))
            row = cur.fetchone()
            return row["id"] if row else None

    def upsert(self, rec: Recording) -> None:
        # The index is a plain file. A recording whose artifacts are encrypted
        # must not leave its transcript or its extracted quotes sitting here in
        # the clear; the vault copy is the one that keeps the words.
        encrypted = rec.is_encrypted
        payload = rec.to_json(
            indent=None,
            include_transcript=not encrypted,
            include_analysis_fields=not encrypted,
        )
        try:
            self._upsert(rec, payload, encrypted)
        except sqlite3.IntegrityError as exc:
            # content_hash is UNIQUE. Two processes -- a `watch` loop and a
            # manual `run`, say -- can both pass the dedupe check for the same
            # file before either writes. The other one won the race and the
            # recording is already indexed; that is a duplicate, not a failure,
            # and it should not be reported to the user as a broken recording.
            log.warning(
                "not indexing %s: another process already recorded this content (%s)",
                rec.source_name, exc,
            )

    def _upsert(self, rec: Recording, payload: str, encrypted: bool) -> None:
        with self.tx() as cur:
            cur.execute(
                """
                INSERT INTO recordings (
                    id, content_hash, source_name, source_path, kind, size_bytes,
                    duration_seconds, recorded_at, ingested_at, stage,
                    governing_profile, sensitivity, consent_status, encrypted,
                    total_cost_usd, payload_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    stage=excluded.stage,
                    duration_seconds=excluded.duration_seconds,
                    governing_profile=excluded.governing_profile,
                    sensitivity=excluded.sensitivity,
                    consent_status=excluded.consent_status,
                    encrypted=excluded.encrypted,
                    total_cost_usd=excluded.total_cost_usd,
                    payload_json=excluded.payload_json
                """,
                (
                    rec.id,
                    rec.content_hash,
                    rec.source_name,
                    rec.source_path,
                    rec.kind,
                    rec.size_bytes,
                    rec.duration_seconds,
                    rec.recorded_at.isoformat() if rec.recorded_at else None,
                    rec.ingested_at.isoformat(),
                    rec.stage.value,
                    rec.compliance.governing_profile,
                    rec.compliance.governing_sensitivity.value,
                    rec.compliance.consent.value,
                    1 if rec.is_encrypted else 0,
                    rec.total_cost_usd,
                    payload,
                ),
            )
            cur.execute("DELETE FROM routes WHERE recording_id=?", (rec.id,))
            for r in rec.routes:
                cur.execute(
                    "INSERT INTO routes(recording_id,profile_id,confidence) VALUES (?,?,?)",
                    (rec.id, r.profile_id, r.confidence),
                )
            cur.execute("DELETE FROM analyses WHERE recording_id=?", (rec.id,))
            for a in rec.analyses:
                cur.execute(
                    """INSERT INTO analyses(recording_id,profile_id,
                       requires_human_attention,llm_model,cost_usd,fields_json,error)
                       VALUES (?,?,?,?,?,?,?)""",
                    (
                        rec.id,
                        a.profile_id,
                        1 if a.requires_human_attention else 0,
                        a.llm_model,
                        a.cost_usd,
                        json.dumps({} if encrypted else a.fields, ensure_ascii=False),
                        a.error or None,
                    ),
                )

    def record_artifact(self, recording_id: str, kind: str, path: str,
                        encrypted: bool, expires_at: datetime | None) -> None:
        with self.tx() as cur:
            cur.execute(
                """INSERT INTO artifacts(recording_id,kind,path,encrypted,created_at,expires_at)
                   VALUES (?,?,?,?,?,?)
                   ON CONFLICT(recording_id,kind) DO UPDATE SET
                     path=excluded.path, encrypted=excluded.encrypted,
                     expires_at=excluded.expires_at""",
                (
                    recording_id,
                    kind,
                    path,
                    1 if encrypted else 0,
                    datetime.now(timezone.utc).isoformat(),
                    expires_at.isoformat() if expires_at else None,
                ),
            )

    def record_spend(self, source: str, cost_usd: float, provider: str = "",
                     model: str = "") -> None:
        """
        Note money spent outside the pipeline.

        A zero is still recorded. A local model costs nothing and a row saying so
        is how you tell "this ran locally" apart from "this never ran", which is
        the same reasoning as the unpriced-provider warning in ADR-014.
        """
        with self.tx() as cur:
            cur.execute(
                "INSERT INTO spend(at,source,provider,model,cost_usd) VALUES (?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(), source, provider, model,
                 float(cost_usd or 0.0)),
            )

    def audit(self, action: str, detail: str = "", recording_id: str | None = None,
              actor: str = "pipeline") -> None:
        with self.tx() as cur:
            cur.execute(
                "INSERT INTO audit(at,recording_id,action,detail,actor) VALUES (?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(), recording_id, action, detail, actor),
            )

    # ---- reads ------------------------------------------------------------
    def load(self, recording_id: str) -> dict[str, Any] | None:
        with self.tx() as cur:
            cur.execute("SELECT payload_json FROM recordings WHERE id=?", (recording_id,))
            row = cur.fetchone()
            return json.loads(row["payload_json"]) if row else None

    def query(self, profile_id: str | None = None, since_days: int | None = None,
              until: datetime | None = None, stage: str | None = None,
              search: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        sql = [
            "SELECT DISTINCT r.id, r.source_name, r.recorded_at, r.ingested_at,",
            "       r.duration_seconds, r.stage, r.governing_profile, r.sensitivity,",
            "       r.consent_status, r.encrypted, r.total_cost_usd, r.payload_json",
            "FROM recordings r",
        ]
        params: list[Any] = []
        if profile_id:
            sql.append("JOIN routes rt ON rt.recording_id = r.id")
        sql.append("WHERE 1=1")
        if profile_id:
            sql.append("AND rt.profile_id = ?")
            params.append(profile_id)
        if since_days is not None:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=since_days)).isoformat()
            sql.append("AND COALESCE(r.recorded_at, r.ingested_at) >= ?")
            params.append(cutoff)
        if until is not None:
            sql.append("AND COALESCE(r.recorded_at, r.ingested_at) <= ?")
            params.append(until.isoformat())
        if stage:
            sql.append("AND r.stage = ?")
            params.append(stage)
        if search:
            # Escape LIKE's own wildcards. Without this, searching for a
            # filename containing "_" matches any character and "%" matches
            # everything, so the results quietly include things you did not ask
            # for -- and "100%.mp3" matches every file you own.
            escaped = search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            sql.append("AND r.source_name LIKE ? ESCAPE '\\'")
            params.append(f"%{escaped}%")
        sql.append("ORDER BY COALESCE(r.recorded_at, r.ingested_at) DESC LIMIT ?")
        params.append(limit)

        with self.tx() as cur:
            cur.execute(" ".join(sql), params)
            return [dict(r) for r in cur.fetchall()]

    def audit_log(self, recording_id: str | None = None, action: str | None = None,
                  actor: str | None = None, since_days: int | None = None,
                  limit: int = 100) -> list[dict[str, Any]]:
        """Newest first. Filters compose; all of them are optional."""
        sql = ["SELECT at, recording_id, action, detail, actor FROM audit WHERE 1=1"]
        params: list[Any] = []
        if recording_id:
            sql.append("AND recording_id = ?")
            params.append(recording_id)
        if action:
            sql.append("AND action = ?")
            params.append(action)
        if actor:
            sql.append("AND actor = ?")
            params.append(actor)
        if since_days is not None:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=since_days)).isoformat()
            sql.append("AND at >= ?")
            params.append(cutoff)
        sql.append("ORDER BY at DESC, id DESC LIMIT ?")
        params.append(limit)

        with self.tx() as cur:
            cur.execute(" ".join(sql), params)
            return [dict(r) for r in cur.fetchall()]

    def expired_artifacts(self, now: datetime | None = None) -> list[dict[str, Any]]:
        now = now or datetime.now(timezone.utc)
        with self.tx() as cur:
            cur.execute(
                "SELECT * FROM artifacts WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (now.isoformat(),),
            )
            return [dict(r) for r in cur.fetchall()]

    def count_recordings(self, profile_id: str | None = None,
                         since_days: int | None = None) -> int:
        """How many rows a query would match, before any limit is applied."""
        sql = ["SELECT COUNT(DISTINCT r.id) c FROM recordings r"]
        params: list[Any] = []
        if profile_id:
            sql.append("JOIN routes rt ON rt.recording_id = r.id")
        sql.append("WHERE 1=1")
        if profile_id:
            sql.append("AND rt.profile_id = ?")
            params.append(profile_id)
        if since_days is not None:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=since_days)).isoformat()
            sql.append("AND COALESCE(r.recorded_at, r.ingested_at) >= ?")
            params.append(cutoff)
        with self.tx() as cur:
            cur.execute(" ".join(sql), params)
            return int(cur.fetchone()["c"])

    def count_audit_before(self, cutoff: datetime) -> int:
        with self.tx() as cur:
            cur.execute("SELECT COUNT(*) c FROM audit WHERE at < ?", (cutoff.isoformat(),))
            return int(cur.fetchone()["c"])

    def delete_audit_before(self, cutoff: datetime | None) -> int:
        """Trim the audit trail. Refuses an open-ended cutoff."""
        if cutoff is None:
            return 0
        with self.tx() as cur:
            cur.execute("DELETE FROM audit WHERE at < ?", (cutoff.isoformat(),))
            return cur.rowcount

    def all_artifacts(self) -> list[dict[str, Any]]:
        with self.tx() as cur:
            cur.execute("SELECT * FROM artifacts ORDER BY recording_id, kind")
            return [dict(r) for r in cur.fetchall()]

    def delete_recording(self, recording_id: str) -> None:
        """
        Remove a recording and everything the index knows about it.

        The audit rows are left alone on purpose. They carry no foreign key, so
        they survive, and they should: "this recording was deleted, by a human,
        at this time" is exactly the kind of thing an audit trail exists to
        record. A trail that forgets deletions is not a trail.
        """
        with self.tx() as cur:
            cur.execute("DELETE FROM recordings WHERE id=?", (recording_id,))

    def drop_artifact(self, recording_id: str, kind: str) -> None:
        with self.tx() as cur:
            cur.execute(
                "DELETE FROM artifacts WHERE recording_id=? AND kind=?", (recording_id, kind)
            )

    def stats(self) -> dict[str, Any]:
        with self.tx() as cur:
            cur.execute("SELECT COUNT(*) c, COALESCE(SUM(duration_seconds),0) d, "
                        "COALESCE(SUM(total_cost_usd),0) k FROM recordings")
            row = cur.fetchone()
            cur.execute("SELECT profile_id, COUNT(*) c FROM routes GROUP BY profile_id")
            by_profile = {r["profile_id"]: r["c"] for r in cur.fetchall()}
            cur.execute("SELECT stage, COUNT(*) c FROM recordings GROUP BY stage")
            by_stage = {r["stage"]: r["c"] for r in cur.fetchall()}
            cur.execute("SELECT source, COALESCE(SUM(cost_usd),0) k, COUNT(*) c "
                        "FROM spend GROUP BY source")
            other = {r["source"]: {"cost_usd": round(r["k"], 4), "calls": r["c"]}
                     for r in cur.fetchall()}
        other_cost = sum(v["cost_usd"] for v in other.values())
        return {
            "recordings": row["c"],
            "audio_hours": round(row["d"] / 3600.0, 2),
            "pipeline_cost_usd": round(row["k"], 4),
            "other_cost_usd": round(other_cost, 4),
            "total_cost_usd": round(row["k"] + other_cost, 4),
            "by_source": other,
            "by_profile": by_profile,
            "by_stage": by_stage,
        }
