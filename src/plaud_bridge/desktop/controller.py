"""
The engine behind the clickable app.

`AppController` is everything the desktop front end needs, with no window and no
web server in it, so it can be driven by a real UI and by a test the same way.
It owns four jobs and nothing else:

  1. Decide where the config and data live for an installed app, and lay down a
     working config from the shipped template on first run.
  2. Switch the analysis "brain" between fully-offline (a local model) and a free
     cloud key -- by choosing which LLM providers the chain may use, which is the
     same lever `pipeline.yaml` already exposes.
  3. Run a preflight (the same questions `run.py doctor` asks) so a person sees
     "you still need X" in the window rather than a stack trace.
  4. Take the audio a person picked, run it through the real `Pipeline`, and
     render the digest they came for.

It deliberately reuses `Config`, `Pipeline`, and `DigestBuilder` unchanged. A
front end that reimplemented any of that would be a second, weaker copy of the
rules this whole tool exists to enforce -- the locality locks, the consent gate,
the encrypted vault -- and those must have exactly one implementation.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from ..config import Config
from ..digest import DigestBuilder, DigestOptions


class Brain(str, Enum):
    """Which providers the analysis is allowed to use."""

    OFFLINE = "offline"   # a local model only; nothing leaves the machine
    CLOUD = "cloud"       # a free cloud key; work recordings only

    @property
    def label(self) -> str:
        return "Offline (fully private)" if self is Brain.OFFLINE else "Free cloud key"


# Which LLM providers each brain permits. The pipeline's own compliance gate is
# still the final word: a locked profile (father/husband) forces local even when
# CLOUD is chosen, because build_llm_chain removes cloud providers under a local
# veto. So CLOUD is "cloud allowed where policy permits", never "cloud always".
_BRAIN_PROVIDERS: dict[Brain, list[str]] = {
    Brain.OFFLINE: ["local"],
    Brain.CLOUD: ["groq", "local"],
}

PASSPHRASE_ENV = "PLAUD_BRIDGE_PASSPHRASE"
GROQ_KEY_ENV = "GROQ_API_KEY"


class LocalLLMStatus(Enum):
    """What a probe of the local model server actually found."""

    NOT_RUNNING = "not_running"     # nothing answering at the configured port
    MODEL_MISSING = "model_missing"  # server up, configured model not pulled
    READY = "ready"                  # server up and the model is there


def _same_model(configured: str, listed: str) -> bool:
    """
    Ollama's `:latest` is implicit: `ollama pull llama3` lists as
    `llama3:latest`, so a person who configured the bare name has the model and
    must not be told to pull it again. Any other tag difference is a real
    difference -- `llama3.3` is not `llama3.3:70b`.
    """
    def canon(name: str) -> str:
        return name[:-len(":latest")] if name.endswith(":latest") else name

    return canon(configured) == canon(listed)


def probe_local_llm(base_url: str, model: str,
                    timeout: float = 2.0) -> tuple[LocalLLMStatus, str]:
    """
    Ask the local model server what is actually true, not what config hopes.

    The provider's own `available()` only reads config -- it cannot tell "Ollama
    was never installed" from "Ollama is up but the model was never pulled",
    and those need different fixes typed by a person who may have never used a
    terminal for anything else. So this makes one real request to the OpenAI-
    compatible model list (`GET <base_url>/models`, which ollama, vLLM and
    llama.cpp all serve) and turns the answer into the exact command to run.

    The timeout is short on purpose: this runs inside the preflight a window is
    waiting on, and a dead route must come back as a red line in a couple of
    seconds, never as a hung UI.
    """
    url = base_url.rstrip("/") + "/models"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
    except urllib.error.HTTPError:
        # Something IS listening -- an HTTP error is a server talking. Treat it
        # as "up, model unconfirmed": the pull command is still the likely fix,
        # and "install Ollama" would be flatly wrong advice here.
        body = b""
    except (TimeoutError, OSError):
        # Connection refused, no route, or nothing answered in time: for the
        # person's purposes the server does not exist.
        return (
            LocalLLMStatus.NOT_RUNNING,
            "Ollama is not installed or not running -- install from ollama.com, "
            f"then run: ollama pull {model}",
        )

    try:
        payload = json.loads(body.decode("utf-8"))
        entries = payload.get("data", []) if isinstance(payload, dict) else []
        listed = [str(e.get("id", "")) for e in entries if isinstance(e, dict)]
    except ValueError:
        listed = []

    if any(_same_model(model, name) for name in listed):
        return LocalLLMStatus.READY, f"ready: local model '{model}' is loaded"
    return (
        LocalLLMStatus.MODEL_MISSING,
        f"Ollama is running but the model '{model}' is not pulled -- "
        f"run: ollama pull {model}",
    )


@dataclass
class PreflightItem:
    """One line of the readiness check, shaped for a UI to colour."""

    ok: bool
    fatal: bool
    name: str
    detail: str


def default_base_dir() -> Path:
    """
    Where an installed app keeps its config and data.

    A per-user, writable location, because the folder the .exe sits in may be
    read-only (Program Files) and two people on one machine should not share a
    vault. `PLAUD_BRIDGE_HOME` overrides it for anyone who wants the data on an
    external drive.
    """
    override = os.environ.get("PLAUD_BRIDGE_HOME", "").strip()
    if override:
        return Path(override).expanduser()
    if sys.platform.startswith("win"):
        root = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or str(Path.home())
        return Path(root) / "PlaudBridge"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "PlaudBridge"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "PlaudBridge"


def _bundled_config_template() -> Path:
    """
    The read-only config that ships inside the app.

    PyInstaller unpacks bundled data under `sys._MEIPASS`; a normal checkout has
    it at the repository root. Either way it is the pristine copy that seeds a
    new install, never the one a person edits.
    """
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidate = Path(meipass) / "config"
        if candidate.is_dir():
            return candidate
    # src/plaud_bridge/desktop/controller.py -> repo root is three parents up.
    return Path(__file__).resolve().parents[3] / "config"


class AppController:
    def __init__(self, base_dir: Path | str | None = None,
                 template_dir: Path | str | None = None):
        self.base_dir = Path(base_dir).expanduser() if base_dir else default_base_dir()
        self.config_dir = self.base_dir / "config"
        self._template_dir = Path(template_dir) if template_dir else _bundled_config_template()

    # ---- first run ------------------------------------------------------
    def ensure_installed(self) -> None:
        """
        Lay down a working config from the template if this is a fresh install.

        Copies rather than symlinks, so the person owns their profiles and an app
        update never silently rewrites the keywords they tuned. If a config is
        already here, it is left exactly as it is.
        """
        if (self.config_dir / "pipeline.yaml").exists():
            return
        if not (self._template_dir / "pipeline.yaml").exists():
            raise FileNotFoundError(
                f"cannot find the bundled config template at {self._template_dir}. "
                "The app was packaged without its config."
            )
        self.base_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self._template_dir, self.config_dir, dirs_exist_ok=True)

    # ---- secrets --------------------------------------------------------
    def set_passphrase(self, passphrase: str) -> None:
        """Hand the vault its key for this session. Never written to disk here."""
        os.environ[PASSPHRASE_ENV] = passphrase or ""

    def set_groq_key(self, key: str) -> None:
        os.environ[GROQ_KEY_ENV] = (key or "").strip()

    # ---- config with a chosen brain ------------------------------------
    def load_config(self, brain: Brain) -> Config:
        """
        The working config, with the LLM chain narrowed to the chosen brain.

        The providers list is the only thing overridden, in memory, so nothing
        else the person configured is touched and the on-disk file is not
        rewritten on every run. ASR already carries its own local fallback, so it
        is left alone.
        """
        self.ensure_installed()
        cfg = Config.load(self.config_dir, root=self.base_dir)
        llm = cfg._d.setdefault("llm", {})
        llm["providers"] = list(_BRAIN_PROVIDERS[brain])
        if brain is Brain.OFFLINE:
            # The template ships llm.local disabled so the CLI never assumes a
            # server nobody started. In the app, flipping the switch to Offline
            # IS that decision -- leaving the flag off would mean the offline
            # brain only works for someone who hand-edits pipeline.yaml, which
            # is exactly the person this window exists to spare. In-memory
            # only, like the providers list above.
            llm.setdefault("local", {})["enabled"] = True
        # The settle window exists so `watch` does not grab a file another program
        # is still copying. The app copies each picked file to completion before
        # it processes, so here it would only add a few seconds of "nothing is
        # happening" after the button is pressed. Zero it for the app's one-shot run.
        cfg._d.setdefault("ingest", {})["settle_seconds"] = 0
        cfg.ensure_dirs()
        return cfg

    # ---- readiness ------------------------------------------------------
    def preflight(self, brain: Brain) -> list[PreflightItem]:
        """
        The same questions `doctor` asks, shaped for a window.

        Returns one item per check. `fatal` marks the ones that stop a run;
        the UI shows those red and keeps the Process button disabled until they
        clear.
        """
        items: list[PreflightItem] = []
        cfg = self.load_config(brain)

        # ffmpeg -- required for any audio at all.
        try:
            from ..audio import AudioPreparer

            AudioPreparer(cfg).check_tools()
            items.append(PreflightItem(True, False, "ffmpeg", "found"))
        except Exception as exc:  # noqa: BLE001
            items.append(PreflightItem(
                False, True, "ffmpeg",
                "not found. Audio needs it. " + str(exc).splitlines()[0][:120],
            ))

        # A local ASR is required or father/husband recordings cannot run.
        from ..asr.registry import build_asr_chain

        local_asr = False
        for provider in build_asr_chain(cfg, cfg.glossary):
            ok, _ = provider.available()
            if ok and not provider.is_cloud:
                local_asr = True
        items.append(PreflightItem(
            local_asr, False, "local transcription",
            "ready" if local_asr else "not installed (needed for private recordings)",
        ))

        # An analysis brain the chosen mode can actually reach.
        if brain is Brain.OFFLINE:
            # For offline, config alone cannot answer "will this run": the
            # provider's available() reads config and would call a dead Ollama
            # "ready". Probe the real server so the red line names the one
            # command that fixes it, not a generic shrug.
            base_url = str(cfg.get("llm.local.base_url", "") or "")
            model = str(cfg.get("llm.local.model", "") or "")
            if base_url and model:
                status, detail = probe_local_llm(base_url, model)
                if status is not LocalLLMStatus.READY:
                    detail += " Or switch to the free cloud key."
                items.append(PreflightItem(
                    status is LocalLLMStatus.READY, True,
                    "analysis brain (offline)", detail,
                ))
            else:
                items.append(PreflightItem(
                    False, True, "analysis brain (offline)",
                    "llm.local in pipeline.yaml is missing base_url or model, "
                    "so there is nothing to probe. Restore those keys, or "
                    "switch to the free cloud key.",
                ))
        else:
            from ..llm.registry import build_llm_chain

            reachable = []
            for provider in build_llm_chain(cfg):
                ok, _ = provider.available()
                if ok:
                    reachable.append(provider.name)
            if reachable:
                items.append(PreflightItem(
                    True, False, f"analysis brain ({brain.value})",
                    "ready: " + ", ".join(reachable),
                ))
            else:
                items.append(PreflightItem(
                    False, True, "analysis brain (cloud)",
                    "no free cloud key set. Paste a Groq key, or switch to offline.",
                ))

        # A passphrase, or the vault refuses to write anything sensitive.
        has_pass = bool(os.environ.get(PASSPHRASE_ENV, "").strip())
        items.append(PreflightItem(
            has_pass, True, "passphrase",
            "set" if has_pass else "not set. It encrypts private recordings; there is no recovery.",
        ))
        return items

    def is_ready(self, brain: Brain) -> bool:
        return not any(i.fatal and not i.ok for i in self.preflight(brain))

    # ---- doing the work -------------------------------------------------
    def add_files(self, paths: list[Path | str]) -> list[Path]:
        """Copy the audio a person picked into the inbox. Returns what landed."""
        cfg = self.load_config(Brain.CLOUD)  # brain is irrelevant to where files go
        inbox = cfg.path("inbox")
        inbox.mkdir(parents=True, exist_ok=True)
        landed: list[Path] = []
        for raw in paths:
            src = Path(raw)
            if not src.is_file():
                continue
            dest = inbox / src.name
            if src.resolve() != dest.resolve():
                shutil.copy2(src, dest)
            landed.append(dest)
        return landed

    def process(self, brain: Brain,
                progress: Callable[[str], None] | None = None) -> dict:
        """
        Run everything waiting in the inbox through the real pipeline.

        `progress` is called with human-readable lines a window can show. The
        return is a small summary dict: processed, quarantined, failed, and what
        the run cost. The Pipeline does the actual work; this only narrates it.
        """
        from ..pipeline import Pipeline

        say = progress or (lambda _msg: None)
        cfg = self.load_config(brain)
        pipe = Pipeline(cfg)
        try:
            say(f"Starting ({brain.label})...")
            stats = pipe.run()
            say(
                f"Done: {stats.processed} processed, "
                f"{stats.quarantined} held for review, {stats.failed} failed."
            )
            return {
                "processed": stats.processed,
                "quarantined": stats.quarantined,
                "failed": stats.failed,
                "skipped": getattr(stats, "skipped", 0),
                "cost_usd": round(getattr(stats, "cost_usd", 0.0), 4),
            }
        finally:
            pipe.close()

    def write_digest(self, *, days: int = 3650, include_personal: bool = False,
                     out_name: str = "digest.html") -> Path:
        """
        Render the digest a person came for and return the file to open.

        Defaults to a long window and work profiles only -- the same default the
        CLI keeps, so a personal recording is not folded into a document meant to
        be glanced at over someone's shoulder unless it is asked for. Rendered
        through the builder's own HTML path so the charts come with it.
        """
        from ..db import Database

        cfg = self.load_config(Brain.CLOUD)
        db = Database(cfg.path("database"))
        try:
            html = DigestBuilder(cfg, db).render_html(
                DigestOptions(days=days, include_personal=include_personal),
                title="Your digest",
            )
        finally:
            db.close()
        out = cfg.path("outbox") / out_name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        return out

    # ---- everything the tabs drive --------------------------------------
    #
    # Each of these opens the index, does one read or one narrow write through
    # the same engine code the CLI uses, and returns plain dicts a page can
    # render. None of them cache: the app is a viewer over live state, and a
    # stale list of quarantined recordings is how someone releases the wrong
    # one. The db handle is opened and closed per call -- SQLite is fine with
    # that, and it keeps the server free of connection lifetime bugs.

    def _stores(self):
        """(cfg, db, archive, vault) for one read; caller closes db."""
        from ..archive import Archive
        from ..db import Database
        from ..storage import Vault

        cfg = self.load_config(Brain.CLOUD)
        db = Database(cfg.path("database"))
        vault = Vault(cfg.path("vault"))
        return cfg, db, Archive(cfg, db, vault), vault

    def recent_recordings(self, limit: int = 50) -> list[dict]:
        """The library view: what is in the archive, newest first."""
        cfg, db, _archive, _vault = self._stores()
        try:
            rows = db.query(limit=limit)
        finally:
            db.close()
        personal = {p.id for p in cfg.profiles.values() if p.exclude_from_combined_export}
        out = []
        for row in rows:
            profile = row["governing_profile"] or "unfiled"
            out.append({
                "id": row["id"],
                "name": row["source_name"],
                "when": ((row["recorded_at"] or row["ingested_at"] or "")[:16]).replace("T", " "),
                "minutes": round(float(row["duration_seconds"] or 0) / 60.0, 1),
                "profile": profile,
                "personal": profile in personal,
                "stage": row["stage"],
                "encrypted": bool(row["encrypted"]),
            })
        return out

    def search(self, query: str) -> dict:
        """
        Search what was actually said, with the honesty fields intact.

        `unopened` and `truncated` ride along so the page can say "3 recordings
        could not be opened" instead of quietly presenting a partial search as a
        complete one -- the same contract the CLI keeps.
        """
        _cfg, db, archive, _vault = self._stores()
        try:
            result = archive.search_content(query)
        finally:
            db.close()
        return {
            "matches": [
                {
                    "recording_id": m.recording_id, "source": m.source_name,
                    "when": m.when, "profile": m.profile_id, "personal": m.personal,
                    "stamp": m.stamp, "speaker": m.speaker, "text": m.text,
                }
                for m in result.matches
            ],
            "scanned": result.scanned,
            "total": result.total,
            "unopened": len(result.unopened),
            "quarantined": len(result.quarantined),
            "complete": result.complete,
        }

    def ask(self, question: str, brain: Brain,
            include_personal: bool = False) -> dict:
        """
        Answer a question from the archive, with citations.

        `ask` never raises for a missing model -- it degrades to ranked excerpts
        and says so -- which is exactly the behaviour a window wants. The chosen
        brain narrows the provider chain the same way processing does.
        """
        from ..archive import Archive
        from ..ask import ask as ask_engine
        from ..db import Database

        cfg = self.load_config(brain)
        db = Database(cfg.path("database"))
        try:
            answer = ask_engine(question, cfg, db, Archive(cfg, db),
                                include_personal=include_personal)
        finally:
            db.close()
        return {
            "text": answer.text,
            "confidence": answer.confidence,
            "unanswered": answer.unanswered,
            "degraded": answer.degraded,
            "note": answer.note,
            "cost_usd": round(answer.cost_usd, 4),
            "citations": [
                {"recording_id": c.recording_id, "stamp": c.stamp,
                 "quote": c.quote, "source": c.source_name}
                for c in answer.citations
            ],
            "unopened": len(answer.unopened),
        }

    def followups(self, status: str | None = "open") -> dict:
        """The worklist: commitments still owed, oldest first."""
        from ..followups import FollowUpError, collect

        cfg, db, archive, vault = self._stores()
        try:
            items = collect(cfg, db, archive, status=status, vault=vault)
        except FollowUpError as exc:
            return {"items": [], "error": str(exc)}
        finally:
            db.close()
        return {"items": [
            {
                "id": item.id, "short_id": item.short_id, "text": item.text,
                "profile": item.profile_id, "age_days": item.age_days,
                "due": item.due, "status": item.status,
                "counterparty": item.counterparty, "source": item.source_name,
            }
            for item in items
        ], "error": ""}

    def followup_mark(self, followup_id: str, status: str) -> dict:
        """Mark one follow-up done/dropped/open, through the engine's rules."""
        from ..followups import FollowUpError, collect, set_status

        cfg, db, archive, vault = self._stores()
        try:
            items = collect(cfg, db, archive, vault=vault)
            match = set_status(cfg, vault, followup_id, status, items=items)
        except FollowUpError as exc:
            return {"ok": False, "error": str(exc)}
        finally:
            db.close()
        return {"ok": True, "id": match.id, "status": match.status, "error": ""}

    def quarantine(self) -> list[dict]:
        """Everything held for review, with the reason distilled and classed."""
        from ..cli import _quarantine_entries

        cfg, db, _archive, _vault = self._stores()
        try:
            entries = _quarantine_entries(cfg, db)
        finally:
            db.close()
        return [
            {
                "id": e["id"], "source": e["source"], "when": e["when"],
                "klass": e["klass"], "reason": e["reason"],
                "released": e["released"], "has_media": bool(e["media"]),
            }
            for e in entries
        ]

    def quarantine_release(self, recording_id: str) -> dict:
        """
        Release one held recording back to the inbox -- unless it is a refusal.

        The CLI lets a refusal be released one at a time behind its own typed
        confirmation; a button is not that. Someone who said "don't record
        this" does not get released by a click, so the app refuses outright and
        points at the deliberate path. Everything else goes through the same
        release helper the CLI uses, same audit, same actor.
        """
        from ..cli import _CLASS_REFUSAL, _quarantine_entries, _release_media

        cfg, db, _archive, _vault = self._stores()
        try:
            entry = next((e for e in _quarantine_entries(cfg, db)
                          if e["id"] == recording_id), None)
            if entry is None:
                return {"ok": False, "error": "no such quarantined recording"}
            if entry["klass"] == _CLASS_REFUSAL:
                return {"ok": False, "error": (
                    "a party objected to being recorded. Policy: refusals are "
                    "not released by a click. If consent truly was obtained, "
                    f"use the command line: run.py release {recording_id}"
                )}
            if not entry["media"]:
                return {"ok": False, "error": "the quarantine folder has no media to release"}
            released = _release_media(cfg, db, recording_id,
                                      detail="released after human review (app)")
        finally:
            db.close()
        return {"ok": True, "released": [p.name for p in released],
                "note": "back in the inbox; press Process to run it", "error": ""}

    def quarantine_forget(self, recording_id: str) -> dict:
        """Delete one held recording permanently, through the real forget."""
        cfg, db, archive, _vault = self._stores()
        try:
            payload = db.load(recording_id)
            qdir = cfg.path("quarantine") / recording_id
            if payload is None and not qdir.is_dir():
                return {"ok": False, "error": "no such recording"}
            removed, failures = archive.forget(recording_id)
        finally:
            db.close()
        if failures:
            return {"ok": False, "error": "; ".join(failures[:3])}
        return {"ok": True, "removed": removed, "error": ""}

    def people(self, include_personal: bool = False) -> dict:
        """The roster: everyone the archive has heard, most present first."""
        from ..people import PeopleError, collect_people

        cfg, db, archive, vault = self._stores()
        try:
            roster = collect_people(cfg, db, archive,
                                    include_personal=include_personal, vault=vault)
        except PeopleError as exc:
            return {"people": [], "error": str(exc)}
        finally:
            db.close()
        return {"people": [p.to_dict() for p in roster], "error": ""}

    def insights(self, days: int = 90, include_personal: bool = False) -> dict:
        """Talk-time arithmetic over the window, shaped for charts."""
        from ..insights import InsightsError, metrics_dict, trend

        cfg, db, archive, _vault = self._stores()
        try:
            report = trend(cfg, db, archive, days=days,
                           include_personal=include_personal)
        except InsightsError as exc:
            return {"report": None, "error": str(exc)}
        finally:
            db.close()
        return {"report": metrics_dict(report), "error": ""}

    def brief_html(self, days: int = 7, include_personal: bool = False,
                   brain: Brain = Brain.CLOUD) -> str:
        """
        The week's memo as a self-contained page.

        `build_brief` never raises for a missing model -- with no brain
        reachable the deterministic skeleton renders, labelled as assembled --
        so this is safe to call before anything is configured. The chosen
        brain narrows the provider chain the same way processing does; the
        brief's own locality rules still force local when personal material
        is included.
        """
        from ..archive import Archive
        from ..brief import build_brief
        from ..brief import render as render_brief
        from ..db import Database

        cfg = self.load_config(brain)
        db = Database(cfg.path("database"))
        try:
            archive = Archive(cfg, db)
            brief = build_brief(cfg, db, archive, days=days,
                                include_personal=include_personal,
                                vault=archive.vault)
        finally:
            db.close()
        return render_brief(brief, fmt="html")

    def transcript(self, recording_id: str) -> dict:
        """The stored segments, shaped for the player's synced transcript."""
        from ..media import transcript_lines

        cfg, db, archive, _vault = self._stores()
        try:
            lines = transcript_lines(cfg, db, archive, recording_id)
        finally:
            db.close()
        if lines is None:
            return {"ok": False, "lines": [], "error": (
                "unknown recording, or its words cannot be opened -- "
                "is the passphrase set?")}
        return {"ok": True, "lines": lines, "error": ""}

    def media_stream(self, recording_id: str,
                     start: int | None = None, end: int | None = None) -> dict | None:
        """
        The original audio as (headers-worth-of-facts + a byte iterator).

        None means no original is kept for that id -- an answer, not an error.
        A Range request on a plaintext original is honoured exactly; on an
        encrypted one the full decrypted stream comes back instead, because
        byte-accurate seeking into the vault's sequential cipher stream would
        require exactly the decrypt-to-temp-file staging `media.py` exists to
        never do. Raises ValueError for an unsatisfiable range (the 416 case).
        """
        from ..media import locate_original, read_range, stream_plaintext

        cfg, db, _archive, vault = self._stores()
        try:
            info = locate_original(cfg, db, recording_id)
        finally:
            db.close()
        if info is None:
            return None
        if start is None or info.encrypted:
            return {
                "encrypted": info.encrypted,
                "content_type": info.content_type,
                "iter": stream_plaintext(cfg, info, vault),
                "total": info.size_bytes,
                "partial": False, "start": 0, "stop": None,
            }
        it, total = read_range(info, start, end, cfg, vault)
        stop = (total - 1) if end is None else min(end, total - 1)
        return {
            "encrypted": False,
            "content_type": info.content_type,
            "iter": it, "total": total,
            "partial": True, "start": start, "stop": stop,
        }

    def install_samples(self) -> dict:
        """
        Drop the fictional sample recordings into the inbox.

        Only writes them; the caller starts the ordinary processing run, so
        the samples take exactly the path a real recording takes. Never
        overwrites a file already in the inbox -- somebody's real recording
        does not get replaced because they clicked "load samples".
        """
        from .. import demo

        cfg = self.load_config(Brain.CLOUD)
        written, skipped = demo.install(cfg)
        return {"ok": True, "written": len(written), "skipped": len(skipped),
                "names": [p.name for p in written], "error": ""}

    def backup(self) -> dict:
        """One encrypted file with everything worth keeping. Fail-closed."""
        from ..backup import BackupError, create_backup, default_backup_path
        from ..storage import VaultError

        cfg = self.load_config(Brain.CLOUD)
        out = self.base_dir / "backups" / default_backup_path().name
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            report = create_backup(cfg, self.config_dir, out)
        except (BackupError, VaultError) as exc:
            return {"ok": False, "error": str(exc), "path": ""}
        return {
            "ok": True, "path": str(out), "error": "",
            "counts": dict(report.counts),
            "size_mb": round(out.stat().st_size / 1_048_576, 1) if out.exists() else 0.0,
        }
