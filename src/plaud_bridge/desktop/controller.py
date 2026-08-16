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

import os
import shutil
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

from ..config import Config
from ..digest import DigestBuilder, DigestOptions
from ..digest.html import to_html


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
        cfg._d.setdefault("llm", {})["providers"] = list(_BRAIN_PROVIDERS[brain])
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
        elif brain is Brain.OFFLINE:
            items.append(PreflightItem(
                False, True, "analysis brain (offline)",
                "no local model reachable. Install Ollama and pull a model, or "
                "switch to the free cloud key.",
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
        be glanced at over someone's shoulder unless it is asked for.
        """
        from ..db import Database

        cfg = self.load_config(Brain.CLOUD)
        db = Database(cfg.path("database"))
        try:
            markdown = DigestBuilder(cfg, db).render_markdown(
                DigestOptions(days=days, include_personal=include_personal)
            )
        finally:
            db.close()
        html = to_html(markdown, title="Your digest")
        out = cfg.path("outbox") / out_name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        return out
