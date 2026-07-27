"""
Offline mode.

The requirement this exists for: install on a machine that has never been
attached to a network, and have everything work. Same code also runs with Groq
and Anthropic when you do have a connection. One switch decides which.

Offline is not a preference here, it is an assertion. With `runtime.offline:
true` the config refuses to load if any cloud provider is enabled, every model
has to resolve to a file on disk, and the ASR and diarization backends are told
never to reach for a download. A tool that claims to be offline and quietly
fetches a 3GB model on first use has told you something false about where your
recordings went.

Two things genuinely cannot happen on an air-gapped machine, and pretending
otherwise would be the same lie:

  - **Installing.** `pip install` needs an index. Build the wheel bundle on a
    networked machine (`scripts/fetch_models.py --wheels`) and carry it over.
  - **Getting the model weights.** Same script, same trip.

Everything after that runs with no network at all, and `doctor --offline` is
the command that proves it rather than assuming it.
"""

from __future__ import annotations

from pathlib import Path

from .logging_setup import get

log = get("runtime")


def is_offline(cfg) -> bool:
    return bool(cfg.get("runtime.offline", False))


def models_dir(cfg) -> Path:
    """Where local model weights live. Relative paths resolve against the project."""
    raw = str(cfg.get("runtime.models_dir", "./models"))
    path = Path(raw)
    return path if path.is_absolute() else (cfg.root / path).resolve()


def model_path(cfg, *parts: str) -> Path:
    return models_dir(cfg).joinpath(*parts)


def resolve_local_model(cfg, configured: str, subdir: str) -> tuple[str, bool]:
    """
    Turn a configured model name into something loadable, and say whether it is
    on disk.

    Returns `(target, local)`. When a directory for it exists under
    `models_dir`, `target` is that path and `local` is True. Otherwise the
    configured name is passed through untouched, which online means "download
    it" and offline means "this will fail, loudly, with a message naming the
    directory it wanted".
    """
    if not configured:
        return configured, False

    candidate = Path(configured)
    if candidate.is_absolute() and candidate.exists():
        return str(candidate), True

    # A model name like "large-v3" or "pyannote/speaker-diarization-3.1" becomes
    # a directory under models_dir. The slash is flattened so the layout stays
    # shallow enough to copy onto a USB stick without surprises.
    local = model_path(cfg, subdir, configured.replace("/", "__"))
    if local.exists():
        return str(local), True
    return configured, False


def require_local(cfg, configured: str, subdir: str, what: str) -> str:
    """
    Resolve a model, refusing to fall back to a download when offline.

    The error names the exact directory it looked in and the command that fills
    it, because the alternative is a stack trace from inside somebody else's
    HTTP client.
    """
    target, local = resolve_local_model(cfg, configured, subdir)
    if local or not is_offline(cfg):
        return target

    wanted = model_path(cfg, subdir, configured.replace("/", "__"))
    raise OfflineError(
        f"runtime.offline is on and the {what} model '{configured}' is not on "
        f"disk.\n"
        f"  Expected: {wanted}\n"
        f"  Fetch it on a networked machine with:\n"
        f"    python scripts/fetch_models.py --{subdir} {configured}\n"
        f"  then copy {models_dir(cfg)} across.\n"
        f"Refusing to download it, because offline means offline."
    )


class OfflineError(RuntimeError):
    """Raised when offline mode cannot be honoured without reaching the network."""


def cloud_providers_enabled(cfg) -> list[str]:
    """Every enabled provider that would talk to somebody else's server."""
    offenders: list[str] = []

    for name in cfg.get("asr.providers", []) or []:
        block = cfg.get(f"asr.{name}") or {}
        if block.get("enabled") and block.get("is_cloud", True):
            offenders.append(f"asr.{name}")

    for name in cfg.get("llm.providers", []) or []:
        block = cfg.get(f"llm.{name}") or {}
        if block.get("enabled") and block.get("is_cloud", True):
            offenders.append(f"llm.{name}")

    return offenders
