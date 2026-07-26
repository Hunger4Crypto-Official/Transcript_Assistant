"""
Configuration loading and validation.

Fails loudly at startup rather than quietly at 2am mid-run. Every invariant
that matters is checked here, including the ones that protect the family
profiles from being accidentally opened up to cloud processing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .models import Sensitivity

# Profiles whose local-only guarantee is enforced by code, not by config.
# Adding a profile here means no config change can send its content to a
# third-party API. Removing one requires a source edit and a commit.
CODE_ENFORCED_LOCAL_ONLY = frozenset({"father", "husband"})

REQUIRED_PROFILE_KEYS = ("id", "name", "sensitivity", "processing", "routing", "extraction")


class ConfigError(Exception):
    """Raised for any configuration problem. Always actionable."""


@dataclass
class FieldSpec:
    key: str
    label: str
    type: str
    description: str
    sensitive: bool = False
    priority: str = "normal"

    @classmethod
    def parse(cls, d: dict[str, Any], profile_id: str) -> FieldSpec:
        for k in ("key", "label", "type", "description"):
            if not d.get(k):
                raise ConfigError(f"profile '{profile_id}': extraction field missing '{k}'")
        return cls(
            key=d["key"],
            label=d["label"],
            type=d["type"],
            description=d["description"],
            sensitive=bool(d.get("sensitive", False)),
            priority=str(d.get("priority", "normal")),
        )


@dataclass
class Profile:
    id: str
    name: str
    short_name: str
    icon: str
    description: str
    sensitivity: Sensitivity

    allow_cloud_asr: bool
    allow_cloud_llm: bool
    require_consent: bool
    encrypt_at_rest: bool
    redact_before_llm: bool
    hard_local_only: bool
    hard_local_reason: str

    transcript_days: int
    raw_audio_days: int
    audit_log_days: int

    min_confidence: float
    keywords: list[str]
    negative_keywords: list[str]
    llm_hint: str

    system_prompt: str
    fields: list[FieldSpec]

    digest_heading: str
    digest_priority: int
    highlight_fields: list[str]
    suppress_fields: list[str]
    exclude_from_combined_export: bool

    consent_gate_key: str = ""
    consent_gate_value: bool = True
    reaffirm_every_days: int = 0

    @property
    def field_keys(self) -> list[str]:
        return [f.key for f in self.fields]

    def field_by_key(self, key: str) -> FieldSpec | None:
        return next((f for f in self.fields if f.key == key), None)

    @classmethod
    def load(cls, path: Path) -> Profile:
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"{path.name}: invalid YAML -> {exc}") from exc

        for key in REQUIRED_PROFILE_KEYS:
            if key not in raw:
                raise ConfigError(f"{path.name}: missing required key '{key}'")

        pid = str(raw["id"]).strip()
        if not pid.isidentifier():
            raise ConfigError(f"{path.name}: id '{pid}' must be a valid identifier")
        if pid != path.stem:
            raise ConfigError(f"{path.name}: id '{pid}' must match filename '{path.stem}'")

        try:
            sens = Sensitivity(str(raw["sensitivity"]).lower())
        except ValueError as exc:
            raise ConfigError(
                f"{path.name}: sensitivity must be one of "
                f"{[s.value for s in Sensitivity]}"
            ) from exc

        proc = raw.get("processing") or {}
        hard = raw.get("hard_lock") or {}
        hard_local = bool(hard.get("local_processing_only", False))

        # Code-enforced lockdown. This is the safeguard that survives a careless
        # config edit at midnight.
        if pid in CODE_ENFORCED_LOCAL_ONLY:
            hard_local = True
            if proc.get("allow_cloud_asr") or proc.get("allow_cloud_llm"):
                raise ConfigError(
                    f"{path.name}: profile '{pid}' is code-enforced local-only. "
                    "Set allow_cloud_asr and allow_cloud_llm to false. "
                    "If you genuinely intend to change this, edit "
                    "CODE_ENFORCED_LOCAL_ONLY in config.py so the change is "
                    "visible in version control."
                )

        allow_cloud_asr = bool(proc.get("allow_cloud_asr", False)) and not hard_local
        allow_cloud_llm = bool(proc.get("allow_cloud_llm", False)) and not hard_local

        ret = raw.get("retention") or {}
        routing = raw.get("routing") or {}
        extraction = raw.get("extraction") or {}
        digest = raw.get("digest") or {}

        fields = [FieldSpec.parse(f, pid) for f in extraction.get("fields", [])]
        if not fields:
            raise ConfigError(f"{path.name}: extraction.fields cannot be empty")
        if len({f.key for f in fields}) != len(fields):
            raise ConfigError(f"{path.name}: duplicate extraction field keys")

        min_conf = float(routing.get("min_confidence", 0.5))
        if not 0.0 <= min_conf <= 1.0:
            raise ConfigError(f"{path.name}: min_confidence must be between 0 and 1")

        # Consent gate: a named boolean in the profile that must be true.
        gate_key, gate_val, reaffirm = "", True, 0
        for candidate in ("family_consent", "spousal_consent"):
            if candidate in raw:
                block = raw[candidate] or {}
                gate_key = candidate
                truthy = [v for k, v in block.items() if isinstance(v, bool)]
                gate_val = all(truthy) if truthy else True
                reaffirm = int(block.get("reaffirm_every_days", 0))
                break

        return cls(
            id=pid,
            name=str(raw["name"]),
            short_name=str(raw.get("short_name", raw["name"])),
            icon=str(raw.get("icon", "circle")),
            description=str(raw.get("description", "")).strip(),
            sensitivity=sens,
            allow_cloud_asr=allow_cloud_asr,
            allow_cloud_llm=allow_cloud_llm,
            require_consent=bool(proc.get("require_consent", False)),
            encrypt_at_rest=bool(proc.get("encrypt_at_rest", sens.rank >= 2)),
            redact_before_llm=bool(proc.get("redact_before_llm", True)),
            hard_local_only=hard_local,
            hard_local_reason=str(hard.get("reason", "")),
            transcript_days=int(ret.get("transcript_days", 365)),
            raw_audio_days=int(ret.get("raw_audio_days", 90)),
            audit_log_days=int(ret.get("audit_log_days", 730)),
            min_confidence=min_conf,
            keywords=[str(k).lower() for k in routing.get("keywords", [])],
            negative_keywords=[str(k).lower() for k in routing.get("negative_keywords", [])],
            llm_hint=str(routing.get("llm_hint", "")).strip(),
            system_prompt=str(extraction.get("system_prompt", "")).strip(),
            fields=fields,
            digest_heading=str(digest.get("heading", raw["name"])),
            digest_priority=int(digest.get("priority", 5)),
            highlight_fields=[str(f) for f in digest.get("highlight_fields", [])],
            suppress_fields=[str(f) for f in digest.get("suppress_fields", [])],
            exclude_from_combined_export=bool(digest.get("exclude_from_combined_export", False)),
            consent_gate_key=gate_key,
            consent_gate_value=gate_val,
            reaffirm_every_days=reaffirm,
        )


@dataclass
class Glossary:
    asr_bias_terms: list[str] = field(default_factory=list)
    corrections: dict[str, str] = field(default_factory=dict)
    proper_nouns: list[str] = field(default_factory=list)

    def asr_prompt(self, max_chars: int = 900) -> str:
        terms = self.asr_bias_terms + self.proper_nouns
        prompt = ", ".join(terms)
        return prompt[:max_chars]

    @classmethod
    def load(cls, path: Path) -> Glossary:
        if not path.exists():
            return cls()
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls(
            asr_bias_terms=[str(t) for t in raw.get("asr_bias_terms", [])],
            corrections={str(k): str(v) for k, v in (raw.get("corrections") or {}).items()},
            proper_nouns=[str(t) for t in raw.get("proper_nouns", [])],
        )


class Config:
    """Loaded pipeline config plus the profile registry."""

    def __init__(self, root: Path, data: dict[str, Any], profiles: dict[str, Profile], glossary: Glossary):
        self.root = root
        self._d = data
        self.profiles = profiles
        self.glossary = glossary

    # ---- generic access -------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._d
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def path(self, name: str) -> Path:
        raw = self.get(f"paths.{name}")
        if raw is None:
            raise ConfigError(f"paths.{name} is not configured")
        p = Path(raw)
        return p if p.is_absolute() else (self.root / p).resolve()

    def ensure_dirs(self) -> None:
        for name in ("inbox", "work", "outbox", "vault", "quarantine", "logs"):
            self.path(name).mkdir(parents=True, exist_ok=True)
        self.path("database").parent.mkdir(parents=True, exist_ok=True)

    def secret(self, env_name: str) -> str | None:
        val = os.environ.get(env_name, "").strip()
        return val or None

    # ---- profile helpers -------------------------------------------------
    def profile(self, pid: str) -> Profile:
        if pid not in self.profiles:
            raise ConfigError(f"unknown profile '{pid}'. Known: {sorted(self.profiles)}")
        return self.profiles[pid]

    def routable_profiles(self) -> list[Profile]:
        fallback = self.get("routing.fallback_profile", "unfiled")
        return [p for pid, p in self.profiles.items() if pid != fallback]

    def cloud_llm_permitted_by_every_profile(self) -> bool:
        """
        True only when no routable profile objects to a cloud LLM.

        Used for decisions taken before routing, where the content could belong
        to any profile. One profile forbidding cloud is enough to forbid it for
        the whole pre-routing stage.
        """
        return all(
            p.allow_cloud_llm and not p.hard_local_only for p in self.routable_profiles()
        )

    def strictest(self, profile_ids: list[str]) -> Profile:
        candidates = [self.profiles[p] for p in profile_ids if p in self.profiles]
        if not candidates:
            return self.profile(self.get("routing.fallback_profile", "unfiled"))
        return max(candidates, key=lambda p: (p.sensitivity.rank, p.hard_local_only))

    # ---- construction ----------------------------------------------------
    @classmethod
    def load(cls, config_dir: str | Path = "config", root: str | Path | None = None) -> Config:
        cfg_dir = Path(config_dir).resolve()
        if not cfg_dir.is_dir():
            raise ConfigError(f"config directory not found: {cfg_dir}")

        pipeline_file = cfg_dir / "pipeline.yaml"
        if not pipeline_file.exists():
            raise ConfigError(f"missing {pipeline_file}")

        try:
            data = yaml.safe_load(pipeline_file.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"pipeline.yaml: invalid YAML -> {exc}") from exc

        profiles_dir = cfg_dir / "profiles"
        if not profiles_dir.is_dir():
            raise ConfigError(f"missing profiles directory: {profiles_dir}")

        profiles: dict[str, Profile] = {}
        for pf in sorted(profiles_dir.glob("*.yaml")):
            prof = Profile.load(pf)
            profiles[prof.id] = prof
        if not profiles:
            raise ConfigError("no profiles found")

        fallback = str((data.get("routing") or {}).get("fallback_profile", "unfiled"))
        if fallback not in profiles:
            raise ConfigError(
                f"routing.fallback_profile '{fallback}' has no matching profile file"
            )

        glossary = Glossary.load(cfg_dir / "glossary.yaml")
        cfg = cls(Path(root).resolve() if root else cfg_dir.parent, data, profiles, glossary)
        cfg._validate()
        return cfg

    def _validate(self) -> None:
        problems: list[str] = []

        if int(self.get("version", 0)) != 1:
            problems.append("pipeline.yaml: version must be 1")

        chunk = float(self.get("audio.chunk_seconds", 0))
        overlap = float(self.get("audio.chunk_overlap_seconds", 0))
        if chunk <= 0:
            problems.append("audio.chunk_seconds must be > 0")
        if overlap < 0 or overlap >= chunk:
            problems.append("audio.chunk_overlap_seconds must be >= 0 and < chunk_seconds")

        # The chunker does not use chunk_seconds directly: it shrinks the window
        # to whatever max_chunk_mb allows. Validating against chunk_seconds alone
        # let a config pass startup and then fail mid-run with an error naming a
        # key this check had just approved.
        rate = float(self.get("audio.target_sample_rate", 16000))
        channels = float(self.get("audio.target_channels", 1))
        max_mb = float(self.get("audio.max_chunk_mb", 20))
        bps = rate * channels * 2
        if chunk > 0 and bps > 0:
            window = max(30.0, min(chunk, (max_mb * 1024 * 1024) / bps))
            if overlap >= window:
                problems.append(
                    f"audio.chunk_overlap_seconds ({overlap:.0f}s) is not smaller than the "
                    f"effective chunk window ({window:.0f}s). The window is capped by "
                    f"audio.max_chunk_mb ({max_mb:.0f}MB at {rate:.0f}Hz), so raise "
                    "max_chunk_mb or lower the overlap."
                )

        asr_chain = self.get("asr.providers", []) or []
        if not asr_chain:
            problems.append("asr.providers cannot be empty")
        for name in asr_chain:
            if self.get(f"asr.{name}") is None:
                problems.append(f"asr.providers lists '{name}' but asr.{name} is not configured")
        if not any(
            self.get(f"asr.{n}.enabled") and not self.get(f"asr.{n}.is_cloud", True)
            for n in asr_chain
        ):
            problems.append(
                "at least one enabled non-cloud ASR provider is required, because "
                "maximum-sensitivity profiles cannot use cloud providers"
            )

        llm_chain = self.get("llm.providers", []) or []
        if not llm_chain:
            problems.append("llm.providers cannot be empty")

        order = self.get("digest.section_order", []) or []
        for pid in order:
            if pid not in self.profiles:
                problems.append(f"digest.section_order references unknown profile '{pid}'")

        halt = float(self.get("cost.halt_usd_per_run", 0))
        warn = float(self.get("cost.warn_usd_per_run", 0))
        if halt <= 0 or warn <= 0 or warn > halt:
            problems.append("cost.warn_usd_per_run must be > 0 and <= cost.halt_usd_per_run")

        for pattern_name, pattern in (self.get("compliance.redact_patterns") or {}).items():
            import re

            try:
                re.compile(pattern)
            except re.error as exc:
                problems.append(f"compliance.redact_patterns.{pattern_name} is not valid regex: {exc}")

        on_missing = self.get("compliance.on_missing_consent", "quarantine")
        if on_missing not in ("quarantine", "flag"):
            problems.append("compliance.on_missing_consent must be 'quarantine' or 'flag'")

        if problems:
            raise ConfigError("configuration problems:\n  - " + "\n  - ".join(problems))
