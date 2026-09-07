"""
Config edges: every refusal names the file and says what is wrong with it.

A config that refuses to load with a clear message is a feature, so each
message is pinned here rather than just "raises ConfigError". The runtime,
logging, and voice helpers that config feeds are pinned alongside because
their edges are the same shape: a hand-edited value that has to fail loudly
or degrade honestly, never silently.
"""

from __future__ import annotations

import logging
import shutil
import textwrap
from pathlib import Path

import pytest
import yaml

from _fixtures import build_sandbox
from plaud_bridge.config import Config, ConfigError, FieldSpec, Glossary, Profile
from plaud_bridge.logging_setup import RedactingFilter
from plaud_bridge.runtime import resolve_local_model
from plaud_bridge.voice import Voice

ROOT = Path(__file__).resolve().parents[1]


def _copy(tmp_path) -> Path:
    cfg_dir = tmp_path / "config"
    shutil.copytree(ROOT / "config", cfg_dir)
    return cfg_dir


def _rewrite(cfg_dir: Path, mutate) -> None:
    """Load pipeline.yaml, let `mutate` edit the dict in place, write it back."""
    path = cfg_dir / "pipeline.yaml"
    data = yaml.safe_load(path.read_text())
    mutate(data)
    path.write_text(yaml.safe_dump(data))


def _refusal(cfg_dir: Path) -> str:
    with pytest.raises(ConfigError) as excinfo:
        Config.load(cfg_dir)
    return str(excinfo.value)


def _profile_text(**overrides) -> str:
    base = {
        "id": "alpha", "name": "Alpha", "sensitivity": "low", "processing": {},
        "routing": {"min_confidence": 0.5},
        "extraction": {"system_prompt": "x", "fields": [
            {"key": "a", "label": "A", "type": "string", "description": "d"},
        ]},
    }
    base.update(overrides)
    return yaml.safe_dump(base)


# =========================================================================
# Files that cannot be read at all
# =========================================================================
def test_a_pipeline_yaml_that_cannot_be_read_says_so_by_name(tmp_path):
    """A directory where the file should be exists() but cannot be read."""
    cfg_dir = _copy(tmp_path)
    (cfg_dir / "pipeline.yaml").unlink()
    (cfg_dir / "pipeline.yaml").mkdir()
    message = _refusal(cfg_dir)
    assert message.startswith("pipeline.yaml: cannot be read")


def test_a_missing_pipeline_yaml_is_named(tmp_path):
    cfg_dir = _copy(tmp_path)
    (cfg_dir / "pipeline.yaml").unlink()
    assert "missing" in _refusal(cfg_dir) and "pipeline.yaml" in _refusal(cfg_dir)


def test_a_missing_profiles_directory_is_named(tmp_path):
    cfg_dir = _copy(tmp_path)
    shutil.rmtree(cfg_dir / "profiles")
    assert "missing profiles directory" in _refusal(cfg_dir)


def test_a_profiles_directory_holding_only_scaffolding_has_no_profiles(tmp_path):
    """_TEMPLATE.yaml is a thing to copy, not a profile, so it does not count."""
    cfg_dir = _copy(tmp_path)
    for path in (cfg_dir / "profiles").glob("*.yaml"):
        if not path.stem.startswith("_"):
            path.unlink()
    assert _refusal(cfg_dir) == "no profiles found"


def test_a_fallback_profile_without_a_file_is_refused(tmp_path):
    cfg_dir = _copy(tmp_path)
    _rewrite(cfg_dir, lambda d: d["routing"].update({"fallback_profile": "ghost"}))
    assert "routing.fallback_profile 'ghost' has no matching profile file" in _refusal(cfg_dir)


# =========================================================================
# Profile files
# =========================================================================
def test_a_profile_missing_a_required_key_names_the_key(tmp_path):
    path = tmp_path / "alpha.yaml"
    text = yaml.safe_load(_profile_text())
    del text["sensitivity"]
    path.write_text(yaml.safe_dump(text))
    with pytest.raises(ConfigError, match="alpha.yaml: missing required key 'sensitivity'"):
        Profile.load(path)


def test_a_profile_id_that_is_not_an_identifier_is_refused(tmp_path):
    path = tmp_path / "alpha.yaml"
    path.write_text(_profile_text(id="not an id"))
    with pytest.raises(ConfigError, match="id 'not an id' must be a valid identifier"):
        Profile.load(path)


def test_an_unknown_sensitivity_lists_the_allowed_values(tmp_path):
    path = tmp_path / "alpha.yaml"
    path.write_text(_profile_text(sensitivity="extreme"))
    with pytest.raises(ConfigError) as excinfo:
        Profile.load(path)
    message = str(excinfo.value)
    assert message.startswith("alpha.yaml: sensitivity must be one of")
    for allowed in ("low", "medium", "high", "maximum"):
        assert allowed in message


def test_duplicate_extraction_field_keys_are_refused(tmp_path):
    path = tmp_path / "alpha.yaml"
    path.write_text(_profile_text(extraction={"system_prompt": "x", "fields": [
        {"key": "a", "label": "A", "type": "string", "description": "d"},
        {"key": "a", "label": "A again", "type": "string", "description": "d"},
    ]}))
    with pytest.raises(ConfigError, match="alpha.yaml: duplicate extraction field keys"):
        Profile.load(path)


@pytest.mark.parametrize("value", [-0.1, 1.5])
def test_min_confidence_outside_zero_to_one_is_refused(tmp_path, value):
    path = tmp_path / "alpha.yaml"
    path.write_text(_profile_text(routing={"min_confidence": value}))
    with pytest.raises(ConfigError, match="min_confidence must be between 0 and 1"):
        Profile.load(path)


@pytest.mark.parametrize("missing", ["key", "label", "type", "description"])
def test_an_extraction_field_missing_a_part_names_it(missing):
    spec = {"key": "a", "label": "A", "type": "string", "description": "d"}
    del spec[missing]
    with pytest.raises(ConfigError, match=f"profile 'alpha': extraction field missing '{missing}'"):
        FieldSpec.parse(spec, "alpha")


def test_a_broken_profile_inside_the_directory_fails_the_whole_load(tmp_path):
    """Config.load walks the directory, so one bad file stops startup by name."""
    cfg_dir = _copy(tmp_path)
    (cfg_dir / "profiles" / "beta.yaml").write_text(textwrap.dedent("""
        id: beta
        name: Beta
        sensitivity: low
        processing: {}
        routing: {min_confidence: 0.5}
        extraction:
          fields:
            - {key: a, label: A, type: string}
    """))
    assert "profile 'beta': extraction field missing 'description'" in _refusal(cfg_dir)


# =========================================================================
# pipeline.yaml validation -- one message per problem, all reported at once
# =========================================================================
def test_every_validation_problem_is_reported_together(tmp_path):
    """
    One run, every complaint. A person fixing a config should not have to
    reload five times to find five mistakes.
    """
    cfg_dir = _copy(tmp_path)

    def mutate(d):
        d["version"] = 2
        d["audio"]["chunk_seconds"] = 0
        d["asr"]["providers"] = []
        d["llm"]["providers"] = []
        d["digest"]["section_order"] = ["insurance_agent", "ghost"]
        d["cost"] = {"warn_usd_per_run": 20.0, "halt_usd_per_run": 10.0}
        d["compliance"]["redact_patterns"] = {"broken": "("}
        d["compliance"]["on_missing_consent"] = "ignore"

    _rewrite(cfg_dir, mutate)
    message = _refusal(cfg_dir)

    assert message.startswith("configuration problems:")
    assert "pipeline.yaml: version must be 1" in message
    assert "audio.chunk_seconds must be > 0" in message
    assert "audio.chunk_overlap_seconds must be >= 0 and < chunk_seconds" in message
    assert "asr.providers cannot be empty" in message
    assert "at least one enabled non-cloud ASR provider is required" in message
    assert "llm.providers cannot be empty" in message
    assert "digest.section_order references unknown profile 'ghost'" in message
    assert "cost.warn_usd_per_run must be > 0 and <= cost.halt_usd_per_run" in message
    assert "compliance.redact_patterns.broken is not valid regex" in message
    assert "compliance.on_missing_consent must be 'quarantine' or 'flag'" in message
    # Each problem is one bullet: the count is the count.
    assert message.count("\n  - ") == 10


def test_an_overlap_at_least_the_chunk_length_is_refused(tmp_path):
    cfg_dir = _copy(tmp_path)
    _rewrite(cfg_dir, lambda d: d["audio"].update({"chunk_seconds": 60, "chunk_overlap_seconds": 60}))
    assert "audio.chunk_overlap_seconds must be >= 0 and < chunk_seconds" in _refusal(cfg_dir)


def test_an_overlap_wider_than_the_size_capped_window_is_refused(tmp_path):
    """
    The chunker shrinks the window to what max_chunk_mb allows, so an overlap
    that clears chunk_seconds can still swallow the window that actually runs.
    At 1MB and 16kHz mono PCM the window is ~33s; a 40s overlap is wider.
    """
    cfg_dir = _copy(tmp_path)
    _rewrite(cfg_dir, lambda d: d["audio"].update(
        {"chunk_seconds": 600, "chunk_overlap_seconds": 40, "max_chunk_mb": 1}))
    message = _refusal(cfg_dir)
    assert "audio.chunk_overlap_seconds (40s) is not smaller than the effective chunk window (33s)" in message
    assert "raise max_chunk_mb or lower the overlap" in message


def test_a_provider_listed_but_not_configured_is_named(tmp_path):
    cfg_dir = _copy(tmp_path)
    _rewrite(cfg_dir, lambda d: d["asr"].update({"providers": ["local", "azure"]}))
    assert "asr.providers lists 'azure' but asr.azure is not configured" in _refusal(cfg_dir)


def test_a_chain_of_only_cloud_asr_is_refused(tmp_path):
    """Maximum-sensitivity profiles cannot use cloud, so a local provider is mandatory."""
    cfg_dir = _copy(tmp_path)
    _rewrite(cfg_dir, lambda d: d["asr"]["local"].update({"enabled": False}))
    assert "at least one enabled non-cloud ASR provider is required" in _refusal(cfg_dir)


# =========================================================================
# Accessors
# =========================================================================
def test_an_unconfigured_path_is_refused_by_name():
    cfg = Config.load(ROOT / "config")
    with pytest.raises(ConfigError, match="paths.nowhere is not configured"):
        cfg.path("nowhere")


def test_an_unknown_profile_is_refused_with_the_known_ones():
    cfg = Config.load(ROOT / "config")
    with pytest.raises(ConfigError) as excinfo:
        cfg.profile("ghost")
    assert "unknown profile 'ghost'" in str(excinfo.value)
    assert "insurance_agent" in str(excinfo.value)


def test_a_secret_is_read_from_the_environment_and_blank_means_none(monkeypatch):
    cfg = Config.load(ROOT / "config")
    monkeypatch.delenv("PB_TEST_SECRET", raising=False)
    assert cfg.secret("PB_TEST_SECRET") is None
    monkeypatch.setenv("PB_TEST_SECRET", "   ")
    assert cfg.secret("PB_TEST_SECRET") is None, "whitespace is not a secret"
    monkeypatch.setenv("PB_TEST_SECRET", "  hunter2  ")
    assert cfg.secret("PB_TEST_SECRET") == "hunter2"


def test_the_asr_prompt_joins_bias_terms_and_proper_nouns_within_the_budget():
    glossary = Glossary(asr_bias_terms=["IUL", "elimination period"], proper_nouns=["Marcus"])
    assert glossary.asr_prompt() == "IUL, elimination period, Marcus"
    assert glossary.asr_prompt(max_chars=8) == "IUL, eli"
    assert len(Config.load(ROOT / "config").glossary.asr_prompt()) <= 900


# =========================================================================
# runtime: model resolution edges
# =========================================================================
def test_an_empty_model_name_resolves_to_nothing_local(tmp_path, monkeypatch):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    assert resolve_local_model(cfg, "", "asr") == ("", False)


def test_an_absolute_path_that_exists_is_used_as_is(tmp_path, monkeypatch):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    weights = tmp_path / "weights" / "large-v3"
    weights.mkdir(parents=True)
    assert resolve_local_model(cfg, str(weights), "asr") == (str(weights), True)
    # And one that does not exist falls through to the models_dir lookup.
    missing = tmp_path / "weights" / "nope"
    assert resolve_local_model(cfg, str(missing), "asr") == (str(missing), False)


# =========================================================================
# logging: the redacting filter's edges
# =========================================================================
def _record(msg, args=(), **extra):
    record = logging.LogRecord("plaud_bridge.test", logging.INFO, __file__, 1, msg, args, None)
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_an_invalid_redaction_pattern_is_skipped_and_the_others_still_apply():
    """config._validate reports the bad pattern; logging is not the place to crash."""
    filt = RedactingFilter(True, {"broken": "(", "ssn": r"\b\d{3}-\d{2}-\d{4}\b"})
    assert [name for name, _ in filt._content_res] == ["ssn"]
    record = _record("ssn 123-45-6789")
    filt.filter(record)
    assert record.getMessage() == "ssn [redacted-ssn]"


def test_a_record_that_cannot_format_is_passed_through_untouched():
    """A bad %-template must not turn a log line into a second exception."""
    record = _record("%d items", ("not a number",))
    assert RedactingFilter(True).filter(record) is True
    assert record.args == ("not a number",), "the record was rewritten despite being unformattable"


def test_a_record_marked_as_content_is_withheld_entirely():
    record = _record("Marcus said: the mortgage is four hundred thousand", content=True)
    RedactingFilter(True).filter(record)
    assert record.getMessage() == "[content withheld from logs]"
    # With redaction off the marker means nothing and the text passes.
    record = _record("Marcus said: hello", content=True)
    RedactingFilter(False).filter(record)
    assert record.getMessage() == "Marcus said: hello"


def test_stack_info_is_scrubbed_like_the_message():
    record = _record("boom", stack_info="frame: token sk-abcdefghijklmnop")
    RedactingFilter(True).filter(record)
    assert "abcdefghijklmnop" not in record.stack_info
    assert "[redacted-key]" in record.stack_info


# =========================================================================
# voice: the pack listing degrades rather than raising
# =========================================================================
def test_listing_voices_in_a_missing_directory_is_empty(tmp_path):
    assert Voice.available(tmp_path / "no-such-dir") == []


def test_a_broken_voice_pack_is_skipped_from_the_listing(tmp_path):
    (tmp_path / "good.yaml").write_text("id: good\nname: Good\ndescription: Fine.\n")
    (tmp_path / "bad.yaml").write_text("this: [is: broken")
    assert Voice.available(tmp_path) == [("good", "Good", "Fine.")]
