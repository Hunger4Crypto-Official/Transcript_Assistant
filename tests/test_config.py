"""Config invariants. These are the guardrails, so they get tested hardest."""

import textwrap
from pathlib import Path

import pytest

from plaud_bridge.config import CODE_ENFORCED_LOCAL_ONLY, Config, ConfigError, Profile
from plaud_bridge.models import Sensitivity

ROOT = Path(__file__).resolve().parents[1]


def test_real_config_loads():
    cfg = Config.load(ROOT / "config")
    assert {"insurance_agent", "sales_trainer", "father", "husband", "unfiled"} <= set(cfg.profiles)


def test_family_profiles_are_locked_local_only():
    cfg = Config.load(ROOT / "config")
    for pid in CODE_ENFORCED_LOCAL_ONLY:
        p = cfg.profile(pid)
        assert p.hard_local_only, f"{pid} lost its local-only lock"
        assert not p.allow_cloud_asr
        assert not p.allow_cloud_llm
        assert p.encrypt_at_rest


def test_config_rejects_cloud_flags_on_locked_profile(tmp_path):
    """The whole point of the lock: a config edit cannot open it up."""
    bad = tmp_path / "father.yaml"
    bad.write_text(textwrap.dedent("""
        id: father
        name: Father
        sensitivity: maximum
        processing:
          allow_cloud_asr: true
          allow_cloud_llm: true
        routing:
          min_confidence: 0.5
        extraction:
          system_prompt: x
          fields:
            - {key: a, label: A, type: string, description: d}
    """))
    with pytest.raises(ConfigError, match="code-enforced local-only"):
        Profile.load(bad)


def test_profile_id_must_match_filename(tmp_path):
    p = tmp_path / "alpha.yaml"
    p.write_text(textwrap.dedent("""
        id: beta
        name: Beta
        sensitivity: low
        processing: {}
        routing: {min_confidence: 0.5}
        extraction:
          system_prompt: x
          fields:
            - {key: a, label: A, type: string, description: d}
    """))
    with pytest.raises(ConfigError, match="must match filename"):
        Profile.load(p)


def test_empty_extraction_fields_rejected(tmp_path):
    p = tmp_path / "alpha.yaml"
    p.write_text(textwrap.dedent("""
        id: alpha
        name: Alpha
        sensitivity: low
        processing: {}
        routing: {min_confidence: 0.5}
        extraction:
          system_prompt: x
          fields: []
    """))
    with pytest.raises(ConfigError, match="cannot be empty"):
        Profile.load(p)


def test_strictest_profile_wins():
    cfg = Config.load(ROOT / "config")
    assert cfg.strictest(["sales_trainer", "husband"]).id == "husband"
    assert cfg.strictest(["sales_trainer", "insurance_agent"]).id == "insurance_agent"
    assert cfg.strictest(["sales_trainer"]).id == "sales_trainer"
    assert cfg.strictest([]).id == "unfiled"


def test_sensitivity_ordering():
    ranks = [Sensitivity.LOW, Sensitivity.MEDIUM, Sensitivity.HIGH, Sensitivity.MAXIMUM]
    assert [s.rank for s in ranks] == sorted(s.rank for s in ranks)


# =========================================================================
# Hand-edited files
#
# Every one of these is a file a person opens in an editor, so every one of
# them will eventually be saved broken. The requirement is not that it works,
# it is that the failure names the file and says what is wrong with it — a
# traceback from inside the YAML parser tells you nothing you can act on.
# =========================================================================
BROKEN = [
    ("does not parse", "this: [is: broken"),
    ("is a list", "- a\n- b\n"),
    ("is a bare string", "just some text\n"),
    ("is a number", "42\n"),
    ("uses tabs", "paths:\n\tinbox: x\n"),
]


@pytest.mark.parametrize("label,body", BROKEN, ids=[b[0] for b in BROKEN])
def test_a_broken_pipeline_yaml_names_itself(tmp_path, label, body):
    import shutil

    shutil.copytree(ROOT / "config", tmp_path / "config")
    (tmp_path / "config" / "pipeline.yaml").write_text(body)

    with pytest.raises(ConfigError) as excinfo:
        Config.load(tmp_path / "config")
    assert "pipeline.yaml" in str(excinfo.value), str(excinfo.value)


@pytest.mark.parametrize("label,body", BROKEN, ids=[b[0] for b in BROKEN])
def test_a_broken_profile_names_itself(tmp_path, label, body):
    import shutil

    shutil.copytree(ROOT / "config", tmp_path / "config")
    (tmp_path / "config" / "profiles" / "father.yaml").write_text(body)

    with pytest.raises(ConfigError) as excinfo:
        Config.load(tmp_path / "config")
    assert "father.yaml" in str(excinfo.value), str(excinfo.value)


@pytest.mark.parametrize("label,body", BROKEN, ids=[b[0] for b in BROKEN])
def test_a_broken_glossary_names_itself(tmp_path, label, body):
    """The glossary is optional, which is not the same as ignorable."""
    import shutil

    shutil.copytree(ROOT / "config", tmp_path / "config")
    (tmp_path / "config" / "glossary.yaml").write_text(body)

    with pytest.raises(ConfigError) as excinfo:
        Config.load(tmp_path / "config")
    assert "glossary.yaml" in str(excinfo.value), str(excinfo.value)


def test_an_empty_config_file_is_not_a_crash(tmp_path):
    """An empty file is a mapping with nothing in it, not a type error."""
    import shutil

    shutil.copytree(ROOT / "config", tmp_path / "config")
    (tmp_path / "config" / "glossary.yaml").write_text("")
    cfg = Config.load(tmp_path / "config")
    assert cfg.glossary.corrections == {}


def test_a_missing_glossary_is_fine(tmp_path):
    import shutil

    shutil.copytree(ROOT / "config", tmp_path / "config")
    (tmp_path / "config" / "glossary.yaml").unlink()
    assert Config.load(tmp_path / "config").glossary.corrections == {}
