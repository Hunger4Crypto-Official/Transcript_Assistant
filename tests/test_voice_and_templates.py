"""
Voice packs, profile scaffolding, and HTML output.

The point of the voice layer is that changing how the digest reads is a config
edit. The point of these tests is that it stays that way, and that a broken
voice file degrades the wording rather than losing you the digest.
"""

import pytest

from _fixtures import CLIENT_CALL, build_sandbox, drop
from plaud_bridge.cli import build_parser
from plaud_bridge.config import Config
from plaud_bridge.digest import DigestBuilder, DigestOptions, to_html
from plaud_bridge.pipeline import Pipeline
from plaud_bridge.voice import DEFAULTS, Voice


def _digest(tmp_path, monkeypatch, preset="plain", overrides=None, **opts):
    # Its own directory per call, so one test can render the same recording
    # through two voices and compare them.
    root = tmp_path / preset
    root.mkdir(parents=True, exist_ok=True)
    cfg, _ = build_sandbox(
        root, monkeypatch,
        overrides={"voice": {"preset": preset, "overrides": overrides or {}}},
    )
    drop(cfg, "client-marcus.txt", CLIENT_CALL)
    pipe = Pipeline(cfg)
    try:
        pipe.run()
        return DigestBuilder(cfg, pipe.db).render_markdown(DigestOptions(days=30, **opts))
    finally:
        pipe.close()


# =========================================================================
# The renderer
# =========================================================================
def test_a_missing_placeholder_renders_empty_rather_than_raising():
    voice = Voice({"digest": {"title": "{nonexistent} Digest"}})
    assert voice.text("digest.title").strip() == "Digest"


def test_a_malformed_template_returns_itself_instead_of_exploding():
    voice = Voice({"digest": {"title": "Digest {unclosed"}})
    assert "unclosed" in voice.text("digest.title")


def test_a_partial_pack_inherits_everything_it_does_not_mention():
    voice = Voice({"digest": {"title": "Weekly"}})
    assert voice.text("digest.title") == "Weekly"
    # untouched keys still come from the defaults
    assert voice.text("digest.glance.heading") == DEFAULTS["digest"]["glance"]["heading"]


def test_an_unknown_preset_falls_back_instead_of_failing(tmp_path):
    voice = Voice.load(tmp_path, preset="does-not-exist")
    assert voice.text("digest.title") == "Digest"


def test_a_corrupt_pack_falls_back_instead_of_failing(tmp_path):
    (tmp_path / "broken.yaml").write_text("digest: [this is: not: valid", encoding="utf-8")
    voice = Voice.load(tmp_path, preset="broken")
    assert voice.text("digest.title") == "Digest"


# =========================================================================
# The packs that ship
# =========================================================================
@pytest.mark.parametrize("preset", ["plain", "brief", "warm"])
def test_every_shipped_pack_renders_a_complete_digest(tmp_path, monkeypatch, preset):
    body = _digest(tmp_path, monkeypatch, preset=preset)
    assert body.startswith("# ")
    # The structure is code, not config, so it is identical across voices.
    assert body.count("## ") >= 3
    assert "Send two quote options by Thursday" in body
    assert "{" not in body, "an unrendered placeholder reached the digest"


def test_the_voice_changes_the_words_and_not_the_structure(tmp_path, monkeypatch):
    plain = _digest(tmp_path, monkeypatch, preset="plain")
    warm = _digest(tmp_path, monkeypatch, preset="warm")

    assert plain != warm
    assert "Needs You" in plain
    assert "Before Anything Else" in warm
    # Same sections, same order, same number of headings.
    assert plain.count("## ") == warm.count("## ")
    assert plain.count("### ") == warm.count("### ")


def test_overrides_beat_the_pack_without_copying_it(tmp_path, monkeypatch):
    body = _digest(
        tmp_path, monkeypatch, preset="plain",
        overrides={"digest": {"needs_you": {"heading": "Deal With This"}}},
    )
    assert "## Deal With This" in body
    assert "Needs You" not in body
    # Everything not overridden still comes from the pack.
    assert "At a Glance" in body


def test_voice_cannot_defeat_the_suppression_rules(tmp_path, monkeypatch):
    """
    Wording is configurable. What renders is not. A voice pack that tried to
    print a suppressed field would be a compliance hole, so the structure stays
    in code and config only supplies strings.
    """
    body = _digest(
        tmp_path, monkeypatch, preset="plain",
        overrides={"digest": {"entry": {"withheld": "{label}: {count}"}}},
    )
    # insurance_agent suppresses financial_disclosures
    assert "four hundred thousand" not in body
    assert "Financial Disclosures: 1" in body


def test_a_profile_supplies_its_own_section_intro(tmp_path, monkeypatch):
    body = _digest(tmp_path, monkeypatch, preset="plain")
    assert "statements_needing_review" in body, (
        "the insurance_agent digest.intro did not render"
    )


# =========================================================================
# Profile scaffolding
# =========================================================================
def test_the_template_is_not_loaded_as_a_profile():
    cfg = Config.load("config")
    assert "_TEMPLATE" not in cfg.profiles
    assert "TEMPLATE" not in cfg.profiles


def test_new_profile_scaffolds_a_loadable_profile(tmp_path, monkeypatch, capsys):
    import shutil

    from plaud_bridge.cli import cmd_new_profile

    shutil.copytree("config", tmp_path / "config")
    args = build_parser().parse_args([
        "--config", str(tmp_path / "config"),
        "new-profile", "mentor", "--name", "Mentor", "--heading", "Mentorship",
    ])
    assert cmd_new_profile(args) == 0

    written = (tmp_path / "config" / "profiles" / "mentor.yaml").read_text()
    assert "id: mentor" in written
    assert 'name: "Mentor"' in written
    assert 'heading: "Mentorship"' in written

    cfg = Config.load(tmp_path / "config")
    assert "mentor" in cfg.profiles
    assert cfg.profiles["mentor"].digest_heading == "Mentorship"


def test_new_profile_name_cannot_inject_yaml(tmp_path, capsys):
    """
    The name lands inside a quoted YAML scalar. A value with a quote and a
    newline would break out and inject structure, or -- more likely for a value
    the user typed -- just write a file that no longer parses. The scaffold must
    stay loadable and must not grow a key nobody asked for.
    """
    import shutil

    from plaud_bridge.cli import cmd_new_profile

    shutil.copytree("config", tmp_path / "config")
    evil = 'Mentor"\nhard_local_only: false\nallow_cloud_llm: true\nx: "'
    args = build_parser().parse_args([
        "--config", str(tmp_path / "config"), "new-profile", "mentor", "--name", evil,
    ])
    assert cmd_new_profile(args) == 0

    # It parses, and the injected keys did not become real settings.
    cfg = Config.load(tmp_path / "config")
    assert "mentor" in cfg.profiles
    assert cfg.profiles["mentor"].name == evil, "the name did not round-trip intact"


def test_new_profile_refuses_to_overwrite(tmp_path, capsys):
    import shutil

    from plaud_bridge.cli import cmd_new_profile

    shutil.copytree("config", tmp_path / "config")
    args = build_parser().parse_args([
        "--config", str(tmp_path / "config"), "new-profile", "father",
    ])
    assert cmd_new_profile(args) == 1
    assert "already exists" in capsys.readouterr().out


@pytest.mark.parametrize("bad", ["_hidden", "not-an-identifier", "9lives"])
def test_new_profile_rejects_an_unusable_id(tmp_path, bad, capsys):
    import shutil

    from plaud_bridge.cli import cmd_new_profile

    shutil.copytree("config", tmp_path / "config")
    args = build_parser().parse_args([
        "--config", str(tmp_path / "config"), "new-profile", bad,
    ])
    assert cmd_new_profile(args) == 1


# =========================================================================
# HTML
# =========================================================================
def test_html_is_self_contained():
    page = to_html("# Digest\n\nSomething happened.\n")
    assert page.startswith("<!doctype html>")
    for external in ("http://", "https://", "<script", "@import"):
        assert external not in page, f"the page reaches out via {external}"


def test_html_renders_the_constructs_the_digest_emits():
    page = to_html(
        "# Title\n\n## Needs You\n\n- **Production** :: do the thing  \n  <sub>a.txt</sub>\n\n"
        "## At a Glance\n\n| Section | Recordings |\n|---|---:|\n| Production | 1 |\n\n"
        "### a.txt\n\n> Analysis error: nope\n\n`meta`\n\n---\n\n_nothing_\n"
    )
    assert "<h1>Title</h1>" in page
    assert "<h2>Needs You</h2>" in page
    assert "<h3>a.txt</h3>" in page
    assert "<strong>Production</strong>" in page
    assert "<table>" in page and "<th>Section</th>" in page and "<td>1</td>" in page
    assert "<blockquote>" in page
    assert "<code>meta</code>" in page
    assert "<hr>" in page
    assert "<em>nothing</em>" in page
    assert "<sub>a.txt</sub>" in page


def test_html_escapes_transcript_content():
    """A quote in a digest is untrusted text, not markup."""
    page = to_html("# D\n\n- He said <script>alert(1)</script> on the call\n")
    assert "<script>" not in page
    assert "&lt;script&gt;" in page


def test_html_survives_a_real_digest(tmp_path, monkeypatch):
    page = to_html(_digest(tmp_path, monkeypatch, preset="warm"))
    assert page.count("<html") == 1
    assert page.rstrip().endswith("</html>")
    assert page.count("<table>") == page.count("</table>")
    assert page.count("<ul>") == page.count("</ul>")
    assert page.count("<blockquote>") == page.count("</blockquote>")
