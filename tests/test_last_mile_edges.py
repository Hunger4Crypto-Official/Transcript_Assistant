"""
The last statements that had never run, after everything else was covered.

Each of these is a small branch a person can actually hit. They live together
because they are the tail of the coverage work, and a file named for that
makes the next measurement easy to read: if something in here starts failing,
the odd corner it pins has moved.
"""

from __future__ import annotations

from _fixtures import build_sandbox
from plaud_bridge.cli import main
from plaud_bridge.pipeline import _parse_text_transcript


def test_a_speaker_prefix_with_nothing_after_it_is_dropped_not_kept_as_an_empty_line():
    """
    A transcript exporter can emit "Marcus:" on a line of its own -- a speaker
    change with no words. That is not a segment. Keeping it would put an empty
    utterance in the archive that search, insights, and the player all have to
    step around; the parser drops it and keeps the clock where it was.
    """
    raw = (
        "Sasson: hello there\n"
        "Sasson:\n"
        "Marcus: hi\n"
        "Marcus: yes\n"
    )
    segments = _parse_text_transcript(raw, ".txt")
    assert [s.text for s in segments] == ["hello there", "hi", "yes"]
    assert [s.speaker for s in segments] == ["Sasson", "Marcus", "Marcus"]


def test_new_profile_refuses_plainly_when_the_template_is_gone(tmp_path, monkeypatch, capsys):
    """
    `new-profile` scaffolds from config/profiles/_TEMPLATE.yaml. If someone
    deleted the template, the command must say which file it wanted and exit
    1 -- not write a half-formed profile from nothing.
    """
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    template = tmp_path / "config" / "profiles" / "_TEMPLATE.yaml"
    template.unlink()

    code = main(["--config", str(tmp_path / "config"), "new-profile", "mentor"])

    assert code == 1
    out = capsys.readouterr().out
    assert "template not found" in out and "_TEMPLATE.yaml" in out
    assert not (tmp_path / "config" / "profiles" / "mentor.yaml").exists()
