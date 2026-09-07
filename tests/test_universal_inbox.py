"""
The inbox takes what note-taking tools actually produce.

"Works with your recorder" and "works with everything that records" are
different products. Zoom, Teams, Fireflies, and YouTube export WebVTT; phones
export .m4a; WhatsApp voice notes are .opus; meeting recordings are .mp4 with
the audio inside. A tool that silently ignores those is Plaud-shaped at the
edges no matter what the rest of it can do.

The VTT tests carry the interesting guarantee: a Teams voice tag is the
platform stating who spoke, from its own per-participant audio channels, so it
flows through as a named speaker with no diarization, no enrollment, and no
model involved.
"""

from __future__ import annotations

import pytest

from _fixtures import CLIENT_CALL, build_sandbox, drop
from plaud_bridge.pipeline import Pipeline, _parse_text_transcript


def vtt(*cues: str) -> str:
    return "WEBVTT\n\n" + "\n\n".join(cues) + "\n"


# ---------------------------------------------------------------------------
# WebVTT parsing
# ---------------------------------------------------------------------------
def test_a_teams_voice_tag_is_a_named_speaker(sandbox):
    body = vtt(
        "00:00:01.000 --> 00:00:04.000\n<v Marcus Reed>I think we should raise the limit.</v>",
        "00:00:04.500 --> 00:00:06.000\n<v Dana Ortiz>Agreed, put it in writing.</v>",
    )
    segments = _parse_text_transcript(body, ".vtt")
    assert [(s.speaker, s.text) for s in segments] == [
        ("Marcus Reed", "I think we should raise the limit."),
        ("Dana Ortiz", "Agreed, put it in writing."),
    ]
    assert segments[0].start == pytest.approx(1.0)
    assert segments[1].end == pytest.approx(6.0)


def test_zoom_style_hourless_stamps_and_srt_style_commas_both_parse(sandbox):
    body = vtt(
        "00:04.000 --> 00:07.250\nShort form without hours.",
        "01:00:04,000 --> 01:00:07,000\nComma milliseconds from a lazy exporter.",
    )
    segments = _parse_text_transcript(body, ".vtt")
    assert segments[0].start == pytest.approx(4.0)
    assert segments[0].end == pytest.approx(7.25)
    assert segments[1].start == pytest.approx(3604.0)


def test_headers_notes_styles_and_cue_identifiers_are_not_speech(sandbox):
    body = (
        "WEBVTT - produced by some meeting tool\n\n"
        "NOTE\n00:00:01.000 --> 00:00:02.000\nghost speech inside a note\n\n"
        "STYLE\n::cue { color: red }\n\n"
        "42\n00:00:01.000 --> 00:00:02.000\nNumbered cue survives.\n\n"
        "intro-slide\n00:00:03.000 --> 00:00:04.000\nNamed cue survives.\n"
    )
    segments = _parse_text_transcript(body, ".vtt")
    assert [s.text for s in segments] == ["Numbered cue survives.", "Named cue survives."]


def test_two_voices_in_one_cue_do_not_merge_into_one_mouth(sandbox):
    body = vtt(
        "00:00:01.000 --> 00:00:05.000\n"
        "<v Marcus>Can you send it today?</v> <v Dana>It goes out tonight.</v>",
    )
    segments = _parse_text_transcript(body, ".vtt")
    assert [(s.speaker, s.text) for s in segments] == [
        ("Marcus", "Can you send it today?"),
        ("Dana", "It goes out tonight."),
    ]


def test_markup_is_stripped_but_speech_is_kept(sandbox):
    body = vtt(
        "00:00:01.000 --> 00:00:04.000\n"
        "<v Marcus><i>Really</i> important: <c.highlight>the</c> "
        "<00:00:02.000>deadline is Thursday.</v>",
    )
    segments = _parse_text_transcript(body, ".vtt")
    assert segments[0].text == "Really important: the deadline is Thursday."
    assert segments[0].speaker == "Marcus"


def test_untagged_cues_still_get_the_name_prefix_heuristic(sandbox):
    body = vtt(
        "00:00:01.000 --> 00:00:03.000\nMarcus: I want the bigger policy.",
        "00:00:03.500 --> 00:00:05.000\nMarcus: And the rider we discussed.",
        "00:00:05.500 --> 00:00:07.000\nNote: this line is a note, not a person.",
    )
    segments = _parse_text_transcript(body, ".vtt")
    assert segments[0].speaker == "Marcus"
    assert segments[1].speaker == "Marcus"
    # "Note" is on the stop-list; the line survives with its colon intact.
    assert segments[2].speaker == "SPEAKER"
    assert segments[2].text.startswith("Note:")


def test_an_empty_or_headerless_vtt_yields_no_phantom_speech(sandbox):
    assert _parse_text_transcript("WEBVTT\n", ".vtt") == []
    assert _parse_text_transcript("", ".vtt") == []
    only_header_and_note = "WEBVTT\n\nNOTE nothing here\n"
    assert _parse_text_transcript(only_header_and_note, ".vtt") == []


# ---------------------------------------------------------------------------
# end to end: a Teams export becomes a recording with named speakers
# ---------------------------------------------------------------------------
def as_teams_vtt(script: str) -> str:
    """CLIENT_CALL re-exported the way Teams would have written it."""
    cues = []
    at = 0.0
    for line in script.strip().splitlines():
        name, text = line.split(":", 1)
        cues.append(
            f"{int(at // 60):02d}:{int(at % 60):02d}.000 --> "
            f"{int((at + 4) // 60):02d}:{int((at + 4) % 60):02d}.000\n"
            f"<v {name.strip()}>{text.strip()}</v>"
        )
        at += 5.0
    return vtt(*cues)


def test_a_meeting_export_flows_through_the_whole_pipeline(sandbox):
    cfg, _ = sandbox
    drop(cfg, "weekly-call.vtt", as_teams_vtt(CLIENT_CALL))

    pipe = Pipeline(cfg)
    try:
        stats = pipe.run()
        assert stats.processed == 1

        import json

        payload = json.loads(pipe.db.query()[0]["payload_json"])
        assert payload["compliance"]["governing_profile"] == "insurance_agent"
        # The platform's own attribution survives into storage: these names came
        # from voice tags, not from diarization or enrollment. This profile
        # encrypts at rest, so the index withholds the segments themselves and
        # keeps the speaker roster -- which is exactly what we want to check.
        assert set(payload["transcript"]["speakers"]) == {"Sasson", "Marcus"}
        assert payload["transcript"]["segments"] == []
        assert payload["transcript"]["segments_withheld"] > 0
    finally:
        pipe.close()


def test_consent_is_detected_from_a_vtt_the_same_as_from_text(sandbox):
    """The consent exchange sits in the opening cues; the gate must see it."""
    cfg, _ = sandbox
    drop(cfg, "weekly-call.vtt", as_teams_vtt(CLIENT_CALL))

    pipe = Pipeline(cfg)
    try:
        assert pipe.run().quarantined == 0
    finally:
        pipe.close()


# ---------------------------------------------------------------------------
# every format a note taker hands you is discovered
# ---------------------------------------------------------------------------
def test_everything_a_note_taker_exports_is_discovered(sandbox):
    """
    The extension list is the only gate on the audio side -- ffmpeg normalises
    any container -- so a format missing from discovery is simply ignored, with
    no error anywhere. This is the test that keeps WhatsApp voice notes and
    meeting videos from silently not existing.
    """
    cfg, _ = sandbox
    exports = [
        "plaud.mp3", "otter.wav", "voicememo.m4a", "audiobook.m4b",
        "lossless.flac", "old-recorder.ogg", "telegram.oga", "whatsapp.opus",
        "stream-rip.aac", "dictaphone.wma", "flip-phone.amr", "studio.aiff",
        "browser-recording.webm", "zoom-meeting.mp4", "iphone-screen.mov",
        "typed-notes.txt", "typed-notes.md", "plaud-export.srt", "teams.vtt",
    ]
    for name in exports:
        drop(cfg, name, "x")

    pipe = Pipeline(cfg)
    try:
        found = {p.name for p in pipe.discover()}
        assert found == set(exports)
        assert pipe.unsupported == []
    finally:
        pipe.close()


def test_what_nobody_exports_is_still_refused_with_a_reason(sandbox):
    cfg, _ = sandbox
    drop(cfg, "spreadsheet.xlsx", "x")
    drop(cfg, "archive.zip", "x")

    pipe = Pipeline(cfg)
    try:
        assert pipe.discover() == []
        assert {p.name for p in pipe.unsupported} == {"spreadsheet.xlsx", "archive.zip"}
    finally:
        pipe.close()


def test_video_containers_are_classified_as_audio_not_text(sandbox):
    """An .mp4 must reach ffmpeg, which extracts its audio track."""
    cfg, _ = sandbox
    audio = {e.lower() for e in cfg.get("ingest.audio_extensions")}
    text = {e.lower() for e in cfg.get("ingest.text_extensions")}
    assert {".mp4", ".mov", ".webm"} <= audio
    assert ".vtt" in text
    assert not audio & text, "an extension in both sets would be classified twice"


def test_an_oversized_text_file_is_refused_before_it_is_read(tmp_path, monkeypatch):
    """
    An imported transcript is read whole into memory. A stray multi-gigabyte
    file dragged into the inbox would be an out-of-memory, so the size is checked
    before the read. The whole run must survive it: the bad file fails, and any
    good file beside it still processes.
    """
    cfg, _ = build_sandbox(tmp_path, monkeypatch, overrides={"ingest": {"max_text_bytes": 2000}})
    drop(cfg, "huge-notes.txt", "word " * 1000)      # 5000 bytes, over the 2000 cap
    drop(cfg, "real-call.txt", CLIENT_CALL)

    pipe = Pipeline(cfg)
    try:
        stats = pipe.run()
        assert stats.failed == 1, "the oversized file should have failed, not crashed the run"
        assert stats.processed == 1, "the good file beside it should still have processed"
        rows = {r["source_name"]: r for r in pipe.db.query(stage=None)}
        assert rows["huge-notes.txt"]["stage"] == "failed"
        reasons = [r["detail"] for r in pipe.db.audit_log(rows["huge-notes.txt"]["id"])]
        assert any("max_text_bytes" in str(d) for d in reasons)
    finally:
        pipe.close()
