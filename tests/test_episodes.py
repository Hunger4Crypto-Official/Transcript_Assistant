"""
Cutting a day into episodes.

The thing being protected here: wear the device from 8am to 6pm and the old
pipeline classified the whole file from a 14,000 character sample and produced
one summary of everything. These tests pin the behaviour that turns a day into a
rundown per profile.
"""

import pytest

from _fixtures import build_sandbox
from plaud_bridge.episodes import Episode, segment_episodes, transcript_for
from plaud_bridge.models import RouteMatch, Segment, Transcript

CLIENT = [
    ("Sasson", "Marcus lets talk about your term policy and the disability elimination period"),
    ("Marcus", "my income is the asset here not the mortgage"),
    ("Sasson", "own occupation definition matters on that disability rider"),
    ("Marcus", "what does the premium run for that coverage"),
    ("Sasson", "I will quote the policy and send underwriting options"),
    ("Marcus", "email the beneficiary paperwork too"),
]
COACHING = [
    ("Sasson", "lets debrief that roleplay your discovery questions were closed"),
    ("Coach", "you rushed the objection handling framework"),
    ("Sasson", "next time I pause and use the rapport script"),
    ("Coach", "closing language needs practice that is the coaching point"),
    ("Sasson", "prospecting activity and dials are fine"),
    ("Coach", "training on mindset next week"),
]
BEDTIME = [
    ("Sasson", "how was practice today buddy"),
    ("Kid", "coach said I am starting saturday sign my permission slip"),
    ("Sasson", "I sign it tonight remind me at bedtime"),
    ("Kid", "can we get pizza after the game"),
    ("Sasson", "pizza after the game deal homework first"),
    ("Kid", "the field trip is next week"),
]


def _block(start, lines, repeats=3, each=6.0):
    segments, cursor = [], start
    for _ in range(repeats):
        for speaker, text in lines:
            segments.append(Segment(cursor, cursor + each, text, speaker))
            cursor += each
    return segments, cursor


def _day(*parts, gap=300.0):
    """Build a transcript from blocks separated by silence."""
    segments, cursor = [], 0.0
    for index, lines in enumerate(parts):
        if index:
            cursor += gap
        block, cursor = _block(cursor, lines)
        segments.extend(block)
    return Transcript(segments=segments, duration_seconds=segments[-1].end)


class _Cfg:
    """Just the episode settings, so these tests do not need a whole sandbox."""

    def __init__(self, **overrides):
        self.values = {
            "enabled": True, "skip_below_seconds": 900, "silence_gap_seconds": 45,
            "speaker_change_above": 0.6, "topic_overlap_below": 0.08,
            "comparison_window_segments": 12, "min_seconds": 120, "max_seconds": 1800,
        }
        self.values.update(overrides)

    def get(self, dotted, default=None):
        return self.values.get(dotted.split(".")[-1], default)


# =========================================================================
# Cutting
# =========================================================================
def test_a_day_becomes_one_episode_per_conversation():
    episodes = segment_episodes(_day(CLIENT, COACHING, BEDTIME), _Cfg())
    assert len(episodes) == 3
    assert [e.speakers for e in episodes] == [
        ["Sasson", "Marcus"], ["Sasson", "Coach"], ["Sasson", "Kid"],
    ]


def test_a_short_recording_is_left_alone():
    """A twenty minute client call is one conversation. Cutting it costs and gains nothing."""
    episodes = segment_episodes(_day(CLIENT), _Cfg())
    assert len(episodes) == 1
    assert episodes[0].reason == "whole recording"


def test_a_silence_beats_the_minimum_length_rule():
    """
    A two minute call with five minutes of driving either side is its own
    conversation however short it is. Merging it into the neighbour would put
    two unrelated conversations in one profile's rundown.
    """
    day = _day(CLIENT, COACHING, BEDTIME)
    assert all(e.duration < 120 for e in segment_episodes(day, _Cfg(min_seconds=600)))
    assert len(segment_episodes(day, _Cfg(min_seconds=600))) == 3


def test_a_weak_fragment_is_folded_into_its_neighbour():
    """A passing remark inside one conversation is not a new episode."""
    segments, cursor = _block(0.0, CLIENT, repeats=6)
    segments.append(Segment(cursor, cursor + 4.0, "anyway", "Sasson"))
    block, _ = _block(cursor + 4.0, COACHING, repeats=6)
    segments.extend(block)

    episodes = segment_episodes(
        Transcript(segments=segments, duration_seconds=segments[-1].end), _Cfg()
    )
    # No silence anywhere, so nothing is strong; the stray line cannot become an
    # episode of its own.
    assert all(e.duration >= 120 or e.strong for e in episodes)
    assert "anyway" in " ".join(s.text for e in episodes for s in e.segments)


def test_an_episode_is_split_when_it_runs_too_long():
    long_day = _day(CLIENT * 40)
    episodes = segment_episodes(long_day, _Cfg(max_seconds=300))
    assert len(episodes) > 1
    assert all(e.duration <= 360 for e in episodes)
    assert any("ran past" in e.reason for e in episodes)


def test_disabling_it_returns_the_whole_recording():
    episodes = segment_episodes(_day(CLIENT, COACHING, BEDTIME), _Cfg(enabled=False))
    assert len(episodes) == 1


def test_nothing_is_lost_in_the_cutting():
    """Every segment lands in exactly one episode."""
    day = _day(CLIENT, COACHING, BEDTIME)
    episodes = segment_episodes(day, _Cfg())
    kept = [s for e in episodes for s in e.segments]
    assert len(kept) == len(day.segments)
    assert [s.text for s in kept] == [s.text for s in day.segments]


def test_an_empty_transcript_produces_no_episodes():
    assert segment_episodes(Transcript(segments=[]), _Cfg()) == []


def test_every_episode_records_why_it_was_cut():
    for episode in segment_episodes(_day(CLIENT, COACHING, BEDTIME), _Cfg()):
        assert episode.reason, "a boundary with no explanation cannot be debugged"


# =========================================================================
# Per-profile rundowns
# =========================================================================
def test_each_profile_only_sees_its_own_episodes():
    """
    The point of the whole exercise. The Insurance Agent rundown must not
    contain bedtime, and the Father rundown must not contain the client call.
    """
    episodes = segment_episodes(_day(CLIENT, COACHING, BEDTIME), _Cfg())
    for episode, pid in zip(episodes, ["insurance_agent", "sales_trainer", "father"],
                            strict=True):
        episode.routes = [RouteMatch(profile_id=pid, confidence=0.9)]

    work = transcript_for("insurance_agent", episodes)
    family = transcript_for("father", episodes)

    assert "elimination period" in work.text
    assert "permission slip" not in work.text
    assert "permission slip" in family.text
    assert "elimination period" not in family.text


def test_a_profile_matching_several_episodes_gets_all_of_them():
    episodes = segment_episodes(_day(CLIENT, COACHING, CLIENT), _Cfg())
    for episode in episodes:
        episode.routes = [RouteMatch(profile_id="insurance_agent", confidence=0.9)] \
            if "policy" in episode.segments[0].text else []

    combined = transcript_for("insurance_agent", episodes)
    assert combined.text.count("term policy") >= 2
    assert "roleplay" not in combined.text


def test_a_profile_with_no_episodes_gets_nothing():
    episodes = segment_episodes(_day(CLIENT, COACHING), _Cfg())
    assert transcript_for("husband", episodes).segments == []


# =========================================================================
# End to end through the pipeline
# =========================================================================
def test_a_long_recording_produces_a_rundown_per_profile(tmp_path, monkeypatch):
    """
    The headline behaviour, through the real pipeline: one file in, several
    profiles out, each analysed from its own part of the day.
    """
    from _fixtures import StubLLM, drop
    from plaud_bridge.pipeline import Pipeline

    class DayStub(StubLLM):
        """Routes each episode by what is actually in it."""

        def __call__(self, cfg, system, user, local_only=False, max_tokens=None):
            self.calls.append({"local_only": local_only})
            if '"scores"' in user:
                lowered = user.lower()
                if "permission slip" in lowered:
                    pid = "father"
                elif "roleplay" in lowered or "coaching point" in lowered:
                    pid = "sales_trainer"
                else:
                    pid = "insurance_agent"
                return {"scores": [{"profile_id": pid, "score": 0.95, "evidence": ["x"]}]}, \
                    self._response()
            return {"requires_human_attention": False, "next_action": "none"}, self._response()

    cfg, stub = build_sandbox(tmp_path, monkeypatch, stub=DayStub())

    # The client block needs a consent exchange or the gate quarantines the
    # whole day before any of this is reached -- which is correct, and exactly
    # what the compliance tests cover.
    consented = [
        ("Sasson", "Before we start Marcus, I record these calls for my notes. Is that okay?"),
        ("Marcus", "Yeah that's fine, no problem."),
        *CLIENT,
    ]

    lines = []
    cursor = 0.0
    for block in (consented, COACHING, BEDTIME):
        for _ in range(3):
            for speaker, text in block:
                lines.append(f"[{int(cursor // 60):02d}:{int(cursor % 60):02d}] {speaker}: {text}")
                cursor += 6.0
        cursor += 300.0
    drop(cfg, "whole-day.txt", "\n".join(lines) + "\n")

    pipe = Pipeline(cfg)
    try:
        assert pipe.run().processed == 1
        import json

        payload = json.loads(pipe.db.query()[0]["payload_json"])
    finally:
        pipe.close()

    assert len(payload["episodes"]) >= 3, "the day was not cut up"
    analysed = {a["profile_id"] for a in payload["analyses"]}
    assert {"insurance_agent", "sales_trainer", "father"} <= analysed, (
        f"one file should have produced a rundown per profile, got {analysed}"
    )


def test_episode_serialisation_survives_the_index(tmp_path, monkeypatch):
    episode = Episode(index=0, segments=[Segment(0.0, 5.0, "hello", "Sasson")],
                      reason="120s silence", strong=True)
    as_dict = episode.to_dict()
    assert as_dict["reason"] == "120s silence"
    assert as_dict["speakers"] == ["Sasson"]
    assert as_dict["segment_count"] == 1
    # No transcript text: the index must not gain a second copy of the words.
    assert "hello" not in str(as_dict)


@pytest.mark.parametrize("gap", [46.0, 120.0, 600.0])
def test_any_gap_past_the_threshold_cuts(gap):
    # Two short blocks, so the "short recordings are one conversation" rule has
    # to be lowered out of the way to test the gap rule on its own.
    episodes = segment_episodes(_day(CLIENT, COACHING, gap=gap),
                                _Cfg(skip_below_seconds=60))
    assert len(episodes) == 2
    assert "silence" in episodes[1].reason
