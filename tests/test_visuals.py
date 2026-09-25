"""
The charts and visuals across every page, and the two bugs found looking at them.

Every page that summarises now draws: the digest, the brief, the people roster
and each person's page, the follow-up worklist, and insights. They share one
module (viz.py), so they share its rules, and these tests pin those rules at
the source and on each page:

  - a chart redraws numbers its page already prints, as inert SVG -- no
    script, nothing fetched, a <title> on every chart;
  - a mark with visible height is never labelled "0" (the first bug: sub-
    minute days and people rounded down to zero);
  - hue never carries meaning alone -- one color per role or profile,
    held across every chart on a page, named once in a legend.

The second bug was not visual but was found looking at the brief: "People
waiting on you" listed whoever SAID each promise, so it ran backwards both
ways. The brief now asks the People engine, which files direction correctly.
"""

from __future__ import annotations

import re
import threading
import urllib.error
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import pytest

from _fixtures import CLIENT_CALL, FAMILY_DINNER, build_sandbox, drop
from plaud_bridge import viz
from plaud_bridge.archive import Archive
from plaud_bridge.brief import Brief, _blocks, build_brief
from plaud_bridge.brief import render as render_brief
from plaud_bridge.cli import main
from plaud_bridge.db import Database
from plaud_bridge.followups import AGE_BUCKETS, FollowUp, aging_chart
from plaud_bridge.followups import render as render_followups
from plaud_bridge.insights import (
    PER_CONVERSATION_LIMIT,
    InsightsError,
    RecordingMetrics,
    SpeakerMetrics,
    TrendReport,
    WindowAggregate,
    render_recording,
    render_trend,
)
from plaud_bridge.people import Appearance, Person, render_person, render_roster

ROOT = Path(__file__).resolve().parents[1]


def _svgs(page: str) -> list[str]:
    return re.findall(r"<svg viewBox.*?</svg>", page, re.S)


def _assert_inert(page: str) -> None:
    """Rule 1, checked wherever a page draws."""
    assert "<script" not in page.lower()
    assert "http://" not in page and "https://" not in page
    for svg in _svgs(page):
        assert 'role="img"' in svg and "<title>" in svg, "a chart has no accessible reading"


# =========================================================================
# viz: the formatting rules
# =========================================================================
@pytest.mark.parametrize("minutes, label", [
    (0, "0"), (-3, "0"), (0.01, "<0.1"), (0.4, "0.4"), (1.0, "1"),
    (2.46, "2.5"), (9.96, "10"), (42.4, "42"),
])
def test_minutes_are_labelled_so_a_visible_mark_is_never_zero(minutes, label):
    assert viz.fmt_minutes(minutes) == label


@pytest.mark.parametrize("fraction, label", [(0, "0%"), (0.001, "<1%"), (0.58, "58%"), (1, "100%")])
def test_shares_keep_the_same_sliver_honesty(fraction, label):
    assert viz.fmt_share(fraction) == label


def test_repeated_axis_labels_are_numbered_so_columns_cannot_be_confused():
    assert viz.unique_ticks(["09-23", "09-24", "09-23", "09-23"]) == [
        "09-23", "09-24", "09-23 (2)", "09-23 (3)"]


# =========================================================================
# viz: the charts
# =========================================================================
def test_every_chart_is_empty_rather_than_an_axis_around_nothing():
    assert viz.labelled_bars([], "c", "t") == ""
    assert viz.share_bars([], "c", "t") == ""
    assert viz.columns([], [], "c", "t") == ""
    assert viz.block("In Charts", ["", ""]) == ""


def test_a_measured_zero_is_labelled_only_where_a_blank_would_mean_missing():
    buckets = [[(0.0, "pb-c0")], [(40.0, "pb-c0")]]
    shown = viz.columns(buckets, ["a", "b"], "c", "t", cap=lambda v: f"{v:.0f}%", show_zero=True)
    hidden = viz.columns(buckets, ["a", "b"], "c", "t", cap=lambda v: f"{v:.0f}%")
    assert ">0%</text>" in shown
    assert ">0%</text>" not in hidden, "a timeline's empty day must stay blank"


def test_a_count_axis_never_shows_half_a_thing():
    chart = viz.columns([[(1.0, "pb-c0")]], ["x"], "c", "t",
                        cap=lambda v: f"{v:.0f}", axis=lambda v: f"{v:g}", integer=True)
    ticks = re.findall(r'text-anchor="end">([^<]*)<', chart)
    assert ticks == ["1"], ticks


def test_share_bars_split_each_row_in_proportion_and_skip_empty_segments():
    rows = [
        ("call", [(0.6, "pb-c0"), (0.4, "pb-cx"), (0.0, "pb-c1")], "you 60%"),
        ("silent", [(0.0, "pb-c0")], "you 0%"),
    ]
    chart = viz.share_bars(rows, "Your share", "t")
    assert chart.count('class="pb-c0"') == 1 and chart.count('class="pb-cx"') == 1
    assert 'class="pb-c1"' not in chart, "a zero segment drew a nub"
    assert "you 60%" in chart and "you 0%" in chart
    # The first segment is square-ended; only the row's last one is rounded.
    assert '<rect class="pb-c0"' in chart


def test_a_legend_names_each_color_once_and_only_when_there_is_more_than_one():
    one = viz.block("In Charts", ["<figure/>"], [("You", "pb-c0")])
    two = viz.block("In Charts", ["<figure/>"], [("You", "pb-c0"), ("Others", "pb-cx")])
    assert 'class="pb-legend"' not in one
    assert two.count('class="pb-key"') == 2


def test_injection_goes_before_or_after_its_marker_and_falls_back_to_the_body():
    page = "<body><h1>x</h1><table></table><h2>End</h2></body>"
    assert viz.inject(page, "C").index("C") > page.index("</table>")
    ahead = viz.inject(page, "C", before="<h2>End</h2>")
    assert ahead.index("C") < ahead.index("<h2>End") and ahead.index("C") > ahead.index("</table>")
    assert viz.inject(page, "C", before="<h2>Missing</h2>").endswith("C\n</body>")
    assert viz.inject("<p>no body</p>", "C") == "<p>no body</p>C"
    assert viz.inject(page, "") == page


# =========================================================================
# The digest: the "0" bars are gone
# =========================================================================
def test_the_digest_labels_a_sub_minute_day_with_its_real_minutes(tmp_path, monkeypatch):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    drop(cfg, "dinner.txt", FAMILY_DINNER)
    assert main(["--config", str(tmp_path / "config"), "run"]) == 0
    out = tmp_path / "d.html"
    assert main(["--config", str(tmp_path / "config"), "digest", "--format", "html",
                 "--include-personal", "--out", str(out)]) == 0
    page = out.read_text()
    caps = re.findall(r'font-size="11" text-anchor="middle">([^<]*)<', page)
    assert caps and all(c != "0" for c in caps), f"a drawn day was labelled zero: {caps}"
    _assert_inert(page)


# =========================================================================
# The brief: direction, then charts
# =========================================================================
@pytest.fixture
def client_week(tmp_path, monkeypatch):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    drop(cfg, "client.txt", CLIENT_CALL)
    assert main(["--config", str(tmp_path / "config"), "run"]) == 0
    db = Database(cfg.path("database"))
    yield cfg, db, Archive(cfg, db)
    db.close()


def test_the_brief_files_each_promise_by_the_way_it_runs(client_week):
    """
    Mutation-style. In CLIENT_CALL the owner (Sasson) promises quotes by
    Thursday, and Marcus promises to email tonight. Marcus is waiting on the
    owner's promise; the owner is waiting on Marcus's. The old section listed
    speakers, so it put Sasson in "people waiting on you" and Marcus there for
    his own promise -- both backwards. Reverting to that makes this fail.
    """
    cfg, db, archive = client_week
    brief = build_brief(cfg, db, archive, days=7)

    waiting = {(e["who"], e["text"]) for e in brief.waiting_on_you}
    owed = {(e["who"], e["text"]) for e in brief.you_are_waiting_on}
    assert ("Marcus", "I'll have them to you by Thursday.") in waiting
    assert ("Marcus", "I'll email you tonight.") in owed
    assert all(e["who"] != "Sasson" for e in brief.waiting_on_you + brief.you_are_waiting_on), (
        "the owner was listed as waiting on themselves")

    text = brief.sections["people"]
    assert text.startswith("Waiting on you: Marcus")
    assert "You are waiting on: Marcus — I'll email you tonight." in text
    assert "Sasson —" not in text
    assert brief.to_dict()["waiting_on_you"] == brief.waiting_on_you


def test_the_material_a_model_reads_states_each_promises_direction(client_week):
    cfg, db, archive = client_week
    items = [
        {"id": "a", "text": "send quotes", "age_days": 2, "profile_id": "insurance_agent",
         "recording_id": "rec_1", "counterparty": "Sasson"},
        {"id": "b", "text": "email tonight", "age_days": 2, "profile_id": "insurance_agent",
         "recording_id": "rec_1", "counterparty": "Marcus"},
        {"id": "c", "text": "sign the slip", "age_days": 1, "profile_id": "father",
         "recording_id": "rec_2", "counterparty": ""},
    ]
    lines = "\n".join(text for _, text in _blocks(cfg, [], items))
    assert "send quotes (the owner promised this)" in lines
    assert "email tonight (Marcus promised this to the owner)" in lines
    assert "sign the slip (the owner promised this)" in lines
    assert "said by" not in lines


def test_a_brief_that_cannot_work_out_direction_says_so_and_still_renders(client_week, monkeypatch):
    from plaud_bridge import brief as brief_module
    from plaud_bridge.people import PeopleError

    def broken(*_a, **_k):
        raise PeopleError("state file locked")

    monkeypatch.setattr(brief_module, "collect_people", broken)
    cfg, db, archive = client_week
    brief = build_brief(cfg, db, archive, days=7)
    assert brief.followups, "the follow-ups themselves must still be there"
    assert brief.waiting_on_you == [] and brief.you_are_waiting_on == []
    assert "Who is waiting on whom could not be worked out: state file locked" in brief.note
    assert "No open follow-up names anyone waiting on you" in brief.sections["people"]


def test_the_brief_page_charts_its_minutes_and_its_aging_in_one_color_per_profile(client_week):
    cfg, db, archive = client_week
    page = render_brief(build_brief(cfg, db, archive, days=7), fmt="html")
    assert "Where the minutes went" in page and "Open follow-ups by age" in page
    # Production is the first profile: the same color class in the legend,
    # the minutes bar, and the aging stack.
    legend = re.search(r'class="pb-legend">(.*?)</p>', page, re.S)
    assert legend is None, "one profile needs no legend"
    minutes_svg, aging_svg = _svgs(page)[:2]
    assert 'class="pb-c0"' in minutes_svg and 'class="pb-c0"' in aging_svg
    _assert_inert(page)
    assert "<svg" not in render_brief(build_brief(cfg, db, archive, days=7))


def test_an_empty_brief_draws_nothing():
    page = render_brief(Brief(), fmt="html")
    assert "In Charts" not in page


# =========================================================================
# Follow-ups: aging
# =========================================================================
def _item(i: int, age: int, profile: str, status: str = "open") -> FollowUp:
    first = (date.today() - timedelta(days=age)).isoformat()
    return FollowUp(id=f"fu_{i}", text=f"promise {i}", profile_id=profile,
                    recording_id=f"rec_{i}", first_seen=first, last_seen=first, status=status)


def test_the_aging_chart_puts_every_boundary_day_in_the_right_bucket():
    ages = [0, 2, 3, 7, 8, 14, 15, 30, 31, 400]
    chart = aging_chart([(a, "p") for a in ages], {"p": "pb-c0"})
    title = re.search(r"<title>(.*?)</title>", chart).group(1)
    expected = "; ".join(f"{label}, 2" for *_, label in AGE_BUCKETS)
    assert title == f"Column chart. Open follow-ups by age: {expected}"
    assert aging_chart([], {}) == ""
    # A profile the caller gave no color to is neutral, never an invented hue.
    assert 'class="pb-cx"' in aging_chart([(1, "stranger")], {})


def test_the_worklist_page_charts_open_items_by_age_and_by_profile():
    items = [_item(1, 1, "insurance_agent"), _item(2, 10, "insurance_agent"),
             _item(3, 40, "father"), _item(4, 3, "father", status="done")]
    page = render_followups(items, fmt="html")
    assert "Open follow-ups by age" in page and "Open follow-ups per profile" in page
    assert "2 open" in page and "1 open" in page, "the closed item was counted"
    assert page.count('class="pb-key"') == 2
    _assert_inert(page)


def test_a_worklist_with_nothing_open_draws_no_charts():
    page = render_followups([_item(1, 3, "father", status="done")], fmt="html")
    assert "In Charts" not in page


# =========================================================================
# People: the roster and one person's page
# =========================================================================
def _person(label, appearances, *, owner=False, bucket=False) -> Person:
    return Person(label=label, display_name=label, is_owner=owner, is_bucket=bucket,
                  appearances=[Appearance(f"rec_{label}_{i}", when, "call.txt", mins, "insurance_agent")
                               for i, (when, mins) in enumerate(appearances)])


def test_the_roster_charts_minutes_by_role_and_who_has_gone_quiet():
    today = date.today()
    ago = lambda d: (today - timedelta(days=d)).isoformat()  # noqa: E731
    people = [
        _person("Sasson", [(ago(0), 3.0)], owner=True),
        _person("Marcus", [(ago(9), 0.4)]),
        _person("Dana", [(ago(0), 1.0), (ago(1), 1.0)]),
        _person("Old", [("not-a-date", 1.0)]),
        _person("(unidentified speakers)", [(ago(2), 0.3)], bucket=True),
    ]
    page = render_roster(people, fmt="html")
    assert "Minutes heard, per person" in page
    assert "Sasson (you)" in page
    assert "0.4 min · 1 conv" in page, "a sub-minute person rounded to zero"
    assert "| 0.4 |" in render_roster(people), "the roster table's own minutes cell"
    start = page.index("How long since you last heard them")
    quiet = page[start:page.index("</figure>", start)]
    assert quiet.index("Marcus") < quiet.index("Dana"), "longest silence first"
    assert "9 days ago" in quiet and "today" in quiet
    assert "Old" not in quiet and "unidentified" not in quiet and "Sasson" not in quiet
    for label in ("You", "People you talk with", "Unidentified speakers"):
        assert f"</svg>{label}</span>" in page
    _assert_inert(page)
    assert "In Charts" not in render_roster([], fmt="html")


def test_a_roster_of_only_the_owner_has_no_quiet_chart_and_no_legend():
    page = render_roster([_person("Sasson", [(date.today().isoformat(), 1.0)], owner=True)],
                         fmt="html")
    assert "How long since" not in page and 'class="pb-legend"' not in page


def test_yesterday_reads_as_yesterday():
    page = render_roster([_person("Kid", [((date.today() - timedelta(days=1)).isoformat(), 1)])],
                         fmt="html")
    assert ">yesterday</text>" in page


def test_a_persons_page_charts_each_conversation_before_the_list_of_them():
    person = _person("Marcus", [("2026-09-23", 0.5), ("2026-09-19", 0.5), ("2026-09-23", 1.2),
                                ("", 2.0)])
    page = render_person(person, fmt="html")
    chart_at = page.index("Minutes of Marcus, conversation by conversation")
    assert chart_at < page.index("<h2>Every time they were heard</h2>")
    ticks = re.findall(r'y="150" font-size="10" text-anchor="middle">([^<]*)<', page)
    assert ticks == ["09-19", "09-23", "09-23 (2)"], "undated left out, dates in order, repeats numbered"
    assert "1.2 min" in page and "2 min" in page
    assert "In Charts" not in render_person(_person("Ghost", [("", 1.0)]), fmt="html")


# =========================================================================
# Insights
# =========================================================================
def _metrics(when, source, owner_share, wpm=150.0, qrate=0.3, owner=True, profile="insurance_agent"):
    speakers = [SpeakerMetrics(speaker="Sasson", is_owner=owner, share=owner_share, seconds=60 * owner_share,
                               words_per_minute=wpm, question_rate=qrate, segment_count=4),
                SpeakerMetrics(speaker="Marcus", share=1 - owner_share, seconds=60 * (1 - owner_share))]
    return RecordingMetrics(recording_id=f"rec_{source}", source_name=source, when=when,
                            profile_id=profile, speakers=speakers)


def _report(recordings, **kw) -> TrendReport:
    return TrendReport(days=30, owner_label="Sasson", owner_recordings=len(recordings),
                       recordings=recordings, overall=WindowAggregate(recordings=len(recordings)),
                       **kw)


def test_insights_draws_you_conversation_by_conversation():
    report = _report([_metrics("2026-09-23", "b.txt", 0.6, qrate=0.0),
                      _metrics("2026-09-19", "a.txt", 0.47),
                      _metrics("2026-09-23", "c.txt", 0.61, owner=False)])
    page = render_trend(report, fmt="html")
    assert page.index("In Charts") > page.index("</ul>"), "charts belong after the headline numbers"
    assert "Your share of each conversation" in page
    rows = re.findall(r'font-size="13">(\d{4}-\d\d-\d\d · [^<]*)<', page)
    assert rows == ["2026-09-19 · a.txt", "2026-09-23 · b.txt"], (
        "oldest first, and a conversation without your voice is left out, not drawn as zero")
    assert ">0%</text>" in page, "a question-free conversation must say 0%, not look missing"
    assert "Your talk share, by profile" not in page, "one profile needs no by-profile chart"
    # The text carries every number the charts draw, with a dash where you were not heard.
    assert "<h2>Per conversation</h2>" in page
    md = render_trend(report)
    assert "| 2026-09-23 | c.txt | - | - | - |" in md
    _assert_inert(page)


def test_insights_compares_this_month_with_the_last_and_splits_by_profile():
    recs = [_metrics("2026-09-20", "a.txt", 0.5, profile="insurance_agent"),
            _metrics("2026-08-10", "b.txt", 0.7, profile="sales_trainer")]
    report = _report(
        recs,
        current=WindowAggregate(focus="owner", share=0.5, question_rate=0.3),
        prior=WindowAggregate(focus="owner", share=0.7, question_rate=0.1),
        deltas={"talk_share": -0.2, "words_per_minute": 0, "question_rate": 0.2,
                "longest_monologue_seconds": 0},
        by_profile={"insurance_agent": WindowAggregate(share=0.5, recordings=1),
                    "sales_trainer": WindowAggregate(focus="everyone", share=0.7, recordings=1)},
    )
    page = render_trend(report, fmt="html")
    assert "This month against the month before" in page
    assert "You: talk share, the 30 before" in page
    assert "Your talk share, by profile" in page and "sales_trainer (everyone)" in page


def test_insights_without_your_voice_says_why_there_is_nothing_about_you():
    report = _report([_metrics("2026-09-20", "a.txt", 0.5, owner=False)],
                     current=WindowAggregate(focus="everyone", share=0.4),
                     prior=WindowAggregate(focus="everyone", share=0.6),
                     deltas={"talk_share": -0.2, "words_per_minute": 0, "question_rate": 0,
                             "longest_monologue_seconds": 0})
    page = render_trend(report, fmt="html")
    assert "Your voice (Sasson) was not identified" in page
    assert "Everyone: talk share, last 30 days" in page
    assert 'class="pb-legend"' not in page, "no 'You' key when there is no you on the page"
    report.owner_label = ""
    assert "diarization.owner_label is unset" in render_trend(report, fmt="html")


def test_insights_shows_the_most_recent_conversations_and_says_how_many_it_left_out():
    recs = [_metrics(f"2026-09-{d:02d}", f"r{d}.txt", 0.5) for d in range(1, PER_CONVERSATION_LIMIT + 4)]
    md = render_trend(_report(recs))
    assert f"The {PER_CONVERSATION_LIMIT} most recent of {len(recs)}" in md
    assert "r1.txt" not in md and f"r{len(recs)}.txt" in md


def test_an_empty_window_draws_nothing_and_a_bad_format_is_refused():
    assert "In Charts" not in render_trend(_report([]), fmt="html")
    with pytest.raises(InsightsError):
        render_trend(_report([]), fmt="pdf")
    with pytest.raises(InsightsError):
        render_recording(_metrics("2026-09-20", "a.txt", 0.5), fmt="pdf")


def test_one_conversation_shows_who_held_the_floor():
    page = render_recording(_metrics("2026-09-20", "a.txt", 0.6), fmt="html")
    assert "Who held the floor" in page and "Sasson (you)" in page and "60% · 0.6 min" in page
    assert page.count('class="pb-key"') == 2
    assert 'class="pb-legend"' not in render_recording(_metrics("2026-09-20", "a.txt", 0.6, owner=False),
                                               fmt="html")
    assert "In Charts" not in render_recording(RecordingMetrics(source_name="empty"), fmt="html")


def test_the_insights_command_writes_the_page_with_charts(tmp_path, monkeypatch, capsys):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    drop(cfg, "client.txt", CLIENT_CALL)
    c = str(tmp_path / "config")
    assert main(["--config", c, "run"]) == 0
    out = tmp_path / "out" / "insights.html"
    assert main(["--config", c, "insights", "--format", "html", "--out", str(out)]) == 0
    assert "wrote" in capsys.readouterr().out
    assert "Your share of each conversation" in out.read_text()
    db = Database(cfg.path("database"))
    rid = db.query(limit=1)[0]["id"]
    db.close()
    assert main(["--config", c, "insights", "--recording", rid, "--format", "html"]) == 0
    assert "Who held the floor" in capsys.readouterr().out


# =========================================================================
# The app: the same pages, token-guarded
# =========================================================================
@pytest.fixture
def app_server(tmp_path, monkeypatch):
    from _fixtures import StubLLM
    from plaud_bridge.desktop import AppController, Brain
    from plaud_bridge.desktop.server import AppServer

    stub = StubLLM()
    for module in ("plaud_bridge.profiles.router", "plaud_bridge.profiles.extractor"):
        monkeypatch.setattr(f"{module}.complete_json", stub)
    monkeypatch.setenv("PLAUD_BRIDGE_PASSPHRASE", "a-long-enough-desktop-passphrase")
    controller = AppController(base_dir=tmp_path / "home", template_dir=ROOT / "config")
    picked = tmp_path / "client.txt"
    picked.write_text(CLIENT_CALL)
    controller.add_files([picked])
    controller.process(Brain.CLOUD)
    srv = AppServer(controller)
    httpd = srv.make_server("127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}", srv.token, controller
    finally:
        httpd.shutdown()
        httpd.server_close()


def _get(base, path, token=None):
    req = urllib.request.Request(base + path, headers={"X-Token": token} if token else {})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


def test_the_app_serves_every_page_with_its_charts(app_server):
    base, token, _ = app_server
    for path, marker in (
        ("/api/insights/page?days=x", "Your share of each conversation"),
        ("/api/people/page", "Minutes heard, per person"),
        ("/api/people/page?name=Marcus", "Minutes of Marcus"),
        ("/api/followups/page?status=all", "Open follow-ups by age"),
        ("/api/followups/page", "Open follow-ups by age"),
    ):
        status, body = _get(base, path, token)
        assert status == 200, (path, body[:200])
        assert marker in body, path


def test_the_chart_pages_refuse_without_the_token(app_server):
    base, _token, _ = app_server
    for path in ("/api/insights/page", "/api/people/page", "/api/followups/page"):
        status, _ = _get(base, path)
        assert status == 403, path


def test_an_unknown_person_is_a_404_and_a_broken_page_is_a_500_with_a_reason(app_server, monkeypatch):
    base, token, controller = app_server
    status, body = _get(base, "/api/people/page?name=Nobody%20Like%20This", token)
    assert status == 404 and "nobody here is called" in body

    def boom(**_kw):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(controller, "insights_html", boom)
    status, body = _get(base, "/api/insights/page", token)
    assert status == 500 and "disk on fire" in body


def test_the_app_page_offers_the_chart_pages(app_server):
    base, _token, _ = app_server
    status, body = _get(base, "/")
    assert status == 200
    assert body.count("Open with charts") == 3
    assert "openPersonPage(" in body and "Open their page with charts" in body
