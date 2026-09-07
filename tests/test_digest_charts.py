"""
Charts in the HTML digest.

The charts are an HTML-layer addition and nothing more: the markdown stays
chart-free, the page stays self-contained, every plotted number matches what
the index holds, and a personal profile stays out of the pictures exactly as
it stays out of the text. These tests pin each of those claims, plus the
degenerate cases — an empty archive and a single recording — that must render
a sensible page rather than a broken SVG.
"""

from __future__ import annotations

from _fixtures import CLIENT_CALL, FAMILY_DINNER, StubLLM, build_sandbox, drop
from plaud_bridge.cli import main
from plaud_bridge.db import Database
from plaud_bridge.digest import DigestBuilder, DigestOptions
from plaud_bridge.digest.charts import charts_html, inject_charts
from plaud_bridge.pipeline import Pipeline


def _processed(tmp_path, monkeypatch, cost=0.0, family=False):
    """A sandbox with the client call (and optionally the family dinner) run."""
    cfg, _ = build_sandbox(tmp_path, monkeypatch, stub=StubLLM(cost_usd=cost))
    drop(cfg, "client-marcus.txt", CLIENT_CALL)
    if family:
        drop(cfg, "dinner-with-kid.txt", FAMILY_DINNER)
    pipe = Pipeline(cfg)
    pipe.run()
    return cfg, pipe


def test_charts_appear_in_the_html_digest_but_never_in_the_markdown(sandbox):
    cfg, _ = sandbox
    drop(cfg, "client-marcus.txt", CLIENT_CALL)
    pipe = Pipeline(cfg)
    try:
        pipe.run()
        builder = DigestBuilder(cfg, pipe.db)
        opts = DigestOptions(days=30)

        markdown = builder.render_markdown(opts)
        assert "<svg" not in markdown
        assert "pb-chart" not in markdown

        page = builder.render_html(opts)
        assert 'class="pb-charts"' in page
        assert page.count('<figure class="pb-chart"') >= 3
        assert "Minutes per section" in page
        assert "Minutes per day" in page
        # Charts sit right after the At a Glance table, inside the body.
        assert page.index("</table>") < page.index('class="pb-charts"')
    finally:
        pipe.close()


def test_the_page_with_charts_stays_self_contained(tmp_path, monkeypatch):
    cfg, pipe = _processed(tmp_path, monkeypatch, cost=0.03, family=True)
    try:
        builder = DigestBuilder(cfg, pipe.db)
        page = builder.render_html(DigestOptions(days=30, include_personal=True))
        for external in ("http://", "https://", "<script", "@import"):
            assert external not in page, f"the page reaches out via {external}"
        assert page.count("<svg") == page.count("</svg>")
        assert page.count("<figure") == page.count("</figure>")
    finally:
        pipe.close()


def test_the_profile_bar_prints_the_minutes_and_count_the_index_holds(tmp_path, monkeypatch):
    cfg, pipe = _processed(tmp_path, monkeypatch)
    try:
        rows = pipe.db.query(profile_id="insurance_agent")
        assert len(rows) == 1
        minutes = round((rows[0]["duration_seconds"] or 0) / 60.0, 1)
        assert minutes > 0, "a text transcript should synthesise a nonzero timeline"

        page = DigestBuilder(cfg, pipe.db).render_html(DigestOptions(days=30))
        assert f"{minutes:.0f} min · 1 rec" in page
        # The bar's identity is text, not hue alone: the section heading is in
        # the SVG's own labels and in its accessible title.
        assert "Bar chart. Minutes per section: Production" in page
    finally:
        pipe.close()


def test_a_personal_profile_stays_out_of_charts_unless_asked_for(tmp_path, monkeypatch):
    cfg, pipe = _processed(tmp_path, monkeypatch, family=True)
    try:
        builder = DigestBuilder(cfg, pipe.db)

        combined = DigestOptions(days=30)
        fragment = charts_html(builder._collect(combined), combined, cfg.voice)
        assert "Production" in fragment
        assert "Family" not in fragment, "a personal section leaked into the charts"

        opened = DigestOptions(days=30, include_personal=True)
        fragment = charts_html(builder._collect(opened), opened, cfg.voice)
        assert "Family" in fragment

        only_dad = DigestOptions(profile_id="father", days=30)
        fragment = charts_html(builder._collect(only_dad), only_dad, cfg.voice)
        assert "Family" in fragment and "Production" not in fragment
    finally:
        pipe.close()


def test_the_cost_chart_prints_real_spend_and_respects_the_costs_toggle(tmp_path, monkeypatch):
    cfg, pipe = _processed(tmp_path, monkeypatch, cost=0.02)
    try:
        rows = pipe.db.query(profile_id="insurance_agent")
        spend = rows[0]["total_cost_usd"]
        assert spend > 0, "the stub was configured to cost money"

        builder = DigestBuilder(cfg, pipe.db)
        page = builder.render_html(DigestOptions(days=30))
        assert "API spend per section" in page
        assert f"${spend:.4f}" in page

        quiet = DigestOptions(days=30, include_costs=False)
        fragment = charts_html(builder._collect(quiet), quiet, cfg.voice)
        assert "API spend" not in fragment
    finally:
        pipe.close()


def test_zero_spend_renders_a_note_rather_than_an_all_zero_chart(tmp_path, monkeypatch):
    cfg, pipe = _processed(tmp_path, monkeypatch, cost=0.0)
    try:
        page = DigestBuilder(cfg, pipe.db).render_html(DigestOptions(days=30))
        assert "No API spend recorded in this window." in page
    finally:
        pipe.close()


def test_an_empty_archive_renders_a_page_without_charts_or_errors(sandbox):
    cfg, _ = sandbox
    db = Database(cfg.path("database"))
    try:
        page = DigestBuilder(cfg, db).render_html(DigestOptions(days=7))
        assert page.rstrip().endswith("</html>")
        assert "pb-charts" not in page, "an empty window should draw nothing"
        assert "No recordings in this window." in page
    finally:
        db.close()


def test_a_single_recording_still_draws_valid_svg(tmp_path, monkeypatch):
    cfg, pipe = _processed(tmp_path, monkeypatch)
    try:
        page = DigestBuilder(cfg, pipe.db).render_html(DigestOptions(days=30))
        assert page.count("<svg") == page.count("</svg>")
        assert "NaN" not in page
        assert 'width="-' not in page and 'height="-' not in page
        # One section means no legend box: the bar's own heading is the key.
        assert 'class="pb-legend"' not in page
    finally:
        pipe.close()


def test_injection_falls_back_to_the_end_of_the_body_without_a_table():
    page = "<!doctype html>\n<html><head></head><body>\n<p>hello</p>\n</body>\n</html>\n"
    out = inject_charts(page, "<section>charts</section>")
    assert "<section>charts</section>" in out
    assert out.index("<p>hello</p>") < out.index("<section>charts</section>")
    assert out.index("<section>charts</section>") < out.index("</body>")
    assert inject_charts(page, "") == page


def test_the_cli_html_route_carries_the_charts(tmp_path, monkeypatch):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    drop(cfg, "client-marcus.txt", CLIENT_CALL)
    assert main(["--config", str(tmp_path / "config"), "run"]) == 0

    out = tmp_path / "digest.html"
    assert main([
        "--config", str(tmp_path / "config"), "digest",
        "--format", "html", "--days", "30", "--out", str(out),
    ]) == 0
    page = out.read_text(encoding="utf-8")
    assert 'class="pb-charts"' in page
    assert page.count('<figure class="pb-chart"') >= 3
    for external in ("http://", "https://", "<script"):
        assert external not in page
