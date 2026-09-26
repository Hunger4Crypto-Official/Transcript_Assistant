"""
Inline SVG charts for the HTML digest.

The digest is text first: markdown is the source of truth, and the HTML page is
that markdown converted (html.py). Charts sit on top as an HTML-only layer that
re-draws numbers the text already prints — the At a Glance counts and minutes,
and the cost footer — so a chart can never say something the text would not.

Two properties are load-bearing:

1. **Same data, same gate.** Charts are computed from the DigestSection list
   the markdown renderer consumed. A profile the options excluded (a personal
   profile in a combined digest) is not in that list, so it cannot appear in a
   chart. No transcript-derived text is drawn; the only words in a chart are
   section headings the digest already prints.

2. **Still inert.** Static SVG generated here in Python: no <script>, no chart
   library, nothing fetched. A digest can hold a client's health disclosures,
   so the page must not reach out when opened (see html.py). There is no hover
   layer to lean on, so every value is printed on or beside its mark, and each
   SVG carries a <title> so the chart has a reading for assistive tech. Print
   works because SVG fills print where CSS backgrounds do not.

Color discipline: six fixed colors, assigned to sections in digest order and
never cycled — a seventh section renders in neutral gray rather than a made-up
hue, and the same section keeps the same color in every chart. The set is the
first six categorical slots of a palette validated for color-vision-deficiency
separation and contrast against both page backgrounds html.py uses (#fff and
#16181c). Three of the light-mode hues sit below 3:1 contrast, which is
acceptable only because hue never carries meaning alone here: every bar has its
own text label, a legend names each color once, and the At a Glance table
holds the same numbers as text.

Cost note: a recording routed to two profiles is counted in both sections,
exactly as the At a Glance table and the cost footer already count it. The
charts reproduce the digest's numbers; they do not introduce a second
bookkeeping.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .. import viz
from ..viz import fmt_minutes
from .builder import DigestOptions, DigestSection

# The drawing lives in viz.py, shared by every page that charts. These names
# stay importable here because the digest's tests reach for them directly.
_n = viz.n
_css = viz.css
_slot = viz.slot
_hbar = viz.hbar
_vcol = viz.vcol
_nice = viz.nice
_labelled_bars = viz.labelled_bars


def _activity_chart(
    sections: list[DigestSection], opts: DigestOptions, now: datetime, caption: str
) -> str:
    """
    Minutes per day across the window, stacked by section.

    Day buckets cover the whole query window (days+1 calendar dates, because a
    cutoff at 10:00 seven days ago still admits that morning's recording).
    Past ~6 weeks the buckets fold to weeks so a long window does not become a
    picket fence of unreadable one-pixel columns. Entries whose row carried no
    timestamp at all are left out of this chart only; they still count in the
    bar charts and in the text.
    """
    group = 7 if opts.days > 45 else 1
    start = (now - timedelta(days=opts.days)).date()
    count = opts.days // group + 1

    per = [[0.0] * len(sections) for _ in range(count)]
    for si, section in enumerate(sections):
        for entry in section.entries:
            try:
                day = datetime.strptime(str(entry["when"])[:10], "%Y-%m-%d").date()
            except ValueError:
                continue
            idx = (day - start).days // group
            if 0 <= idx < count:
                per[idx][si] += entry["minutes"]

    unit = "week starting" if group == 7 else "day"
    total = sum(sum(bucket) for bucket in per)
    title = (
        f"Column chart. Minutes per {unit}, {start:%Y-%m-%d} to {now:%Y-%m-%d}: "
        f"{fmt_minutes(total)} minutes in total."
    )
    buckets = [[(m, _slot(si)) for si, m in enumerate(bucket)] for bucket in per]
    ticks = [f"{start + timedelta(days=i * group):%m-%d}" for i in range(count)]
    return viz.columns(buckets, ticks, caption, title)


def charts_html(
    sections: list[DigestSection],
    opts: DigestOptions,
    voice,
    now: datetime | None = None,
) -> str:
    """
    The whole chart block, or "" when there is nothing to draw.

    An empty window draws nothing rather than an empty chart: the digest's own
    empty note already says why the page is quiet, and axes around a void would
    contradict it. Headings and captions go through voice.get with built-in
    defaults, so a voice pack may reword them but a missing key cannot break
    the page.
    """
    if not sections:
        return ""
    now = now or datetime.now(timezone.utc)

    def say(key: str, default: str) -> str:
        value = voice.get(f"digest.charts.{key}", default)
        return value if isinstance(value, str) and value.strip() else default

    parts: list[str] = []

    minute_rows = []
    for i, section in enumerate(sections):
        mins = sum(e["minutes"] for e in section.entries)
        count = len(section.entries)
        minute_rows.append(
            (section.heading, mins, f"{fmt_minutes(mins)} min · {count} rec", _slot(i))
        )
    title = "Bar chart. Minutes per section: " + "; ".join(
        f"{label}, {printed}" for label, _, printed, _ in minute_rows
    )
    parts.append(_labelled_bars(minute_rows, say("minutes_caption", "Minutes per section"), title))

    parts.append(
        _activity_chart(sections, opts, now, say("activity_caption", "Minutes per day"))
    )

    if opts.include_costs:
        caption = say("cost_caption", "API spend per section")
        costs = [sum(e["cost"] for e in s.entries) for s in sections]
        if sum(costs) > 0:
            cost_rows = [
                (s.heading, c, f"${c:.4f}", _slot(i))
                for i, (s, c) in enumerate(zip(sections, costs, strict=True))
            ]
            title = "Bar chart. API spend per section: " + "; ".join(
                f"{label}, {printed}" for label, _, printed, _ in cost_rows
            )
            parts.append(_labelled_bars(cost_rows, caption, title))
        else:
            # An all-zero bar chart reads as broken. Say the good news instead.
            note = say("no_spend", "No API spend recorded in this window.")
            parts.append(viz.quiet(caption, note))

    # One legend for the whole block: every chart uses the same section-to-
    # color assignment, so each color is named exactly once. A single section
    # needs no legend at all — its bar carries its own heading.
    keys = [(s.heading, _slot(i)) for i, s in enumerate(sections)]
    return viz.block(say("heading", "In Charts"), parts, keys)


def inject_charts(page: str, fragment: str) -> str:
    """
    Place the chart block into a rendered page, after the At a Glance table.

    The glance table is the only table the digest emits, so "after the first
    </table>" is that spot without parsing HTML or guessing at a voice pack's
    heading text. A page with no table takes the block just before </body>.
    """
    return viz.inject(page, fragment, after="</table>")
