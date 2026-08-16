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

import html as _h
import math
from datetime import datetime, timedelta, timezone

from .builder import DigestOptions, DigestSection

# Light/dark steps of the same six hues. Index = section order in the digest,
# fixed, never cycled. Validated (CVD separation, contrast) against #fff and
# #16181c — see the module docstring.
_LIGHT = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
_DARK = ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300")

_VB_W = 640        # shared viewBox width; the page column is ~46rem
_BAR_H = 18        # bar/column thickness, kept under the 24px cap
_ROW_H = 48        # one labelled bar row: heading line + bar + air
_BAR_MAX_W = 500   # leaves room for the value label at the tip
_PLOT_H = 120      # column chart plot height
_END_R = 4         # rounded data-end radius; baseline corners stay square


def _n(v: float) -> str:
    """A number for an SVG attribute: short, no float noise."""
    return f"{v:.2f}".rstrip("0").rstrip(".")


def _css() -> str:
    light = "".join(f"--pb-c{i}:{c};" for i, c in enumerate(_LIGHT))
    dark = "".join(f"--pb-c{i}:{c};" for i, c in enumerate(_DARK))
    fills = "".join(
        f".pb-charts .pb-c{i}{{fill:var(--pb-c{i});}}" for i in range(len(_LIGHT))
    )
    return (
        "<style>\n"
        f".pb-charts{{{light}--pb-cx:#767676;--pb-muted:#6a6a6a;--pb-grid:#e5e5e5;}}\n"
        "@media (prefers-color-scheme: dark){"
        f".pb-charts{{{dark}--pb-cx:#8b929c;--pb-muted:#949aa4;--pb-grid:#2c2f36;}}}}\n"
        "@media print{.pb-chart{break-inside:avoid;}}\n"
        ".pb-chart{margin:1.1rem 0 1.9rem;}\n"
        ".pb-chart figcaption{font-size:.85rem;font-weight:600;color:var(--pb-muted);margin:0 0 .5rem;}\n"
        ".pb-chart svg{display:block;width:100%;height:auto;}\n"
        ".pb-charts text{fill:currentColor;font-family:inherit;}\n"
        ".pb-charts .pb-mut{fill:var(--pb-muted);}\n"
        ".pb-charts .pb-grid-line{stroke:var(--pb-grid);stroke-width:1;}\n"
        f"{fills}.pb-charts .pb-cx{{fill:var(--pb-cx);}}\n"
        ".pb-legend{font-size:.85rem;margin:.1rem 0 1.3rem;}\n"
        ".pb-legend .pb-key{display:inline-flex;align-items:center;gap:.4rem;margin:0 1.1rem .2rem 0;}\n"
        ".pb-quiet{color:var(--pb-muted);font-size:.9rem;margin:.2rem 0;}\n"
        "</style>"
    )


def _slot(i: int) -> str:
    """Color class for section i. Past the palette: neutral gray, never a cycle."""
    return f"pb-c{i}" if i < len(_LIGHT) else "pb-cx"


def _hbar(x: float, y: float, w: float, h: float, cls: str) -> str:
    """Horizontal bar: square at the baseline, 4px rounded data-end."""
    if w <= 0.5:
        return ""
    r = min(_END_R, w / 2, h / 2)
    return (
        f'<path class="{cls}" d="M{_n(x)} {_n(y)}h{_n(w - r)}'
        f"q{_n(r)} 0 {_n(r)} {_n(r)}v{_n(h - 2 * r)}"
        f'q0 {_n(r)} -{_n(r)} {_n(r)}h-{_n(w - r)}z"/>'
    )


def _vcol(x: float, y: float, w: float, h: float, cls: str, rounded: bool) -> str:
    """Column segment: rounded on top only when it is the top of its stack."""
    if h <= 0.5 or w <= 0:
        return ""
    r = min(_END_R, w / 2, h / 2) if rounded else 0.0
    if r <= 0:
        return f'<rect class="{cls}" x="{_n(x)}" y="{_n(y)}" width="{_n(w)}" height="{_n(h)}"/>'
    return (
        f'<path class="{cls}" d="M{_n(x)} {_n(y + h)}v-{_n(h - r)}'
        f"q0 -{_n(r)} {_n(r)} -{_n(r)}h{_n(w - 2 * r)}"
        f'q{_n(r)} 0 {_n(r)} {_n(r)}v{_n(h - r)}z"/>'
    )


def _nice(v: float) -> float:
    """The smallest 1/2/5-shaped number >= v, so axis ticks read clean."""
    if v <= 0:
        return 1.0
    exp = math.floor(math.log10(v))
    for m in (1, 2, 5, 10):
        cand = m * 10.0**exp
        if cand >= v - 1e-9:
            return cand
    return 10.0 ** (exp + 1)  # pragma: no cover - (1,2,5,10) always hits


def _labelled_bars(rows: list[tuple[str, float, str, str]], caption: str, title: str) -> str:
    """
    A labelled bar list: heading above each bar, value printed at the tip.

    The heading rides its own line rather than a left gutter so a long profile
    heading can never collide with its bar, and a zero draws no bar at all —
    just its "0" label — rather than a misleading nub.
    """
    maxv = max((v for _, v, _, _ in rows), default=0.0)
    height = _ROW_H * len(rows)
    parts = [
        f'<figure class="pb-chart"><figcaption>{_h.escape(caption)}</figcaption>',
        f'<svg viewBox="0 0 {_VB_W} {height}" width="{_VB_W}" height="{height}" role="img">',
        f"<title>{_h.escape(title)}</title>",
    ]
    for i, (label, value, printed, cls) in enumerate(rows):
        y = i * _ROW_H
        w = round(_BAR_MAX_W * value / maxv, 2) if maxv > 0 else 0.0
        parts.append(f'<text x="0" y="{y + 13}" font-size="13">{_h.escape(label)}</text>')
        bar = _hbar(0, y + 20, w, _BAR_H, cls)
        if bar:
            parts.append(bar)
        parts.append(
            f'<text class="pb-mut" x="{_n(w + 8)}" y="{y + 33}" font-size="12">'
            f"{_h.escape(printed)}</text>"
        )
    parts.append("</svg></figure>")
    return "".join(parts)


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
    n = opts.days // group + 1

    per = [[0.0] * len(sections) for _ in range(n)]
    for si, section in enumerate(sections):
        for entry in section.entries:
            try:
                day = datetime.strptime(str(entry["when"])[:10], "%Y-%m-%d").date()
            except ValueError:
                continue
            idx = (day - start).days // group
            if 0 <= idx < n:
                per[idx][si] += entry["minutes"]

    totals = [sum(bucket) for bucket in per]
    maxv = _nice(max(totals, default=0.0))

    left, right, top = 34.0, 6.0, 16.0
    plot_w = _VB_W - left - right
    base = top + _PLOT_H
    height = base + 20
    slot = plot_w / n
    col_w = min(24.0, slot * 0.72)

    unit = "week starting" if group == 7 else "day"
    title = (
        f"Column chart. Minutes per {unit}, {start:%Y-%m-%d} to {now:%Y-%m-%d}: "
        f"{sum(totals):.0f} minutes in total."
    )
    parts = [
        f'<figure class="pb-chart"><figcaption>{_h.escape(caption)}</figcaption>',
        f'<svg viewBox="0 0 {_VB_W} {_n(height)}" width="{_VB_W}" height="{_n(height)}" role="img">',
        f"<title>{_h.escape(title)}</title>",
    ]

    # Recessive chrome: hairline gridlines at the half and full tick, plus the
    # baseline. The ticks carry the values the cap labels do not.
    for frac in (0.5, 1.0):
        gy = base - _PLOT_H * frac
        parts.append(
            f'<line class="pb-grid-line" x1="{_n(left)}" y1="{_n(gy)}" '
            f'x2="{_VB_W - int(right)}" y2="{_n(gy)}"/>'
        )
        parts.append(
            f'<text class="pb-mut" x="{_n(left - 6)}" y="{_n(gy + 3.5)}" '
            f'font-size="10" text-anchor="end">{_n(maxv * frac)}</text>'
        )
    parts.append(
        f'<line class="pb-grid-line" x1="{_n(left)}" y1="{_n(base)}" '
        f'x2="{_VB_W - int(right)}" y2="{_n(base)}"/>'
    )

    # Cap labels: all of them while they fit, otherwise only the busiest bucket
    # — a number on every crowded column is noise, and the ticks still carry
    # the scale.
    nonzero = [i for i, t in enumerate(totals) if t > 0]
    label_all = slot >= 22 or len(nonzero) <= 10
    peak = max(totals, default=0.0)

    for i, bucket in enumerate(per):
        x = left + i * slot + (slot - col_w) / 2
        # Exact stacked boundaries, then a 2px surface gap carved between
        # touching segments (1px off each side of the shared edge).
        bounds = [base]
        for m in bucket:
            bounds.append(bounds[-1] - (m / maxv) * _PLOT_H)
        drawn = [si for si, m in enumerate(bucket) if m > 0]
        for si in drawn:
            seg_top = bounds[si + 1] + (1.0 if si != drawn[-1] else 0.0)
            seg_bot = bounds[si] - (1.0 if si != drawn[0] else 0.0)
            parts.append(
                _vcol(x, seg_top, col_w, seg_bot - seg_top, _slot(si), rounded=si == drawn[-1])
            )
        if totals[i] > 0 and (label_all or totals[i] == peak):
            ly = max(bounds[-1] - 4, 10.0)
            parts.append(
                f'<text x="{_n(x + col_w / 2)}" y="{_n(ly)}" font-size="11" '
                f'text-anchor="middle">{totals[i]:.0f}</text>'
            )

    step = max(1, math.ceil(n / 8))
    for i in range(0, n, step):
        day = start + timedelta(days=i * group)
        parts.append(
            f'<text class="pb-mut" x="{_n(left + i * slot + slot / 2)}" y="{_n(base + 14)}" '
            f'font-size="10" text-anchor="middle">{day:%m-%d}</text>'
        )

    parts.append("</svg></figure>")
    return "".join(parts)


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

    parts = [_css(), '\n<section class="pb-charts">']
    parts.append(f"<h2>{_h.escape(say('heading', 'In Charts'))}</h2>")

    # One legend for the whole block: every chart uses the same section-to-
    # color assignment, so each color is named exactly once. A single section
    # needs no legend at all — its bar carries its own heading.
    if len(sections) > 1:
        keys = "".join(
            '<span class="pb-key">'
            f'<svg width="11" height="11" viewBox="0 0 11 11" aria-hidden="true">'
            f'<rect width="11" height="11" rx="3" class="{_slot(i)}"/></svg>'
            f"{_h.escape(s.heading)}</span>"
            for i, s in enumerate(sections)
        )
        parts.append(f'<p class="pb-legend">{keys}</p>')

    minute_rows = []
    for i, section in enumerate(sections):
        mins = sum(e["minutes"] for e in section.entries)
        count = len(section.entries)
        minute_rows.append(
            (section.heading, mins, f"{mins:.0f} min · {count} rec", _slot(i))
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
            parts.append(
                f'<figure class="pb-chart"><figcaption>{_h.escape(caption)}</figcaption>'
                f'<p class="pb-quiet">{_h.escape(note)}</p></figure>'
            )

    parts.append("</section>")
    return "".join(parts)


def inject_charts(page: str, fragment: str) -> str:
    """
    Place the chart block into a rendered page, after the At a Glance table.

    The glance table is the only table the digest emits, so "after the first
    </table>" is that spot without parsing HTML or guessing at a voice pack's
    heading text. A page with no table (an empty digest never gets a fragment,
    but be safe) takes the block just before </body>.
    """
    if not fragment:
        return page
    marker = "</table>"
    at = page.find(marker)
    if at >= 0:
        at += len(marker)
        return page[:at] + "\n" + fragment + page[at:]
    at = page.rfind("</body>")
    if at >= 0:
        return page[:at] + fragment + "\n" + page[at:]
    return page + fragment
