"""
Inline SVG charts, shared by every page that draws one.

The digest drew the first charts; the brief, the people roster, a person's
page, the follow-up worklist, and insights draw theirs with the same pieces,
so every chart in the product obeys one set of rules:

1. **A chart redraws numbers its page already prints.** Each page renders its
   text first; charts are an HTML-only layer over those same numbers, so a
   chart can never say something the text would not, and a reader who cannot
   see the chart loses nothing.

2. **Inert.** Static SVG generated in Python: no <script>, no chart library,
   nothing fetched. These pages hold client disclosures and family
   conversations, so opening one must never reach out anywhere. With no hover
   layer, every value is printed on or beside its mark, and each SVG carries a
   <title> so it has a reading for assistive tech. SVG fills print; CSS
   backgrounds do not.

3. **Hue never carries meaning alone.** Six fixed colors validated for
   color-vision-deficiency separation and contrast against both page
   backgrounds (#fff, #16181c), assigned in order and never cycled -- a
   seventh series is neutral gray, not an invented hue. Every mark also has a
   text label, and a legend names each color once.

4. **Small numbers stay honest.** `fmt_minutes` exists because a bar that
   visibly stands tall must never be labelled "0". Anything under ten minutes
   keeps a decimal; anything under a tenth of a minute says so.
"""

from __future__ import annotations

import html as _h
import math

# Light/dark steps of the same six hues. Index = series order, fixed, never
# cycled. Validated (CVD separation, contrast) against #fff and #16181c.
LIGHT = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
DARK = ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300")
NEUTRAL = "pb-cx"

VB_W = 640        # shared viewBox width; the page column is ~46rem
BAR_H = 18        # bar/column thickness, kept under the 24px cap
ROW_H = 48        # one labelled bar row: heading line + bar + air
BAR_MAX_W = 500   # leaves room for the value label at the tip
PLOT_H = 120      # column chart plot height
END_R = 4         # rounded data-end radius; baseline corners stay square


# =========================================================================
# Formatting
# =========================================================================
def n(v: float) -> str:
    """A number for an SVG attribute: short, no float noise."""
    return f"{v:.2f}".rstrip("0").rstrip(".")


def fmt_minutes(minutes: float) -> str:
    """
    Minutes as a label a reader can trust at any size.

    Whole minutes past ten, one decimal under ten (trailing ".0" dropped),
    "<0.1" for a sliver that would otherwise round to a lie, and "0" only for
    a true zero. The rule it enforces: a mark with visible height is never
    labelled zero.
    """
    m = max(0.0, float(minutes))
    if m == 0:
        return "0"
    if m < 0.05:
        return "<0.1"
    if m < 10:
        return f"{m:.1f}".removesuffix(".0")
    return f"{m:.0f}"


def unique_ticks(labels: list[str]) -> list[str]:
    """
    Axis labels that cannot be confused: a repeat gains " (2)", " (3)"...

    Two conversations on one day would otherwise sit under identical labels,
    and a reader could not say which column is which.
    """
    seen: dict[str, int] = {}
    out = []
    for label in labels:
        seen[label] = seen.get(label, 0) + 1
        out.append(label if seen[label] == 1 else f"{label} ({seen[label]})")
    return out


def fmt_share(fraction: float) -> str:
    """A 0..1 fraction as a whole percent, with the same sliver honesty."""
    f = max(0.0, float(fraction))
    if f == 0:
        return "0%"
    if f < 0.005:
        return "<1%"
    return f"{f * 100:.0f}%"


# =========================================================================
# Page chrome
# =========================================================================
def css() -> str:
    light = "".join(f"--pb-c{i}:{c};" for i, c in enumerate(LIGHT))
    dark = "".join(f"--pb-c{i}:{c};" for i, c in enumerate(DARK))
    fills = "".join(
        f".pb-charts .pb-c{i}{{fill:var(--pb-c{i});}}" for i in range(len(LIGHT))
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


def slot(i: int) -> str:
    """Color class for series i. Past the palette: neutral gray, never a cycle."""
    return f"pb-c{i}" if 0 <= i < len(LIGHT) else NEUTRAL


def legend(items: list[tuple[str, str]]) -> str:
    """One key per (label, color class), each color named exactly once."""
    keys = "".join(
        '<span class="pb-key">'
        '<svg width="11" height="11" viewBox="0 0 11 11" aria-hidden="true">'
        f'<rect width="11" height="11" rx="3" class="{cls}"/></svg>'
        f"{_h.escape(label)}</span>"
        for label, cls in items
    )
    return f'<p class="pb-legend">{keys}</p>'


def block(heading: str, parts: list[str], keys: list[tuple[str, str]] | None = None) -> str:
    """
    A titled chart section, or "" when no chart in it drew anything.

    An empty section returns nothing rather than a heading over a void: the
    page's own text already says why it is quiet.
    """
    drawn = [p for p in parts if p]
    if not drawn:
        return ""
    out = [css(), '\n<section class="pb-charts">', f"<h2>{_h.escape(heading)}</h2>"]
    if keys and len(keys) > 1:
        out.append(legend(keys))
    out.extend(drawn)
    out.append("</section>")
    return "".join(out)


def quiet(caption: str, note: str) -> str:
    """A captioned note where a chart would be misleading (all zero, one point)."""
    return (
        f'<figure class="pb-chart"><figcaption>{_h.escape(caption)}</figcaption>'
        f'<p class="pb-quiet">{_h.escape(note)}</p></figure>'
    )


def inject(page: str, fragment: str, after: str = "</table>", *, before: str = "") -> str:
    """
    Place a chart block into a rendered page.

    With `before`, the block goes just ahead of the first occurrence of that
    marker; otherwise just after the first `after` marker. Pages put their
    summary table first, so "after the first </table>" is the natural default
    without parsing HTML. A page without the marker takes the block just
    before </body>.
    """
    if not fragment:
        return page
    if before:
        at = page.find(before)
        if at >= 0:
            return page[:at] + fragment + "\n" + page[at:]
    else:
        at = page.find(after)
        if at >= 0:
            at += len(after)
            return page[:at] + "\n" + fragment + page[at:]
    at = page.rfind("</body>")
    if at >= 0:
        return page[:at] + fragment + "\n" + page[at:]
    return page + fragment


# =========================================================================
# Marks
# =========================================================================
def hbar(x: float, y: float, w: float, h: float, cls: str, *, rounded: bool = True) -> str:
    """Horizontal bar: square at the baseline, 4px rounded data-end."""
    if w <= 0.5:
        return ""
    r = min(END_R, w / 2, h / 2) if rounded else 0.0
    if r <= 0:
        return f'<rect class="{cls}" x="{n(x)}" y="{n(y)}" width="{n(w)}" height="{n(h)}"/>'
    return (
        f'<path class="{cls}" d="M{n(x)} {n(y)}h{n(w - r)}'
        f"q{n(r)} 0 {n(r)} {n(r)}v{n(h - 2 * r)}"
        f'q0 {n(r)} -{n(r)} {n(r)}h-{n(w - r)}z"/>'
    )


def vcol(x: float, y: float, w: float, h: float, cls: str, rounded: bool) -> str:
    """Column segment: rounded on top only when it is the top of its stack."""
    if h <= 0.5 or w <= 0:
        return ""
    r = min(END_R, w / 2, h / 2) if rounded else 0.0
    if r <= 0:
        return f'<rect class="{cls}" x="{n(x)}" y="{n(y)}" width="{n(w)}" height="{n(h)}"/>'
    return (
        f'<path class="{cls}" d="M{n(x)} {n(y + h)}v-{n(h - r)}'
        f"q0 -{n(r)} {n(r)} -{n(r)}h{n(w - 2 * r)}"
        f'q{n(r)} 0 {n(r)} {n(r)}v{n(h - r)}z"/>'
    )


def nice(v: float) -> float:
    """The smallest 1/2/5-shaped number >= v, so axis ticks read clean."""
    if v <= 0:
        return 1.0
    exp = math.floor(math.log10(v))
    for m in (1, 2, 5, 10):
        cand = m * 10.0**exp
        if cand >= v - 1e-9:
            return cand
    return 10.0 ** (exp + 1)  # pragma: no cover - (1,2,5,10) always hits


def _open(caption: str, height: float, title: str) -> list[str]:
    return [
        f'<figure class="pb-chart"><figcaption>{_h.escape(caption)}</figcaption>',
        f'<svg viewBox="0 0 {VB_W} {n(height)}" width="{VB_W}" height="{n(height)}" role="img">',
        f"<title>{_h.escape(title)}</title>",
    ]


# =========================================================================
# Charts
# =========================================================================
def labelled_bars(rows: list[tuple[str, float, str, str]], caption: str, title: str) -> str:
    """
    A labelled bar list: heading above each bar, value printed at the tip.

    rows are (label, value, printed, color class). The heading rides its own
    line rather than a left gutter so a long label can never collide with its
    bar, and a zero draws no bar at all -- just its printed value -- rather
    than a misleading nub.
    """
    if not rows:
        return ""
    maxv = max((v for _, v, _, _ in rows), default=0.0)
    height = ROW_H * len(rows)
    parts = _open(caption, height, title)
    for i, (label, value, printed, cls) in enumerate(rows):
        y = i * ROW_H
        w = round(BAR_MAX_W * value / maxv, 2) if maxv > 0 else 0.0
        parts.append(f'<text x="0" y="{y + 13}" font-size="13">{_h.escape(label)}</text>')
        bar = hbar(0, y + 20, w, BAR_H, cls)
        if bar:
            parts.append(bar)
        parts.append(
            f'<text class="pb-mut" x="{n(w + 8)}" y="{y + 33}" font-size="12">'
            f"{_h.escape(printed)}</text>"
        )
    parts.append("</svg></figure>")
    return "".join(parts)


def share_bars(rows: list[tuple[str, list[tuple[float, str]], str]],
               caption: str, title: str) -> str:
    """
    One full-width bar per row, split into proportional segments.

    rows are (label, [(fraction, color class), ...], printed). Fractions are
    of that row's whole (they should sum to ~1); each bar spans the same
    width, so the reader compares proportions, not totals -- the right shape
    for "how much of this conversation was you". Segments meet with a 2px
    surface gap so adjacent colors never read as one.
    """
    if not rows:
        return ""
    height = ROW_H * len(rows)
    parts = _open(caption, height, title)
    for i, (label, segments, printed) in enumerate(rows):
        y = i * ROW_H
        parts.append(f'<text x="0" y="{y + 13}" font-size="13">{_h.escape(label)}</text>')
        total = sum(max(0.0, f) for f, _ in segments)
        drawn = [(f, c) for f, c in segments if f > 0]
        x = 0.0
        for k, (frac, cls) in enumerate(drawn):
            w = BAR_MAX_W * frac / total if total > 0 else 0.0
            last = k == len(drawn) - 1
            gap = 0.0 if last else 2.0
            seg = hbar(x, y + 20, max(0.0, w - gap), BAR_H, cls, rounded=last)
            if seg:
                parts.append(seg)
            x += w
        parts.append(
            f'<text class="pb-mut" x="{n(BAR_MAX_W + 8)}" y="{y + 33}" font-size="12">'
            f"{_h.escape(printed)}</text>"
        )
    parts.append("</svg></figure>")
    return "".join(parts)


def columns(buckets: list[list[tuple[float, str]]], ticks: list[str], caption: str,
            title: str, *, cap=fmt_minutes, axis=None, max_ticks: int = 8,
            integer: bool = False, show_zero: bool = False) -> str:
    """
    A column chart, stacked when a bucket holds more than one series.

    buckets[i] is [(value, color class), ...] bottom to top; ticks[i] its axis
    label. Cap labels print every total while they fit, otherwise only the
    peak -- a number on every crowded column is noise, and the gridline ticks
    still carry the scale. `cap` formats cap labels (minutes by default);
    `axis` formats the gridline ticks (the plain number by default). With
    `integer`, a gridline that would sit at a fraction is left out -- a count
    axis never shows "0.5 follow-ups". With `show_zero`, a bucket whose value
    is a measured zero prints its zero at the baseline: on a chart where every
    column is a real observation (a conversation), a blank would read as
    missing data, not as "none". Off by default, because on a timeline a
    blank day really is nothing.
    """
    if not buckets:
        return ""
    axis = axis or n
    totals = [sum(v for v, _ in b) for b in buckets]
    maxv = nice(max(totals, default=0.0))
    count = len(buckets)

    left, right, top = 34.0, 6.0, 16.0
    plot_w = VB_W - left - right
    base = top + PLOT_H
    height = base + 20
    slot_w = plot_w / count
    col_w = min(24.0, slot_w * 0.72)

    parts = _open(caption, height, title)

    # Recessive chrome: hairline gridlines at the half and full tick, plus the
    # baseline. The ticks carry the values the cap labels do not.
    for frac in (0.5, 1.0):
        if integer and (maxv * frac) % 1:
            continue
        gy = base - PLOT_H * frac
        parts.append(
            f'<line class="pb-grid-line" x1="{n(left)}" y1="{n(gy)}" '
            f'x2="{VB_W - int(right)}" y2="{n(gy)}"/>'
        )
        parts.append(
            f'<text class="pb-mut" x="{n(left - 6)}" y="{n(gy + 3.5)}" '
            f'font-size="10" text-anchor="end">{_h.escape(axis(maxv * frac))}</text>'
        )
    parts.append(
        f'<line class="pb-grid-line" x1="{n(left)}" y1="{n(base)}" '
        f'x2="{VB_W - int(right)}" y2="{n(base)}"/>'
    )

    nonzero = [i for i, t in enumerate(totals) if t > 0]
    label_all = slot_w >= 22 or len(nonzero) <= 10
    peak = max(totals, default=0.0)

    for i, bucket in enumerate(buckets):
        x = left + i * slot_w + (slot_w - col_w) / 2
        # Exact stacked boundaries, then a 2px surface gap carved between
        # touching segments (1px off each side of the shared edge).
        bounds = [base]
        for value, _ in bucket:
            bounds.append(bounds[-1] - (value / maxv) * PLOT_H)
        drawn = [k for k, (value, _) in enumerate(bucket) if value > 0]
        for k in drawn:
            seg_top = bounds[k + 1] + (1.0 if k != drawn[-1] else 0.0)
            seg_bot = bounds[k] - (1.0 if k != drawn[0] else 0.0)
            parts.append(
                vcol(x, seg_top, col_w, seg_bot - seg_top, bucket[k][1], rounded=k == drawn[-1])
            )
        if totals[i] > 0 and (label_all or totals[i] == peak):
            ly = max(bounds[-1] - 4, 10.0)
            parts.append(
                f'<text x="{n(x + col_w / 2)}" y="{n(ly)}" font-size="11" '
                f'text-anchor="middle">{_h.escape(cap(totals[i]))}</text>'
            )
        elif show_zero and totals[i] == 0:
            parts.append(
                f'<text class="pb-mut" x="{n(x + col_w / 2)}" y="{n(base - 4)}" '
                f'font-size="11" text-anchor="middle">{_h.escape(cap(0.0))}</text>'
            )

    step = max(1, math.ceil(count / max_ticks))
    for i in range(0, count, step):
        parts.append(
            f'<text class="pb-mut" x="{n(left + i * slot_w + slot_w / 2)}" y="{n(base + 14)}" '
            f'font-size="10" text-anchor="middle">{_h.escape(ticks[i])}</text>'
        )

    parts.append("</svg></figure>")
    return "".join(parts)
