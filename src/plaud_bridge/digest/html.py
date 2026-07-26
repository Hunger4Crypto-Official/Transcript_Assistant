"""
Markdown to HTML, for the digest only.

This is not a general markdown implementation and should not grow into one. It
handles exactly the constructs `DigestBuilder.render_markdown` emits, which is a
short and closed list, so that HTML output and markdown output cannot drift into
saying different things. One renderer, one voice, two formats.

The page it produces is self-contained: no fonts, no scripts, no external
stylesheet. A digest can contain a client's health disclosures, so it must not
make a network request when opened, and it must survive being saved to a phone
and read on a plane.
"""

from __future__ import annotations

import html
import re

# Escaped first, restored after. The digest emits <sub> deliberately; nothing
# else is allowed through, so a transcript containing "<script>" stays inert.
_ALLOWED = ("sub", "br")

_BOLD = re.compile(r"\*\*(.+?)\*\*")
_CODE = re.compile(r"`([^`]+)`")
_ITALIC = re.compile(r"(?<![\w*])_([^_]+)_(?![\w*])")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_TABLE_DIVIDER = re.compile(r"^\|[\s:|-]+\|$")

_CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body {
  margin: 0 auto; padding: 2.5rem 1.25rem 5rem; max-width: 46rem;
  font: 16px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  color: #1a1a1a; background: #fff;
  -webkit-text-size-adjust: 100%;
}
h1 { font-size: 1.9rem; line-height: 1.2; margin: 0 0 .3rem; letter-spacing: -.02em; }
h2 {
  font-size: 1.15rem; margin: 2.6rem 0 .9rem; padding-bottom: .35rem;
  border-bottom: 1px solid #e5e5e5; letter-spacing: -.01em;
}
h3 { font-size: 1rem; margin: 1.9rem 0 .4rem; }
p { margin: .7rem 0; }
ul { margin: .5rem 0 1rem; padding-left: 1.2rem; }
li { margin: .3rem 0; }
blockquote {
  margin: .9rem 0; padding: .6rem .9rem; border-left: 3px solid #d0d0d0;
  background: #fafafa; color: #444;
}
code {
  font: .85em/1.4 ui-monospace, SFMono-Regular, Menlo, monospace;
  background: #f2f2f2; padding: .12em .38em; border-radius: 3px;
}
sub { display: inline-block; font-size: .78rem; color: #6a6a6a; vertical-align: baseline; }
hr { border: 0; border-top: 1px solid #e5e5e5; margin: 2.5rem 0 1rem; }
a { color: #0b62c4; }
table { border-collapse: collapse; width: 100%; margin: .8rem 0 1.4rem; font-size: .93rem; }
th, td { padding: .42rem .6rem; border-bottom: 1px solid #ececec; text-align: left; }
th { font-weight: 600; color: #555; }
td:not(:first-child), th:not(:first-child) { text-align: right; width: 6.5rem; }
/* Anything the digest wants you to act on. */
h2 + ul li { line-height: 1.5; }
@media (prefers-color-scheme: dark) {
  body { color: #e6e6e6; background: #16181c; }
  h2 { border-bottom-color: #2c2f36; }
  blockquote { background: #1d2026; border-left-color: #3a3f48; color: #b8bcc4; }
  code { background: #23262c; }
  sub { color: #949aa4; }
  hr, th, td { border-color: #2c2f36; }
  a { color: #6fb3ff; }
  th { color: #a8adb6; }
}
@media print {
  body { padding: 0; max-width: none; color: #000; background: #fff; }
  h2 { page-break-after: avoid; }
  h3 { page-break-after: avoid; }
}
""".strip()


def _inline(text: str) -> str:
    """Escape, then restore the small set of markup the digest actually uses."""
    out = html.escape(text, quote=False)
    for tag in _ALLOWED:
        out = out.replace(f"&lt;{tag}&gt;", f"<{tag}>").replace(f"&lt;/{tag}&gt;", f"</{tag}>")
        out = out.replace(f"&lt;{tag}/&gt;", f"<{tag}/>").replace(f"&lt;{tag} /&gt;", f"<{tag} />")

    out = _CODE.sub(r"<code>\1</code>", out)
    out = _BOLD.sub(r"<strong>\1</strong>", out)
    out = _ITALIC.sub(r"<em>\1</em>", out)
    out = _LINK.sub(r'<a href="\2">\1</a>', out)
    return out


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def to_html(markdown: str, title: str = "Digest") -> str:
    lines = markdown.splitlines()
    body: list[str] = []

    list_open = False
    quote_open = False
    table: list[list[str]] | None = None

    def close_list() -> None:
        nonlocal list_open
        if list_open:
            body.append("</ul>")
            list_open = False

    def close_quote() -> None:
        nonlocal quote_open
        if quote_open:
            body.append("</blockquote>")
            quote_open = False

    def close_table() -> None:
        nonlocal table
        if table is None:
            return
        head, *rest = table
        body.append("<table>")
        body.append("<thead><tr>" + "".join(f"<th>{_inline(c)}</th>" for c in head) + "</tr></thead>")
        if rest:
            body.append("<tbody>")
            for row in rest:
                body.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in row) + "</tr>")
            body.append("</tbody>")
        body.append("</table>")
        table = None

    def close_all() -> None:
        close_list()
        close_quote()
        close_table()

    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()

        if not stripped:
            close_all()
            continue

        # Tables: a run of pipe rows, with the |---| divider discarded.
        if stripped.startswith("|") and stripped.endswith("|"):
            close_list()
            close_quote()
            if _TABLE_DIVIDER.match(stripped):
                continue
            if table is None:
                table = []
            table.append(_cells(stripped))
            continue
        close_table()

        if stripped.startswith("### "):
            close_all()
            body.append(f"<h3>{_inline(stripped[4:])}</h3>")
        elif stripped.startswith("## "):
            close_all()
            body.append(f"<h2>{_inline(stripped[3:])}</h2>")
        elif stripped.startswith("# "):
            close_all()
            body.append(f"<h1>{_inline(stripped[2:])}</h1>")
        elif stripped == "---":
            close_all()
            body.append("<hr>")
        elif stripped.startswith("> "):
            close_list()
            if not quote_open:
                body.append("<blockquote>")
                quote_open = True
            body.append(f"<p>{_inline(stripped[2:])}</p>")
        elif stripped.startswith("- "):
            close_quote()
            if not list_open:
                body.append("<ul>")
                list_open = True
            # A hard line break inside a bullet is how the digest attaches the
            # source filename under an action.
            item = stripped[2:]
            if raw.endswith("  "):
                item = item + "<br>"
            body.append(f"<li>{_inline(item)}</li>")
        else:
            close_list()
            close_quote()
            # A trailing double space is markdown's hard break.
            suffix = "<br>" if raw.endswith("  ") else ""
            body.append(f"<p>{_inline(stripped)}{suffix}</p>")

    close_all()

    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<meta name="robots" content="noindex, nofollow">\n'
        f"<title>{html.escape(title)}</title>\n"
        f"<style>\n{_CSS}\n</style>\n"
        "</head>\n<body>\n"
        + "\n".join(body)
        + "\n</body>\n</html>\n"
    )
