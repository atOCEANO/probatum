"""What both reports are built of: the document around them, a figure held inline, a
table and a folded part, every text escaped."""

import html
import re

from . import _figures, _look


def escape(text):
    return html.escape(str(text), quote=True)


def document(title, body):
    """A whole page of ``body``, titled ``title``, its look in one style block."""
    return (f"<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
            f"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            f"<title>{escape(title)}</title>\n<style>{_look.PAGE}</style>\n</head>\n"
            f"<body>\n<main>\n{body}\n</main>\n</body>\n</html>\n")


class Figures:
    """The figures of one page, each SVG's ids made its own, since a page holds them
    side by side and an id or a reference to one would otherwise reach another's."""


    def __init__(self):
        self.count = 0


    def held(self, make, *given):
        """``make(*given)`` as a figure held inline, or nothing where it does not
        apply."""
        _, svg = _figures.drawn(make, *given)
        if svg is None:
            return ""

        self.count += 1
        tag = f"f{self.count}-"
        svg = re.sub(r'\bid="', f'id="{tag}', svg)
        svg = re.sub(r'href="#', f'href="#{tag}', svg)
        svg = re.sub(r"url\(#", f"url(#{tag}", svg)
        return f"<figure>{svg}</figure>\n"


def table(heads, rows, numbers=()):
    """A table of ``rows`` under ``heads``, the columns at ``numbers`` set as
    numbers."""
    top  = "".join(f"<th>{escape(head)}</th>" for head in heads)
    body = "".join(
        "<tr>" + "".join(f"<td class=\"number\">{escape(cell)}</td>" if at in numbers
                         else f"<td>{escape(cell)}</td>" for at, cell in enumerate(row))
        + "</tr>\n" for row in rows)
    return (f"<div class=\"scroll\"><table>\n<thead><tr>{top}</tr></thead>\n<tbody>\n"
            f"{body}</tbody>\n</table></div>\n")


def folded(summary, body, open_=False):
    """``body`` folded under ``summary``, open where ``open_``."""
    shown = " open" if open_ else ""
    return f"<details{shown}><summary>{escape(summary)}</summary>\n{body}</details>\n"


def section(title, body):
    return f"<section>\n<h2>{escape(title)}</h2>\n{body}</section>\n" if body else ""


def paragraph(text, kind=None):
    marked = f" class=\"{kind}\"" if kind else ""
    return f"<p{marked}>{escape(text)}</p>\n"


def preformatted(text):
    return f"<pre>{escape(text)}</pre>\n"
