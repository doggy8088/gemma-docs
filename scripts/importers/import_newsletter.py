#!/usr/bin/env python3
"""
Import a visual guide from newsletter.maartengrootendorst.com as markdown.

    python scripts/importers/import_newsletter.py <post-url> <output.md> --title "A Visual Guide to X"

Extracts the article body verbatim -- headings, prose, lists, quotes and images -- rather than
summarising it. The guides are Maarten's own writing, so the words have to survive the import
exactly; a paraphrase would misrepresent them.

Images keep their Substack CDN URLs for now, so the pages can be reviewed before anyone spends
time vendoring artwork. `scripts/lint_docs.py --check images` allows that host explicitly and
flags it as pending -- see PLAN.md Step 6.1.
"""

from __future__ import annotations

import argparse
import html
import re
import sys
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

# `<div dir="auto" class="body markup">` -- class is not necessarily the first attribute.
BODY_START = re.compile(r'<div[^>]*\bclass="[^"]*\bbody\b[^"]*\bmarkup\b[^"]*"[^>]*>')

# Substack chrome that sits inside the article flow but is not part of the writing.
SKIP_CLASSES = (
    "subscription-widget",
    "subscribe-widget",
    "button-wrapper",
    "captioned-image-container-caption",
    "poll",
    "footnote-anchor",
    "pencraft",
)


class ArticleToMarkdown(HTMLParser):
    """Convert Substack's article HTML to markdown, keeping structure and image URLs."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.buffer: list[str] = []
        self.depth = 0                 # nesting depth of the article div
        self.skip_depth = 0            # >0 while inside skipped chrome
        self.heading: str | None = None
        self.list_stack: list[str] = []
        self.in_quote = False
        self.in_code = False
        self.seen_images: set[str] = set()

    # --- helpers ---------------------------------------------------------------------------
    def emit(self, text: str) -> None:
        self.out.append(text)

    def flush(self) -> None:
        text = "".join(self.buffer).strip()
        self.buffer.clear()
        if not text:
            return
        text = re.sub(r"[ \t]+", " ", text)
        if self.heading:
            self.emit(f"{'#' * int(self.heading[1])} {text}\n")
        elif self.list_stack:
            marker = "-" if self.list_stack[-1] == "ul" else "1."
            indent = "    " * (len(self.list_stack) - 1)
            self.emit(f"{indent}{marker} {text}")
        elif self.in_quote:
            self.emit("\n".join(f"> {line}" for line in text.splitlines()) + "\n")
        else:
            self.emit(text + "\n")

    # --- parser callbacks ------------------------------------------------------------------
    def handle_starttag(self, tag: str, attrs: list) -> None:
        attributes = dict(attrs)
        classes = attributes.get("class") or ""

        if self.skip_depth:
            self.skip_depth += 1
            return
        if any(name in classes for name in SKIP_CLASSES):
            self.flush()
            self.skip_depth = 1
            return

        if tag == "img":
            src = attributes.get("src") or ""
            # Substack serves a low-res placeholder plus the real asset in srcset.
            if src and "substackcdn.com" in src and src not in self.seen_images:
                self.seen_images.add(src)
                self.flush()
                alt = (attributes.get("alt") or "").strip()
                self.emit(f"![{alt}]({src})\n")
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.flush()
            self.heading = tag
        elif tag in ("ul", "ol"):
            self.flush()
            self.list_stack.append(tag)
        elif tag == "li":
            self.flush()
        elif tag == "blockquote":
            self.flush()
            self.in_quote = True
        elif tag in ("strong", "b"):
            self.buffer.append("**")
        elif tag in ("em", "i"):
            self.buffer.append("*")
        elif tag == "code":
            self.buffer.append("`")
            self.in_code = True
        elif tag == "a":
            self.buffer.append("[")
            self._href = attributes.get("href") or ""
        elif tag == "br":
            self.buffer.append(" ")
        elif tag == "hr":
            self.flush()
            self.emit("---\n")

    def handle_endtag(self, tag: str) -> None:
        if self.skip_depth:
            self.skip_depth -= 1
            return

        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.flush()
            self.heading = None
        elif tag in ("ul", "ol"):
            self.flush()
            if self.list_stack:
                self.list_stack.pop()
            self.emit("")
        elif tag in ("li", "p", "div"):
            self.flush()
        elif tag == "blockquote":
            self.flush()
            self.in_quote = False
        elif tag in ("strong", "b"):
            self.buffer.append("**")
        elif tag in ("em", "i"):
            self.buffer.append("*")
        elif tag == "code":
            self.buffer.append("`")
            self.in_code = False
        elif tag == "a":
            href = getattr(self, "_href", "")
            self.buffer.append(f"]({href})" if href else "]")

    def handle_data(self, data: str) -> None:
        if self.skip_depth:
            return
        self.buffer.append(data)


def extract_body(page: str) -> str:
    """Slice out the article div, balancing tags so trailing page chrome is excluded."""
    match = BODY_START.search(page)
    if not match:
        raise SystemExit("could not find the article body; Substack markup may have changed")

    start = match.end()
    depth = 1
    for token in re.finditer(r"<(/?)div\b", page[start:]):
        depth += -1 if token.group(1) else 1
        if depth == 0:
            return page[start : start + token.start()]
    return page[start:]


def normalise_headings(text: str) -> str:
    """Re-level the article's headings to sit under the page title, and de-style them.

    Substack posts start their sections at `#`, which would give the page a second H1 and break
    the table of contents. Everything is shifted so the article's own top level becomes `##`,
    preserving the relative hierarchy. Emphasis inside heading text (`### **Foo**`) is stripped:
    the heading level already carries the emphasis.
    """
    levels = [len(m.group(1)) for m in re.finditer(r"^(#+) \S", text, re.MULTILINE)]
    if not levels:
        return text
    shift = 2 - min(levels)

    def relevel(match: re.Match) -> str:
        level = min(len(match.group(1)) + shift, 6)
        title = re.sub(r"[*_]{1,3}", "", match.group(2)).strip()
        title = re.sub(r"\s+", " ", title)
        return f"{'#' * level} {title}"

    return re.sub(r"^(#+) +(.+?)\s*$", relevel, text, flags=re.MULTILINE)


def tidy(markdown: str) -> str:
    text = html.unescape(markdown)

    # Substack wraps every image in a link to its full-size version. Keep the image, drop the
    # wrapper -- a link whose only content is an image is chrome, not a reference.
    text = re.sub(r"\[\s*(!\[[^\]]*\]\([^)\s]+\))\s*\]\([^)\s]+\)", r"\1", text)

    # Emphasis must hug its text: `**bold **` renders literally. Move any whitespace that sits
    # just inside the markers to the outside, in one pass so the rules can't fight each other.
    def hug(match: re.Match) -> str:
        lead, inner, trail = match.groups()
        return f"{' ' if lead else ''}**{inner}**{' ' if trail else ''}"

    text = re.sub(r"\*\*(\s*)([^*\n]+?)(\s*)\*\*", hug, text)

    text = re.sub(r"\*\*\s*\*\*", "", text)      # emptied emphasis
    text = re.sub(r"\[\]\(\)", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = normalise_headings(text)
    return text.strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("url")
    parser.add_argument("output")
    parser.add_argument("--title", required=True)
    parser.add_argument("--description", default="")
    args = parser.parse_args()

    request = urllib.request.Request(args.url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response:
        page = response.read().decode("utf-8", errors="replace")

    converter = ArticleToMarkdown()
    converter.feed(extract_body(page))
    converter.flush()
    body = tidy("\n".join(converter.out))

    description = args.description or f"{args.title} — a visual walkthrough."
    front = (
        "---\n"
        f"title: {args.title}\n"
        f"description: {description}\n"
        f"source_url: {args.url}\n"
        "---\n\n"
        f"# {args.title}\n\n"
        f"!!! note\n"
        f"    Originally published on\n"
        f"    [Exploring Language Models]({args.url}).\n\n"
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(front + body, encoding="utf-8")
    print(f"  {out}: {len(body):,} chars, {len(converter.seen_images)} images")
    return 0


if __name__ == "__main__":
    sys.exit(main())
