#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Convert Google AI dev .md.txt notebook sources into Zensical-formatted markdown.

Transformations applied:
  1. Strip metadata header (badge table, "<br />" lines, "Run in Google Colab", etc.)
  2. Remove "This notebook will run on ..." lines
  3. Convert 4-space/tab indented blocks → ```python / ```bash / ```text fenced code blocks
  4. Convert bare ``` fenced blocks (output) → ``` { .no-copy .output }
  5. Convert blockquote-wrapped notes/warnings/important →  !!! admonitions
  6. Fix markdown tables (ensure valid headers and separator line placement)
  7. Style "Get it on Kaggle / Get it on Hugging Face" links as cookbook-card blocks
  8. Ensure markdown list sub-bullets are indented by 4 spaces (additional tab)
  9. Collapse triple+ blank lines into double blank lines
  10. Detect shell-only code blocks (pip install / gcloud / git) and use ```bash
  11. Rejoin wrapped list-item continuation lines instead of eating them as code
  12. Extract base64 data-URI images to a real asset directory

Content fidelity
----------------
This is a FORMAT converter, not an editor. It must never reword, reorder, or drop prose --
see the "Content fidelity" section in PLAN.md. If output differs from the source in anything
other than markup, that is a bug in this script.

Usage:
    python convert_notebook.py <input.md.txt|url> <output.md> [--title "Default Title"]
                               [--assets-dir docs/assets/<section>] [--assets-url ../assets/<section>]

Tests:
    python tests/test_convert_notebook.py
"""

import re
import sys
import argparse
import urllib.request
from pathlib import Path


def fetch_content(source: str) -> str:
    """Read content from a file path or URL."""
    if source.startswith("http://") or source.startswith("https://"):
        req = urllib.request.Request(source, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            return resp.read().decode("utf-8")
    return Path(source).read_text(encoding="utf-8")


def is_badge_line(line: str) -> bool:
    """Detect 'Run in Google Colab', badge tables, or coding platform header buttons."""
    s = line.strip()
    badge_keywords = (
        "View on ai.google.dev",
        "Run in Google Colab",
        "Run in Kaggle",
        "Open in Vertex AI",
        "View source on GitHub",
        "notebook-site-button",
        "colab_logo",
        "kaggle-logo",
        "cloud-icon.svg",
        "GitHub-Mark",
    )
    if any(kw in s for kw in badge_keywords):
        return True
    return False


def is_shell_block(lines: list[str]) -> bool:
    """Detect if a code block is shell commands (pip, gcloud, git, etc.)."""
    code_lines = [l for l in lines if l.strip() and not l.strip().startswith("#")]
    if not code_lines:
        return False
    shell_prefixes = (
        "pip ", "pip3 ", "apt-get ", "apt ", "conda ", "brew ", "npm ",
        "wget ", "curl ", "git ", "gcloud ", "chmod ", "mkdir ", "cd ", "export ",
        "sudo ", "source ", "bash ", "sh ", "cat ", "ls ", "echo ", "gsutil ",
    )
    all_shell = True
    for line in code_lines:
        s = line.strip()
        if s.startswith("!"):
            s = s[1:].strip()
        if not any(s.startswith(p) for p in shell_prefixes) and not s.startswith("!"):
            all_shell = False
            break
    return all_shell


def clean_shell_lines(lines: list[str]) -> list[str]:
    """Clean up shell lines by stripping leading exclamation marks."""
    cleaned = []
    for line in lines:
        s = line.lstrip()
        indent = line[:len(line) - len(s)]
        if s.startswith("!") and not s.startswith("!="):
            cleaned.append(indent + s[1:])
        else:
            cleaned.append(line)
    return cleaned


def is_sub_bullet(line: str) -> bool:
    """Check if an indented line is actually a markdown list bullet item."""
    s = line.strip()
    return bool(re.match(r"^([-*+]|\d+\.)\s+", s))


LIST_ITEM = re.compile(r"^(\s*)([-*+]|\d+\.)\s+")

# Syntax-level Python signals. Deliberately avoids bare English words like "for", "with" and
# "from" on their own -- prompt templates and prose contain those constantly, and treating them
# as code is exactly the bug this replaces.
PY_SIGNAL = re.compile(
    r"^\s*(?:import|from)\s+\w"                                   # import statements
    r"|^\s*(?:def|class)\s+\w"                                    # definitions
    r"|^\s*(?:return|yield|raise|assert|pass|break|continue)\b"    # statement keywords
    r"|^\s*[#@]"                                                  # comment or decorator
    r"|\w\s*=\s*\S"                                               # assignment
    r"|\w\("                                                      # function call
    r"|\)\s*:"                                                    # block opener
    r"|\bself\.|\bNone\b|\bTrue\b|\bFalse\b",
    re.MULTILINE,
)

# Markdown image whose source is an inline base64 payload. Anchored on the `![...](...)` form so
# that a data URI appearing inside real code (e.g. `return f"data:image/jpeg;base64,{b64}"`)
# is left alone.
DATA_URI_IMAGE = re.compile(
    r"!\[([^\]]*)\]\(\s*data:image/([A-Za-z0-9.+-]+);base64,([A-Za-z0-9+/=\s]+?)\s*\)"
)


def enclosing_list_indent(lines: list[str], i: int) -> int | None:
    """Return the indent of the list item that line ``i`` continues, or None.

    Devsite wraps long list items across lines, indenting the continuation. Without this the
    4-space-indent rule below mistakes that continuation for a code block and fences prose --
    the defect seen at models/gemma-4/index.md:117.

    A blank line ends a continuation: an indented block after a blank line is a code block.
    """
    current = lines[i]
    current_indent = len(current) - len(current.lstrip())
    j = i - 1
    while j >= 0:
        previous = lines[j]
        if not previous.strip():
            return None
        match = LIST_ITEM.match(previous)
        if match:
            indent = len(match.group(1))
            return indent if indent < current_indent else None
        previous_indent = len(previous) - len(previous.lstrip())
        if previous_indent == 0 or previous_indent < current_indent:
            return None
        j -= 1  # another continuation line; keep walking back to find the bullet
    return None


def looks_like_python(lines: list[str]) -> bool:
    """Whether an indented block is Python rather than prose or a prompt template."""
    code = "\n".join(lines).strip()
    return bool(code) and bool(PY_SIGNAL.search(code))


def detect_language(lines: list[str]) -> str:
    """Pick a fence language. Falls back to ``text`` rather than mislabelling prose as python."""
    if is_shell_block(lines):
        return "bash"
    if looks_like_python(lines):
        return "python"
    return "text"


def normalised_indent(indent: int) -> int:
    """The indent a bullet ends up with once fix_list_indentation has run."""
    if 1 <= indent <= 3:
        return 4
    if 5 <= indent <= 7:
        return 8
    return indent


def list_content_indent(lines: list[str], index: int, base_indent: int) -> int:
    """Column where the enclosing list item's content starts; 0 when not inside a list.

    A fence must sit at *exactly* this column. At column 0 it breaks out of the list, so
    everything after it starts a new one (embeddinggemma/inference.md's RAG section). One column
    too far and Markdown stops treating it as part of the item and emits the backticks
    literally -- which is what `4. ` items did, their content starting at column 3, not 4.
    """
    j = index - 1
    while j >= 0:
        previous = lines[j]
        if not previous.strip():
            j -= 1
            continue
        match = LIST_ITEM.match(previous)
        if match:
            indent = len(match.group(1))
            if indent >= base_indent:
                return 0
            marker_width = len(match.group(0)) - indent  # "- " -> 2, "4. " -> 3
            return normalised_indent(indent) + marker_width
        if len(previous) - len(previous.lstrip()) == 0:
            return 0
        j -= 1
    return 0


# A devsite card grid, which the upstream .md.txt export mangles into an empty bullet followed
# by a heading and a paragraph. Collapsed to a plain bullet -- see PLAN.md Step 0.7.
CARD_ITEM = re.compile(
    r"^-[ \t]*(?:\[\]\([^)]*\))?[ \t]*\n"      # bullet, optionally an empty link
    r"\n+"
    r"[ \t]+#{3,6}[ \t]+(?P<title>.+?)[ \t]*\n"  # the card's heading
    r"\n+"
    r"(?P<body>[ \t]+\S.*(?:\n[ \t]+\S.*)*)",   # the card's paragraph
    re.MULTILINE,
)

# `[Video](https://youtube...)` on a line of its own: render it, don't link it.
VIDEO_LINK = re.compile(
    r"^\[Video\]\(\s*https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)"
    r"(?P<id>[A-Za-z0-9_-]{6,})[^)]*\)[ \t]*$",  # not \s*$ -- that eats the following blank lines
    re.MULTILINE,
)

# A model-card header: pipe-separated links, then the License/Authors line. Upstream centres
# this with `<p align="center">`, which the .md.txt export drops.
CARD_HEADER = re.compile(
    r"\A(?P<banner>!\[[^\]]*\]\([^)]*\)\n+)?"
    r"(?P<links>\[[^\]]+\]\([^)]+\)(?:[ \t]*\\\|\n?[ \t]*\[[^\]]+\]\([^)]+\))+)\n+"
    r"(?P<meta>\*\*License\*\*[^\n]*\n)",
)

# Notes inside a blockquote. Covers the labelled forms (`> NOTE:`, `> **Note:**`, `> *Note:`)
# and the unlabelled one (`> Note that many libraries ...`), which upstream also uses.
BLOCKQUOTE_NOTE = re.compile(
    r"^>[ \t]*[*_]{0,2}"
    r"(?P<label>NOTE|Note|IMPORTANT|Important|WARNING|Warning|TIP|Tip|CAUTION|Caution)"
    r"[*_]{0,2}"
    # Either a colon (the label is a prefix and drops out) or a word boundary (the label is
    # part of the sentence, as in "Note that many libraries…", and must be kept).
    # `**Note:**` closes its emphasis *after* the colon, so allow markers on both sides.
    r"(?:(?P<colon>[ \t]*:[*_]{0,2})|(?=[ \t]|$))"
    r"[ \t]*(?P<first>.*)$",
)


def collapse_card_grids(text: str) -> str:
    """Turn devsite card grids into plain bullets: `- **Title** -- description`."""

    def replace(match: re.Match) -> str:
        title = match.group("title").strip()
        body = " ".join(line.strip() for line in match.group("body").splitlines())
        return f"- **{title}** — {body}"

    return CARD_ITEM.sub(replace, text)


def merge_output_blocks(text: str) -> str:
    """Join consecutive output blocks separated only by blank lines.

    Upstream splits a single stream of terminal output across several fences -- Keras
    `.summary()` emits a title line and a table, and the source fences them separately. Rendered
    as two boxes it reads as two results. Only `{ .no-copy .output }` blocks are merged: two
    adjacent *code* blocks may legitimately be separate cells, but output is one stream.
    """
    pattern = re.compile(
        r"^(?P<pad>[ \t]*)```[ \t]*\n"          # closing fence of the first block
        r"(?:[ \t]*\n)+"                        # blank lines between
        r"(?P=pad)``` \{ \.no-copy \.output \}[ \t]*\n",  # reopening
        re.MULTILINE,
    )
    previous = None
    while previous != text:                     # runs of 3+ blocks collapse in passes
        previous = text
        text = pattern.sub("", text)
    return text


def tighten_nested_lists(text: str) -> str:
    """Drop the blank line between a list item and its own nested list.

    The blank makes Markdown render the list "loose", wrapping every item in <p> and spacing
    them inconsistently against neighbouring items that lack the blank.
    """
    return re.sub(
        r"^(\s*(?:[-*+]|\d+\.)\s+.*)\n\n+(?=(\s+)(?:[-*+]|\d+\.)\s)",
        r"\1\n",
        text,
        flags=re.MULTILINE,
    )


def embed_videos(text: str) -> str:
    """Replace a standalone `[Video](youtube-url)` with a responsive privacy-mode embed."""

    def replace(match: re.Match) -> str:
        return (
            '<div class="video-embed">\n'
            f'  <iframe src="https://www.youtube.com/embed/{match.group("id")}"\n'
            '          title="Video" loading="lazy" allowfullscreen\n'
            '          allow="accelerometer; encrypted-media; gyroscope; picture-in-picture"\n'
            "          referrerpolicy=\"strict-origin-when-cross-origin\"></iframe>\n"
            "</div>"
        )

    return VIDEO_LINK.sub(replace, text)


def center_card_header(text: str) -> str:
    """Restore the centred model-card header that upstream renders with `<p align="center">`."""

    def replace(match: re.Match) -> str:
        banner = (match.group("banner") or "").strip()
        parts = ['<div class="model-card-header" markdown>']
        if banner:
            parts.append(banner)
        parts.append(match.group("links").strip())
        parts.append(match.group("meta").strip())
        parts.append("</div>")
        return "\n\n".join(parts) + "\n"

    return CARD_HEADER.sub(replace, text, count=1)


def convert_blockquote_notes(lines: list[str]) -> list[str]:
    """Convert `> NOTE: ...` blockquotes into `!!! note` admonitions."""
    out: list[str] = []
    i = 0
    while i < len(lines):
        match = BLOCKQUOTE_NOTE.match(lines[i])
        if not match:
            out.append(lines[i])
            i += 1
            continue

        first = match.group("first").strip()
        if not match.group("colon"):
            # No colon means the label is part of the prose, not a prefix.
            first = f"{match.group('label')} {first}".strip()
        body = [first]
        i += 1
        while i < len(lines) and lines[i].lstrip().startswith(">"):
            body.append(re.sub(r"^\s*>[ \t]?", "", lines[i]).strip())
            i += 1

        out.append(f"!!! {match.group('label').capitalize()}")
        out.extend(f"    {line}" for line in body if line)
        out.append("")
    return out


def extract_data_uri_images(
    text: str, assets_dir: Path | None, assets_url: str, stem: str
) -> tuple[str, int]:
    """Write inline base64 images out to ``assets_dir`` and rewrite the references.

    Returns the rewritten text and the number of images extracted. With no ``assets_dir`` the
    text is returned untouched, so the caller can warn instead of silently inlining megabytes.
    """
    if assets_dir is None or not DATA_URI_IMAGE.search(text):
        return text, 0

    import base64

    assets_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    def replace(match: re.Match) -> str:
        nonlocal count
        count += 1
        alt, fmt, payload = match.group(1), match.group(2).lower(), match.group(3)
        suffix = "jpg" if fmt in ("jpeg", "jpg") else fmt
        filename = f"{stem}-{count:02d}.{suffix}"
        (assets_dir / filename).write_bytes(base64.b64decode("".join(payload.split())))
        return f"![{alt}]({assets_url.rstrip('/')}/{filename})"

    return DATA_URI_IMAGE.sub(replace, text), count


def fix_list_indentation(lines: list[str]) -> list[str]:
    """Ensure markdown list sub-bullets use 4-space tab indentation."""
    out = []
    for line in lines:
        # Check if line has bullet indented by 1 to 3 spaces (sub-bullet level 1)
        m = re.match(r"^(\s+)([-*+]|\d+\.)\s+", line)
        if m:
            num_spaces = len(m.group(1))
            if 1 <= num_spaces <= 3:
                # Replace with 4 spaces
                line = "    " + line.lstrip()
            elif 5 <= num_spaces <= 7:
                # Replace with 8 spaces
                line = "        " + line.lstrip()
        out.append(line)
    return out


def fix_tables(lines: list[str]) -> list[str]:
    """Fix markdown tables so header rows and separator lines conform to markdown specs."""
    out = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if line.strip().startswith("|") and line.strip().endswith("|"):
            table_lines = []
            j = i
            while j < n and lines[j].strip().startswith("|") and lines[j].strip().endswith("|"):
                table_lines.append(lines[j].strip())
                j += 1

            non_sep_lines = [l for l in table_lines if not re.match(r"^\|\s*:?[-=]+:?\s*(\|\s*:?[-=]+:?\s*)+\|$", l)]

            if len(non_sep_lines) >= 1:
                header = non_sep_lines[0]
                cols = [c.strip() for c in header.split("|")[1:-1]]
                num_cols = max(len(cols), 1)
                sep_row = "|" + "|".join(["---"] * num_cols) + "|"

                out.append(header)
                out.append(sep_row)
                for row in non_sep_lines[1:]:
                    out.append(row)
            else:
                out.extend(table_lines)
            i = j
        else:
            out.append(line)
            i += 1
    return out


def parse_get_it_on_link(line: str):
    """Parse a line like '[Get it on Kaggle](url)' -> (platform, url)."""
    m = re.search(r"\[Get it on\s+(.*?)\]\((.*?)\)", line, re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None, None


def convert(
    source_text: str,
    default_title: str = None,
    assets_dir: Path | None = None,
    assets_url: str = "../assets",
    assets_stem: str = "image",
) -> str:
    raw_lines = source_text.splitlines()
    lines = []
    for line in raw_lines:
        l_lstrip = line.lstrip("\t")
        num_tabs = len(line) - len(l_lstrip)
        lines.append(("    " * num_tabs) + l_lstrip)

    output = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # --- Skip metadata/badge lines ---
        if is_badge_line(line):
            i += 1
            continue

        if stripped.startswith("|---"):
            prev_line = lines[i - 1].strip() if i > 0 else ""
            next_line = lines[i + 1].strip() if i + 1 < n else ""
            if is_badge_line(prev_line) or is_badge_line(next_line):
                i += 1
                continue

        if stripped in ("<br />", "<br/>", "<br>"):
            i += 1
            continue

        if re.match(r"^This notebook will run on\b", stripped, re.IGNORECASE):
            i += 1
            continue

        # --- Convert "Get it on Kaggle / Get it on Hugging Face" blocks to cookbook cards ---
        platform, url = parse_get_it_on_link(stripped)
        if platform and url:
            links = [(platform, url)]
            j = i + 1
            while j < n:
                next_strip = lines[j].strip()
                p2, u2 = parse_get_it_on_link(next_strip)
                if p2 and u2:
                    links.append((p2, u2))
                    j += 1
                elif next_strip == "" and j + 1 < n and parse_get_it_on_link(lines[j + 1].strip())[0]:
                    # Skip blank line between Get it on links
                    j += 1
                else:
                    break

            output.append('<div class="cookbook-grid">')
            for p_name, p_url in links:
                output.append(f'<a class="cookbook-card" href="{p_url}" target="_blank" rel="noopener">')
                output.append(f'  <div class="cookbook-card-title">{p_name} <span class="cookbook-card-ext">↗</span></div>')
                output.append(f'  <p class="cookbook-card-desc">Get it on {p_name}</p>')
                output.append('</a>')
            output.append('</div>')
            output.append('')
            i = j
            continue

        # --- Convert blockquote-wrapped admonitions (even if single > or double >>) ---
        match = re.match(r"^>?\s*>?\s*\[!(NOTE|IMPORTANT|WARNING|TIP|CAUTION)\]", stripped, re.IGNORECASE)
        if match:
            admonition_type = match.group(1).capitalize()
            admonition_lines = []
            j = i + 1
            while j < n:
                s_line = lines[j].strip()
                if not s_line:
                    # check if next line is still part of blockquote
                    if j + 1 < n and (lines[j + 1].strip().startswith(">") or lines[j + 1].strip().startswith("> >")):
                        j += 1
                        continue
                    else:
                        break
                if s_line.startswith(">") or s_line.startswith("> >"):
                    # remove leading > characters
                    content = re.sub(r"^(\s*>)+", "", s_line).strip()
                    content = re.sub(r"^\*\*(Note|Important|Warning|Tip|Caution):\*\*\s*", "", content, flags=re.IGNORECASE)
                    admonition_lines.append(content)
                    j += 1
                else:
                    break

            # If there was a preceding empty > line, we can clean up output if needed
            if output and output[-1].strip() == ">":
                output.pop()

            output.append(f"!!! {admonition_type}")
            for al in admonition_lines:
                output.append(f"    {al}")
            output.append("")
            i = j
            continue

        # --- Rejoin wrapped list-item continuations (must precede the code-block rule) ---
        # Devsite wraps long bullets; the continuation is indented and would otherwise be
        # swallowed as a code block. Joining restores the original sentence exactly -- markdown
        # renders "foo\n  bar" inside a list as "foo bar", so nothing about the content changes.
        if (
            line.startswith("    ")
            and not is_sub_bullet(line)
            and enclosing_list_indent(lines, i) is not None
        ):
            if output and output[-1].strip():
                output[-1] = output[-1].rstrip() + " " + line.strip()
            else:
                output.append(line)
            i += 1
            continue

        # --- Convert 4-space indented blocks to fenced code blocks ---
        # NOTE: the previous-line test compares raw indentation. Calling .strip() first (as an
        # earlier version did) removes the very whitespace being tested, so the guard never fired.
        if line.startswith("    ") and not is_sub_bullet(line) and (
            i == 0 or not lines[i - 1].startswith("    ") or is_sub_bullet(lines[i - 1])
        ):
            code_lines = []
            j = i
            while j < n:
                # A bullet may not *start* a block (that would eat indented lists), but once a
                # block is underway an indented bullet is block content -- e.g. the formatting
                # rules inside an ASR prompt template. Ending the block there would split one
                # prompt into a fence plus orphaned bullets.
                started = bool(code_lines)
                if lines[j].startswith("    ") and (started or not is_sub_bullet(lines[j])):
                    code_lines.append(lines[j])  # dedented below, once the block's base is known
                    j += 1
                elif lines[j].strip() == "":
                    # Skip the whole run of blank lines, not just one. Notebook exports
                    # routinely leave two blank lines mid-cell; stopping at the first one
                    # splits a single function across two fences (seen at
                    # models/paligemma/inference.md, `binary_mask = ...`).
                    k = j
                    while k < n and not lines[k].strip():
                        k += 1
                    if (
                        k < n
                        and lines[k].startswith("    ")
                        and (started or not is_sub_bullet(lines[k]))
                    ):
                        code_lines.extend([""] * (k - j))  # preserve the original spacing
                        j = k
                    else:
                        break
                else:
                    break

            while code_lines and code_lines[-1].strip() == "":
                code_lines.pop()

            if code_lines:
                # Dedent by the block's own indent rather than a fixed 4, and indent the fence
                # to match its list item -- otherwise nested code escapes the list.
                base = min(
                    len(line) - len(line.lstrip())
                    for line in code_lines
                    if line.strip()
                )
                code_lines = [line[base:] if line.strip() else "" for line in code_lines]
                pad = " " * list_content_indent(lines, i, base)

                lang = detect_language(code_lines)
                if lang == "bash":
                    code_lines = clean_shell_lines(code_lines)
                output.append(f"{pad}```{lang}")
                output.extend(pad + line if line else "" for line in code_lines)
                output.append(f"{pad}```")
                i = j
                continue

        # --- Convert bare ``` output blocks ---
        if stripped == "```" and i + 1 < n:
            content_lines = []
            j = i + 1
            while j < n and lines[j].strip() != "```":
                content_lines.append(lines[j])
                j += 1

            if j < n:
                # Devsite indents output blocks to match their list item. Emitting at column 0
                # split the list, and the *following* item's fence then rendered literally.
                fence_indent = len(line) - len(line.lstrip())
                pad = " " * (list_content_indent(lines, i, fence_indent + 1) or fence_indent)
                body_base = min(
                    (len(c) - len(c.lstrip()) for c in content_lines if c.strip()),
                    default=0,
                )
                output.append(f"{pad}``` {{ .no-copy .output }}")
                output.extend(
                    pad + c[body_base:] if c.strip() else "" for c in content_lines
                )
                output.append(f"{pad}```")
                i = j + 1
                continue

        # --- Default: pass through ---
        output.append(line)
        i += 1

    # Apply table fixes and list indentation fixes
    output = convert_blockquote_notes(output)
    output = fix_tables(output)
    output = fix_list_indentation(output)

    result = "\n".join(output)

    # Whole-document cleanups. Order matters: collapse the card grids into bullets before
    # tightening lists, so the generated bullets are tightened too.
    result = collapse_card_grids(result)
    result = merge_output_blocks(result)
    result = tighten_nested_lists(result)
    result = embed_videos(result)
    result = center_card_header(result)

    # Extract inline base64 images before anything else measures the document's size.
    result, extracted = extract_data_uri_images(result, assets_dir, assets_url, assets_stem)
    if extracted:
        print(f"  extracted {extracted} inline image(s) to {assets_dir}")
    elif assets_dir is None and DATA_URI_IMAGE.search(result):
        print("  WARNING: inline base64 images left in place -- rerun with --assets-dir")

    # Collapse 3+ consecutive blank lines into 2
    result = re.sub(r"\n{4,}", "\n\n\n", result)

    stripped_res = result.lstrip()
    if default_title and not stripped_res.startswith("# "):
        result = f"# {default_title}\n\n" + stripped_res

    # Ensure file ends with single newline
    result = result.rstrip("\n") + "\n"

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Input file path or URL (.md.txt)")
    parser.add_argument("output", help="Output Markdown file path")
    parser.add_argument("--title", default=None, help="Default title heading if missing")
    parser.add_argument(
        "--assets-dir", default=None,
        help="directory to write extracted base64 images into (e.g. docs/assets/paligemma)",
    )
    parser.add_argument(
        "--assets-url", default=None,
        help="URL prefix used to reference extracted images (default: derived from --assets-dir)",
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    assets_dir = Path(args.assets_dir) if args.assets_dir else None
    if assets_dir and not args.assets_url:
        # Derive a page-relative URL, e.g. docs/models/x.md + docs/assets/y -> ../assets/y
        import os
        args.assets_url = os.path.relpath(assets_dir, out_path.parent).replace("\\", "/")

    print(f"Reading from: {args.source}")
    raw = fetch_content(args.source)

    print("Converting...")
    converted = convert(
        raw,
        default_title=args.title,
        assets_dir=assets_dir,
        assets_url=args.assets_url or "../assets",
        assets_stem=out_path.stem,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(converted, encoding="utf-8")
    print(f"Written to: {out_path}")


if __name__ == "__main__":
    main()
