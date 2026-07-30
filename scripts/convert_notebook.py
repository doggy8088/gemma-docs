#!/usr/bin/env python3
"""
Convert Google AI dev .md.txt notebook sources into Zensical-formatted markdown.

Transformations applied:
  1. Strip metadata header (badge table, "<br />" lines, "Run in Google Colab", etc.)
  2. Remove "This notebook will run on ..." lines
  3. Convert 4-space/tab indented blocks → ```python or ```bash fenced code blocks
  4. Convert bare ``` fenced blocks (output) → ``` { .no-copy .output }
  5. Convert blockquote-wrapped notes/warnings/important →  !!! admonitions
  6. Fix markdown tables (ensure valid headers and separator line placement)
  7. Style "Get it on Kaggle / Get it on Hugging Face" links as cookbook-card blocks
  8. Ensure markdown list sub-bullets are indented by 4 spaces (additional tab)
  9. Collapse triple+ blank lines into double blank lines
  10. Detect shell-only code blocks (pip install / gcloud / git) and use ```bash

Usage:
    python convert_notebook.py <input.md.txt|url> <output.md> [--title "Default Title"]
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


def convert(source_text: str, default_title: str = None) -> str:
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

        # --- Convert 4-space indented blocks to fenced code blocks ---
        if line.startswith("    ") and not is_sub_bullet(line) and (
            i == 0 or not lines[i - 1].strip().startswith("    ") or is_sub_bullet(lines[i - 1])
        ):
            code_lines = []
            j = i
            while j < n:
                if lines[j].startswith("    ") and not is_sub_bullet(lines[j]):
                    code_lines.append(lines[j][4:])
                    j += 1
                elif lines[j].strip() == "" and j + 1 < n and lines[j + 1].startswith("    ") and not is_sub_bullet(lines[j + 1]):
                    code_lines.append("")
                    j += 1
                else:
                    break

            while code_lines and code_lines[-1].strip() == "":
                code_lines.pop()

            if code_lines:
                lang = "bash" if is_shell_block(code_lines) else "python"
                if lang == "bash":
                    code_lines = clean_shell_lines(code_lines)
                output.append(f"```{lang}")
                output.extend(code_lines)
                output.append("```")
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
                output.append("``` { .no-copy .output }")
                output.extend(content_lines)
                output.append("```")
                i = j + 1
                continue

        # --- Default: pass through ---
        output.append(line)
        i += 1

    # Apply table fixes and list indentation fixes
    output = fix_tables(output)
    output = fix_list_indentation(output)

    result = "\n".join(output)

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
    args = parser.parse_args()

    print(f"Reading from: {args.source}")
    raw = fetch_content(args.source)

    print("Converting...")
    converted = convert(raw, default_title=args.title)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(converted, encoding="utf-8")
    print(f"Written to: {out_path}")


if __name__ == "__main__":
    main()
