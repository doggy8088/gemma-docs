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
  7. Collapse triple+ blank lines into double blank lines
  8. Detect shell-only code blocks (pip install / gcloud / git) and use ```bash

Usage:
    python convert_notebook.py <input.md.txt|url> <output.md> [--title "Default Title"]

Examples:
    python convert_notebook.py source.md.txt docs/get-started/capabilities/function-calling.md
    python convert_notebook.py https://ai.google.dev/gemma/docs/capabilities/text/function-calling-gemma4.md.txt docs/get-started/capabilities/function-calling.md
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
    # Check if lines start with shell commands or !shell commands
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


def fix_tables(lines: list[str]) -> list[str]:
    """Fix markdown tables so header rows and separator lines conform to markdown specs."""
    out = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if line.strip().startswith("|") and line.strip().endswith("|"):
            # We are at the start of a potential table block
            table_lines = []
            j = i
            while j < n and lines[j].strip().startswith("|") and lines[j].strip().endswith("|"):
                table_lines.append(lines[j].strip())
                j += 1
            
            # Analyze table_lines
            # Separate separator lines (|---|) from data/header lines
            non_sep_lines = [l for l in table_lines if not re.match(r"^\|\s*:?[-=]+:?\s*(\|\s*:?[-=]+:?\s*)+\|$", l)]
            
            if len(non_sep_lines) >= 1:
                header = non_sep_lines[0]
                # Count columns in header
                cols = [c.strip() for c in header.split("|")[1:-1]]
                num_cols = max(len(cols), 1)
                sep_row = "|" + "|".join(["---"] * num_cols) + "|"
                
                # Reconstruct clean table
                out.append(header)
                out.append(sep_row)
                for row in non_sep_lines[1:]:
                    out.append(row)
            else:
                # Just separators or empty table, skip or leave
                out.extend(table_lines)
            i = j
        else:
            out.append(line)
            i += 1
    return out


def convert(source_text: str, default_title: str = None) -> str:
    # Normalize tabs at beginning of lines to 4 spaces
    raw_lines = source_text.splitlines()
    lines = []
    for line in raw_lines:
        # Replace leading tabs with 4 spaces each
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

        # Skip badge table separator rows (|---|---|...) adjacent to badges
        if stripped.startswith("|---"):
            prev_line = lines[i - 1].strip() if i > 0 else ""
            next_line = lines[i + 1].strip() if i + 1 < n else ""
            if is_badge_line(prev_line) or is_badge_line(next_line):
                i += 1
                continue

        # Skip <br /> lines at root level
        if stripped in ("<br />", "<br/>", "<br>"):
            i += 1
            continue

        # --- Remove "This notebook will run on..." lines ---
        if re.match(r"^This notebook will run on\b", stripped, re.IGNORECASE):
            i += 1
            continue

        # --- Convert blockquote-wrapped admonitions ---
        if stripped == ">" and i + 1 < n:
            match = re.match(
                r"^>\s*>\s*\[!(NOTE|IMPORTANT|WARNING|TIP|CAUTION)\]",
                lines[i + 1].strip() if i + 1 < n else ""
            )
            if match:
                admonition_type = match.group(1).capitalize()
                admonition_lines = []
                j = i + 2
                while j < n:
                    s_line = lines[j].strip()
                    if s_line == ">":
                        j += 1
                        break
                    if s_line.startswith("> >"):
                        content = s_line[3:].strip()
                        content = re.sub(
                            r"^\*\*(Note|Important|Warning|Tip|Caution):\*\*\s*",
                            "", content
                        )
                        admonition_lines.append(content)
                    else:
                        break
                    j += 1

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

    # Apply table fixes
    output = fix_tables(output)

    result = "\n".join(output)

    # Collapse 3+ consecutive blank lines into 2
    result = re.sub(r"\n{4,}", "\n\n\n", result)

    # Ensure title heading if default_title provided and missing
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
