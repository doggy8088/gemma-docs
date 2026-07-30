#!/usr/bin/env python3
"""
Convert Google AI dev .md.txt notebook sources into Zensical-formatted markdown.

Transformations applied:
  1. Strip metadata header (badge table, "<br />" lines)
  2. Remove "This notebook will run on ..." lines
  3. Convert 4-space indented blocks → ```python fenced code blocks
  4. Convert bare ``` fenced blocks (output) → ``` { .no-copy .output }
  5. Convert blockquote-wrapped notes/warnings/important →  !!! admonitions
  6. Collapse triple+ blank lines into double blank lines
  7. Detect shell-only code blocks (pip install / apt-get) and use ```bash

Usage:
    python convert_notebook.py <input.md.txt> <output.md>
    python convert_notebook.py <url> <output.md>

Examples:
    python convert_notebook.py source.md.txt docs/get-started/capabilities/function-calling.md
    python convert_notebook.py https://ai.google.dev/gemma/docs/capabilities/text/function-calling-gemma4.md.txt docs/get-started/capabilities/function-calling.md
"""

import re
import sys
import urllib.request
from pathlib import Path


def fetch_content(source: str) -> str:
    """Read content from a file path or URL."""
    if source.startswith("http://") or source.startswith("https://"):
        with urllib.request.urlopen(source) as resp:
            return resp.read().decode("utf-8")
    return Path(source).read_text()


def is_shell_block(lines: list[str]) -> bool:
    """Detect if a code block is shell commands (pip, apt-get, etc.)."""
    code_lines = [l for l in lines if l.strip() and not l.strip().startswith("#")]
    if not code_lines:
        return False
    shell_prefixes = ("pip ", "pip3 ", "apt-get ", "apt ", "conda ", "brew ", "npm ",
                      "wget ", "curl ", "git ", "chmod ", "mkdir ", "cd ", "export ",
                      "sudo ", "source ", "bash ", "sh ")
    return all(
        any(l.strip().startswith(p) for p in shell_prefixes)
        for l in code_lines
    )


def convert(source_text: str) -> str:
    lines = source_text.split("\n")
    output = []

    i = 0

    while i < len(lines):
        line = lines[i]

        # --- Skip metadata/badge lines (always, regardless of position) ---
        # Skip badge table rows (|---|---|...|  and  | [![...]... | lines)
        # Only skip |---| lines if they're part of a badge table (adjacent to [![ rows)
        if line.strip().startswith("|") and "[![" in line:
            i += 1
            continue
        if line.strip().startswith("|---"):
            # Check if this separator is part of a badge table (next or prev line has [![)
            prev_line = lines[i - 1].strip() if i > 0 else ""
            next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
            if "[![" in prev_line or "[![" in next_line:
                i += 1
                continue
        # Skip <br /> lines
        if line.strip() in ("<br />", "<br/>", "<br>"):
            i += 1
            continue

        # --- Remove "This notebook will run on..." lines ---
        if re.match(r"^This notebook will run on\b", line.strip()):
            i += 1
            continue

        # --- Convert blockquote-wrapped admonitions ---
        # Pattern:  > > [!NOTE]  or  > > [!IMPORTANT]  or  > > [!WARNING]
        if line.strip() == ">" and i + 1 < len(lines):
            # Look ahead for > > [!TYPE] pattern
            match = re.match(
                r"^>\s*>\s*\[!(NOTE|IMPORTANT|WARNING|TIP|CAUTION)\]", 
                lines[i + 1].strip() if i + 1 < len(lines) else ""
            )
            if match:
                admonition_type = match.group(1).capitalize()
                # Collect the admonition content
                admonition_lines = []
                j = i + 2  # skip the > and > > [!TYPE] lines
                while j < len(lines):
                    stripped = lines[j].strip()
                    if stripped == ">":
                        # End of admonition block
                        j += 1
                        break
                    if stripped.startswith("> >"):
                        content = stripped[3:].strip()
                        # Strip leading **Note:** / **Important:** etc
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
        if line.startswith("    ") and (
            i == 0 or not lines[i - 1].strip().startswith("    ")
        ):
            # Collect all contiguous indented lines (including blank lines within)
            code_lines = []
            j = i
            while j < len(lines):
                if lines[j].startswith("    "):
                    code_lines.append(lines[j][4:])  # strip 4-space indent
                    j += 1
                elif lines[j].strip() == "" and j + 1 < len(lines) and lines[j + 1].startswith("    "):
                    # Blank line within indented block
                    code_lines.append("")
                    j += 1
                else:
                    break

            # Strip trailing blank lines from code block
            while code_lines and code_lines[-1].strip() == "":
                code_lines.pop()

            # Determine language
            lang = "bash" if is_shell_block(code_lines) else "python"
            output.append(f"```{lang}")
            output.extend(code_lines)
            output.append("```")
            i = j
            continue

        # --- Convert bare ``` output blocks ---
        if line.strip() == "```" and i + 1 < len(lines):
            # This is the opening of a bare fenced block (output)
            content_lines = []
            j = i + 1
            while j < len(lines) and lines[j].strip() != "```":
                content_lines.append(lines[j])
                j += 1
            
            if j < len(lines):  # found closing ```
                output.append("``` { .no-copy .output }")
                output.extend(content_lines)
                output.append("```")
                i = j + 1
                continue

        # --- Default: pass through ---
        output.append(line)
        i += 1

    result = "\n".join(output)

    # Collapse 3+ consecutive blank lines into 2
    result = re.sub(r"\n{4,}", "\n\n\n", result)

    # Ensure file ends with single newline
    result = result.rstrip("\n") + "\n"

    return result


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    source = sys.argv[1]
    output_path = sys.argv[2]

    print(f"Reading from: {source}")
    raw = fetch_content(source)
    
    print("Converting...")
    converted = convert(raw)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(converted)
    print(f"Written to: {output_path}")


if __name__ == "__main__":
    main()
