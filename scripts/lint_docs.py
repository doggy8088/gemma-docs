#!/usr/bin/env python3
"""
Documentation linter for the Gemma docs port.

Guards the invariants in PLAN.md so cleanup work can't silently regress. Every check is a
small, independent function registered in CHECKS; add a new one by writing the function and
registering it, nothing else.

    uv run python scripts/lint_docs.py                    # all checks, full report
    uv run python scripts/lint_docs.py --check links      # one check
    uv run python scripts/lint_docs.py --check fences,admonitions
    uv run python scripts/lint_docs.py --list             # show available checks
    uv run python scripts/lint_docs.py --format json      # machine-readable

Baseline
--------
The port starts with a large backlog of known violations. A baseline file records them so CI
fails only on *new* violations while the backlog is burned down:

    uv run python scripts/lint_docs.py --write-baseline   # snapshot current violations
    uv run python scripts/lint_docs.py --baseline .lintbaseline.json

Findings in the baseline are reported as "known" and do not affect the exit code. Findings that
have been fixed since the baseline was written are reported too, as a nudge to shrink it.
Shrinking the baseline to zero is the definition of done for Phase 0.

Exit codes: 0 = no new findings, 1 = new findings, 2 = usage error.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
CONFIG_FILE = REPO_ROOT / "zensical.toml"
DEFAULT_BASELINE = REPO_ROOT / ".lintbaseline.json"

# --- Thresholds -------------------------------------------------------------------------------

# Body text below this is a stub, not a page.
MIN_BODY_CHARS = 400

# Above this, something went wrong in conversion (usually inlined base64 images).
MAX_FILE_BYTES = 60_000

# Frontmatter keys required today. Phase 3.1 expands this to the full agent schema
# (estimated_tokens, prerequisites, frameworks, code_languages, models, last_updated,
# source_url, ported_at) -- add them here when that step lands.
REQUIRED_FRONTMATTER = ("title", "description")

# The site being replaced. Any link here is a leak.
UPSTREAM_DOCS = "ai.google.dev/gemma/docs"

# Devsite callouts that should have become admonitions during conversion.
DEVSITE_CALLOUTS = ("Note:", "Key Point:", "Caution:", "Warning:", "Important:", "Tip:")

# GitHub-style callouts, which devsite also emits: `> [!NOTE]`, sometimes double-quoted as
# `> > [!NOTE]`. These must become `!!! note` too.
GITHUB_CALLOUT = re.compile(r"^\s*(?:>\s*)+\[!(NOTE|IMPORTANT|WARNING|TIP|CAUTION)\]", re.IGNORECASE)

# Signals that a code block really is Python. Absence of all of these in a block tagged
# `python` means prose or a prompt template got captured by the converter.
PY_SIGNALS = re.compile(
    r"\b(import|from|def|class|return|print|lambda|None|True|False|self|async|await|yield)\b"
    r"|[=<>]=?\s"
    r"|\w\("
    r"|\)\s*[:.]"
    r"|^\s*[#@]"
    r"|\[[\"'\d]",
    re.MULTILINE,
)

# Two or more consecutive word-like tokens -- i.e. it reads as a sentence.
PROSE_SIGNAL = re.compile(r"[A-Za-z]{2,}\s+[A-Za-z]{2,}")


# --- Model ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    """One violation, anchored to a file and (where meaningful) a line."""

    check: str
    path: str  # repo-relative, forward slashes
    line: int  # 1-indexed; 0 when the finding is about the file as a whole
    message: str

    def key(self) -> str:
        """Stable identity for baseline matching.

        Deliberately excludes the line number so that unrelated edits above a known finding
        don't resurrect it as "new".
        """
        return f"{self.check}|{self.path}|{self.message}"

    def location(self) -> str:
        return f"{self.path}:{self.line}" if self.line else self.path


@dataclass
class Page:
    """A markdown page with its frontmatter split out."""

    path: Path
    rel: str  # relative to docs/, forward slashes
    text: str
    frontmatter: dict = field(default_factory=dict)
    has_frontmatter: bool = False
    body: str = ""
    body_offset: int = 0  # lines consumed by the frontmatter block

    @property
    def repo_rel(self) -> str:
        return f"docs/{self.rel}"

    def body_lines(self) -> Iterator[tuple[int, str]]:
        """Yield (1-indexed file line number, line) for the body only."""
        for i, line in enumerate(self.body.splitlines(), start=self.body_offset + 1):
            yield i, line


@dataclass
class Context:
    """Everything the checks need, loaded once."""

    pages: list[Page]
    nav_entries: list[str]  # docs-relative paths referenced by zensical.toml nav
    config: dict

    @property
    def page_paths(self) -> set[str]:
        return {p.rel for p in self.pages}


# --- Loading ----------------------------------------------------------------------------------


def parse_frontmatter(text: str) -> tuple[dict, bool, str, int]:
    """Split a YAML frontmatter block off the top of a document.

    Returns (mapping, had_frontmatter, body, lines_consumed). Parsing is intentionally shallow:
    we only need top-level scalar keys, so a full YAML dependency isn't warranted here.
    """
    if not text.startswith("---"):
        return {}, False, text, 0

    lines = text.splitlines()
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            block = lines[1:idx]
            mapping: dict = {}
            for raw in block:
                # Top-level keys only -- indented lines belong to nested structures.
                if not raw or raw[0].isspace() or raw.lstrip().startswith("#"):
                    continue
                if ":" in raw:
                    key, _, value = raw.partition(":")
                    mapping[key.strip()] = value.strip()
            body = "\n".join(lines[idx + 1 :])
            return mapping, True, body, idx + 1

    # Unterminated block: treat the whole file as body.
    return {}, False, text, 0


def load_pages() -> list[Page]:
    pages = []
    for path in sorted(DOCS_DIR.rglob("*.md")):
        text = path.read_text(encoding="utf-8", errors="replace")
        fm, has_fm, body, offset = parse_frontmatter(text)
        pages.append(
            Page(
                path=path,
                rel=path.relative_to(DOCS_DIR).as_posix(),
                text=text,
                frontmatter=fm,
                has_frontmatter=has_fm,
                body=body,
                body_offset=offset,
            )
        )
    return pages


def walk_nav(node) -> Iterator[str]:
    """Yield every markdown path in zensical.toml's nested nav structure."""
    if isinstance(node, str):
        if node.endswith(".md"):
            yield node
    elif isinstance(node, list):
        for item in node:
            yield from walk_nav(item)
    elif isinstance(node, dict):
        for value in node.values():
            yield from walk_nav(value)


def load_context() -> Context:
    config = tomllib.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    nav = config.get("project", {}).get("nav", [])
    return Context(pages=load_pages(), nav_entries=list(walk_nav(nav)), config=config)


def iter_code_blocks(page: Page) -> Iterator[tuple[int, str, str]]:
    """Yield (start line, info string, block content) for each fenced code block."""
    fence = re.compile(r"^(\s*)```(.*)$")
    open_at: int | None = None
    info = ""
    buf: list[str] = []
    for lineno, line in page.body_lines():
        m = fence.match(line)
        if not m:
            if open_at is not None:
                buf.append(line)
            continue
        if open_at is None:
            open_at, info, buf = lineno, m.group(2).strip(), []
        else:
            yield open_at, info, "\n".join(buf)
            open_at = None


def outside_code(page: Page) -> Iterator[tuple[int, str]]:
    """Yield body lines that are not inside a fenced code block."""
    in_fence = False
    for lineno, line in page.body_lines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            yield lineno, line


# --- Checks -----------------------------------------------------------------------------------

MD_LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
MD_IMAGE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)")
HTML_SRC = re.compile(r"<img[^>]+src=[\"']([^\"']+)[\"']")


def check_links(ctx: Context) -> Iterable[Finding]:
    """No prose links back to the documentation site we're replacing.

    Code blocks are exempt, and deliberately so. Every upstream URL inside a snippet here is
    either sample data the reader is meant to fetch (`img_url = "https://.../cat.png"`) or a
    BibTeX citation, which must keep pointing at the canonical public URL. Rewriting those to
    local paths would break copy-paste and misattribute citations.
    """
    for page in ctx.pages:
        for lineno, line in outside_code(page):
            if UPSTREAM_DOCS in line:
                for match in re.finditer(rf"https?://{re.escape(UPSTREAM_DOCS)}[^\s)\"'>]*", line):
                    yield Finding(
                        "links", page.repo_rel, lineno,
                        f"link to upstream docs: {match.group(0)}",
                    )


def check_images(ctx: Context) -> Iterable[Finding]:
    """Images must be vendored into docs/assets/, not hotlinked.

    Scans outside code blocks only -- markdown image syntax inside a snippet is code, not
    an image reference (e.g. an f-string like `{render_inline(image, resize=(64,64))}`).
    """
    for page in ctx.pages:
        for lineno, line in outside_code(page):
            for pattern in (MD_IMAGE, HTML_SRC):
                for match in pattern.finditer(line):
                    src = match.group(1)
                    if src.startswith("data:"):
                        yield Finding(
                            "images", page.repo_rel, lineno,
                            "inline base64 image -- extract to docs/assets/",
                        )
                    elif src.startswith(("http://", "https://")):
                        yield Finding(
                            "images", page.repo_rel, lineno,
                            f"hotlinked image: {src}",
                        )
                    else:
                        target = (page.path.parent / src.split("#")[0]).resolve()
                        if not target.exists():
                            yield Finding(
                                "images", page.repo_rel, lineno,
                                f"missing local image: {src}",
                            )


def check_internal(ctx: Context) -> Iterable[Finding]:
    """Relative links must resolve to a file that exists. Prose only -- see check_images."""
    for page in ctx.pages:
        for lineno, line in outside_code(page):
            for match in MD_LINK.finditer(line):
                target = match.group(1)
                if target.startswith(("http://", "https://", "#", "mailto:", "data:", "/")):
                    continue
                if line[match.start()] == "!":  # images handled by check_images
                    continue
                resolved = (page.path.parent / target.split("#")[0]).resolve()
                if not resolved.exists():
                    yield Finding(
                        "internal", page.repo_rel, lineno,
                        f"broken internal link: {target}",
                    )


def check_stubs(ctx: Context) -> Iterable[Finding]:
    """Every page in the nav must have real content."""
    for page in ctx.pages:
        if "Content coming soon" in page.body:
            yield Finding("stubs", page.repo_rel, 0, "placeholder text: 'Content coming soon'")
            continue
        if "example.com" in page.body:
            yield Finding("stubs", page.repo_rel, 0, "placeholder link: example.com")
        # Strip headings and blank lines before measuring.
        prose = "\n".join(
            line for line in page.body.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
        if len(prose) < MIN_BODY_CHARS:
            yield Finding(
                "stubs", page.repo_rel, 0,
                f"stub: {len(prose)} chars of content (minimum {MIN_BODY_CHARS})",
            )


def check_frontmatter(ctx: Context) -> Iterable[Finding]:
    """Every page carries agent-readable frontmatter (see PLAN.md Phase 3.1)."""
    for page in ctx.pages:
        if not page.has_frontmatter:
            yield Finding("frontmatter", page.repo_rel, 0, "no frontmatter block")
            continue
        for key in REQUIRED_FRONTMATTER:
            if key not in page.frontmatter:
                yield Finding("frontmatter", page.repo_rel, 0, f"missing required key: {key}")


def check_orphans(ctx: Context) -> Iterable[Finding]:
    """Nav and filesystem must agree."""
    nav = set(ctx.nav_entries)
    for rel in sorted(ctx.page_paths - nav):
        yield Finding("orphans", f"docs/{rel}", 0, "page exists but is not in nav")
    for rel in sorted(nav - ctx.page_paths):
        yield Finding("orphans", "zensical.toml", 0, f"nav entry has no file: {rel}")
    seen: set[str] = set()
    for rel in ctx.nav_entries:
        if rel in seen:
            yield Finding("orphans", "zensical.toml", 0, f"page listed twice in nav: {rel}")
        seen.add(rel)


def check_fences(ctx: Context) -> Iterable[Finding]:
    """Catch converter damage: prose captured as Python, and code blocks split in half."""
    for page in ctx.pages:
        if page.body.count("```") % 2:
            yield Finding("fences", page.repo_rel, 0, "unbalanced code fences")

        # Two same-language blocks separated by nothing but blank lines is the signature of a
        # block that got split -- usually by blank lines inside a notebook cell.
        blocks = list(iter_code_blocks(page))
        body = page.body.splitlines()
        for (start_a, info_a, code_a), (start_b, info_b, _) in zip(blocks, blocks[1:]):
            if info_a != info_b or not info_a:
                continue
            end_a = start_a + len(code_a.splitlines()) + 1  # closing fence
            between = body[end_a - page.body_offset : start_b - page.body_offset - 1]
            if all(not line.strip() for line in between):
                yield Finding(
                    "fences", page.repo_rel, start_b,
                    f"`{info_a}` block split in two -- likely blank lines inside one block",
                )

        for lineno, info, code in blocks:
            lang = info.split()[0].lower() if info else ""
            if lang not in ("python", "py"):
                continue
            stripped = code.strip()
            if not stripped or PY_SIGNALS.search(stripped) or not PROSE_SIGNAL.search(stripped):
                continue
            preview = " ".join(stripped.split())[:60]
            kind = "prompt template" if "{" in stripped else "prose"
            yield Finding(
                "fences", page.repo_rel, lineno,
                f"{kind} tagged as python: '{preview}...'",
            )


def check_admonitions(ctx: Context) -> Iterable[Finding]:
    """Devsite and GitHub callouts should both be Zensical admonitions."""
    for page in ctx.pages:
        for lineno, line in outside_code(page):
            github = GITHUB_CALLOUT.match(line)
            if github:
                yield Finding(
                    "admonitions", page.repo_rel, lineno,
                    f"unconverted callout: '[!{github.group(1).upper()}]' -- use !!! admonition",
                )
                continue
            stripped = line.lstrip("> ").lstrip()
            for callout in DEVSITE_CALLOUTS:
                if stripped.startswith(callout):
                    yield Finding(
                        "admonitions", page.repo_rel, lineno,
                        f"unconverted devsite callout: '{callout}' -- use !!! admonition",
                    )
                    break


def check_size(ctx: Context) -> Iterable[Finding]:
    """Oversized files mean conversion inlined something it shouldn't have."""
    for page in ctx.pages:
        size = page.path.stat().st_size
        if size > MAX_FILE_BYTES:
            yield Finding(
                "size", page.repo_rel, 0,
                f"{size:,} bytes exceeds {MAX_FILE_BYTES:,} -- check for inlined assets",
            )


def check_modeldata(ctx: Context) -> Iterable[Finding]:
    """Keep data/models.yml in step with the ported memory table it was copied from.

    models/gemma-4/index.md is a straight port, so its table can't be generated from the data
    file -- re-porting would overwrite the template. Instead the two are compared here, so
    upstream revising a figure shows up as a finding rather than a silent contradiction.
    """
    data_file = REPO_ROOT / "data" / "models.yml"
    page_path = DOCS_DIR / "models" / "gemma-4" / "index.md"
    if not data_file.exists() or not page_path.exists():
        return

    # Minimal parse: `bf16: 11.4` style keys under each size's `memory:` block, in file order.
    text = data_file.read_text(encoding="utf-8")
    declared: list[tuple[str, dict[str, str]]] = []
    current: str | None = None
    for line in text.splitlines():
        name = re.match(r"^\s*name:\s*(\S.*?)\s*$", line)
        if name:
            current = name.group(1)
            declared.append((current, {}))
            continue
        figure = re.match(r"^\s*(bf16|sfp8|q4_0):\s*([\d.]+)\s*$", line)
        if figure and declared:
            declared[-1][1][figure.group(1)] = figure.group(2)

    # Table rows look like: | Gemma 4 E2B | 11.4 GB | 5.7 GB | 2.9 GB | ...
    rows = {}
    for line in page_path.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) >= 4 and cells[0].startswith("Gemma 4 "):
            rows[cells[0].removeprefix("Gemma 4 ")] = [
                c.removesuffix(" GB").strip() for c in cells[1:4]
            ]

    for name, figures in declared:
        row = rows.get(name)
        if row is None:
            yield Finding(
                "modeldata", "data/models.yml", 0,
                f"'{name}' has no row in models/gemma-4/index.md",
            )
            continue
        for key, expected in zip(("bf16", "sfp8", "q4_0"), row):
            actual = figures.get(key)
            if actual and actual != expected:
                yield Finding(
                    "modeldata", "data/models.yml", 0,
                    f"{name} {key}: data says {actual}, ported table says {expected}",
                )


CHECKS: dict[str, Callable[[Context], Iterable[Finding]]] = {
    "links": check_links,
    "images": check_images,
    "internal": check_internal,
    "stubs": check_stubs,
    "frontmatter": check_frontmatter,
    "orphans": check_orphans,
    "fences": check_fences,
    "admonitions": check_admonitions,
    "size": check_size,
    "modeldata": check_modeldata,
}

CHECK_HELP = {
    "links": "links back to ai.google.dev/gemma/docs",
    "images": "hotlinked, inlined, or missing images",
    "internal": "relative links that don't resolve",
    "stubs": "placeholder pages and leftover example.com",
    "frontmatter": "missing or incomplete agent frontmatter",
    "orphans": "pages missing from nav, nav entries missing files",
    "fences": "prose or prompts captured as python code blocks",
    "admonitions": "unconverted devsite Note:/Caution: callouts",
    "size": "files large enough to indicate inlined assets",
    "modeldata": "data/models.yml vs the ported memory table",
}


# --- Reporting --------------------------------------------------------------------------------


def load_baseline(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return set(json.loads(path.read_text(encoding="utf-8")).get("known", []))


def report_text(new: list[Finding], known: list[Finding], fixed: set[str], selected: list[str]) -> None:
    if new:
        print("NEW FINDINGS\n" + "=" * 78)
        by_check: dict[str, list[Finding]] = {}
        for f in new:
            by_check.setdefault(f.check, []).append(f)
        for check in selected:
            items = by_check.get(check)
            if not items:
                continue
            print(f"\n{check}  ({len(items)}) -- {CHECK_HELP[check]}")
            for f in items:
                print(f"  {f.location()}\n      {f.message}")
        print()

    print("SUMMARY\n" + "=" * 78)
    width = max(len(c) for c in selected)
    new_by_check: dict[str, int] = {}
    known_by_check: dict[str, int] = {}
    for f in new:
        new_by_check[f.check] = new_by_check.get(f.check, 0) + 1
    for f in known:
        known_by_check[f.check] = known_by_check.get(f.check, 0) + 1

    print(f"  {'check':<{width}}   new    known")
    for check in selected:
        n, k = new_by_check.get(check, 0), known_by_check.get(check, 0)
        flag = "" if not n else "  <--"
        print(f"  {check:<{width}}  {n:>4}  {k:>7}{flag}")
    print(f"  {'-' * (width + 16)}")
    print(f"  {'total':<{width}}  {len(new):>4}  {len(known):>7}")

    if fixed:
        print(f"\n  {len(fixed)} baseline finding(s) no longer present -- rerun with "
              f"--write-baseline to shrink the baseline.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Lint the Gemma docs against the invariants in PLAN.md.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--check", default="all",
        help="comma-separated check names, or 'all' (default). See --list.",
    )
    parser.add_argument("--list", action="store_true", help="list available checks and exit")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument(
        "--baseline", nargs="?", const=str(DEFAULT_BASELINE), default=None,
        help=f"treat findings in this file as known (default file: {DEFAULT_BASELINE.name})",
    )
    parser.add_argument(
        "--write-baseline", action="store_true",
        help="write current findings to the baseline file and exit 0",
    )
    args = parser.parse_args(argv)

    if args.list:
        width = max(len(c) for c in CHECKS)
        for name in CHECKS:
            print(f"  {name:<{width}}  {CHECK_HELP[name]}")
        return 0

    if args.check == "all":
        selected = list(CHECKS)
    else:
        selected = [c.strip() for c in args.check.split(",") if c.strip()]
        unknown = [c for c in selected if c not in CHECKS]
        if unknown:
            parser.error(f"unknown check(s): {', '.join(unknown)}. Use --list to see options.")

    if not CONFIG_FILE.exists():
        parser.error(f"config not found: {CONFIG_FILE}")

    ctx = load_context()
    findings = [f for name in selected for f in CHECKS[name](ctx)]
    findings.sort(key=lambda f: (f.check, f.path, f.line))

    if args.write_baseline:
        # A partial run would drop every unselected check's entries from the baseline.
        if set(selected) != set(CHECKS):
            parser.error("--write-baseline requires all checks; drop --check and rerun.")
        DEFAULT_BASELINE.write_text(
            json.dumps(
                {
                    "_comment": "Known violations at baseline time. Shrink to zero; never grow. "
                                "Regenerate with: uv run python scripts/lint_docs.py --write-baseline",
                    "checks": selected,
                    "count": len(findings),
                    "known": sorted({f.key() for f in findings}),
                },
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote baseline with {len(findings)} finding(s) to {DEFAULT_BASELINE.name}")
        return 0

    baseline = load_baseline(Path(args.baseline)) if args.baseline else set()
    new = [f for f in findings if f.key() not in baseline]
    known = [f for f in findings if f.key() in baseline]

    # Only baseline entries belonging to the checks we actually ran can be judged fixed --
    # otherwise `--check links` would report every unrelated baseline entry as resolved.
    scoped_baseline = {k for k in baseline if k.split("|", 1)[0] in selected}
    fixed = scoped_baseline - {f.key() for f in findings}

    if args.format == "json":
        print(json.dumps(
            {
                "new": [f.__dict__ for f in new],
                "known_count": len(known),
                "fixed_count": len(fixed),
            },
            indent=2,
        ))
    else:
        report_text(new, known, fixed, selected)

    return 1 if new else 0


if __name__ == "__main__":
    sys.exit(main())
