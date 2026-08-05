"""Fill in missing frontmatter across docs/ without touching a single line of body text.

Every page is hand-maintained (PLAN.md, "Import once, then hand-maintain"), so this script
is deliberately conservative:

  * It never rewrites a key that already exists -- yours wins, always.
  * It never touches the body. The block is rebuilt and the rest of the file is copied
    through verbatim, which `--check` verifies.
  * Existing keys keep their original raw lines, so nested structures (`cards:`, `hide:`)
    and any comments survive exactly as written. Only the key *order* is normalised.

Descriptions are derived from the page's opening paragraph. That is a starting point for
review, not an authoring system -- run with `--report` to see which ones were guessed.

    uv run python scripts/backfill_frontmatter.py --check     # report, write nothing
    uv run python scripts/backfill_frontmatter.py             # write
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

import yaml

from frontmatter_schema import (
    DESCRIPTION_MAX,
    DESCRIPTION_MIN,
    DOCS_DIR,
    REPO_ROOT,
    SCHEMA,
    THEME_KEYS,
    models_with_modality,
    ported_sources,
)

# Emission order: required keys first, then optional authored keys, then theme keys.
KEY_ORDER = [f.name for f in SCHEMA] + sorted(THEME_KEYS)

FENCE = re.compile(r"^\s*(```|~~~)")
HEADING = re.compile(r"^#\s+(.*)$")
ATTR_LIST = re.compile(r"\s*\{[^}]*\}\s*$")
MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
MD_EMPHASIS = re.compile(r"[*_`]+")
HTML_TAG = re.compile(r"<[^>]+>")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

# Lines that open a paragraph but are not prose.
NON_PROSE_PREFIX = ("!!!", "???", "|", ">", "-", "*", "+", "<", "![", "=== ", ":", "{")


# --- Frontmatter block surgery ----------------------------------------------------------------


def split_block(text: str) -> tuple[list[str], str]:
    """Return (frontmatter lines, rest-of-file). Rest is byte-identical to the input tail."""
    if not text.startswith("---"):
        return [], text
    lines = text.split("\n")
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            return lines[1:idx], "\n".join(lines[idx + 1 :])
    return [], text  # unterminated: treat it all as body


def segment(block: list[str]) -> dict[str, list[str]]:
    """Group a frontmatter block into {top-level key: its raw lines}.

    Continuation lines (indented, blank) and preceding comments stay with their key, so a
    nested `cards:` list round-trips exactly.
    """
    segments: dict[str, list[str]] = {}
    pending: list[str] = []
    current: str | None = None
    for raw in block:
        stripped = raw.strip()
        is_top_level = raw[:1] not in (" ", "\t") and ":" in raw and not stripped.startswith("#")
        if is_top_level:
            current = raw.partition(":")[0].strip()
            segments[current] = [*pending, raw]
            pending = []
        elif current is None or not stripped:
            pending.append(raw)
        else:
            segments[current].append(raw)
    return segments


def parse(block: list[str]) -> dict:
    try:
        return yaml.safe_load("\n".join(block)) or {}
    except yaml.YAMLError:
        return {}


def emit(key: str, value) -> list[str]:
    if isinstance(value, list):
        inline = ", ".join(str(v) for v in value)
        return [f"{key}: [{inline}]"]
    text = str(value)
    if any(ch in text for ch in ':#"') or text != text.strip():
        return [f'{key}: "{text}"']
    return [f"{key}: {text}"]


def render(segments: dict[str, list[str]], additions: dict[str, object]) -> str:
    """Rebuild the block: known keys in canonical order, anything unrecognised last."""
    lines: list[str] = []
    ordered = [k for k in KEY_ORDER if k in segments or k in additions]
    trailing = [k for k in segments if k not in KEY_ORDER]
    for key in ordered + trailing:
        lines.extend(segments[key] if key in segments else emit(key, additions[key]))
    return "---\n" + "\n".join(lines) + "\n---\n"


# --- Inference --------------------------------------------------------------------------------


def strip_fences(lines: list[str]) -> list[str]:
    """Blank out fenced blocks. Shell comments (`# Install PyTorch`) are not headings, and
    reading them as one is how `diffusiongemma/inference.md` acquired the wrong title."""
    out, in_fence = [], False
    for line in lines:
        if FENCE.match(line):
            in_fence = not in_fence
            out.append("")
        else:
            out.append("" if in_fence else line)
    return out


def infer_title(body: str, rel: str, nav_labels: dict[str, str]) -> str:
    for line in strip_fences(body.split("\n")):
        match = HEADING.match(line)
        if match:
            text = ATTR_LIST.sub("", match.group(1))
            text = MD_EMPHASIS.sub("", MD_LINK.sub(r"\1", text)).strip()
            if text:
                return text
    # No H1 on the page. The nav label is hand-curated, so it beats a filename guess.
    return nav_labels.get(rel) or Path(rel).stem.replace("-", " ").capitalize()


def lead_paragraphs(body: str, limit: int = 3) -> list[str]:
    """The opening prose paragraphs, skipping fences, HTML embeds, images, tables, callouts.

    Non-prose openers skip to the next blank line rather than advancing one line: an
    `<iframe` spans several lines, and only its first one starts with `<`.
    """
    lines = strip_fences(body.split("\n"))
    found: list[str] = []
    idx = 0
    while idx < len(lines) and len(found) < limit:
        stripped = lines[idx].strip()
        if not stripped or stripped.startswith("#"):
            idx += 1
            continue
        if stripped.startswith(NON_PROSE_PREFIX):
            while idx < len(lines) and lines[idx].strip():
                idx += 1
            continue
        collected = []
        while idx < len(lines) and lines[idx].strip():
            collected.append(lines[idx].strip())
            idx += 1
        paragraph = " ".join(collected)
        # Link rows ("Hugging Face | GitHub | Docs") and stray tag attributes are not prose.
        if '="' not in paragraph and paragraph.count("|") < 2:
            found.append(paragraph)
    return found


# Upstream pages open with a boilerplate "Gemma is a family of..." paragraph and only then
# say what the page covers. That second paragraph is the description we actually want.
SCOPE_OPENER = re.compile(r"^This (document|guide|page|tutorial|notebook|section)\b", re.I)


def lead_paragraph(body: str) -> str:
    paragraphs = lead_paragraphs(body)
    for paragraph in paragraphs:
        if SCOPE_OPENER.match(paragraph):
            return paragraph
    return paragraphs[0] if paragraphs else ""


def infer_description(body: str, title: str) -> tuple[str, str]:
    """Return (description, quality) where quality is "derived", "truncated" or "fallback"."""
    paragraph = lead_paragraph(body)
    paragraph = HTML_TAG.sub("", MD_LINK.sub(r"\1", paragraph))
    paragraph = MD_EMPHASIS.sub("", paragraph).replace("\\|", "|").strip()

    if paragraph:
        sentences = SENTENCE_END.split(paragraph)
        text = sentences[0].strip()
        # A single short sentence says nothing useful; pull in the next one.
        for extra in sentences[1:]:
            if len(text) >= DESCRIPTION_MIN:
                break
            text = f"{text} {extra.strip()}".strip()
        if len(text) > DESCRIPTION_MAX:
            trimmed = text[: DESCRIPTION_MAX - 1].rsplit(" ", 1)[0].rstrip(",;:")
            return trimmed + ".", "truncated"
        if len(text) >= DESCRIPTION_MIN:
            return text, "derived"

    return f"Reference documentation for {title} in the Gemma model documentation.", "fallback"


def infer_models(rel: str) -> list[str]:
    """Which models a page is about. An empty list means "not about a model" -- community,
    tooling and site pages, where naming a model would be noise."""
    parts = rel.split("/")

    # `models/<family>/…` names a family; `models/releases.md` sits at the top level and is
    # about the family as a whole, so the directory check has to be real.
    if parts[0] == "models" and len(parts) > 2:
        return [parts[1]]

    if rel == "tutorials/visual-guides/diffusiongemma.md":
        return ["diffusiongemma"]
    if rel == "tutorials/visual-guides/gemma-4-12b.md":
        return ["gemma-4/12b"]

    # Audio is E2B/E4B/12B only -- the large sparse models take text, image and video.
    if rel == "get-started/capabilities/audio.md":
        return list(models_with_modality("audio"))

    if parts[0] in {"get-started", "deployment", "tuning", "tutorials", "learn"}:
        return ["gemma-4"]

    return []


# Engines and libraries a page is *about*, keyed by filename. Pages that merely mention a
# framework in passing are left blank rather than guessed at from body text.
PATH_FRAMEWORKS: dict[str, list[str]] = {
    "deployment/ollama.md": ["ollama"],
    "deployment/llamacpp.md": ["llamacpp"],
    "deployment/lmstudio.md": ["lmstudio"],
    "deployment/mlx.md": ["mlx"],
    "deployment/huggingface.md": ["transformers"],
    "deployment/keras.md": ["keras"],
    "deployment/litert.md": ["litert"],
    "deployment/web.md": ["webgpu", "transformers"],
    "deployment/mobile.md": ["litert", "android", "ios"],
    "deployment/vllm.md": ["vllm"],
    "deployment/sglang.md": ["sglang"],
    "deployment/google-cloud.md": ["google-cloud", "vllm"],
    "deployment/langchain.md": ["langchain", "ollama"],
    "tuning/qlora.md": ["transformers", "peft", "trl"],
    "tuning/full-finetune.md": ["transformers", "trl"],
    "tuning/vision-qlora.md": ["transformers", "peft", "trl"],
    "tuning/keras-lora.md": ["keras"],
    "tuning/distributed.md": ["jax", "keras"],
    "models/embeddinggemma/inference.md": ["sentence-transformers", "transformers"],
    "models/embeddinggemma/fine-tuning.md": ["sentence-transformers"],
    "models/functiongemma/inference.md": ["transformers"],
    "models/functiongemma/fine-tuning.md": ["transformers", "peft", "trl"],
    "models/paligemma/inference.md": ["keras"],
    "models/paligemma/fine-tuning.md": ["jax"],
    "models/recurrentgemma/inference.md": ["jax"],
    "models/recurrentgemma/fine-tuning.md": ["jax"],
    "models/diffusiongemma/inference.md": ["transformers"],
}


def nav_labels() -> dict[str, str]:
    """docs-relative path -> the hand-written nav label, where zensical.toml gives one."""
    config = tomllib.loads((REPO_ROOT / "zensical.toml").read_text(encoding="utf-8"))
    labels: dict[str, str] = {}

    def walk(node, label: str | None = None) -> None:
        if isinstance(node, str):
            if node.endswith(".md") and label and label not in {"Overview", "Index"}:
                labels.setdefault(node, label)
        elif isinstance(node, list):
            # A label attached to a *list* names the section, not any page inside it, so it
            # must not propagate -- that is how `models/releases.md` was titled "Models".
            for item in node:
                walk(item)
        elif isinstance(node, dict):
            for key, value in node.items():
                walk(value, key)

    walk(config["project"]["nav"])
    return labels


# --- Driver -----------------------------------------------------------------------------------


def process(path: Path, labels: dict[str, str]) -> tuple[str, dict[str, object], str]:
    """Return (new text, keys added, description quality)."""
    text = path.read_text(encoding="utf-8")
    rel = path.relative_to(DOCS_DIR).as_posix()
    block, rest = split_block(text)
    segments = segment(block)
    existing = parse(block)

    additions: dict[str, object] = {}
    quality = ""

    title = existing.get("title") or infer_title(rest, rel, labels)
    if "title" not in existing:
        additions["title"] = title

    if "description" not in existing:
        additions["description"], quality = infer_description(rest, title)

    if "models" not in existing:
        inferred = infer_models(rel)
        if inferred:  # an empty list means "not about a model" -- omit the key entirely
            additions["models"] = inferred

    if "frameworks" not in existing and rel in PATH_FRAMEWORKS:
        additions["frameworks"] = PATH_FRAMEWORKS[rel]

    source = ported_sources().get(rel) or existing.get("source_url")
    if source and "source_url" not in existing:
        additions["source_url"] = source

    if not additions:
        return text, {}, ""

    return render(segments, additions) + rest, additions, quality


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="report without writing")
    args = parser.parse_args()

    labels = nav_labels()
    touched = 0
    needs_review: list[tuple[str, str]] = []
    seen_descriptions: dict[str, str] = {}

    for path in sorted(DOCS_DIR.rglob("*.md")):
        rel = path.relative_to(DOCS_DIR).as_posix()
        original = path.read_text(encoding="utf-8")
        new_text, additions, quality = process(path, labels)
        if not additions:
            continue

        # The body must survive untouched -- this script only ever edits the block.
        if split_block(new_text)[1] != split_block(original)[1]:
            print(f"ABORT {rel}: body would change", file=sys.stderr)
            return 1

        touched += 1
        description = str(additions.get("description", ""))
        if quality in {"truncated", "fallback"}:
            needs_review.append((rel, quality))
        elif description and description in seen_descriptions:
            needs_review.append((rel, f"duplicate of {seen_descriptions[description]}"))
        elif description:
            seen_descriptions[description] = rel

        print(f"{'would add' if args.check else 'added':>9}  {rel}: {', '.join(additions)}")
        if not args.check:
            path.write_text(new_text, encoding="utf-8")

    print(f"\n{touched} pages updated; {len(needs_review)} descriptions need a human")
    for rel, why in needs_review:
        print(f"  {rel:55} {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
