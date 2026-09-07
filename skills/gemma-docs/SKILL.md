---
name: gemma-docs
description: Authoring, editing, and maintaining the Google DeepMind Gemma documentation website built with Zensical. Use when adding or updating documentation pages, modifying model/framework metadata, creating tutorials with executable notebooks, styling with design tokens and components, or running verification checks.
---

# Gemma Documentation Authoring & Maintenance

This skill guides coding agents and contributors in maintaining and authoring documentation for the Google DeepMind Gemma documentation repository built on [Zensical](https://github.com/zensical/zensical).

## Core Principles

1. **Single Source of Truth**: A fact belongs in exactly one place under `data/`. Pages and scripts read from that source; never duplicate or hardcode model specs, framework lists, or repeated URLs in pages.
2. **Import Once, Then Hand-Maintain**: Imported pages from upstream sources are recorded in `scripts/importers/sources.toml` and frozen (`frozen = true`). Never bulk-reconvert frozen pages.
3. **Markdown-First Notebooks**: Markdown is the single source of truth for tutorials. Jupyter notebooks (`docs/notebooks/*.ipynb`) are generated build artifacts, never authored directly.
4. **Zero-Warning CI**: All lint checks (`scripts/lint_docs.py`), tests (`tests/`), and builds (`scripts/build_docs.py`) must pass with zero new or unresolved findings.

---

## Repository Structure

```text
├── data/                  # Single sources of truth (YAML)
│   ├── models.yml         # Model families, sizes, context, memory, lineage, checkpoint IDs
│   ├── frameworks.yml     # Supported engines, runtimes, tuning paths, and blurbs
│   ├── links.yml          # Repeated external links (appended to every page via snippets)
│   └── glossary.yml       # Acronym definitions (opt-in abbreviation tooltips)
├── docs/                  # Documentation source pages (Markdown)
│   ├── index.md           # Homepage
│   ├── get-started/       # Quickstart, capabilities, prompt formatting, QAT
│   ├── models/            # Family overviews, model cards, specialized variants
│   ├── deployment/        # Run guides (local, serving, cloud, frameworks)
│   ├── tuning/            # Fine-tuning guides (QLoRA, full, vision, distributed, etc.)
│   ├── learn/             # Task guides and visual walkthroughs
│   ├── resources/         # Tools, token calculator, community, skills
│   ├── contributing/      # Contributor guides: components.md and conventions.md
│   ├── stylesheets/       # CSS tokens and component styles
│   ├── javascripts/       # Interactive scripts (model switcher, generated model data)
│   └── notebooks/         # Exported .ipynb notebooks for Colab
├── overrides/             # HTML templates and partials
├── scripts/               # Build, generation, linting, and importer tooling
│   ├── build_docs.py      # Production build entry point (Zensical + agent routes)
│   ├── lint_docs.py       # Comprehensive documentation and invariant linter
│   ├── agent_docs.py      # Generates llms.txt and raw Markdown routes
│   ├── notebooks.py       # Exports registered Markdown tutorials to .ipynb
│   ├── generate_model_names.py # Compiles model data into JS and HTML partials
│   ├── generate_snippets.py    # Compiles links.md and glossary.md snippets
│   ├── vendor_icons.py    # Vendors SVGs referenced in pages
│   └── frontmatter_schema.py   # Shared schema and validation for YAML frontmatter
├── tests/                 # Unit tests (test_agent_docs.py, test_notebooks.py, etc.)
├── zensical.toml          # Site configuration, navigation tree, and theme settings
└── PLAN.md                # Living project roadmap and architectural guidelines
```

---

## Where Facts Live & How to Update Them

| Data File | Governs | Generator / Consumer |
|---|---|---|
| `data/models.yml` | Model families, parameter sizes, context lengths, memory requirements, framework-specific checkpoint IDs | `scripts/generate_model_names.py` -> `docs/javascripts/model-data.js` and `overrides/partials/model-names.html`; `scripts/notebooks.py` -> `docs/notebooks/*.ipynb` |
| `data/frameworks.yml` | Engine metadata, blurbs, icons, URLs, `tuning_paths` | Rendered by Jinja macros (`{{ engines(...) }}`, `{{ paths(...) }}`) in pages |
| `data/links.yml` | URLs used on more than one page, model version tags | `scripts/generate_snippets.py` -> `snippets/links.md` (auto-appended to all pages) |
| `data/glossary.yml` | Acronyms and term definitions | `scripts/generate_snippets.py` -> `snippets/glossary.md` (opt-in via `--8<-- "glossary.md"`) |

### Regenerating Data Artifacts
Whenever editing files in `data/`, run:
```bash
uv run --frozen python scripts/generate_model_names.py
uv run --frozen python scripts/generate_snippets.py
uv run --frozen python scripts/agent_docs.py
uv run --frozen python scripts/notebooks.py
```
`scripts/lint_docs.py --check generated` will fail if generated files are out of sync with `data/`.

---

## Authoring Documentation Pages

### Frontmatter Conventions
Every page must have valid YAML frontmatter matching `scripts/frontmatter_schema.py`:

```yaml
---
title: Page Title
description: Concise 1-2 sentence description for search and previews.
models: [gemma-4]          # Optional: list of relevant model family IDs
frameworks: [transformers] # Optional: list of relevant framework IDs
requires: audio            # Optional: required input modality (e.g. audio, vision)
colab: https://colab.research.google.com/github/google-gemma/docs/blob/main/docs/notebooks/tutorial-gemma-4-e2b-it.ipynb # Optional: Colab link
prerequisites:             # Optional: list of prerequisite topics
  - get-started/index.md
source_url: "https://..."  # Mandatory on imported pages; omitted on original pages
---
```
- **Authored keys only**: Do NOT add computed keys (e.g. token counts).
- **Vocabularies**: Model and framework identifiers must exist in `data/models.yml` and `data/frameworks.yml`.
- **Modality gating**: Use `requires:` to restrict model selection and notebook export to models supporting that modality (e.g., `audio` excludes 26B/31B).
- **Colab link**: For swappable tutorials, link to the default model notebook (`-gemma-4-e2b-it.ipynb`). The model switcher updates the link dynamically when readers select other models.

### Headings and Anchor Links
- Use standard markdown headings (`# H1`, `## 1. Step`, `### Substep`).
- When referencing numbered sections (e.g. `## 1. Choose a framework`), remember Python-Markdown generates slugs with the number (e.g., `index.md#1-choose-a-framework`).
- Verify relative links and anchors with `uv run --frozen python scripts/lint_docs.py --check internal`.

### Model Switching Code Blocks
To enable interactive model switching in code examples:
```python { .model-swap }
# Use default model name in authoring; the JS switcher replaces it dynamically:
model_id = "google/gemma-4-E2B-it"
```

### Shared Links & Citations
- For URLs referenced on multiple pages, add to `data/links.yml` and reference by ID: `[Link text][link-id]`.
- For model-pinned external links:
  ```markdown
  [Kaggle](https://www.kaggle.com/models/google/gemma-4){ data-model="gemma-4" }
  ```

---

## Markdown-First Notebook Workflows

To create or edit a tutorial that exports to a Jupyter notebook:

1. **Executable Code Cells**: Add `{ .notebook-cell }` to fences that should execute in Jupyter:
   ````markdown
   ```python { .notebook-cell }
   import torch
   from transformers import AutoModelForCausalLM
   ```
   ````
2. **Model-Swappable Cells**: Combine with `{ .model-swap }` to export multi-model notebooks:
   ````markdown
   ```python { .notebook-cell .model-swap }
   model_id = "google/gemma-4-E2B-it"
   ```
   ````
   During export, `scripts/notebooks.py` detects swappable engines in marked blocks and candidate models matching any `requires:` frontmatter. For swappable tutorials, it generates model-specific notebooks named `<stem>-gemma-4-<size>-it.ipynb` (e.g., `thinking-gemma-4-e2b-it.ipynb`, `thinking-gemma-4-12b-it.ipynb`). Non-swappable tutorials generate a single `<stem>.ipynb`.
3. **Display-Only Code**: Normal fences (without `.notebook-cell`) stay in Markdown and become text cells in notebooks.
4. **Outputs**: Fences marked `{ .output }` are omitted during notebook export.
5. **Site-Only Blocks**: Wrap web-only elements (such as interactive widgets or card grids) in skip comments:
   ```markdown
   <!-- notebook:skip-start -->
   {{ engines(["ollama", "vllm"]) }}
   <!-- notebook:skip-end -->
   ```
6. **Registration**: Register new notebook source tutorials in `SOURCES` in `scripts/notebooks.py`.
7. **Colab Link**: For swappable tutorials, add the default model notebook's Colab URL to frontmatter (`colab: .../<stem>-gemma-4-e2b-it.ipynb`). The page context switcher dynamically updates the link when readers select a different model.
8. **Export & Verify**: Run `uv run --frozen python scripts/notebooks.py` to regenerate `docs/notebooks/*.ipynb`, and verify with `uv run --frozen python scripts/notebooks.py --check`.

---

## Design System & Styling Rules

- **Tokens**: All colors, spacing, and border-radii live in `docs/stylesheets/gemma.css`.
  - Brand blue is controlled solely by `--gemma-blue-seed: #0B57D0`.
  - Test new seeds with `uv run --frozen python scripts/check_contrast.py --seed "#0B57D0"`.
  - **No raw hex colors** are allowed in any stylesheet other than `gemma.css` (enforced by `scripts/lint_docs.py --check css`).
- **Components**: Use existing components from `docs/contributing/components.md` (cards, engine grids, steps lists, tabs, admonitions).

---

## Verification & CI Workflow

Always run the full test and lint suite before submitting changes:

```bash
# 1. Run unit tests
uv run --frozen python -m unittest discover -s tests -p "test_*.py"

# 2. Run documentation linter across all 15 checks
uv run --frozen python scripts/lint_docs.py --baseline

# 3. Build documentation site and generate raw agent routes / llms.txt
uv run --frozen python scripts/build_docs.py
```

### Individual Linter Checks
Run specific checks during development:
```bash
uv run --frozen python scripts/lint_docs.py --check internal,links,frontmatter,orphans
```
Available checks: `links`, `references`, `icons`, `images`, `internal`, `stubs`, `frontmatter`, `orphans`, `fences`, `model-swap`, `admonitions`, `size`, `modeldata`, `css`, `generated`.
