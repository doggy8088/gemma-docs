# Gemma Docs — Port & Re-imagination Plan

Porting and re-imagining the Gemma documentation onto [Zensical](https://github.com/zensical/zensical).
Source: <https://ai.google.dev/gemma/docs> (append `.md.txt` to any page for raw markdown).

**Goal:** the best open-weights developer experience on the web. Not just accurate reference —
an experience that makes people *want* to build with Gemma, welcomes newcomers, and gives
partners a place to showcase their work.

**Gemma 4 is front and center.** All other models are documented, but the docs are
implicitly about Gemma 4 unless stated otherwise.

---

## Working agreement

- **Bite-sized steps.** Each step below is independently testable and roughly one session of work.
- **Every step ends with a verification command** and a "done when" condition. Run it before moving on.
- **Maarten does all commits and pushes.** Agents stage nothing and push nothing — finish the step,
  report what changed, and stop.
- **Handover-friendly.** Each step is self-contained enough that a fresh agent can pick it up
  from this file alone. If you complete a step, tick its box and note anything surprising
  under *Notes* at the bottom.
- **Foundation before fun.** Phases 0–2 must be complete and clean before Phase 4 work starts.
  A WebGPU playground next to 34 "Content coming soon" pages reads as a prototype, not a product.

### Content fidelity — non-negotiable

**Ported content must not change.** The original documentation has been validated for accuracy;
that validation is only transferable if the words survive the port intact.

- A port is a **format** migration, not an editorial one. Convert devsite syntax to Zensical
  syntax, fix links and image paths, repair converter damage — and change nothing else.
- Do **not** reword, tighten, restructure, "improve", modernize, or re-order ported prose.
  Do not silently correct something that looks wrong: flag it under *Notes* and ask.
- Code snippets are copied verbatim. Changing a flag or an argument invalidates the validation.
- **New pages may remix.** Combining snippets from several source pages into a new page is fine
  and encouraged — but the snippets themselves are still copied, not rewritten.
- **Start with a straight port.** Port a page as-is first, confirm it renders well, and only
  then consider whether it needs a companion page. Don't port and redesign in one motion.
- Every ported page records its source in frontmatter (`source_url`, `ported_at`) so any page
  can be diffed against upstream later.

Rule of thumb: if a diff against the source `.md.txt` shows a prose change, it needs a reason
written down.

Local preview:

```bash
uv run zensical serve
```

---

## Current state (audit, Aug 2026)

| Metric | Value |
|---|---|
| Markdown files | 80 |
| Nav entries | 76 |
| Real ported content (>1.5 KB) | 44 |
| One-line stubs (<250 B) | 34 |
| Pages with frontmatter | 9 / 80 |
| Links still pointing at `ai.google.dev/gemma/docs` | 122 |
| Images hotlinked to `ai.google.dev/static` | 18 (0 vendored) |
| Orphan pages (exist, not in nav) | 4 |
| Orphan stylesheet (exists, not in `extra_css`) | 1 (`custom.css`) |

Known defects:

- Malformed code fences from the notebook converter — e.g. `docs/models/gemma-4/index.md:117`
  wraps prose in a ```` ```python ```` fence mid-sentence.
- 7 prompt-template blocks tagged `python` that are plain text.
- ~16 unconverted devsite `Note:` / `Key Point:` / `Caution:` callouts.
- `docs/models/paligemma/fine-tuning.md` is 435 KB (base64 images inlined by the converter).
- `docs/community/gemmaverse.md` still ships `example.com` placeholder cards.
- Modality data disagrees across `quickstart.md`, `models/index.md:14`, and `models/gemma-4/index.md:45`.

---

## Target navigation

Eight top-level tabs. Capabilities stay under **Get started** — they are fundamental Gemma 4
capabilities and belong in one place where the full picture is visible.

```
Home
Get started    Quickstart · Install & setup · Choose your model
               Prompt & chat format
               Capabilities: Text · Thinking · Function calling
                             Vision (Image, Video) · Audio
                             Multi-Token Prediction · Long context
Models         Gemma 4 (overview, model card, benchmarks)
               Variants: Gemma 3n · DiffusionGemma · FunctionGemma
                         EmbeddingGemma · PaliGemma · ShieldGemma · CodeGemma
               Previous generations: Gemma 3 · 2 · 1
               Research: RecurrentGemma · DataGemma · Gemma Scope
               Releases
Run            Local:  Ollama · llama.cpp · LM Studio · MLX · Transformers
                       LiteRT · WebGPU · Mobile
               Serve:  vLLM · SGLang · TGI · Google Cloud (GKE / Cloud Run)
               Quantization & QAT · Speculative decoding
Fine-tune      Overview · QLoRA · Full fine-tune · Vision QLoRA
               Keras LoRA · Distributed · Dataset prep · Evaluation
Learn          Visual guides · Tutorials · Notebooks · Videos
Demos          Interactive tools · Apps & Spaces
Ecosystem      Partners · Gemmaverse · Discord · Skills · Contribute
```

**Models stays model-card-centric.** It closely mirrors the upstream structure on purpose —
model cards are canonical reference and shouldn't be creatively restructured. The improvement
is in presentation, consistency, and cross-linking, not reorganization.

**Demos absorbs Tools.** "Interactive tools" = things we built (VRAM calculator, model selector,
tokenizer playground). "Apps & Spaces" = embedded HF Spaces and showcase apps. This gives Demos
real substance instead of one stub page.

Decisions made: no "Migrating from Gemma 3 → 4" page (users swap one model string).
Visual guides get fully ported pages, not outbound links.

---

## Design system

Standardized components throughout — styled boxes, frames, and animations reused everywhere
rather than invented per page.

**Official Gemma palette:**

| Token | Hex | Use |
|---|---|---|
| `--gemma-blue` | `#3186FF` | Primary — links, active nav, primary buttons |
| `--gemma-blue-deep` | `#346BF1` | Hover/pressed states, gradient anchor |
| `--gemma-blue-light` | `#4FA0FF` | Accents, highlights, dark-mode primary |
| `--gemma-white` | `#FFFFFF` | Light surface |
| `--gemma-ink` | `#202124` | Dark surface, body text on light |

All new CSS derives from these. No ad-hoc hex values outside the token file.

---

## Phase 0 — Foundation: make the existing content correct

*Nothing here is glamorous. Everything else compounds on top of it, and CI checks stop the
debt from coming back.*

### Step 0.1 — Repo hygiene and CI guardrails ✅
- [x] `scripts/lint_docs.py` with 9 checks: `links`, `images`, `internal`, `stubs`,
      `frontmatter`, `orphans`, `fences`, `admonitions`, `size`.
- [x] Baseline mechanism (`.lintbaseline.json`) so CI fails only on *new* violations while the
      backlog is burned down. **Shrinking the baseline to zero is the definition of done for Phase 0.**
- [x] `.github/workflows/lint.yml` — runs on push, PR, and dispatch. Stdlib-only, no install step.
- [x] Deleted `docs/stylesheets/custom.css` (dead: not in `extra_css`, fully superseded by
      `layout.css`, and would have conflicted on `.md-grid` max-width and sidebar width).
- **Verify:** `uv run python scripts/lint_docs.py --baseline` → exit 0.
      `uv run python scripts/lint_docs.py` → full 300-finding backlog.
- **Done when:** ✅ regression test confirmed — adding an upstream link exits 1, reverting exits 0.

**Baseline at 2026-08-03 (the Phase 0 burn-down list):**

| Check | Count | Notes |
|---|---|---|
| `frontmatter` | 81 | 71 pages with no block + missing keys elsewhere |
| `links` | 122 | 5 of these are inside code blocks |
| `images` | 52 | 18 hotlinked, 34 inline base64 (all in `paligemma/fine-tuning.md`) |
| `stubs` | 33 | |
| `fences` | 7 | prompt templates tagged `python` |
| `orphans` | 4 | |
| `size` | 1 | `paligemma/fine-tuning.md` at 435 KB |
| `internal` | 0 | all relative links resolve |
| `admonitions` | 0 | **converter already handles these correctly** — 36 converted |
| **Total** | **300** | |

Useful commands:

```bash
uv run python scripts/lint_docs.py --list
```

```bash
uv run python scripts/lint_docs.py --check links,images
```

```bash
uv run python scripts/lint_docs.py --write-baseline
```

### Step 0.2 — Fix the notebook converter ✅
- [x] **Wrapped list continuations** no longer eaten as code. Devsite wraps long bullets with an
      indented continuation line; the 4-space rule fenced it as `python`. The guard meant to
      prevent this (`lines[i-1].strip().startswith("    ")`) could never fire — `.strip()`
      removes the very indentation it tests. Continuations are now rejoined onto their bullet,
      which restores the original sentence exactly (markdown renders `foo\n  bar` in a list as
      `foo bar`, so nothing about the content changes).
- [x] **Language detection** gained a `text` fallback: `bash` → `python` → `text`. Detection now
      uses Python *syntax* signals, deliberately not bare English words like "for"/"with"/"from",
      which is why prompt templates were being classed as code.
- [x] **Indented bullets no longer split a block.** A bullet may not *start* a block (that would
      eat indented lists), but once a block is underway an indented bullet is block content —
      previously one ASR prompt became a fence plus orphaned bullets.
- [x] **Base64 extraction** via `--assets-dir` / `--assets-url`. Anchored on `![...](...)` so a
      data URI inside real code is left alone.
- [x] Golden tests: `scripts/test_convert_notebook.py` + 4 fixtures in `scripts/fixtures/`,
      each a minimised reproduction of a real defect. 11 tests, wired into CI.
- **Verify:** `uv run python scripts/test_convert_notebook.py` → 11 passing.
- **Done when:** ✅ re-converting `core.md.txt` produces the QAT bullets intact with **zero**
      python fences; `audio.md.txt` yields 8 `python`, 2 `bash`, 2 `text` fences, both ASR
      prompts correctly `text`.

**Content-fidelity check** (word multiset, source vs. output):

| Source | Dropped | Verdict |
|---|---|---|
| `core.md.txt` | `NOTE` ×1 | the `> [!NOTE]` → `!!! Note` conversion. Additions are cookbook-card markup. |
| `audio.md.txt` | 111, all badge/metadata | `colab_logo`, `notebook-site-button.png`, `View`, `Run`, `<br>` — the intended stripping. |

**Zero prose lost in either.** Regenerate goldens with `--update`, then review the diff —
those files are the specification.

```bash
uv run python scripts/test_convert_notebook.py
```

### Step 0.3 — Re-port Models, targeted repair for Get Started

Two different strategies, because the two sections have different histories:

- **Models is a straight port.** No hand-authoring, so re-converting is safe and correct. ✅ Done.
- **Get Started is hand-edited and reviewed.** `quickstart.md` has a hand-authored model table,
  the `npx skills add` admonition, and `hide: toc` frontmatter — none of it in upstream's
  `get_started.md.txt`. Re-converting would destroy that. Repair defects in place.

#### 0.3a — Porting infrastructure ✅
- [x] `scripts/sources.toml` — manifest mapping each ported page to its upstream source.
      **A page absent from the manifest must never be re-converted.** Also the origin of the
      `source_url` frontmatter in Step 3.1.
- [x] `scripts/check_fidelity.py` — word-multiset comparison proving a conversion changed markup
      and nothing else. Compares *source minus what the converter is designed to drop* vs. output,
      reusing the converter's own `is_badge_line` rather than a parallel word blocklist, so the
      two can't drift apart.
- [x] `scripts/port_docs.py` — batch porter. **Fidelity-gated: a page that drops prose is never
      written.** Caches sources under `.sources/` (gitignored).
- [x] New converter fix: **consecutive blank lines no longer split a code block.** Notebook
      exports routinely leave two blank lines mid-cell; the old lookahead checked only one line
      ahead, so one function became two fences (found at `paligemma/inference.md`, `binary_mask`).
- [x] New linter check: two same-language fences separated only by blank lines — the signature of
      a split block. Found **9 across 7 files**, well beyond the one reported.

#### 0.3b — Models re-ported ✅
- [x] All **37** manifest pages re-ported. Fidelity: 37 faithful, 0 with dropped prose.
      Re-running is idempotent (all "unchanged").
- [x] `paligemma/fine-tuning.md`: **435 KB → 51 KB**, 34 images extracted to `docs/assets/paligemma/`.
- [x] `models/releases.md`: 58-byte stub → full port (+4.2 KB).
- **Verify:** `uv run python scripts/port_docs.py --filter models/ --dry-run` → 37 ok, 0 failed.

#### 0.3c — Get Started targeted repair ✅

Diffing each page against a fresh conversion first confirmed both carry real hand-edits, so
repair-in-place was the correct call:

| Page | Hand-edits preserved |
|---|---|
| `capabilities/image.md` | title recased, `## Summary` edit, `<!-- TODO: Port over -->` block |
| `capabilities/audio.md` | hand-added `# Audio Understanding`, `## Summary` edit, TODO block |

- [x] `capabilities/audio.md` — two ASR prompt templates `python` → `text`.
- [x] `capabilities/audio.md:81` — unconverted `> > [!NOTE]` callout → `!!! Note`.
- [x] `capabilities/image.md` — three split blocks rejoined. Diff is exactly 6 lines
      (the fence pairs); no content touched.
- [x] New linter coverage: GitHub-style `> [!NOTE]` callouts. The `admonitions` check only
      knew about devsite `Note:` prefixes, so this defect was invisible.
- **Verify:** `uv run python scripts/lint_docs.py --check fences,admonitions` → zero.
- **Done when:** ✅ zero fence and admonition findings site-wide; remaining diff vs. a fresh
      conversion is hand-edits only.

**Technique worth reusing:** diffing a page against a fresh conversion cleanly separates
*bugs* from *hand-edits*. Do this before touching any page whose history is unclear — it is the
manual form of the `--divergence` check and it is what proved re-porting would have been wrong here.

### Phase 0 progress: 300 → 285

| Check | Start | Now | |
|---|---|---|---|
| `fences` | 7 | **0** | ✅ cleared |
| `size` | 1 | **0** | ✅ cleared |
| `admonitions` | 0 | **0** | ✅ 1 found and fixed |
| `internal` | 0 | **0** | ✅ |
| `images` | 52 | **18** | base64 done; 18 hotlinked remain → Step 0.4 |
| `stubs` | 33 | **32** | → Steps 2.2–2.4 |
| `orphans` | 4 | **4** | → Step 2.1 |
| `frontmatter` | 81 | **81** | → Step 3.1 |
| `links` | 122 | **150** | +28 from `releases.md` gaining content → Step 0.5 |

**Backlog after the Models port — 300 → 290:**

| Check | Was | Now | |
|---|---|---|---|
| `images` | 52 | **18** | 34 base64 extracted |
| `size` | 1 | **0** | cleared |
| `fences` | 7 | **5** | model-card fences fixed; 5 left, all in Get Started |
| `stubs` | 33 | **32** | `releases.md` now real content |
| `links` | 122 | **150** | +28 from `releases.md` gaining content — Step 0.5 rewrites them |
| `frontmatter` / `orphans` / `internal` / `admonitions` | 81 / 4 / 0 / 0 | unchanged | later steps |

```bash
uv run python scripts/port_docs.py --filter models/ --dry-run
```

```bash
uv run python scripts/check_fidelity.py --manifest --cache-dir .sources
```

### Step 0.4 — Vendor all images ✅
- [x] `scripts/vendor_images.py` — downloads remote images into `docs/assets/<section>/` and
      rewrites the references. Section = the page's parent directory, matching the layout
      `port_docs.py` already uses. Original filenames are kept so any asset traces back to its
      upstream URL. A failed download leaves the reference untouched (a broken local path is
      worse than a working remote one) and the script is idempotent, so re-running retries.
- [x] All **18** hotlinked images vendored, 18/18 succeeded. `docs/assets/` is now 8.2 MB.
- [x] Fidelity checker taught to ignore image *targets* on both sides — only alt text is prose.
      Without this, vendoring read as 6 pages losing prose and would have **blocked those pages
      from ever being re-ported**, since `port_docs.py` is fidelity-gated.
- [ ] ~~Dark-mode variants~~ — **deferred to Phase 1** with the rest of the design system.
- **Verify:** ✅ zero `ai.google.dev/static` references in the built `site/`; all 214 local
      `<img>` sources resolve on disk.
- **Done when:** ✅ the site renders fully with no network access to Google.

!!! warning "Ordering: port, then vendor"
    Conversion restores the upstream image URLs, so **re-porting a page undoes its vendoring**
    (confirmed: re-porting `models/gemma-4/model-card.md` would revert one image, +21 bytes).
    Always run `vendor_images.py` after `port_docs.py`. `port_docs.py` prints this reminder, and
    since the `images` check now sits at zero, a missed run fails CI.

```bash
uv run python scripts/vendor_images.py --dry-run
```

### Step 0.4b — CI hardening (zizmor / org policy) ✅

The org runs [zizmor](https://docs.zizmor.sh/) on workflows and **fails on any Medium/High
finding**, not just the mandatory-check list. Both workflows now pass clean.

- [x] All 6 action references pinned to commit SHAs in `lint.yml` and `deploy.yml`
      (`unpinned-uses`, High — this is what broke the build).
- [x] `lint.yml` given an explicit `permissions: contents: read` block.
- [x] `deploy.yml` permissions narrowed: `contents: read` at workflow level, with
      `pages: write` / `id-token: write` scoped to the `deploy` job alone. At workflow level they
      were also being handed to the `build` job, which only needs to read (`excessive-permissions`, High).
- [x] `persist-credentials: false` on both checkouts (`artipacked`).
- **Verify:** `uvx zizmor --persona=regular .github/workflows/` → "No findings to report."

!!! note "Updating a pin"
    Pins carry a `# v4` comment so the intended version stays readable. To move one:

    ```bash
    gh api repos/actions/checkout/git/ref/tags/v4 --jq '.object.sha'
    ```

    For annotated tags, dereference: `gh api repos/<repo>/git/tags/<sha> --jq '.object.sha'`.

### Step 0.5 — Rewrite internal links
- [ ] Map all 122 `ai.google.dev/gemma/docs/*` links to local paths.
- [ ] For pages not yet ported, point at the eventual local path and let the linter flag it as
      a known-pending target (don't leave live external links).
- **Verify:** `uv run python scripts/lint_docs.py --check links`
- **Done when:** zero outbound links to the site we're replacing.

### Step 0.6 — Model capability matrix as single source of truth
- [ ] Create `data/models.yml`: every Gemma 4 size with parameters, modalities in/out, context
      window, memory at each quantization, MTP support, target platform.
- [ ] Resolve the modality contradiction (`quickstart.md` vs `models/index.md:14` vs
      `models/gemma-4/index.md:45`) — confirm 26B A4B and 31B are text+image only.
- [ ] Generate the model tables in `quickstart.md`, `models/index.md`, and
      `models/gemma-4/index.md` from this file via the macros plugin.
- **Verify:** change one value in `data/models.yml`, rebuild, confirm all three tables update.
- **Done when:** no hand-maintained model spec table remains anywhere.

---

## Phase 1 — Design system

### Step 1.1 — Token layer

**Stylesheets stay modular — many small files, not one monolith.** Each file owns one concern,
opens with a comment saying what it owns and what it must not touch, and stays small enough for
an agent to read end-to-end before editing. Adding a seventh or eighth stylesheet is the right
move whenever a new concern appears; growing `custom.css` into a 900-line catch-all is not.

Target layout (`docs/stylesheets/`):

| File | Owns |
|---|---|
| `gemma.css` | Palette + spacing/radius/shadow scale. **The only file with raw hex values.** |
| `layout.css` | Page grid, sidebars, content width |
| `components.css` | Cards, callouts, badges, figures, walkthroughs |
| `code.css` | Code blocks, copy buttons, tabs |
| `tokens.css` | Chat-token badges (`<start_of_turn>` etc.) |
| `home.css` | Landing page only |
| `motion.css` | Animations + `prefers-reduced-motion` |

- [ ] Create `gemma.css` with the palette above as CSS custom properties, light and dark
      variants, and the spacing/radius/shadow scale.
- [ ] Rebind Zensical's `--md-*` primary/accent variables to the Gemma tokens.
- [ ] Refactor the existing sheets to consume tokens — remove hardcoded hex values.
- [ ] Resolve `custom.css` (currently not in `extra_css`): fold into `layout.css` or delete.
- [ ] Keep `extra_css` in `zensical.toml` in the same order as the table above — token layer first.
- **Verify:** visual pass on 3 pages in both light and dark mode.
- **Done when:** `grep -E '#[0-9a-fA-F]{6}' docs/stylesheets/*.css` only matches `gemma.css`,
  and no stylesheet exceeds ~250 lines.

### Step 1.2 — Standard component library
- [ ] Card grid (generalize the existing `cookbook-card`).
- [ ] Callout/box variants: note, tip, warning, "requires GPU", "Gemma 4 only".
- [ ] Model badge chips (size, modality, quantization) driven by `data/models.yml`.
- [ ] Framed figure component with caption and light/dark image swap.
- [ ] Step-by-step / numbered walkthrough component.
- [ ] Document every component in `docs/contributing/components.md` with live examples.
- **Verify:** the components page renders all variants correctly in both themes.
- **Done when:** new pages can be built from components without writing new CSS.

### Step 1.3 — Motion
- [ ] Standardized entrance/hover animations, `prefers-reduced-motion` respected throughout.
- [ ] Apply consistently to cards, callouts, and nav.
- **Verify:** enable reduced-motion in the OS; confirm animations disable.

---

## Phase 2 — Navigation & content completion

### Step 2.1 — Restructure nav
- [ ] Rewrite the `nav` block in `zensical.toml` to the target structure above.
- [ ] Resolve the 4 orphans: fold `get-started/index.md` into `quickstart.md`,
      make `capabilities/vision.md` a real parent page, add `models/index.md` and
      `tools/index.md` to nav.
- **Verify:** `uv run python scripts/lint_docs.py --check orphans`
- **Done when:** zero orphans, zero broken nav entries, 8 top-level tabs.

### Step 2.2 — Local engine pages (the differentiator)

These are *written*, not ported — upstream's versions are three paragraphs each. One step per
engine so they can be tested and handed over independently.

Template each page must follow: one-line install → one-line run → model tags for every
size × quant → server mode with OpenAI-compatible endpoint → multimodal examples (gated by
actual model support) → context/KV-cache flags → troubleshooting → measured tok/s on named hardware.

- [ ] 2.2a Ollama
- [ ] 2.2b llama.cpp
- [ ] 2.2c LM Studio
- [ ] 2.2d MLX
- [ ] 2.2e Transformers
- [ ] 2.2f vLLM
- [ ] 2.2g SGLang
- [ ] 2.2h LiteRT / mobile / WebGPU
- [ ] 2.2i Google Cloud (GKE, Cloud Run)
- **Verify per page:** every command runs as written on a clean machine.
- **Done when:** the page is genuinely more useful than the upstream equivalent.

### Step 2.3 — Fill remaining stubs
- [ ] Fine-tune pages (QLoRA, full FT, vision QLoRA, Keras LoRA, distributed).
- [ ] Quantization & QAT page — move the QAT content currently buried in
      `models/gemma-4/index.md` into its own page under Run.
- [ ] Multi-Token Prediction page — port `mtp/overview` + `mtp/mtp`. **Biggest content gap:**
      MTP is a flagship Gemma 4 feature with zero documentation today.
- [ ] Ecosystem pages — replace the `example.com` placeholders in `gemmaverse.md` with real
      data-driven cards.
- **Verify:** `uv run python scripts/lint_docs.py --check stubs` → zero.

### Step 2.4 — Port remaining upstream pages
- [ ] CodeGemma (5 pages, currently absent from nav entirely).
- [ ] `setup`, `run`, `gemma_chat`, `gemma_library`, `keras_inference`, `pytorch_gemma`,
      `huggingface_inference`, `lora_tuning`, `prompt-structure`.
- [ ] Application tutorials → Learn: `business-email-assistant`, `personal-code-assistant`,
      `meme-generator`.
- [ ] Decide case-by-case on: `gemma-aps`, `spoken-language/task-specific-tuning`,
      `conversions/hf-to-mediapipe-task`, `mobile-actions`.

---

## Phase 3 — Agent-first infrastructure

*The headline requirement from the original plan, currently ~11% built.*

### Step 3.1 — Frontmatter schema
- [ ] Define and document the schema: `name`, `title`, `description` (richer than llms.txt),
      `estimated_tokens`, `prerequisites`, `frameworks`, `code_languages`, `models`, `last_updated`,
      plus `source_url` and `ported_at` on every ported page (see *Content fidelity*) so any page
      can be diffed against upstream later.
- [ ] Backfill all 80 pages.
- [ ] Enforce in `scripts/lint_docs.py`.
- **Verify:** linter passes with the frontmatter check enabled.

### Step 3.2 — Computed metadata
- [ ] Build hook computing `estimated_tokens`, `code_languages`, and `last_updated`
      automatically so they can't drift.
- **Verify:** add a code block to a page, rebuild, confirm metadata updates.

### Step 3.3 — Machine-readable surfaces
- [ ] Generate `/llms.txt` (index with descriptions) and `/llms-full.txt` (everything).
- [ ] Serve raw markdown at `<page>.md` for every page.
- **Verify:** `curl` the built site for `llms.txt` and one `.md` endpoint.

### Step 3.4 — Copy-for-agent affordances
- [ ] "Copy page as Markdown" button in the page header.
- [ ] "Open in Claude / ChatGPT" links.
- **Verify:** click the button, paste, confirm clean markdown with frontmatter.

### Step 3.5 — Docs as a skill
- [ ] Package the docs as an installable skill (`npx skills add ...`) and/or MCP server so
      agents can query them directly.

---

## Phase 4 — Interactive tools

### Step 4.1 — VRAM & context calculator
Replace the static table in `docs/tools/vram-calculator.md` with a real calculator.

- [ ] Inputs: model, quantization, context length, batch size, KV-cache dtype, GPU preset
      (4090, 5090, A100, H100, M-series unified memory, …).
- [ ] Outputs: fits / doesn't fit with headroom, estimated tok/s, recommended engine,
      **and the exact copy-paste command for that engine**.
- [ ] URL-encoded state (`?model=12b&quant=q4_0&gpu=rtx4090&ctx=32k`) so results are shareable.
- [ ] Reads from `data/models.yml`.
- **Verify:** results match the memory table in `models/gemma-4/index.md` at matching settings.
- **Done when:** a shared URL reproduces the exact same state.

### Step 4.2 — Global engine + model switcher

The highest-leverage UX win: pick "MLX" once and the entire site shows MLX; pick "E4B" and
every snippet on every page rewrites its model ID.

- [ ] Header control for engine and model, persisted in `localStorage`.
- [ ] Extend Zensical's `content.tabs.link` to sync site-wide.
- [ ] **Capability-aware gating, driven by `data/models.yml`:** audio examples must not offer
      26B A4B or 31B (text+image only). When the selected model can't do what the page shows,
      show an inline explanation and the nearest valid model — never a broken snippet, and
      never silently swap the user's choice.
- [ ] Graceful fallback when a model/engine pair has no published weights.
- **Verify:** select 31B, navigate to `capabilities/audio.md`, confirm the gating message
      appears and no invalid command is shown.
- **Done when:** no combination of switcher state produces a command that would fail.

### Step 4.3 — "Choose your model" decision flow
- [ ] Four questions → recommendation + copy-paste command for the user's engine.
- [ ] Embeds the VRAM calculator.

### Step 4.4 — Tokenizer & chat-template playground
- [ ] Paste a conversation → exact `<start_of_turn>` token stream, token count.
- [ ] Image token budget for a given resolution and aspect ratio.
- [ ] Reuses the existing `tokens.css` badge styling.
- Pairs directly with `get-started/prompt-formatting.md`.

---

## Phase 5 — Demos & embeds

### Step 5.1 — First HF Space embed (spike)
- [ ] Embed <https://huggingface.co/spaces/google/gemma4_vision_token_budget> on a Demos page.
- [ ] Lazy-load (Spaces are heavy) and design a fallback for sleeping Spaces — a dead iframe
      looks worse than no iframe.
- [ ] Evaluate: sizing, theme match, mobile behavior, load time.
- **Verify:** view on mobile and desktop, light and dark. Test with the Space asleep.
- **Done when:** we know whether to standardize on iframe or `<gradio-app>`, documented here.

### Step 5.2 — Demos section build-out
- [ ] Concurrent inference demo (currently a stub).
- [ ] Additional Spaces once 5.1 establishes the pattern.

---

## Phase 6 — Learn

### Step 6.1 — Visual guides, fully ported
Not outbound links — complete pages hosted here. Maarten provides raw SVG for all images.

- [ ] 6.1a A Visual Guide to Gemma 4
- [ ] 6.1b A Visual Guide to Gemma 4 12B
- [ ] 6.1c A Visual Guide to MTP for Gemma 4
- [ ] 6.1d A Visual Guide to DiffusionGemma
- [ ] Establish the reading experience: wide layout, framed figures, dark-mode SVG handling,
      canonical link back to the newsletter.
- [ ] **Contextual cross-linking** — link the MoE guide from the 26B A4B section, the
      Quantization guide from QAT, the Reasoning LLMs guide from `capabilities/thinking.md`.
      Related guides worth linking (not porting): MoE, Quantization, Reasoning LLMs, LLM Agents.
- **Verify:** SVGs render crisply at all widths, legible in both themes.

### Step 6.2 — Notebooks as source of truth
- [ ] Keep `.ipynb` canonical, render to pages at build time.
- [ ] Colab / Kaggle / local badges on every tutorial.
- [ ] Execute notebooks in CI — a "snippets are tested" badge is a real credibility marker.

### Step 6.3 — Task-first entry points
- [ ] "I want to…" layer: build a RAG app · extract structured JSON · caption images ·
      classify content · build a tool-using agent. Each is a curated path across existing pages.

---

## Phase 7 — The wow tier

Only after Phases 0–2 are clean.

- [ ] **In-browser playground** — E2B q4 via WebGPU/transformers.js, running in the docs page.
      ~1.5–2 GB download, so explicit opt-in and clear progress UI. "Try Gemma 4 with zero
      install, right here" is unmatched for an open-weights model.
- [ ] **Ask Gemma about Gemma** — docs chatbot on transformers.js or an HF Space endpoint.
      Doubles as a live demo and a search upgrade.
- [ ] **Interactive Gemma 4 architecture explainer** — clickable/scrollytelling coverage of PLE
      (why E2B's memory exceeds its effective params), MoE routing on 26B A4B, MTP draft heads,
      and the 12B's encoder-free linear projections. These four concepts are exactly what
      confuses people, and prose isn't fixing it.
- [ ] **MTP animated explainer** — draft-vs-verify animation with an acceptance-rate slider
      showing the resulting speedup.
- [ ] **DiffusionGemma denoising animation** — text unmasking step by step.
- [ ] **Community throughput leaderboard** — measured tok/s per (model × quant × engine ×
      hardware), PR-based submissions. Exists nowhere authoritative; high maintenance, high payoff.
- [ ] **Cost calculator** — GPU $/hr → $/1M tokens, self-host vs. API.

---

## Phase 8 — Maintainer skill & handover

*Finalized once the docs are done, but **drafted incrementally** — every phase that establishes
a convention adds its section immediately, while the reasoning is fresh. A skill written from
scratch at the end will miss the things that were obvious in the moment.*

### Step 8.1 — `gemma-docs` authoring skill
A skill that lets any agent update these docs correctly without rediscovering the conventions.
Must cover:

- [ ] **Content fidelity rules** — the non-negotiable section above, stated first and plainly.
- [ ] **How to port a page** — fetch `.md.txt`, run `scripts/convert_notebook.py`, what the
      converter handles vs. what needs manual repair, required frontmatter including `source_url`.
- [ ] **How to add a page** — where it goes in nav, frontmatter schema, which components to use.
- [ ] **The design system** — which stylesheet owns what, the token layer, the rule that raw hex
      only lives in `gemma.css`, when to add a new stylesheet instead of growing one.
- [ ] **Component catalog** — every card/callout/badge/figure with copy-paste examples.
- [ ] **`data/models.yml`** — that it's the single source of truth, what reads from it
      (tables, VRAM calculator, switcher gating), and why hand-editing a model table is a bug.
- [ ] **Verification** — `scripts/lint_docs.py` checks, how to run the build, what CI enforces.
- [ ] **Gotchas** — Zensical version quirks, features in `zensical.toml` that aren't implemented,
      converter edge cases, anything accumulated in *Notes* below.
- [ ] **What not to do** — don't reword ported prose, don't hotlink images, don't add hex outside
      the token file, don't commit or push (that's Maarten's).
- **Verify:** hand a fresh agent the skill and a page to port. It should succeed with no
  additional context and pass the linter first try.
- **Done when:** that test passes.

### Step 8.2 — Contributor docs
- [ ] `docs/contributing/` covering the same ground for humans, linking the components page
      from Step 1.2.

---

## Open decisions

- Whether visual-guide diagrams can also appear inline in the reference docs (with attribution),
  which would be a much stronger integration than standalone guide pages alone.
- Whether the throughput leaderboard is worth its ongoing maintenance cost.

---

## Pipeline order

Steps that rewrite pages must run in this order, because each stage can undo the previous one:

```
port_docs.py  ->  vendor_images.py  ->  lint_docs.py --baseline
```

`check_fidelity.py --manifest` can run at any point; it ignores image targets by design.

## Notes

*Agents: record surprises, deviations, and anything that changes the plan here.*

- Zensical version pinned via `uv.lock` (`zensical>=0.0.51`). Some Material features listed in
  `zensical.toml` may not be implemented yet — verify before relying on them.

**2026-08-03 (Step 0.1)**

- ⚠️ **Two workflows both deploy to GitHub Pages on push to `main`** — `docs.yml` and
  `deploy.yml`. They will race; `deploy.yml` has a `concurrency: pages` group, `docs.yml`
  doesn't. They also install zensical differently (`docs.yml` → plain PyPI, `deploy.yml` → uv +
  `pypi.zensical.org`). **Needs a decision from Maarten** — `deploy.yml` looks like the intended
  one. Left both in place rather than deleting unilaterally.
- **Correction to the original audit:** devsite callouts are *not* an issue. The converter
  already converts them (36 admonitions present, 0 unconverted). The earlier "~16" figure came
  from a combined grep pattern that also matched `.md.txt` and `colab.research` strings.
  Step 0.2's admonition work is therefore already done — verify with a fixture rather than rewrite.
- `check_images` and `check_internal` scan **outside code blocks only**. Markdown-like syntax
  inside snippets is code, not a reference (e.g. an f-string `{render_inline(image, resize=(64,64))}`
  in `paligemma/fine-tuning.md:515` was a false positive).
- `check_links` scans **everywhere**, including code blocks — 5 upstream links live inside
  snippets. Some may be intentional (e.g. downloading a sample asset); review case by case in Step 0.5.
- `Finding.key()` deliberately excludes line numbers so edits above a known finding don't
  resurrect it as "new".
- The 34 base64 images are all in `paligemma/fine-tuning.md`. Fixing that one file clears the
  `size` check and 34 of the 52 `images` findings at once — good first target for Step 0.3/0.4.

**2026-08-03 (Step 0.2)**

- ⚠️ **Pages have been hand-edited since their port.** There is currently no way to tell a
  straight port from a hand-edited page automatically — hence `--divergence` in Step 0.3 and
  `source_url` frontmatter in Step 3.1. Until one of those lands, **assume any page may contain
  hand-work and never re-convert in bulk.**
- **Get Started is reviewed and approved by Maarten.** Repair defects in place; do not re-convert.
- Only 6 files have converter-class defects. The other 294 backlog findings are links, images,
  frontmatter, and nav — all outside the converter's remit.
- The converter's `text` fallback matters more than it looks: an early version of the Python
  detector matched `\bfor\b` inside "for formatting the answer", which is exactly how prose got
  classified as code. Keep the detector on *syntax* signals, never bare English words.

**2026-08-03 (Step 0.3)**

- **`scripts/sources.toml` is the authority on what may be re-converted.** Absent from the
  manifest = hand-authored or hand-edited = never re-convert. Add a page to it only after
  confirming it is a straight port.
- Fidelity checking must compare against *source minus designed drops*, not raw source.
  A naive comparison flags every Colab/Kaggle badge as lost prose — that produced 10 false
  failures before `strip_designed_drops` reused the converter's own `is_badge_line`.
- `check_fidelity.py` and `port_docs.py` **share `cache_name()`**. They must: keying the source
  cache on the bare filename collided every `index.md` across all model directories onto one
  entry, making the two tools disagree (0 failures vs 26).
- `--dry-run` extracts images to a temp directory. Without that it wrote 34 files to
  `docs/assets/` while claiming to change nothing.
- Two blank lines mid-cell is common in notebook exports — worth checking for whenever a new
  converter bug is suspected.

**2026-08-03 (Step 0.4)**

- **Conversion and vendoring fight each other.** Converting restores upstream image URLs;
  vendoring replaces them with local paths. Always port first, then vendor. See *Pipeline order*.
- Any check that gates writes (as `port_docs.py` does on fidelity) must be **precise about what
  is allowed to change**, or it locks pages out of future edits. Vendoring tripped the fidelity
  check on 6 pages until image targets were excluded from the word comparison.
- `docs/assets/` is 8.2 MB, most of it `capabilities/` (7 notebook-output PNGs, 5.5 MB). If repo
  size becomes a concern, those are the candidates for recompression — not the diagrams.
- The site's `404.html` uses root-absolute `/assets/` paths. That is correct (a 404 can be served
  from any depth) and will look like a broken link to any checker resolving paths relative to the
  file. Don't "fix" it.
- ⚠️ **Any new workflow must be zizmor-clean before it is pushed.** Pin every action to a SHA,
  set an explicit `permissions:` block, and scope write permissions to the job that needs them.
  Check locally with `uvx zizmor --persona=regular .github/workflows/` — it catches everything
  CI does and takes seconds. The org fails on Medium/High, which is broader than the published
  mandatory-check list.
- **Why zizmor appeared only now, on this branch.** The scanner is "GitHub Actions Scan"
  (`.github/workflows/github_actions_scan.yml`), injected org-wide by `google-gemma` — it is not
  in this repo. It triggers on **`pull_request_target`**, and every commit before this branch went
  *straight to main via push*, so it had never run. `foundation` is the repo's first pull request.
  The pre-existing `docs.yml` and `deploy.yml` did not pass the policy; they were never scanned.
- **The scan is diff-scoped.** It flagged only `lint.yml` while `deploy.yml` sat unmodified on the
  same branch with four unpinned actions. Pre-existing workflow debt therefore stays invisible
  until a PR touches that file — worth auditing a workflow *before* you need to edit it under
  time pressure.
