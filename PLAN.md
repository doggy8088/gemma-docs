# Gemma Docs — Port & Re-imagination Plan

Porting and re-imagining the Gemma documentation onto [Zensical](https://github.com/zensical/zensical).
Source: <https://ai.google.dev/gemma/docs> (append `.md.txt` to any page for raw markdown).

**Goal:** the best open-weights developer experience on the web. Not just accurate reference —
an experience that makes people *want* to build with Gemma, welcomes newcomers, and gives
partners a place to showcase their work.

**Gemma 4 is front and center.** All other models are documented, but the docs are
implicitly about Gemma 4 unless stated otherwise.

> Completed work, outcome tables, and the running log of lessons live in
> [`PLAN-ARCHIVE.md`](PLAN-ARCHIVE.md). This file is only what is still ahead.

---

## Working agreement

- **Bite-sized steps.** Each step is independently testable and roughly one session of work.
- **Every step ends with a verification command** and a "done when" condition. Run it before
  moving on.
- **Maarten does all commits and pushes.** Agents stage nothing and push nothing — finish the
  step, report what changed, and stop.
- **Handover-friendly.** Each step is self-contained enough that a fresh agent can pick it up
  from this file alone. Record anything surprising in the archive's *Notes* log.
- **Foundation before fun.** Phases 0–2 must be clean before Phase 4 work starts. A WebGPU
  playground next to "Content coming soon" pages reads as a prototype, not a product.

### Content fidelity — non-negotiable

**Ported content must not change.** The original documentation has been validated for accuracy;
that validation is only transferable if the words survive the port intact.

- A port is a **format** migration, not an editorial one. Convert devsite syntax to Zensical
  syntax, fix links and image paths, repair converter damage — and change nothing else.
- Do **not** reword, tighten, restructure, "improve", modernize, or re-order ported prose.
  Do not silently correct something that looks wrong: flag it and ask.
- Code snippets are copied verbatim. Changing a flag or an argument invalidates the validation.
- **New pages may remix.** Combining snippets from several source pages into a new page is fine
  and encouraged — but the snippets themselves are still copied, not rewritten.
- **Start with a straight port.** Port a page as-is first, confirm it renders well, and only then
  consider whether it needs a companion page. Don't port and redesign in one motion.
- Every ported page records `source_url` in frontmatter, so any page can be diffed against
  upstream later.

Rule of thumb: if a diff against the source `.md.txt` shows a prose change, it needs a reason
written down.

### Import once, then hand-maintain — non-negotiable

**Every page is hand-maintained the moment it is imported.** Maarten edits pages continuously to
get them looking right; re-running an importer over one would silently discard that work.

All **63** manifest pages carry `frozen = true`. There are no unfrozen pages, and that is the
steady state — not a backlog to work through.

- `port_docs.py` skips frozen pages. `check_fidelity.py` skips them too: divergence from
  upstream is now *intended*, so comparing would produce noise rather than signal.
- The `source` line stays for provenance.

**Importing a new page** is the only time a converter runs:

1. Add it to `scripts/sources.toml` *without* `frozen`.
2. Run the pipeline: `port_docs.py` → `vendor_images.py` → `rewrite_links.py`.
3. Review the result.
4. **Set `frozen = true` immediately.** Before any hand-editing, not after.

!!! warning "Converter improvements do not reach existing pages"
    A fix to `convert_notebook.py` only affects future imports. Existing pages need the
    equivalent change applied by hand. Weigh that before adding converter transforms aimed at
    pages that are already in.

**Visual guides** use `scripts/import_newsletter.py` instead (`importer = "import_newsletter"`).
`gemma-4.md` and `mtp.md` were split by hand after import — re-importing either would merge MTP
back into the parent guide and drop the pointer.

### Standing rules

Learned the hard way; each one cost a debugging session. Full stories in the archive.

- **Never re-convert in bulk.** See above. There is no automatic way to tell hand-work from a
  straight port.
- **Any new workflow must be zizmor-clean before it is pushed.** Pin every action to a SHA, set
  an explicit `permissions:` block, scope write permissions to the job that needs them. Check
  with `uvx zizmor --persona=regular .github/workflows/` — seconds, and it catches what CI does.
  The scan is diff-scoped, so pre-existing workflow debt stays invisible until a PR touches it.
- **Measure the rendered page, don't reason about the renderer.** Four layout bugs in Step 1.4
  and one in Step 3.1 were invisible in the source and obvious in one DOM query.
- **`data/models.yml` and `data/frameworks.yml` are the single sources of truth.** Hand-editing a
  model table in a page is a bug. Adding a framework name that isn't in the data file is a bug.
- **Frontmatter keys shadow macros variables.** `models:` frontmatter silently broke
  `models.gemma4.sizes` in both generated tables. The macros data is exposed as `modeldata.*`
  for exactly this reason — never name a frontmatter key after an `include_yaml` variable.

Local preview:

```bash
uv run zensical serve
```

---

## Current status

| Phase | State |
|---|---|
| 0 — Foundation | ✅ complete |
| 1 — Design system | ✅ complete |
| 2 — Navigation & content | 🟡 nav and Run section done; 7 stubs and 6 keep-or-drop decisions left |
| 3 — Agent-first infrastructure | 🟡 3.1 done; 3.2–3.5 open |
| 4–8 | not started |

89 pages · 63 ported and frozen · 9 stylesheets · lint backlog **7** (all stubs).

**Remaining Phase 2 work**, all blocked on content decisions rather than engineering:

| Page | Blocked on |
|---|---|
| `community/gemmaverse.md` | real project data (still ships `example.com` cards) |
| `community/discord.md` | port `/discord` |
| `community/gemma-skills.md` | content — no upstream source |
| `cookbook.md` | content decision |
| `demos/concurrent.md` | content — the app lives in the cookbook repo |
| `tools/index.md`, `tools/model-selector.md` | **Phase 4** — these *are* the interactive tools |

**Upstream pages not ported (6):** `/core/gemma_library`, `/core/pytorch_gemma`,
`/core/gemma_on_gemini_api`, `/core/deploy_to_cloud_run_from_ai_studio`, `/gemma_chat`,
`/meme-generator`. Each needs a keep-or-drop decision rather than a straight port —
`/meme-generator` in particular is a demo app, not reference material.

Deliberately not ported and **not** gaps: `/integrations/{ollama,llamacpp,lmstudio,mlx}`
(written as quick references instead), `/core/gke` (folded into Google Cloud),
`/codegemma/*` (dropped by decision).

---

## Target navigation

Eight top-level tabs. Capabilities stay under **Get started** — they are fundamental Gemma 4
capabilities and belong in one place where the full picture is visible.

```
Home
Get started    Quickstart · Prompt & chat format
               Capabilities: Text · Thinking · Function calling
                             Vision (Image, Video) · Audio · Multi-Token Prediction
Models         Gemma 4 (overview, model card)
               Variants: Gemma 3n · DiffusionGemma · FunctionGemma
                         EmbeddingGemma · PaliGemma · ShieldGemma
               Previous generations: Gemma 3 · 2 · 1
               Research: RecurrentGemma · DataGemma · Gemma Scope · Gemma APS
               Releases
Run            Local:  Ollama · llama.cpp · LM Studio · MLX · Transformers
                       Keras · LiteRT · Web · Mobile
               Serve:  vLLM · SGLang · Google Cloud
               Frameworks: LangChain · Quantization & QAT
Fine-tune      Overview · QLoRA · Full fine-tune · Vision QLoRA
               Keras LoRA · Distributed · Spoken language · Mobile actions
Learn          Visual guides · Tutorials · Notebooks
Demos          Interactive tools · Apps & Spaces
Ecosystem      Gemmaverse · Discord · Skills · Contribute
```

**Models stays model-card-centric.** It mirrors the upstream structure on purpose — model cards
are canonical reference and shouldn't be creatively restructured. The improvement is in
presentation, consistency, and cross-linking, not reorganization.

`docs/models/` stays **flat**; grouping is nav-only, so a model's URL survives reclassification.

Decisions made: no "Migrating from Gemma 3 → 4" page (users swap one model string). Visual
guides get fully ported pages, not outbound links. Nav labels are short ("Ollama"); page H1s
stay as written ("Run Gemma with Ollama").

---

## Design system

**Official Gemma palette:**

| Token | Hex | Use |
|---|---|---|
| `--gemma-blue` | `#3186FF` | **Fill only** — buttons, bars, icons. 3.51:1 on white fails AA for text. |
| `--gemma-blue-deep` | `#346BF1` | Text on light surfaces (4.65:1) |
| `--gemma-blue-light` | `#4FA0FF` | Text on dark surfaces (5.98:1) |
| `--gemma-white` | `#FFFFFF` | Light surface |
| `--gemma-ink` | `#202124` | Dark surface, body text on light |

Use `--gemma-accent`, which resolves to `blue-deep` in light and `blue-light` in dark, rather
than naming a brand colour directly.

**Stylesheets** — 9 files, each under 250 lines, each opening with what it owns and what it must
not touch. Adding a ninth is the right move when a new concern appears; growing one into a
catch-all is not.

| File | Owns |
|---|---|
| `gemma.css` | Tokens. **Only file with raw hex.** |
| `layout.css` | Page grid, sidebars, typography scale |
| `nav.css` | Header, tabs, sidebars, table of contents |
| `content.css` | Tables, admonitions, code, content tabs |
| `components.css` | Cards, badges, figures, steps, embeds |
| `page-context.css` | The applicability bar above every H1 |
| `tokens.css` | Chat-token badges |
| `home.css` | Landing page |
| `motion.css` | Transitions + reduced-motion opt-out |

Enforced by `lint_docs.py --check css`: raw hex only in `gemma.css`, every sheet registered in
`extra_css`, none over 250 lines, no `<style>` blocks in templates.

Live catalogue with copy-paste markup: `docs/contributing/components.md`.

---

## Frontmatter schema

Defined in `scripts/frontmatter_schema.py` — the single source of truth, imported by both the
linter and the backfill script so they cannot disagree. Validated by
`lint_docs.py --check frontmatter`.

**Required on every page:**

| Key | Type | Notes |
|---|---|---|
| `title` | string | See *title is sometimes content* below |
| `description` | string | One sentence, 40–200 chars. Feeds llms.txt, search, page cards. |

**Optional, validated when present:**

| Key | Type | Notes |
|---|---|---|
| `models` | list | Models the page is about. **Omit when it is about none.** |
| `requires` | string | One input modality, narrowing `models` to the sizes that support it |
| `frameworks` | list | Engines/libraries the page shows code for |
| `prerequisites` | list | docs-relative paths; must resolve to real pages |
| `name` | string | Stable kebab-case id for agents to reference |
| `source_url` | string | Upstream page, from `scripts/sources.toml` |

**Reserved** for the Step 3.2 build hook and rejected if hand-written: `estimated_tokens`,
`code_languages`, `last_updated`. Theme keys (`template`, `hide`, `cards`, …) pass through.

### Authored vs computed

A key is either **authored** (only a person knows it) or **computed** (derivable from the file),
never both. Hand-writing a token count guarantees it is wrong the next time anyone edits the
page, so authoring a computed key is a lint *error*, not a style preference.

### Vocabularies

Both are read from data files, so adding an entry there is all that is needed:

- **`models`** — `family` or `family/size`. Families come from the `docs/models/` directories;
  Gemma 4 sizes come from `data/models.yml`. A bare `12b` is rejected: Gemma 3 and Gemma 4 both
  ship a 12B, so the family is required. Only Gemma 4 has per-size data today; other families
  are addressable at family level until someone adds sizes for them.

    ```yaml
    models: [gemma-4]                                  # the whole family
    models: [gemma-4/e2b, gemma-4/e4b, gemma-4/12b]    # the audio-capable sizes
    models: [embeddinggemma]                           # a variant
    ```

    **Omitting `models` means the page is not about a model** — community, tooling and site
    pages. There is no "applies to everything" wildcard: that and "applies to nothing" were
    being written identically, which made the key useless for the page context header.

    The audio page's list is generated from the `modalities` field rather than typed out, so it
    cannot drift from the model data.

- **`frameworks`** — ids from `data/frameworks.yml`, which also carries display names,
  categories and the docs page for each. A controlled vocabulary because the Step 4.2 switcher
  cannot offer a free-text value, and `transformers` / `huggingface` / `hf` spread across 89
  pages would make the feature unbuildable.

### `title` is sometimes content

Zensical synthesises an H1 from `title` (`<h1 id="__skip">`) **only when the page has no H1 of
its own**; it never replaces an existing heading. That gives the key two roles:

- **Page has an H1** — `title` reaches only the browser tab and llms.txt, and may be *more*
  specific than the heading. This is why nine pages headed "Model card" carry titles like
  `EmbeddingGemma model card`: the heading on screen is untouched, the tab and the
  machine-readable index become useful.
- **Page has no H1** — `title` *is* the visible heading and must say what the page already said.
  Five pages are in this state: `deployment/index.md`, `deployment/google-cloud.md`,
  `models/gemma-4/model-card.md`, `models/diffusiongemma/inference.md`, `models/releases.md`.
  The linter requires `title` on any page lacking an H1.

Giving those five a real H1 would be a content change, so it is Maarten's call, not the port's.

---

## Page context header — **done 2026-08-05**

A Stripe-style bar at the top of **every page** — see <https://docs.stripe.com/billing/scripts>.
Not silence-by-default: the slot is always present, so readers learn where to look.

It carries three things, in this order of arrival:

1. **Applicability** — ✅ built.
2. **LLM tools** (Step 3.4) — Copy as Markdown, Open in Claude/ChatGPT, View raw.
3. **Model + engine selector** (Step 4.2) — page-scoped context belongs here, not site chrome.

- [x] `overrides/main.html` renders the bar on every page, inside `.md-content__inner` so it
      shares one left edge with the breadcrumb, H1 and body.
- [x] Applicability badges from `models`, display names from `data/models.yml`.
- [x] Empty state holds its height, so the breadcrumb→H1 distance never changes between pages.
- **Verified:** audio shows `Gemma 4 · E2B E4B 12B`, discord shows nothing, both 28px tall with
  a 16px gap to the H1; all four left edges land on 304px; contrast passes AA in both schemes
  (light 4.68/6.73/9.00, dark 6.64/9.25/12.12).
- **Done when:** ✅ adding a size to `data/models.yml` changed the audio page without the page
  being touched — confirmed with a throwaway `7b-probe` size, then reverted.

### Applicability is declared two ways

| Frontmatter | Shows |
|---|---|
| *(no `models`)* | nothing — the slot stays empty |
| `models: [gemma-3]` | `Gemma 3` |
| `models: [gemma-4/e2b, gemma-4/e4b]` | `Gemma 4 · E2B E4B` |
| `models: [gemma-4]` + `requires: audio` | `Gemma 4 · E2B E4B 12B`, resolved at build time |

**Prefer `requires` whenever the set *is* "the models that can do X".** A listed set is a copy
of the model data and goes stale the day a new size ships; a derived one cannot. `requires`
takes one input modality, validated against `data/models.yml`, and narrows the family named in
`models` — so it needs `models` to be present.

This was the gap that testing the "done when" exposed: the audio page's three sizes had been
written into its frontmatter by the Step 3.1 backfill, which is generation at *authoring* time,
not build time. It looked data-driven and wasn't.

### Design

Measured off `docs.stripe.com`, not approximated. Their page toolbar is `display:flex`,
`gap:16px`, `align-items:baseline`, `padding-bottom:12px`, a 1px hairline rule beneath, and
between items a `<span role="separator">` **1px wide with `align-self: stretch`** in a tone
deliberately darker than that rule. Nothing sits above the H1 at all.

```
Models  Gemma 4  │  Uses  Ollama  │  Examples  Gemma 4 12B ⌄
──────────────────────────────────────────────────────────────
```

Stripe's toolbar items are peer actions at one weight. Ours are label→value pairs, a structure
they never have to solve — so two rules do the work their flat treatment cannot:

1. **Spacing is hierarchical.** 6px inside a pair, 16px between pairs. An 8px/16px ratio was
   too weak and the whole line read as one run of grey.
2. **Every pair gets the same label/value contrast.** Emphasising the model value but not the
   framework values made `Frameworks` and `Sentence Transformers` indistinguishable.

Model names link to their local canonical page. Tool names link to the project home. Both stay
neutral-coloured; the selector is the only control in the strip.

### Subject vs. selection

The strip carries two different kinds of thing, and conflating them was the original design
error:

| Item | Source | Answers |
|---|---|---|
| `Models` | `models:` frontmatter | what the page is **about** |
| `Uses` | `frameworks:` frontmatter | what it uses |
| `Examples` | the page's marked code blocks | what the **reader** wants the commands to target |

Driving the selector from `models:` looked reasonable and was wrong: the Ollama page is about
Gemma 4, but its commands could target anything Ollama publishes, and Quick Start ended up
claiming `E2B` on a page that covers all five sizes. The selector is now built in JavaScript
from the identifiers actually present in the page's `.model-swap` blocks:

- a model must have a published id for **every** engine on the page, so a switch cannot rewrite
  some snippets and leave others behind;
- `requires:` narrows it further, so an audio page can never offer a model without ears;
- with fewer than two candidates there is no control at all.

This also gives the extension point for later generations: add Gemma 3's Ollama tags to
`data/models.yml` and it appears in the Run pages' selector with no template change.

### How the data reaches the template

Zensical renders with **minijinja**, exposes `page.meta` to templates but *not* the macros
plugin's YAML, and has no hook mechanism to inject any. So `data/models.yml` is compiled into
`overrides/partials/model-names.html` by `scripts/generate_model_names.py`, and
`lint_docs.py --check generated` fails if the two drift apart. Committed build output, verified
in CI — the same arrangement as `.lintbaseline.json`.

Writing template logic here: minijinja has no `.get()` and no Python string methods. Use
`dict[key] | default(...)`, the `split` filter, and the `startingwith` test.

**Interaction decision.** The picker is local to pages with swappable examples, but its full
model choice is remembered globally and mirrored in `?model=` for shareable URLs. Code blocks
opt in with `.model-swap`; there are no competing per-block selectors. If the remembered model
is unavailable, the authored page default remains active and the menu names the incompatibility.
Silent substitution is rejected.

---

## Phase 3 — Agent-first infrastructure (3.1 ✅)

### Step 3.2 — Computed metadata
- [ ] Build hook computing `estimated_tokens`, `code_languages`, and `last_updated`
      automatically so they cannot drift.
- **Verify:** add a code block to a page, rebuild, confirm metadata updates.

### Step 3.3 — Machine-readable surfaces
- [ ] Generate `/llms.txt` (index with descriptions) and `/llms-full.txt` (everything).
- [ ] Serve raw markdown at `<page>.md` for every page.
- **Verify:** `curl` the built site for `llms.txt` and one `.md` endpoint.

Compose entry labels from `title` + `models` rather than adding a second title key —
`models: [gemma-4]` plus `title: Model card` gives "Gemma 4 — Model card" without touching the
page. Every input this needs already exists.

### Step 3.4 — Copy-for-agent affordances
- [ ] "Copy page as Markdown" button, in the page context header.
- [ ] "Open in Gemini / Claude / ChatGPT" links.
- **Verify:** click the button, paste, confirm clean markdown with frontmatter.

### Step 3.5 — Docs as a skill
- [ ] Package the docs as an installable skill and/or MCP server so agents can query them
      directly.

---

## Phase 4 — Interactive tools

### Step 4.1 — VRAM & context calculator
Replace the static table in `docs/tools/vram-calculator.md` with a real calculator.

- [ ] Inputs: model, quantization, context length, batch size, KV-cache dtype, GPU preset
      (4090, 5090, A100, H100, M-series unified memory, …).
- [ ] Outputs: fits / doesn't fit with headroom, estimated tok/s, recommended engine,
      **and the exact copy-paste command for that engine**.
- [ ] URL-encoded state (`?model=gemma-4/12b&quant=q4_0&gpu=rtx4090&ctx=32k`) so results are
      shareable.
- [ ] Reads from `data/models.yml`.
- **Verify:** results match the memory table in `models/gemma-4/index.md` at matching settings.
- **Done when:** a shared URL reproduces the exact same state.

### Step 4.2 — Global model switcher, then engine preference

The model half is implemented: pick "E4B" and marked examples across the site rewrite their model
IDs. A global engine preference remains a separate follow-up; pages currently show the runtimes
their tutorial actually covers.

- [x] Lean control in the page context header, persisted in `localStorage` and the URL.
- [ ] Add a global engine preference once pages have equivalent, verified examples for each engine.
- [ ] Extend Zensical's `content.tabs.link` to sync site-wide.
- [x] **Capability-aware gating, driven by `data/models.yml`:** audio examples must not offer
      26B A4B or 31B. When the selected model can't do what the page shows, show an inline
      explanation and the nearest valid model — never a broken snippet, never a silent swap.
- [x] Graceful fallback when a model/engine or identifier-flavour pair has no published weights.
- [x] Opt-in `.model-swap` blocks across the current Gemma 4 quickstart, deployment, capability,
      QAT, and QLoRA pages; base, instruction, QAT, GGUF, Ollama, and MLX IDs stay semantically
      correct.
- **Verify:** select 31B, navigate to `capabilities/audio.md`, confirm the gating message appears
      and no invalid command is shown.
- **Model selector verified:** 31B → audio shows the gating message and retains the authored E2B
      example; multi-engine pages only offer models published for every marked block.
- **Done when:** engine selection and linked tabs are implemented and no combination of model and
      engine state produces a command that would fail.

### Step 4.3 — "Choose your model" decision flow
- [ ] Four questions → recommendation + copy-paste command for the user's engine.
- [ ] Embeds the VRAM calculator.

### Step 4.4 — Tokenizer & chat-template playground
- [ ] Paste a conversation → exact `<start_of_turn>` token stream, token count.
- [ ] Image token budget for a given resolution and aspect ratio.
- [ ] Reuses the existing `tokens.css` badge styling. Pairs with `get-started/prompt-formatting.md`.

---

## Phase 5 — Demos & embeds

### Step 5.1 — First HF Space embed (spike)
- [ ] Embed <https://huggingface.co/spaces/google/gemma4_vision_token_budget> on a Demos page.
- [ ] Lazy-load (Spaces are heavy) and design a fallback for sleeping Spaces — a dead iframe
      looks worse than no iframe.
- **Verify:** view on mobile and desktop, light and dark. Test with the Space asleep.
- **Done when:** we know whether to standardize on iframe or `<gradio-app>`, documented here.

### Step 5.2 — Demos section build-out
- [ ] Concurrent inference demo (currently a stub).
- [ ] Additional Spaces once 5.1 establishes the pattern.

---

## Phase 6 — Learn

### Step 6.1 — Visual guides: vendor the images
The four guides are imported and hand-edited. Their images still point at the Substack CDN.

- [ ] Replace remote images with local assets (Maarten provides raw SVG).
- [ ] Empty `PENDING_IMAGE_HOSTS` in `lint_docs.py` when done — it exists only for this.
- [ ] **Contextual cross-linking** — link the MoE guide from the 26B A4B section, Quantization
      from QAT, Reasoning LLMs from `capabilities/thinking.md`.
- **Verify:** SVGs render crisply at all widths, legible in both themes; `--check images` → 0.

### Step 6.2 — Notebooks as source of truth
- [ ] Keep `.ipynb` canonical, render to pages at build time.
- [ ] Colab / Kaggle / local badges on every tutorial.
- [ ] Execute notebooks in CI — a "snippets are tested" badge is a real credibility marker.

### Step 6.3 — Task-first entry points
- [ ] "I want to…" layer: build a RAG app · extract structured JSON · caption images · classify
      content · build a tool-using agent. Each is a curated path across existing pages.

---

## Phase 7 — The wow tier

Only after Phases 0–2 are clean.

- [ ] **In-browser playground** — E2B q4 via WebGPU/transformers.js, running in the docs page.
      ~1.5–2 GB download, so explicit opt-in and clear progress UI.
- [ ] **Ask Gemma about Gemma** — docs chatbot on transformers.js or an HF Space endpoint.
- [ ] **Interactive Gemma 4 architecture explainer** — PLE, MoE routing on 26B A4B, MTP draft
      heads, the 12B's encoder-free linear projections. These four confuse people and prose
      isn't fixing it.
- [ ] **MTP animated explainer** — draft-vs-verify with an acceptance-rate slider.
- [ ] **DiffusionGemma denoising animation** — text unmasking step by step.
- [ ] **Community throughput leaderboard** — measured tok/s per (model × quant × engine ×
      hardware), PR-based. Exists nowhere authoritative; high maintenance, high payoff.
- [ ] **Cost calculator** — GPU $/hr → $/1M tokens, self-host vs. API.

---

## Phase 8 — Maintainer skill & handover

*Finalized once the docs are done, but **drafted incrementally** — every phase that establishes a
convention adds its section immediately, while the reasoning is fresh.*

### Step 8.1 — `gemma-docs` authoring skill
Must cover:

- [ ] **Content fidelity rules** — stated first and plainly.
- [ ] **How to port a page** — fetch `.md.txt`, run the pipeline, what the converter handles vs.
      what needs manual repair, and freezing immediately.
- [ ] **How to add a page** — nav placement, frontmatter schema, which components to use.
- [ ] **The design system** — which stylesheet owns what, the token layer, when to add a sheet.
- [ ] **Component catalog** — every card/callout/badge/figure with copy-paste examples.
- [ ] **The data files** — what reads `data/models.yml` and `data/frameworks.yml`, and why
      hand-editing a model table is a bug.
- [ ] **Verification** — the lint checks, how to run the build, what CI enforces.
- [ ] **Gotchas** — everything in the archive's *Notes*.
- [ ] **What not to do** — don't reword ported prose, don't hotlink images, don't add hex outside
      the token file, don't commit or push.
- **Verify:** hand a fresh agent the skill and a page to port. It should succeed with no
  additional context and pass the linter first try.

### Step 8.2 — Contributor docs
- [ ] `docs/contributing/` covering the same ground for humans.

---

## Tooling lifecycle

Scripts are **migration tooling or permanent infrastructure**, and the two have opposite
lifespans. Every entry below has a stated end date so the directory doesn't quietly become an
archive of one-shot scripts nobody dares delete.

| Script | Lines | Fate |
|---|---|---|
| `lint_docs.py` | 804 | **Permanent.** CI gate. |
| `frontmatter_schema.py` | 233 | **Permanent.** Imported by the linter. |
| `generate_model_names.py` | 86 | **Permanent.** Compiles `data/models.yml` into the template partial the page context header reads; CI verifies it is current. |
| `convert_notebook.py` + `test_convert_notebook.py` | 896 | **Delete when Phase 2.4 closes** — i.e. once the 6 keep-or-drop decisions are made and nothing more will be imported from upstream. |
| `port_docs.py` | 135 | **Delete with the converter.** All 63 manifest pages are frozen, so today it can only skip. |
| `check_fidelity.py` | 223 | **Delete with the converter.** Same reason: it reports `0 faithful, 63 frozen` and has nothing left to check. |
| `rewrite_links.py` | 253 | **Delete with the converter.** One-shot link migration; `links` and `internal` have been 0 for weeks. |
| `sources.toml` + `link_map.toml` | 577 | **Delete with the converter.** ⚠️ `source` is now duplicated in every page's `source_url` frontmatter — the manifest's only unique data is `frozen`, `assets` and `importer`. |
| `backfill_frontmatter.py` | 373 | **Delete when Phase 3 closes.** One-shot; frontmatter is complete and hand-maintained from here. New pages get theirs from the Phase 8 skill. |
| `import_newsletter.py` | 259 | **Keep while more visual guides may arrive**; revisit at Phase 6.1. |
| `vendor_images.py` | 136 | **Keep until Phase 6.1** — the Substack images still need vendoring. |

Roughly **2,450 of 3,877 lines are scheduled for deletion**, most of it gated on one decision:
whether anything else gets ported from upstream. Making that call early shrinks the surface a
maintainer has to understand.

---

## Pipeline order

Steps that rewrite pages must run in this order, because each stage can undo the previous one
(conversion restores upstream image URLs; vendoring replaces them with local paths):

```
port_docs.py  ->  vendor_images.py  ->  rewrite_links.py  ->  lint_docs.py --baseline
```

`check_fidelity.py --manifest` can run at any point; it ignores image targets by design.

---

## Open decisions

- **Will anything else be ported from upstream?** Gates the deletion of ~2,450 lines of tooling
  (see above) and closes Phase 2.4.
- The 6 unported upstream pages — keep or drop, individually.
- Whether visual-guide diagrams can also appear inline in the reference docs (with attribution),
  which would be a stronger integration than standalone guide pages alone.
- Whether the throughput leaderboard is worth its ongoing maintenance cost.
- The five pages with no H1 — leave as-is, or add real headings.
- **Unresolved: Ollama download sizes.** `ollama.com/library/gemma4/tags` states E4B 9.6 GB and
  12B 7.6 GB, making E4B *larger* than 12B, which contradicts the validated Q4_0 column in
  `models/gemma-4/index.md`. Verified verbatim twice. Most likely the default tag per size uses a
  different quantization, but no explanation appears on the page — so the figures stay
  unexplained rather than guessed at.
