---
title: Components
description: The reusable building blocks these docs are made from, with live examples and the markup to copy.
---

# Components

Every visual element on this site comes from this catalogue. Build pages from these rather than
writing new CSS — if something you need isn't here, add it to
`docs/stylesheets/components.css` and document it on this page.

!!! note
    Colours, spacing, radii and shadows are **tokens**, defined once in
    `docs/stylesheets/gemma.css`. It is the only stylesheet allowed to contain a raw hex value,
    and `scripts/lint_docs.py --check css` enforces that.

## Colour tokens

| Token | Use |
|---|---|
| `--gemma-blue` | Fills: buttons, bars, icons, step markers. **Not body text** — 3.51:1 on white fails WCAG AA. |
| `--gemma-blue-deep` | Text on light surfaces. 4.65:1 on white, passes AA. |
| `--gemma-blue-light` | Text on dark surfaces. 5.98:1 on ink, passes AA. |
| `--gemma-white`, `--gemma-ink` | Surfaces. |
| `--gemma-accent` | Whichever of the above is text-safe in the current scheme. **Prefer this.** |
| `--gemma-accent-fill` | Scheme-aware fill colour. |

Scales: `--gemma-space-1…7`, `--gemma-radius-sm|md|lg|pill`, `--gemma-shadow-sm|md|lg`.

## Cards

A grid of links. Used on the cookbook, model overviews and ecosystem pages.

<div class="card-grid">
<a class="card" href="#cards">
  <div class="card-title">Card title <span class="card-ext">↗</span></div>
  <p class="card-desc">One or two lines describing where this card leads.</p>
</a>
<a class="card" href="#cards">
  <div class="card-title">Second card</div>
  <p class="card-desc">The grid reflows automatically and collapses to one column on mobile.</p>
</a>
</div>

```html
<div class="card-grid">
<a class="card" href="…">
  <div class="card-title">Card title <span class="card-ext">↗</span></div>
  <p class="card-desc">One or two lines describing where this card leads.</p>
</a>
</div>
```

`.cookbook-grid` / `.cookbook-card` are aliases kept for converter output and existing pages.
Use `.card-grid` / `.card` in anything new.

## Badges

Inline chips for model attributes. Drive them from `data/models.yml` so a badge can never
disagree with the tables.

<p>
  <span class="badge">12B</span>
  <span class="badge">256K context</span>
  <span class="badge badge--accent">Recommended</span>
  <span class="badge badge--muted">Audio</span>
</p>

```html
<span class="badge">12B</span>
<span class="badge badge--accent">Recommended</span>
<span class="badge badge--muted">Audio</span>   <!-- capability this model lacks -->
```

`--muted` is how the model switcher (Step 4.2) will show a capability the selected model does
not have — struck through rather than hidden, so the reader learns the limitation.

## Steps

A numbered walkthrough. Built on an ordered list, so it still reads correctly without CSS.

<ol class="steps" markdown>
<li markdown>
**Install the dependencies**

Anything can go inside a step, including code blocks.
</li>
<li markdown>
**Run the model**

The connecting rail is drawn automatically between steps.
</li>
</ol>

```html
<ol class="steps" markdown>
<li markdown>
**Step title**

Body content, including code blocks.
</li>
</ol>
```

The `markdown` attribute is required — it tells `md_in_html` to keep parsing Markdown inside
the tag.

## Figures

A framed diagram with a caption.

```html
<figure class="figure" markdown>
![Alt text](../assets/example.png)
<figcaption>What the diagram shows.</figcaption>
</figure>
```

For a diagram that needs different artwork per theme, mark the pair and let CSS choose:

```html
![Alt text](../assets/diagram-light.png){ .only-light }
![Alt text](../assets/diagram-dark.png){ .only-dark }
```

## Video

Emitted automatically by `scripts/convert_notebook.py` from a standalone `[Video](youtube…)`
link. Full content width, 16:9, lazy-loaded, privacy-mode host.

```html
<div class="video-embed">
  <iframe src="https://www.youtube-nocookie.com/embed/VIDEO_ID"
          title="Video" loading="lazy" allowfullscreen></iframe>
</div>
```

## Model-card header

Centres the banner, resource links and License/Authors line on a model card. Also emitted by
the converter — you shouldn't need to write it by hand.

```html
<div class="model-card-header" markdown>
![Banner](../assets/banner.png)

[Hugging Face](…) \| [GitHub](…)

**License** : [Apache 2.0](…)
</div>
```

## Admonitions

Standard Zensical admonitions, used throughout. The converter produces these from upstream's
`> NOTE:` blockquotes and `> [!NOTE]` callouts.

!!! note
    Neutral supporting information.

!!! tip
    A recommendation or shortcut.

!!! warning
    Something that will cost the reader time or money if ignored.

## Motion

Shared component transitions live in `docs/stylesheets/motion.css`, including a blanket
`prefers-reduced-motion` opt-out — anything added later inherits it. Motion is feedback
(something is interactive, something changed), never decoration.

The isolated homepage animation also checks that preference before it starts.

To check the reduced-motion path, enable *Reduce motion* in your OS accessibility settings and
reload; hover effects should become instant.

## Stylesheet layout

| File | Owns |
|---|---|
| `gemma.css` | Tokens. The only file with raw hex. |
| `layout.css` | Page grid, sidebars, typography scale |
| `nav.css` | Header, tabs, sidebars, table of contents |
| `content.css` | Tables, admonitions, code, content tabs |
| `components.css` | Everything on this page |
| `tokens.css` | Chat-token badges (`<start_of_turn>` etc.) |
| `home.css` | Landing page: chrome, backdrop, hero |
| `motion.css` | Transitions and the reduced-motion opt-out |

Each stays under 250 lines so it can be read end-to-end before editing. Adding a stylesheet
for a new concern is the right move; growing one into a catch-all is not.
