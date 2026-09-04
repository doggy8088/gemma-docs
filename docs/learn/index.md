---
title: Learn
description: Illustrated walkthroughs of how the Gemma models work, from the architecture down to individual mechanisms.
---

# Learn

This section explains how the Gemma models actually work. Where the rest of the documentation
shows you what to run, these pages take an architecture apart and draw it — what the layers do,
why a design choice was made, and where a model's behaviour comes from.

Read them in any order. Nothing here is a prerequisite for using Gemma.

## What's next

<div class="card-grid">
<a class="card" href="../tutorials/visual-guides/gemma-4.md">
  <span class="card-icon">{{ icon('lucide/book-open') }}</span>
  <div class="card-title">A Visual Guide to Gemma 4</div>
  <p class="card-desc">How the architecture works, drawn out step by step.</p>
</a>
<a class="card" href="../tutorials/visual-guides/gemma-4-12b.md">
  <span class="card-icon">{{ icon('lucide/book-open') }}</span>
  <div class="card-title">A Visual Guide to Gemma 4 12B</div>
  <p class="card-desc">What the unified 12B model does differently.</p>
</a>
<a class="card" href="../tutorials/visual-guides/mtp.md">
  <span class="card-icon">{{ icon('lucide/book-open') }}</span>
  <div class="card-title">A Visual Guide to MTP</div>
  <p class="card-desc">Drafter models and speculative decoding, illustrated.</p>
</a>
<a class="card" href="../tutorials/visual-guides/diffusiongemma.md">
  <span class="card-icon">{{ icon('lucide/book-open') }}</span>
  <div class="card-title">A Visual Guide to DiffusionGemma</div>
  <p class="card-desc">How denoising produces text instead of pixels.</p>
</a>
</div>

## Previous Gemma Explained Series

<div class="card-grid">
<a class="card" href="https://developers.googleblog.com/gemma-explained-overview-gemma-model-family-architectures" target="_blank" rel="noopener">
  <span class="card-icon">{{ icon('lucide/book-open') }}</span>
  <div class="card-title">Overview of Gemma Architectures <span class="card-ext">↗</span></div>
  <p class="card-desc">Text-to-text models and specialization for coding.</p>
</a>
<a class="card" href="https://developers.googleblog.com/gemma-explained-new-in-gemma-2/" target="_blank" rel="noopener">
  <span class="card-icon">{{ icon('lucide/book-open') }}</span>
  <div class="card-title">What’s New in Gemma 2 <span class="card-ext">↗</span></div>
  <p class="card-desc">Updated architecture with knowledge distillation for 2B and 9B models.</p>
</a>
<a class="card" href="https://developers.googleblog.com/gemma-explained-recurrentgemma-architecture/" target="_blank" rel="noopener">
  <span class="card-icon">{{ icon('lucide/book-open') }}</span>
  <div class="card-title">RecurrentGemma Architecture <span class="card-ext">↗</span></div>
  <p class="card-desc">The Griffin architecture mixing local attention and linear recurrences for fast long-sequence inference.</p>
</a>
<a class="card" href="https://developers.googleblog.com/gemma-explained-paligemma-architecture/" target="_blank" rel="noopener">
  <span class="card-icon">{{ icon('lucide/book-open') }}</span>
  <div class="card-title">PaliGemma Architecture <span class="card-ext">↗</span></div>
  <p class="card-desc">Vision-language model that takes text and images as input to generate text outputs.</p>
</a>
<a class="card" href="https://developers.googleblog.com/gemma-explained-whats-new-in-gemma-3/" target="_blank" rel="noopener">
  <span class="card-icon">{{ icon('lucide/book-open') }}</span>
  <div class="card-title">What’s New in Gemma 3 <span class="card-ext">↗</span></div>
  <p class="card-desc">Updated vision-language model with 5-to-1 interleaved attention and 128k context handling.</p>
</a>
<a class="card" href="https://developers.googleblog.com/gemma-explained-embeddinggemma-architecture-and-recipe/" target="_blank" rel="noopener">
  <span class="card-icon">{{ icon('lucide/book-open') }}</span>
  <div class="card-title">EmbeddingGemma Architecture <span class="card-ext">↗</span></div>
  <p class="card-desc">Text embedding model for search and retrieval.</p>
</a>
</div>

--8<-- "glossary.md"

