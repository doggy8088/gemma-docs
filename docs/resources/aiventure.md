---
title: AIventure
description: Learning about vibe-coding, agents and Gen AI with Gemma 4.
models: [gemma-4]
hide:
  - toc
---

Learn how to create an AI-powered educational game focused on vibe-coding and agentic characters using Angular, PhaserJS, Gemini and Gemma.

<div class="demo-embed">
  <div class="demo-disclaimer">
    <div class="demo-disclaimer-card" role="dialog" aria-label="Demo disclaimer">
      <p>
        This demo uses Mediapipe with WebGPU to load and run a Gemma 4 model locally. <strong>This will require a 4 gigabyte download</strong> the first time it runs which will then be cached. Performance varies depending on the device's GPU.
      </p>
      <p>
        Note: There is some overhead from the browser's WebGPU API which can reduce performance compared to running the model natively via Ollama or LM Studio. Additionally, this model is quantized. Explore this demo and then download the code yourself to compare the performance with LM Studio.
      </p>
      <div class="demo-disclaimer-actions">
        <button
          type="button"
          class="demo-disclaimer-btn"
          onclick="const embed = this.closest('.demo-embed'); const iframe = embed.querySelector('iframe'); iframe.src = iframe.dataset.src; iframe.hidden = false; this.closest('.demo-disclaimer').hidden = true;"
        >
          Accept and Continue
        </button>
      </div>
    </div>
  </div>
  <iframe
    data-src="https://bebechien.github.io/AIventure/index.html"
    title="AIventure"
    loading="lazy"
    allow="clipboard-read; clipboard-write"
    hidden
  ></iframe>
</div>

If the embedded demo does not load, [open it directly from our solutions page](https://developers.google.com/solutions/learn/aiventure){ data-model="gemma-4" }.
