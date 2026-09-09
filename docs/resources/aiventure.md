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
        <strong>Note:</strong> There is some overhead from the browser's WebGPU API which can reduce performance compared to running the model natively via Ollama or LM Studio. Additionally, this model is quantized. Explore this demo and then download the code yourself to compare the performance with LM Studio.
      </p>
      <div class="demo-disclaimer-actions">
        <button
          type="button"
          class="demo-disclaimer-btn"
          onclick="const embed = this.closest('.demo-embed'); const iframe = embed.querySelector('iframe'); iframe.src = iframe.dataset.src; iframe.hidden = false; const fsBtn = embed.querySelector('.demo-fullscreen-btn'); if (fsBtn) fsBtn.hidden = false; this.closest('.demo-disclaimer').hidden = true;"
        >
          Accept and Continue
        </button>
      </div>
    </div>
  </div>
  <button
    type="button"
    class="demo-fullscreen-btn"
    aria-label="Toggle full screen"
    title="Toggle full screen"
    hidden
    onclick="const embed = this.closest('.demo-embed'); if (!document.fullscreenElement) { embed.requestFullscreen(); } else { document.exitFullscreen(); }"
  >
    <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true" fill="currentColor">
      <path d="M7 14H5v5h5v-2H7v-3zm-2-4h2V7h3V5H5v5zm12 7h-3v2h5v-5h-2v3zM14 5v2h3v3h2V5h-5z"/>
    </svg>
    <span>Full screen</span>
  </button>
  <iframe
    data-src="https://bebechien.github.io/AIventure/index.html"
    title="AIventure"
    loading="lazy"
    allow="clipboard-read; clipboard-write; fullscreen"
    allowfullscreen
    hidden
  ></iframe>
</div>

If the embedded demo does not load, [open it directly from our solutions page](https://developers.google.com/solutions/learn/aiventure){ data-model="gemma-4" }.
