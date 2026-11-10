---
title: AIventure
description: 透過 Gemma 4 學習氛圍編程（vibe-coding）、代理與生成式 AI。
models: [gemma-4]
hide:
  - toc
---

AIventure 是一款互動式 AI 驅動的教育遊戲，使用 Angular、PhaserJS、Gemini 與 Gemma 4 打造。它示範如何使用 MediaPipe 與 WebGPU，直接在瀏覽器中執行本機 Gemma 4 模型，驅動動態的代理型非玩家角色（NPC），同時介紹氛圍編程與生成式 AI 的概念。

### 如何使用這個示範

1. 按一下下方的「**接受並繼續**」載入示範。首次執行時會在瀏覽器中下載並快取 Gemma 4 E4B 模型。
2. 在遊戲世界中移動並與角色互動，查看由裝置 GPU 在本機產生的回應。
3. 使用示範視窗左上角的「**全螢幕**」按鈕展開檢視。

如要深入瞭解架構、原始碼，以及打造自己的代理型遊戲的逐步說明，請造訪 [AIventure 解決方案頁面](https://developers.google.com/solutions/learn/aiventure){ data-model="gemma-4" }。

<div class="demo-embed">
  <div class="demo-disclaimer">
    <div class="demo-disclaimer-card" role="dialog" aria-label="Demo disclaimer">
      <p>
        此示範使用 Mediapipe 搭配 WebGPU，在本機載入並執行 Gemma 4 模型。<strong>首次執行需要下載 4 GB 的資料</strong>，之後會快取起來。效能會依裝置的 GPU 而有所不同。
      </p>
      <p>
        <strong>注意：</strong>瀏覽器的 WebGPU API 有一些額外負擔，與透過 Ollama 或 LM Studio 原生執行模型相比，效能可能會降低。此外，這個模型是量化版本。歡迎探索這個示範，然後自行下載程式碼，與 LM Studio 比較效能。
      </p>
      <div class="demo-disclaimer-actions">
        <button
          type="button"
          class="demo-disclaimer-btn"
          onclick="const embed = this.closest('.demo-embed'); const iframe = embed.querySelector('iframe'); iframe.src = iframe.dataset.src; iframe.hidden = false; const fsBtn = embed.querySelector('.demo-fullscreen-btn'); if (fsBtn) fsBtn.hidden = false; this.closest('.demo-disclaimer').hidden = true;"
        >
          接受並繼續
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
    <span>全螢幕</span>
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
