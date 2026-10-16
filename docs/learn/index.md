---
title: 學習
description: 用圖解方式說明 Gemma 模型如何運作：從整體架構深入到注意力機制、嵌入等個別關鍵機制。
---

# 學習

這個章節說明 Gemma 模型的實際運作方式。文件其他部分告訴您要執行什麼，而這些頁面會把架構拆解並畫出來——層做了什麼、為何做出某個設計選擇，以及模型行為來自何處。

閱讀順序不限。這裡的內容都不是使用 Gemma 的先決條件。

## 接下來

<div class="card-grid">
<a class="card" href="../tutorials/visual-guides/gemma-4.md">
  <span class="card-icon">{{ icon('lucide/book-open') }}</span>
  <div class="card-title">Gemma 4 圖解指南</div>
  <p class="card-desc">逐步繪製架構的運作方式。</p>
</a>
<a class="card" href="../tutorials/visual-guides/gemma-4-12b.md">
  <span class="card-icon">{{ icon('lucide/book-open') }}</span>
  <div class="card-title">Gemma 4 12B 圖解指南</div>
  <p class="card-desc">統一的 12B 模型有何不同。</p>
</a>
<a class="card" href="../tutorials/visual-guides/mtp.md">
  <span class="card-icon">{{ icon('lucide/book-open') }}</span>
  <div class="card-title">MTP 圖解指南</div>
  <p class="card-desc">用圖解說明草稿模型與推測解碼。</p>
</a>
<a class="card" href="../tutorials/visual-guides/diffusiongemma.md">
  <span class="card-icon">{{ icon('lucide/book-open') }}</span>
  <div class="card-title">DiffusionGemma 圖解指南</div>
  <p class="card-desc">去噪如何產生文字而非像素。</p>
</a>
</div>

## 先前的 Gemma 解說系列

<div class="card-grid">
<a class="card" href="../tutorials/gemma-explained/overview.md">
  <span class="card-icon">{{ icon('lucide/book-open') }}</span>
  <div class="card-title">Gemma 架構總覽</div>
  <p class="card-desc">文字轉文字模型與程式碼專門化。</p>
</a>
<a class="card" href="../tutorials/gemma-explained/gemma-2.md">
  <span class="card-icon">{{ icon('lucide/book-open') }}</span>
  <div class="card-title">Gemma 2 新功能</div>
  <p class="card-desc">2B 與 9B 模型採用知識蒸餾的更新架構。</p>
</a>
<a class="card" href="../tutorials/gemma-explained/recurrentgemma.md">
  <span class="card-icon">{{ icon('lucide/book-open') }}</span>
  <div class="card-title">RecurrentGemma 架構</div>
  <p class="card-desc">混合局部注意力與線性遞迴的 Griffin 架構，實現快速長序列推論。</p>
</a>
<a class="card" href="../tutorials/gemma-explained/paligemma.md">
  <span class="card-icon">{{ icon('lucide/book-open') }}</span>
  <div class="card-title">PaliGemma 架構</div>
  <p class="card-desc">以文字與圖片為輸入、產生文字輸出的視覺語言模型。</p>
</a>
<a class="card" href="../tutorials/gemma-explained/gemma-3.md">
  <span class="card-icon">{{ icon('lucide/book-open') }}</span>
  <div class="card-title">Gemma 3 新功能</div>
  <p class="card-desc">5 比 1 交錯注意力與 128k 上下文處理的更新視覺語言模型。</p>
</a>
<a class="card" href="../tutorials/gemma-explained/embeddinggemma.md">
  <span class="card-icon">{{ icon('lucide/book-open') }}</span>
  <div class="card-title">EmbeddingGemma 架構</div>
  <p class="card-desc">用於搜尋與檢索的文字嵌入模型。</p>
</a>
</div>

--8<-- "glossary.md"
