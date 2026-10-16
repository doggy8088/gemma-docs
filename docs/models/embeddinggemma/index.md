---
title: EmbeddingGemma 模型概覽
description: EmbeddingGemma 2 是一個輕量、開源的 740M 參數模型，專為統一多模態嵌入而設計。
models: [embeddinggemma]
source_url: "https://ai.google.dev/gemma/docs/embeddinggemma.md.txt"
---
# EmbeddingGemma 模型概覽 {: #hidden-title style="display: none;" }

<div class="video-embed">
  <iframe src="https://www.youtube.com/embed/anPsS6huQk0"
          title="Video" loading="lazy" allowfullscreen
          allow="accelerometer; encrypted-media; gyroscope; picture-in-picture"
          referrerpolicy="strict-origin-when-cross-origin"></iframe>
</div>

**EmbeddingGemma 2** 是一個輕量、開源的 740M 參數模型，專為統一多模態嵌入而設計。它基於 Gemma 4 解碼器架構，能將**文字、圖像、音訊和影片**映射到統一的 768 維向量空間。您可以在消費級 GPU 或 CPU 上，於本機離線執行要求嚴苛的跨模態工作流程（例如多模態檢索增強生成（RAG）、語義搜尋和零樣本分類）。

<div class="cookbook-grid">
<a class="cookbook-card" href="https://huggingface.co/collections/google/embeddinggemma-2" target="_blank" rel="noopener">
  <div class="cookbook-card-title">Hugging Face <span class="cookbook-card-ext">↗</span></div>
  <p class="cookbook-card-desc">前往 Hugging Face 取得</p>
</a>
</div>

EmbeddingGemma 2 以開放權重提供，並採用 [Apache 2.0 授權](../../resources/apache-2.md)，允許您在自己的專案與應用中微調與部署。

[試用 EmbeddingGemma 2](inference.md)

[微調 EmbeddingGemma 2](fine-tuning.md)

## 主要特色

EmbeddingGemma 2 相較於 EmbeddingGemma 有重大改進：

- **原生多模態**：在單一統一向量空間中，直接支援零樣本的文字、程式碼、圖像、影片與音訊搜尋。
- **文字與程式碼品質更優**：在 Code MTEB（大規模文字嵌入基準測試）上超越 EmbeddingGemma 1。
- **更小的基礎體積**：將基礎文字與程式碼模型從 3 億參數縮減至 **2.7 億參數**。
- **高儲存效率**：Matryoshka Representation Learning (MRL) 可動態截斷輸出維度（**128、256、512 與 768 維**），在品質幾乎不受影響的情況下，將向量資料庫儲存成本降低高達 **6 倍**。
- **更長的上下文長度**：支援高達 **8,192 個詞元**的文字與原始碼，涵蓋 100 多種語言。

!!! tip
    **提示：** 搭配 Gemma 4 E2B 部署 EmbeddingGemma 2，打造情境相關、行動優先的檢索增強生成（RAG）管線與聊天機器人。請參閱我們的[快速入門 RAG notebook](https://github.com/google-gemma/cookbook/blob/main/tutorials/RAG_with_EmbeddingGemma.ipynb){ data-model="embeddinggemma-2" } 開始使用。

## 先前版本

### EmbeddingGemma

EmbeddingGemma 是一個基於 Gemma 3 的 308M 參數多語言文字嵌入模型。它針對日常裝置（例如手機、筆記型電腦和平板）進行了最佳化。該模型產生文字的數值表示，可用於資訊檢索、語義相似度搜尋、分類和分群等下游任務。

EmbeddingGemma 包含以下主要特色：

- **多語言支援**：具備廣泛的語言資料理解能力，支援 100 多種語言訓練。
- **彈性輸出維度**：使用 Matryoshka Representation Learning (MRL)，可將輸出維度從 768 自訂為 128，以在速度與儲存之間取捨。
- **2K 詞元上下文**：直接在您的硬體上處理文字資料與文件的充足輸入上下文。
- **儲存高效**：量化後在不到 200MB 的 RAM 上即可執行。
- **低延遲**：在 EdgeTPU 上不到 22ms 即可產生嵌入，應用快速流暢。
- **離線安全**：直接在您的硬體上產生文件嵌入，無需網路連線，讓敏感資料保持安全。

--8<-- "glossary.md"
