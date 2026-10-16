---
title: PaliGemma
description: PaliGemma 2 與 PaliGemma 是輕量級開放視覺語言模型（VLM），受 PaLI-3 啟發，並基於 SigLIP 視覺模型與 Gemma 語言模型等開放元件構建。
models: [paligemma]
source_url: "https://ai.google.dev/gemma/docs/paligemma.md.txt"
---
# PaliGemma

PaliGemma 2 與 PaliGemma 是輕量級開放*視覺語言模型*（VLM），受 [PaLI-3](https://arxiv.org/abs/2310.09199) 啟發，並基於 [SigLIP 視覺模型](https://arxiv.org/abs/2303.15343)與 [Gemma 語言模型](https://arxiv.org/abs/2403.08295)等開放元件構建。PaliGemma 以圖片與文字作為輸入，能結合細節與上下文回答與圖片相關的問題，這意味著 PaliGemma 可以對圖片進行更深入的分析並提供實用的見解，例如圖片與短片字幕、物件偵測，以及讀取圖片中嵌入的文字。


PaliGemma 2 提供 3B、10B 與 28B 參數規模，分別基於 Gemma 2 的 2B、9B 與 27B 模型。原始的 PaliGemma 模型提供 3B 規模。如需更多關於 Gemma 模型變體的資訊，請參閱 [Gemma 模型總覽](../index.md)。PaliGemma 模型變體支援不同的圖片輸入像素解析度，包括 224 x 224、448 x 448 與 896 x 896 像素。


您可以從以下網站查看並下載 PaliGemma 模型：

- 從 [Kaggle](https://www.kaggle.com/models?query=paligemma){ data-model="paligemma" } 下載。
- 從 [Hugging Face](https://huggingface.co/models?search=google/paligemma){ data-model="paligemma" } 下載。


!!! Important
    **預訓練**（Pretrained）的 PaliGemma 模型*需要*經過調校才能產生實用的結果。在將這些 PaliGemma 模型部署給終端使用者*之前*，請務必對它們執行微調並測試輸出。


## 模型類別

PaliGemma 模型分為三個類別：

- **PaliGemma PT** - 通用預訓練模型，可在各種任務上進行微調。
- **PaliGemma FT** - 研究導向模型，在特定研究資料集上進行微調。
- **PaliGemma mix** - 針對任務混合調校的模型，可開箱即用於常見使用案例。


主要優勢包括：

- **多模態能力** — 同時處理圖片與文字輸入。
- **多用途基礎模型** — 可在廣泛的視覺語言任務上進行微調。
- **開箱即用探索** — 隨附在任務混合上微調的檢查點，可立即用於研究。

## 深入了解

### [在 Colab 中執行](inference.md)

在 Colab 中使用 PaliGemma 試用偵測與內容生成功能。

### [在 Colab 中調校](fine-tuning.md)

在 Colab 中使用 JAX，以圖片資料微調 PaliGemma 模型。

### [在 Kaggle 上查看](https://www.kaggle.com/models/google/paligemma-2){ data-model="paligemma" }

在 Kaggle 上查看更多關於 PaliGemma 的程式碼、Colab 筆記本、資訊與討論。

--8<-- "glossary.md"
