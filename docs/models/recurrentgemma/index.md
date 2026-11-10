---
title: RecurrentGemma
description: RecurrentGemma 是基於 Griffin 的開放模型；Griffin 是一種混合模型架構，結合門控線性遞迴（gated linear recurrences）與局部滑動視窗注意力（local sliding window attention）。
models: [recurrentgemma]
source_url: "https://ai.google.dev/gemma/docs/recurrentgemma.md.txt"
---
# RecurrentGemma

RecurrentGemma 是基於
[Griffin](https://arxiv.org/abs/2402.19427) 的開放模型；Griffin 是一種混合模型架構，
結合門控線性遞迴與局部滑動視窗注意力。


如同 Gemma，RecurrentGemma 非常適合
多種文字生成任務，包括問答、摘要與
推理。不過，RecurrentGemma 獨特的架構還帶來以下額外
優勢：

- **記憶體用量降低** — 較低的記憶體需求讓記憶體有限的裝置（如單張 GPU 或 CPU）也能生成較長的樣本。
- **吞吐量更高** — RecurrentGemma 可以用顯著更高的批次大小執行推論，意味著每秒能生成更多詞元——尤其是在生成長序列時。
- **高效能** — RecurrentGemma 在記憶體需求更低、推論速度更快的同時，效能與 Gemma 相當。

## 更多資源

### [查看模型卡](model-card.md)

RecurrentGemma 的模型卡包含模型的詳細資訊，包括實作資訊、評估資訊、模型用途與限制等。

### [在 Kaggle 上查看](https://www.kaggle.com/models/google/recurrentgemma){ data-model="recurrentgemma" }

在 Kaggle 上查看更多關於 RecurrentGemma 的程式碼、Colab 筆記本、資訊與討論。

### [在 GitHub 上執行](https://github.com/google-deepmind/recurrentgemma/blob/main/colabs/){ data-model="recurrentgemma" }

在 GitHub 上執行 JAX 與 PyTorch 的範例 Colab 筆記本。
