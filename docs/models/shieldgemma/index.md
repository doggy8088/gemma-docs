---
title: ShieldGemma
description: ShieldGemma 是一組指令微調（instruction-tuned）模型，用於依據一組定義好的安全政策，評估文字與圖片的安全性。
models: [shieldgemma]
source_url: "https://ai.google.dev/gemma/docs/shieldgemma.md.txt"
---
# ShieldGemma {: #hidden-title style="display: none;" }

<div class="video-embed">
  <iframe src="https://www.youtube.com/embed/BgWO2iVPlyw"
          title="影片" loading="lazy" allowfullscreen
          allow="accelerometer; encrypted-media; gyroscope; picture-in-picture"
          referrerpolicy="strict-origin-when-cross-origin"></iframe>
</div>


ShieldGemma 是一組指令微調模型，用於依據一組定義好的安全政策，評估文字與圖片的安全性。您可以將這個模型納入生成式 AI 應用的較大實作中，協助評估並防止生成式 AI 應用違反安全政策。ShieldGemma 系列模型以開放權重提供，讓您可以針對特定使用場景進行微調。


ShieldGemma 2 是為圖片安全標註而建構的 40 億（4B）參數模型。
ShieldGemma 1 建構於 [Gemma 2](../gemma-4/index.md) 之上，提供 2B、9B 與 27B 三種參數規模。


- **內容安全評估** — 依據一組定義好的安全政策，評估提示輸入與輸出回應的安全性。
- **[可調校的開放模型](https://huggingface.co/collections/google/shieldgemma-67d130ef8da6af884072a789){ data-model="shieldgemma" }** — ShieldGemma 模型以開放權重提供，可針對您的特定使用場景進行微調。

## ShieldGemma 2

### [檢視模型卡](model-card-v2.md)

ShieldGemma 2 的模型卡包含關於模型實作、評估、模型用途與限制等詳細資訊。

### [在 Kaggle 上查看](https://www.kaggle.com/models/google/shieldgemma-2){ data-model="shieldgemma" }

在 Kaggle 上查看更多關於 ShieldGemma 的程式碼、Colab 筆記本、資訊與討論。

### [在 HuggingFace 上查看](https://huggingface.co/google/shieldgemma-2-4b-it){ data-model="shieldgemma" }

執行一個可用的範例，學習如何使用 ShieldGemma 評估圖片。

## ShieldGemma 1

### [檢視模型卡](model-card-v1.md)

ShieldGemma 的模型卡包含關於模型實作、評估、模型用途與限制等詳細資訊。

### [在 Kaggle 上查看](https://www.kaggle.com/models/google/shieldgemma){ data-model="shieldgemma" }

在 Kaggle 上查看更多關於 ShieldGemma 的程式碼、Colab 筆記本、資訊與討論。

### [在 Colab 執行](https://colab.research.google.com/github/google-gemini/gemma-cookbook/blob/main/responsible/shieldgemma_on_keras.ipynb){ data-model="shieldgemma" }

執行一個可用的範例，學習如何使用 ShieldGemma 評估文字提示輸入與輸出。
