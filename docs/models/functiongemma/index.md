---
title: FunctionGemma 模型總覽
description: FunctionGemma 是我們的 Gemma 3 270M 模型的專門版本，針對函式呼叫微調。
models: [functiongemma]
source_url: "https://ai.google.dev/gemma/docs/functiongemma.md.txt"
---
# FunctionGemma 模型總覽 {: #hidden-title style="display: none;" }

<div class="video-embed">
  <iframe src="https://www.youtube.com/embed/-Tgc_9uYJLI"
          title="Video" loading="lazy" allowfullscreen
          allow="accelerometer; encrypted-media; gyroscope; picture-in-picture"
          referrerpolicy="strict-origin-when-cross-origin"></iframe>
</div>

FunctionGemma 是我們的 Gemma 3 270M 模型的專門版本，針對函式呼叫微調。它被設計為強大的基礎，可進一步訓練成自訂、快速、私有、本機的代理程式，把自然語言翻譯成可執行的 API 動作。

## 何時選擇 FunctionGemma

FunctionGemma 是自然語言與軟體執行之間的橋樑。在以下情況它是正確的選擇：

- **您有明確定義的 API 範圍：**您的應用程式有明確定義的動作集合（例如智慧家庭、媒體、導航）。
- **您準備好微調：**您需要來自特定資料微調的穩定、可確定的行為，而非零樣本提示（zero-shot prompting）的變異性。
- **您優先考慮本機優先部署：**您的應用程式需要近乎即時的延遲與完全的資料隱私，並在邊緣裝置的運算與電池限制內有效率地執行。
- **您正在建構複合系統：**您需要輕量級的邊緣模型處理本機動作，讓系統能在裝置端處理常見指令，只把更複雜的任務交給更大的模型（如 Gemma 3 27B）。

<div class="cookbook-grid">
<a class="cookbook-card" href="https://huggingface.co/google/functiongemma-270m-it" target="_blank" rel="noopener">
  <div class="cookbook-card-title">Hugging Face <span class="cookbook-card-ext">↗</span></div>
  <p class="cookbook-card-desc">在 Hugging Face 上取得</p>
</a>
<a class="cookbook-card" href="https://www.kaggle.com/models/google/functiongemma" target="_blank" rel="noopener">
  <div class="cookbook-card-title">Kaggle <span class="cookbook-card-ext">↗</span></div>
  <p class="cookbook-card-desc">在 Kaggle 上取得</p>
</a>
</div>

[在 Vertex 上使用](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/functiongemma){ data-model="functiongemma" }

與其他 Gemma 模型一樣，FunctionGemma 以開放權重提供，並授權用於負責任的[商業用途][gemma-terms]，讓您能在自己的專案與應用中微調與部署。

[格式與最佳實務](formatting.md)
[試用 FunctionGemma](inference.md)

[微調 FunctionGemma](fine-tuning.md)
