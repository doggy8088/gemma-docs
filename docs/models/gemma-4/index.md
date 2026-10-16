---
title: Gemma 4
description: Gemma 是生成式人工智慧模型的家族，可用於各種生成任務，包括問答、摘要與推理。
models: [gemma-4]
source_url: "https://ai.google.dev/gemma/docs/core.md.txt"
---
# Gemma 4 {: #hidden-title style="display: none;" }

<div class="video-embed">
  <iframe src="https://www.youtube.com/embed/jZVBoFOJK-Q"
          title="影片" loading="lazy" allowfullscreen
          allow="accelerometer; encrypted-media; gyroscope; picture-in-picture"
          referrerpolicy="strict-origin-when-cross-origin"></iframe>
</div>

Gemma 是生成式人工智慧模型的家族，可用於各種生成任務，包括問答、摘要與推理。Gemma 模型提供開放權重，並允許負責任的[商業用途][gemma-terms]，讓您能在自己的專案與應用中微調與部署它們。

Gemma 4 模型家族包含四種為特定硬體需求量身打造的架構：

- **小型：**2B 與 4B 有效參數模型，為超行動、邊緣與瀏覽器部署打造（例如 Pixel、Chrome）。
- **稠密（Dense）：**強大的 31B 參數稠密模型，彌合伺服器級效能與本地執行之間的差距。
- **混合專家：**高效率的 26B MoE 模型，為高吞吐量、進階推理而設計。
- **統一（Unified）：**12B 參數的無編碼器多模態任務模型，以輸入的直接線性投影取代視覺與音訊編碼器。

您可以從
[Kaggle](https://www.kaggle.com/models?query=gemma-4&publisher=google){ data-model="gemma-4" } 與
[Hugging Face][hf-gemma4] 下載 Gemma 4 模型。
如需更多 Gemma 4 的技術細節，請見
[模型卡](model-card.md)與
[技術報告](https://goo.gle/Gemma4Report){ data-model="gemma-4" }。
舊版 Gemma 核心模型亦可下載。詳情請見[舊版 Gemma 模型](index.md#舊版-gemma-模型)。

<div class="cookbook-grid">
<a class="cookbook-card" href="https://www.kaggle.com/models?query=gemma-4&publisher=google" target="_blank" rel="noopener">
  <div class="cookbook-card-title">Kaggle <span class="cookbook-card-ext">↗</span></div>
  <p class="cookbook-card-desc">在 Kaggle 上取得</p>
</a>
<a class="cookbook-card" href="https://huggingface.co/collections/google/gemma-4" target="_blank" rel="noopener">
  <div class="cookbook-card-title">Hugging Face <span class="cookbook-card-ext">↗</span></div>
  <p class="cookbook-card-desc">在 Hugging Face 上取得</p>
</a>
</div>


## 功能

- **推理：**此家族所有模型皆設計為高能力的推理者，並支援可設定的[思考模式](../../get-started/capabilities/thinking.md)。
- **擴展多模態：**可處理文字、[圖像](../../get-started/capabilities/image.md)（所有模型均支援可變長寬比與解析度）、[影片](../../get-started/capabilities/video.md)與[音訊](../../get-started/capabilities/audio.md)（E2B、E4B 與 12B 模型原生支援）。
- **更大的上下文視窗：**小型模型具備 128K 上下文視窗，中型模型支援 256K。
- **增強的程式設計與智慧代理能力：**在程式設計基準測試上取得顯著進步，並內建[函式呼叫支援](../../get-started/capabilities/function-calling.md)，驅動高能力的自主智慧代理。
- **原生系統提示支援：**Gemma 4 內建對 system 角色的支援，實現更結構化、可控的對話。
- **[多詞元預測](../../get-started/capabilities/mtp.md)：**所有 Gemma 4 模型（E2B、E4B、12B、31B 與 26B A4B）皆包含用於推測解碼的專用草稿模型，可在不損失品質的情況下顯著加速推論。

## 參數規模與量化

Gemma 4 模型提供 5 種參數規模：E2B、E4B、12B、31B 與 26B A4B。模型可以預設精度（16 位元）使用，也可以透過量化以較低精度使用。不同的規模與精度代表您的 AI 應用的一組取捨：參數與位元數較高（精度較高）的模型通常能力更強，但執行成本更高，包括運算週期、記憶體成本與功耗。參數與位元數較低（精度較低）的模型能力較弱，但可能已足以應付您的 AI 任務。

### Gemma 4 推論記憶體需求

下表詳列執行各尺寸 Gemma 4 模型版本推論所需的 GPU 或 TPU 記憶體約略值。

!!! Note
    這些數字可能因您的推論工具與環境而異。


| 參數量 | BF16 (16-bit) | SFP8 (8-bit) | Q4_0 (4-bit) | 行動版 | 行動版（僅文字） |
|---|---|---|---|---|---|
| Gemma 4 E2B | 11.4 GB | 5.7 GB | 2.9 GB | 1.1 GB | 0.84 GB |
| Gemma 4 E4B | 17.9 GB | 8.9 GB | 4.5 GB | 2.5 GB | 2.2 GB |
| Gemma 4 12B | 26.7 GB | 13.4 GB | 6.7 GB | - | - |
| Gemma 4 26B A4B | 57.7 GB | 28.8 GB | 14.4 GB | - | - |
| Gemma 4 31B | 69.9 GB | 34.9 GB | 17.5 GB | - | - |

**表 1.** 根據參數量、量化等級，載入 Gemma 4 模型所需的 GPU 或 TPU 記憶體約略值（含額外載入項目的 20% 開銷）。行動版使用 LiteRT-LM。

### 記憶體規劃要點

- **高效架構（E2B 與 E4B）：**「E」代表「有效」參數。小型模型採用逐層嵌入（PLE）以最大化裝置端部署的參數效率。PLE 不為模型增加更多層，而是讓每個解碼器層為每個詞元擁有自己的小型嵌入。這些嵌入表雖然龐大，但僅用於快速查表，這也是載入靜態權重所需的總記憶體高於有效參數量所暗示的原因。
- **MoE 架構（26B A4B）：**26B 是混合專家模型。雖然它在生成期間每個詞元僅啟用 40 億個參數，但**全部 260 億個參數**都必須載入記憶體，才能維持快速的路由與推論速度。這也是其基準記憶體需求更接近稠密 26B 模型而非 4B 模型的原因。
- **僅基礎權重：**上表中的估計*僅*計入載入靜態模型權重所需的記憶體，不包括支援軟體或上下文視窗所需的額外 VRAM。
- **上下文視窗（KV 快取）：**記憶體消耗會根據提示中的詞元總數與生成的回應動態增加。較大的上下文視窗需要在基礎模型權重之外使用顯著更多的 VRAM。
- **微調開銷：***微調* Gemma 模型所需的記憶體遠高於標準推論。您的實際用量將高度取決於開發框架、批次大小，以及使用全精度微調或參數高效微調（PEFT）方法（如低秩適應，LoRA）。

### 量化感知訓練（QAT）

對於要求最高效率且品質妥協最小的部署，Gemma 提供官方的**量化感知訓練（Quantization-Aware Training, QAT）**模型。

與標準的訓練後量化（PTQ）不同——後者壓縮已完全訓練的模型，可能導致品質下降——QAT 將量化模擬整合到訓練過程本身。這讓模型學會補償精度損失，產出的小型模型表現幾乎與高精度基線相同。

#### 快速路由表

| 目標部署引擎 | 下載後綴 | 主要使用情境 |
|---|---|---|
| llama.cpp / LM Studio（本機） | `{model-name}-qat-q4_0-gguf` | 在 CPU、Apple Silicon 或消費級 GPU 上零設定本機部署。 |
| vLLM / SGLang | 伺服器版：`{model-name}-qat-w4a16-ct` 行動版：`{model-name}-qat-mobile-ct` | 使用 4 位元權重與 16 位元激活的高吞吐量推論。 |
| 推測解碼 | 主模型：`{model-name}-qat-q4_0-unquantized` 草稿模型：`{model-name}-qat-q4_0-unquantized-assistant` | 將主模型與其對應的 MTP 草稿模型並行運行，大幅加速詞元生成。模型必須經過量化。 |
| 其他格式 | `{model-name}-qat-q4_0-unquantized` | 用於轉換為其他格式（例如 MLX）的未量化權重 |
| 行動部署（Transformers） | `{model-name}-qat-mobile-transformers` | 針對行動使用情境優化的邊緣權重，可作為其他格式的參考。 |

Hugging Face 上的官方 QAT 集合

- **[collections/google/gemma-4-qat-q4-0][hf-gemma4-qat-q40]**
    - **未量化 QAT 檢查點（`-unquantized` / `-assistant`）：**直接從 QAT 流程提取的半精度權重。非常適合自訂下游編譯、研究，或使用輔助草稿模型進行推測解碼。*適用於 Gemma 4 E2B、E4B、12B、26B A4B 與 31B。*
    - **GGUF（`-gguf`）：**可直接相容於本機 LLM 生態系的檢查點。*適用於 Gemma 4 E2B、E4B、12B、26B A4B 與 31B。*
    - **Compressed Tensors（`-w4a16-ct`）：**以 `compressed-tensors` 標準原生序列化，為優化、高併發的雲端服務而設計。*適用於 Gemma 4 E2B、E4B、12B 與 31B。*
- **[collections/google/gemma-4-qat-mobile][hf-gemma4-qat-mobile]**
    - **行動優化（`-mobile-transformers` / `-mobile-ct`）：**基於專為行動硬體限制設計的自訂 `wNa8o8` 結構。利用針對性的 2 位元解碼層、優化的 KV 快取與靜態激活，在不拖慢邊緣處理器的前提下，最大化裝置端 RAM 節省。*適用於 Gemma 4 E2B 與 E4B。*

所有官方 Gemma 4 QAT 檢查點也可直接從
[Kaggle](https://www.kaggle.com/models/google/gemma-4/transformers){ data-model="gemma-4" } 取得。

## 舊版 Gemma 模型

您也可以使用舊世代的 Gemma 模型，這些模型同樣可從 [Kaggle](https://www.kaggle.com/models?query=gemma) 與
[Hugging Face](https://huggingface.co/google/collections) 取得。
如需舊版 Gemma 模型的更多技術細節，請見以下模型卡頁面：

- Gemma 3 [模型卡](../gemma-3/index.md)
- Gemma 2 [模型卡](../gemma-2/index.md)
- Gemma 1 [模型卡](../gemma-1/index.md)

準備好開始打造了嗎？
[立即開始](../../get-started/index.md)
使用 Gemma 模型！

--8<-- "glossary.md"
