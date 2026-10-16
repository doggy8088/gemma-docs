---
title: RecurrentGemma 模型卡
description: RecurrentGemma 是在 Google 開發的全新遞迴架構上打造的開放語言模型家族。
models: [recurrentgemma]
source_url: "https://ai.google.dev/gemma/docs/recurrentgemma/model_card.md.txt"
---
# 模型卡

**模型頁面：** [RecurrentGemma](index.md)

**資源與技術文件：**

- [負責任的生成式 AI 工具包][responsible-ai]
- [Kaggle 上的 RecurrentGemma](https://www.kaggle.com/models/google/recurrentgemma){ data-model="recurrentgemma" }

**使用條款：** [條款](https://www.kaggle.com/models/google/gemma/license/consent)

**作者：** Google

## 模型資訊

### 模型摘要

#### 說明

RecurrentGemma 是基於 Google 開發的[全新遞迴
架構](https://arxiv.org/abs/2402.19427) 的開放語言模型家族。提供預訓練與指令微調版本，皆為英文。

如同 Gemma，RecurrentGemma 模型非常適合多種文字
生成任務，包括問答、摘要與推理。
由於其新穎的架構，RecurrentGemma 比
Gemma 需要更少的記憶體，並在生成長序列時達到更快的推論速度。

#### 輸入與輸出

- **輸入：** 文本字串（例如問題、提示，或要摘要的文件）。
- **輸出：** 回應輸入的生成英文文本（例如問題的答案、文件的摘要）。

#### 引用

```python
@article{recurrentgemma_2024,
    title={RecurrentGemma},
    url={},
    DOI={},
    publisher={Kaggle},
    author={Griffin Team, Alexsandar Botev and Soham De and Samuel L Smith and Anushan Fernando and George-Christian Muraru and Ruba Haroun and Leonard Berrada et al.},
    year={2024}
}
```

### 模型資料

#### 訓練資料集與資料處理

RecurrentGemma 使用與 Gemma 模型家族相同的訓練資料與資料處理方式。完整說明請參閱 [Gemma
模型卡](../gemma-1/index.md#model-data)。

## 實作資訊

### 訓練期間使用的硬體與框架

如同
[Gemma](../gemma-1/index.md#實作資訊)，
RecurrentGemma 使用
[TPUv5e](https://cloud.google.com/tpu/docs/intro-to-tpu?_gl=1*18wi411*_ga*MzE3NDU5OTY1LjE2MzQwNDA4NDY.*_ga_WH2QY8WWF5*MTcxMTA0MjUxMy4xNy4wLjE3MTEwNDI1MTkuMC4wLjA.&_ga=2.239449409.-317459965.1634040846)，
以 [JAX](https://github.com/jax-ml/jax) 與 [ML
Pathways](https://blog.google/technology/ai/introducing-pathways-next-generation-ai-architecture/) 訓練。

## 評估資訊

### 基準測試結果

#### 評估方法

這些模型針對大量不同的資料集與
指標進行評估，以涵蓋文字生成的不同面向：

#### 評估結果

| 基準測試 | 指標 | RecurrentGemma 2B | RecurrentGemma 9B |
|---|---|---|---|
| [MMLU](https://arxiv.org/abs/2009.03300) | 5-shot, top-1 | 38.4 | 60.5 |
| [HellaSwag](https://arxiv.org/abs/1905.07830) | 0-shot | 71.0 | 80.4 |
| [PIQA](https://arxiv.org/abs/1911.11641) | 0-shot | 78.5 | 81.3 |
| [SocialIQA](https://arxiv.org/abs/1904.09728) | 0-shot | 51.8 | 52.3 |
| [BoolQ](https://arxiv.org/abs/1905.10044) | 0-shot | 71.3 | 80.3 |
| [WinoGrande](https://arxiv.org/abs/1907.10641) | partial score | 67.8 | 73.6 |
| [CommonsenseQA](https://arxiv.org/abs/1811.00937) | 7-shot | 63.7 | 73.2 |
| [OpenBookQA](https://arxiv.org/abs/1809.02789) |   | 47.2 | 51.8 |
| [ARC-e](https://arxiv.org/abs/1911.01547) |   | 72.9 | 78.8 |
| [ARC-c](https://arxiv.org/abs/1911.01547) |   | 42.3 | 52.0 |
| [TriviaQA](https://arxiv.org/abs/1705.03551) | 5-shot | 52.5 | 70.5 |
| [Natural Questions](https://github.com/google-research-datasets/natural-questions) | 5-shot | 11.5 | 21.7 |
| [HumanEval](https://arxiv.org/abs/2107.03374) | pass@1 | 21.3 | 31.1 |
| [MBPP](https://arxiv.org/abs/2108.07732) | 3-shot | 28.8 | 42.0 |
| [GSM8K](https://arxiv.org/abs/2110.14168) | maj@1 | 13.4 | 42.6 |
| [MATH](https://arxiv.org/abs/2103.03874) | 4-shot | 11.0 | 23.8 |
| [AGIEval](https://arxiv.org/abs/2304.06364) |   | 23.8 | 39.3 |
| [BIG-Bench](https://arxiv.org/abs/2206.04615) |   | 35.3 | 55.2 |
| **平均** |   | 44.6 | 56.1 |

## 倫理與安全

### 倫理與安全評估

#### 評估方法

我們的評估方法包括結構化評估，以及針對相關內容政策的內部紅隊
測試。紅隊測試由多個不同的團隊執行，各自有不同的目標與人工評估指標。這些
模型針對倫理與安全相關的多個類別進行了評估，包括：

- **文字轉文字內容安全：** 針對安全政策（包括兒童性虐待與剝削、騷擾、暴力與血腥，以及仇恨言論）提示的人工評估。
- **文字轉文字再現性傷害：** 以 WinoBias 與 BBQ 資料集等相關學術資料集進行的基準測試。
- **記憶化：** 對訓練資料記憶的自動化評估，包括個人可識別資訊外洩的風險。
- **大規模傷害：** 針對「危險能力」的測試，例如化學、生物、放射性與核（CBRN）風險；以及說服與欺騙、網路安全與自主複製的測試。

#### 評估結果

倫理與安全評估的結果落在可接受的門檻內，符合兒童安全、內容安全、再現性傷害、
記憶化、大規模傷害等類別的[內部
政策](https://storage.googleapis.com/gweb-uniblog-publish-prod/documents/2023_Google_AI_Principles_Progress_Update.pdf#page=11)。
除了穩健的內部評估，BBQ、Winogender、WinoBias、
RealToxicity 與 TruthfulQA 等知名安全基準測試的結果也列於此處。

| 基準測試 | 指標 | RecurrentGemma 2B | RecurrentGemma 2B IT | RecurrentGemma 9B | RecurrentGemma 9B IT |
|---|---|---|---|---|---|
| [RealToxicity](https://arxiv.org/abs/2009.11462) | avg | 9.8 | 7.60 | 10.3 | 8.8 |
| [BOLD](https://arxiv.org/abs/2101.11718) |   | 39.3 | 52.3 | 39.8 | 47.9 |
| [CrowS-Pairs](https://aclanthology.org/2020.emnlp-main.154/) | top-1 | 41.1 | 43.4 | 38.7 | 39.5 |
| [BBQ Ambig](https://arxiv.org/abs/2110.08193v2) | top-1 | 62.6 | 71.1 | 95.9 | 67.1 |
| [BBQ Disambig](https://arxiv.org/abs/2110.08193v2) | top-1 | 58.4 | 50.8 | 78.6 | 78.9 |
| [Winogender](https://arxiv.org/abs/1804.09301) | top-1 | 55.1 | 54.7 | 59.0 | 64.0 |
| [TruthfulQA](https://arxiv.org/abs/2109.07958) |   | 35.1 | 42.7 | 38.6 | 47.7 |
| [WinoBias 1_2](https://arxiv.org/abs/1804.06876) |   | 58.4 | 56.4 | 61.5 | 60.6 |
| [WinoBias 2_2](https://arxiv.org/abs/1804.06876) |   | 90.0 | 75.4 | 90.2 | 90.3 |
| [Toxigen](https://arxiv.org/abs/2203.09509) |   | 56.7 | 50.0 | 58.8 | 64.5 |

## 模型用途與限制

### 已知限制

這些模型有一些使用者應注意的限制：

- **訓練資料**
    - 訓練資料的品質與多樣性會顯著影響模型的能力。訓練資料中的偏見或缺口可能導致模型回應的限制。
    - 訓練資料集的範圍決定了模型能有效處理的主題領域。
- **脈絡與任務複雜度**
    - LLM 更擅長可以用明確提示與指示框定的任務。開放式或高度複雜的任務可能具有挑戰性。
    - 模型的效能可能受提供的脈絡量影響（在一定範圍內，較長的脈絡通常會帶來較好的輸出）。
- **語言歧義與細微差異**
    - 自然語言本質上很複雜。LLM 可能難以掌握細微的差異、諷刺或比喻性語言。
- **事實準確性**
    - LLM 根據從訓練資料集學到的資訊生成回應，但它們不是知識庫。它們可能生成不正確或過時的事實陳述。
- **常識**
    - LLM 依賴語言中的統計模式。在某些情況下，它們可能缺乏運用常識推理的能力。

### 倫理考量與風險

大型語言模型（LLM）的開發引發了若干倫理議題。
在建立開放模型時，我們審慎考量了以下幾點：

- **偏見與公平性**
    - 以大規模真實世界文本資料訓練的 LLM 可能反映訓練素材中嵌入的社會文化偏見。這些模型經過仔細審查，本卡片說明了輸入資料預處理，並報告了後續評估。
- **錯誤資訊與濫用**
    - LLM 可能被濫用以生成虛假、誤導或有害的文字。
    - 本模型提供了負責任使用的指南，請參閱 [負責任的生成式 AI 工具包][responsible-ai]。
- **透明度與問責**
    - 本模型卡彙整了模型的架構、能力、限制與評估流程的詳細資訊。
    - 負責任開發的開放模型提供了分享創新的機會，讓 AI 生態系中的開發者與研究人員都能使用 LLM 技術。

已識別的風險與緩解措施：

- **偏見的延續：** 鼓勵在模型訓練、微調與其他使用案例期間進行持續監控（使用評估指標、人工審查），並探索去偏見技術。
- **有害內容的生成：** 內容安全的機制與指南至關重要。鼓勵開發者謹慎行事，並根據其特定的產品政策與應用使用案例實施適當的內容安全防護措施。
- **用於惡意目的的濫用：** 技術限制以及開發者與終端使用者的教育，有助於緩解 LLM 的惡意應用。提供了教育資源與使用者舉報濫用的機制。Gemma 模型的禁止用途列於我們的[使用
  條款](https://www.kaggle.com/models/google/gemma/license/consent)。
- **隱私侵犯：** 模型以經過篩選、移除 PII（個人可識別資訊）的資料訓練。鼓勵開發者以保護隱私的技術遵守隱私法規。

## 預期用途

### 應用

開放大型語言模型（LLM）在各行各業與領域有廣泛的應用。
以下潛在用途清單並不完整。其目的是提供背景資訊，說明模型建立者在模型訓練
與開發過程中考量的可能使用案例。

- **內容創作與溝通**
    - **文字生成：** 這些模型可用於生成創意文字格式，例如詩歌、劇本、程式碼、行銷文案、電子郵件草稿等。
    - **聊天機器人與對話式 AI：** 為客戶服務、虛擬助理或互動式應用程式提供對話介面。
    - **文字摘要：** 為文本語料庫、研究論文或報告生成簡潔摘要。
- **研究與教育**
    - **自然語言處理（NLP）研究：** 這些模型可作為研究人員實驗 NLP 技術、開發演算法並推動領域進展的基礎。
    - **語言學習工具：** 支援互動式語言學習體驗，協助文法修正或提供寫作練習。
    - **知識探索：** 透過生成摘要或回答特定主題的問題，協助研究人員探索大量文本。

### 效益

在發布時，相較於類似規模的模型，這個模型家族提供了專為負責任 AI 開發從頭設計的高效能開放
大型語言模型實作。

使用本文件所述的基準測試評估指標，這些模型
的表現優於其他同等規模的開放模型替代方案。

特別是，RecurrentGemma 模型的效能可與 Gemma
模型媲美，但推論速度更快、記憶體需求更低，尤其是在長序列上。
