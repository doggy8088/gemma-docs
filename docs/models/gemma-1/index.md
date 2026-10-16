---
title: Gemma 模型卡
description: "第一代 Gemma 文字轉文字模型的模型卡：模型說明、輸入與輸出格式、訓練資料、評測結果與使用限制。"
models: [gemma-1]
source_url: "https://ai.google.dev/gemma/docs/core/model_card.md.txt"
---
# 模型卡

**模型頁面**：[Gemma](index.md)

**資源與技術文件**：

- [負責任生成式 AI 工具包][responsible-ai]
- [Kaggle 上的 Gemma](https://www.kaggle.com/models/google/gemma)
- [Vertex Model Garden 上的 Gemma](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/335)

**使用條款**：[條款][gemma-terms]

**作者**：Google

## 模型資訊

摘要說明，以及輸入與輸出的簡要定義。

### 說明

Gemma 是 Google 推出的輕量級、最先進的開放模型家族，採用與打造 Gemini 模型相同的技術與研究成果。這些模型是文字轉文字的僅解碼器大型語言模型，提供英文版本，具備開放權重，並有預訓練與指令微調版本。Gemma 模型非常適合各種文字生成任務，包括問答、摘要與推理。相對較小的體積使其能夠部署在資源有限的環境中，例如筆記型電腦、桌機或您自己的雲端基礎設施，讓最先進的 AI 模型更加普及，並促進人人創新。

### 輸入與輸出

- **輸入：**文字字串，例如問題、提示或待摘要的文件。
- **輸出：**針對輸入生成的英文文字，例如問題的答案或文件摘要。

### 引用

```python
@article{gemma_2024,
    title={Gemma},
    url={https://www.kaggle.com/m/3301},
    DOI={10.34740/KAGGLE/M/3301},
    publisher={Kaggle},
    author={Gemma Team, Thomas Mesnard and Cassidy Hardin and Robert Dadashi and Surya Bhupatiraju and Laurent Sifre and Morgane Rivière and Mihir Sanjay Kale and Juliette Love and Pouya Tafti and Léonard Hussenot and et al.},
    year={2024}
}
```

## 模型資料

用於模型訓練的資料，以及資料的處理方式。

### 訓練資料集

這些模型在包含各種來源的文字資料集上進行訓練，共計 6 兆個詞元。主要組成如下：

- 網路文件：多元的網路文字蒐集，讓模型接觸廣泛的語言風格、主題與詞彙。主要為英文內容。
- 程式碼：讓模型接觸程式碼有助於學習程式語言的語法與模式，進而提升生成程式碼或理解程式相關問題的能力。
- 數學：以數學文字訓練有助於模型學習邏輯推理、符號表示，並處理數學問題。

這些多元資料來源的組合，對於訓練出能處理各種任務與文字格式的強大語言模型至關重要。

### 資料預處理

以下為應用於訓練資料的主要資料清理與過濾方法：

- CSAM 過濾：在資料準備過程的多個階段實施嚴格的 CSAM（兒童性虐待內容）過濾，以確保排除有害與非法內容。
- 敏感資料過濾：為使 Gemma 預訓練模型安全可靠，使用自動化技術從訓練集中過濾掉部分個人資訊與其他敏感資料。
- 其他方法：依據內容品質與安全性進行過濾，符合[我們的政策](https://storage.googleapis.com/gweb-uniblog-publish-prod/documents/2023_Google_AI_Principles_Progress_Update.pdf#page=11)。

## 實作資訊

模型內部機制的詳細資訊。

### 硬體

Gemma 使用最新一代的[張量處理單元（Tensor Processing Unit, TPU）](https://cloud.google.com/tpu/docs/intro-to-tpu)硬體（TPUv5e）進行訓練。

訓練大型語言模型需要龐大的運算能力。TPU 專為機器學習中常見的矩陣運算而設計，在此領域具有多項優勢：

- 效能：TPU 專為處理訓練 LLM 所涉及的大規模運算而設計。與 CPU 相比，可大幅加速訓練。
- 記憶體：TPU 通常配備大量高頻寬記憶體，可在訓練期間處理大型模型與批次大小，有助於提升模型品質。
- 可擴展性：TPU Pod（大型 TPU 叢集）為應對日益複雜的大型基礎模型提供了可擴展的解決方案。您可以將訓練分散到多個 TPU 裝置，以更快、更有效率地處理。
- 成本效益：在許多情境下，與基於 CPU 的基礎設施相比，TPU 為訓練大型模型提供了更具成本效益的方案，尤其是考量到更快的訓練所節省的時間與資源。
- 這些優勢與[Google 永續營運的承諾](https://sustainability.google/operating-sustainably/)相符。

### 軟體

訓練使用 [JAX](https://github.com/jax-ml/jax) 與 [ML Pathways](https://blog.google/technology/ai/introducing-pathways-next-generation-ai-architecture/) 進行。

JAX 讓研究人員得以利用最新一代硬體（包括 TPU），更快、更有效率地訓練大型模型。

ML Pathways 是 Google 為打造能跨多個任務泛化的人工智慧系統所做的最新努力。這特別適合基礎模型，包括這類大型語言模型。

JAX 與 ML Pathways 的共同使用方式如[關於 Gemini 模型家族的論文](https://arxiv.org/abs/2312.11805)所述：「Jax 與 Pathways 的『單一控制器』程式設計模型，讓單一 Python 行程即可編排整個訓練流程，大幅簡化了開發工作流程。」

## 評測

模型評測指標與結果。

### 基準測試結果

這些模型使用大量不同的資料集與指標進行評測，涵蓋文字生成的各個面向：

| 基準測試 | 指標 | Gemma PT 2B | Gemma PT 7B |
|---|---|---|---|
| [MMLU](https://arxiv.org/abs/2009.03300) | 5-shot, top-1 | 42.3 | 64.3 |
| [HellaSwag](https://arxiv.org/abs/1905.07830) | 0-shot | 71.4 | 81.2 |
| [PIQA](https://arxiv.org/abs/1911.11641) | 0-shot | 77.3 | 81.2 |
| [SocialIQA](https://arxiv.org/abs/1904.09728) | 0-shot | 49.7 | 51.8 |
| [BoolQ](https://arxiv.org/abs/1905.10044) | 0-shot | 69.4 | 83.2 |
| [WinoGrande](https://arxiv.org/abs/1907.10641) | 部分分數 | 65.4 | 72.3 |
| [CommonsenseQA](https://arxiv.org/abs/1811.00937) | 7-shot | 65.3 | 71.3 |
| [OpenBookQA](https://arxiv.org/abs/1809.02789) |   | 47.8 | 52.8 |
| [ARC-e](https://arxiv.org/abs/1911.01547) |   | 73.2 | 81.5 |
| [ARC-c](https://arxiv.org/abs/1911.01547) |   | 42.1 | 53.2 |
| [TriviaQA](https://arxiv.org/abs/1705.03551) | 5-shot | 53.2 | 63.4 |
| [Natural Questions](https://github.com/google-research-datasets/natural-questions) | 5-shot | 12.5 | 23.0 |
| [HumanEval](https://arxiv.org/abs/2107.03374) | pass@1 | 22.0 | 32.3 |
| [MBPP](https://arxiv.org/abs/2108.07732) | 3-shot | 29.2 | 44.4 |
| [GSM8K](https://arxiv.org/abs/2110.14168) | maj@1 | 17.7 | 46.4 |
| [MATH](https://arxiv.org/abs/2103.03874) | 4-shot | 11.8 | 24.3 |
| [AGIEval](https://arxiv.org/abs/2304.06364) |   | 24.2 | 41.7 |
| [BIG-Bench](https://arxiv.org/abs/2206.04615) |   | 35.2 | 55.1 |
| **平均** |   | **44.9** | **56.4** |

## 倫理與安全

倫理與安全評測方法及結果。

### 評測方法

我們的評測方法包括結構化評測，以及針對相關內容政策的內部紅隊測試。紅隊測試由多個不同團隊執行，各自有不同的目標與人工評測指標。這些模型針對與倫理及安全相關的多個類別進行評測，包括：

- 文字轉文字內容安全：針對涵蓋安全政策的提示進行人工評測，包括兒童性虐待與剝削、騷擾、暴力與血腥，以及仇恨言論。
- 文字轉文字表徵危害：以相關學術資料集進行基準測試，例如 [WinoBias](https://arxiv.org/abs/1804.06876) 與 [BBQ 資料集](https://arxiv.org/abs/2110.08193v2)。
- 記憶化：對訓練資料記憶化進行自動評測，包括個人可識別資訊外洩的風險。
- 大規模危害：測試「危險能力」，例如化學、生物、放射性與核（CBRN）風險。

### 評測結果

倫理與安全評測結果在兒童安全、內容安全、表徵危害、記憶化、大規模危害等類別上，均符合[內部政策](https://storage.googleapis.com/gweb-uniblog-publish-prod/documents/2023_Google_AI_Principles_Progress_Update.pdf#page=11)的可接受閾值。除了嚴謹的內部評測，BBQ、BOLD、Winogender、Winobias、RealToxicity、TruthfulQA 等知名安全基準測試的結果如下。

#### Gemma 1.0

| 基準測試 | 指標 | Gemma 1.0 IT 2B | Gemma 1.0 IT 7B |
|---|---|---|---|
| [RealToxicity](https://arxiv.org/abs/2009.11462) | 平均 | 6.86 | 7.90 |
| [BOLD](https://arxiv.org/abs/2101.11718) |   | 45.57 | 49.08 |
| [CrowS-Pairs](https://aclanthology.org/2020.emnlp-main.154/) | top-1 | 45.82 | 51.33 |
| [BBQ Ambig](https://arxiv.org/abs/2110.08193v2) | 1-shot, top-1 | 62.58 | 92.54 |
| [BBQ Disambig](https://arxiv.org/abs/2110.08193v2) | top-1 | 54.62 | 71.99 |
| [Winogender](https://arxiv.org/abs/1804.09301) | top-1 | 51.25 | 54.17 |
| [TruthfulQA](https://arxiv.org/abs/2109.07958) |   | 44.84 | 31.81 |
| [Winobias 1_2](https://arxiv.org/abs/1804.06876) |   | 56.12 | 59.09 |
| [Winobias 2_2](https://arxiv.org/abs/1804.06876) |   | 91.10 | 92.23 |
| [Toxigen](https://arxiv.org/abs/2203.09509) |   | 29.77 | 39.59 |

#### Gemma 1.1

| 基準測試 | 指標 | Gemma 1.1 IT 2B | Gemma 1.1 IT 7B |
|---|---|---|---|
| [RealToxicity](https://arxiv.org/abs/2009.11462) | 平均 | 7.03 | 8.04 |
| [BOLD](https://arxiv.org/abs/2101.11718) |   | 47.76 |   |
| [CrowS-Pairs](https://aclanthology.org/2020.emnlp-main.154/) | top-1 | 45.89 | 49.67 |
| [BBQ Ambig](https://arxiv.org/abs/2110.08193v2) | 1-shot, top-1 | 58.97 | 86.06 |
| [BBQ Disambig](https://arxiv.org/abs/2110.08193v2) | top-1 | 53.90 | 85.08 |
| [Winogender](https://arxiv.org/abs/1804.09301) | top-1 | 50.14 | 57.64 |
| [TruthfulQA](https://arxiv.org/abs/2109.07958) |   | 44.24 | 45.34 |
| [Winobias 1_2](https://arxiv.org/abs/1804.06876) |   | 55.93 | 59.22 |
| [Winobias 2_2](https://arxiv.org/abs/1804.06876) |   | 89.46 | 89.2 |
| [Toxigen](https://arxiv.org/abs/2203.09509) |   | 29.64 | 38.75 |

## 使用方式與限制

這些模型存在使用者應注意的特定限制。

### 預期用途

開放大型語言模型（LLM）在各產業與領域有廣泛應用。以下潛在用途清單並不詳盡，目的在於提供背景資訊，說明模型建立者在模型訓練與開發過程中所考量的可能使用情境。

- 內容創作與溝通
    - 文字生成：這些模型可用於生成詩歌、劇本、程式碼、行銷文案與電子郵件草稿等創意文字格式。
    - 聊天機器人與對話式 AI：為客戶服務、虛擬助理或互動式應用程式提供對話介面。
    - 文字摘要：為文字語料庫、研究論文或報告生成精簡摘要。
- 研究與教育
    - 自然語言處理（NLP）研究：這些模型可作為研究人員的基礎，用於實驗 NLP 技術、開發演算法，並推動該領域的發展。
    - 語言學習工具：支援互動式語言學習體驗，協助文法修正或提供寫作練習。
    - 知識探索：協助研究人員探索大量文字，透過生成摘要或回答特定主題的問題。

### 限制

- 訓練資料
    - 訓練資料的品質與多樣性會顯著影響模型的能力。訓練資料中的偏見或缺口可能導致模型回應受限。
    - 訓練資料集的範圍決定了模型能有效處理的主題領域。
- 上下文與任務複雜度
    - LLM 較擅長能以明確提示與指示框架化的任務。開放式或高度複雜的任務可能較具挑戰性。
    - 模型的效能可能受提供的上下文量影響（上下文越長通常輸出越好，但有其上限）。
- 語言歧義與細微差別
    - 自然語言本質上很複雜。LLM 可能難以掌握細微差別、諷刺或比喻性語言。
- 事實準確性
    - LLM 根據從訓練資料集學到的資訊生成回應，但它們並非知識庫，可能會生成錯誤或過時的事實陳述。
- 常識
    - LLM 依賴語言中的統計模式，在某些情境下可能缺乏應用常識推理的能力。

### 倫理考量與風險

大型語言模型（LLM）的開發引發了若干倫理議題。在打造開放模型時，我們審慎考量了以下事項：

- 偏見與公平性
    - 以大規模真實世界文字資料訓練的 LLM 可能反映訓練資料中蘊含的社會文化偏見。這些模型經過嚴格審查，輸入資料預處理如上所述，後續評測結果載於本卡。
- 錯誤資訊與濫用
    - LLM 可能被濫用來生成虛假、誤導或有害的文字。
    - 我們提供負責任使用模型的指南，詳見 [負責任生成式 AI 工具包][responsible-ai]。
- 透明度與問責制：
    - 本模型卡彙整了模型的架構、能力、限制與評測流程的詳細資訊。
    - 一個負責任地開發的開放模型，讓整個 AI 生態系的開發者與研究人員都能取用 LLM 技術，提供了共享創新的機會。

已識別的風險與緩解措施：

- 偏見的延續：鼓勵在模型訓練、微調與其他使用情境中持續監控（使用評測指標、人工審查），並探索去偏見技術。
- 有害內容的生成：內容安全機制與指南至關重要。鼓勵開發者謹慎行事，並依據其特定產品政策與應用情境實施適當的內容安全防護。
- 惡意用途的濫用：技術限制，以及開發者與終端使用者的教育，有助於緩解 LLM 的惡意應用。我們提供教育資源與回報機制，供使用者舉報濫用。Gemma 模型的禁用用途載於 [Gemma 禁用用途政策][prohibited-use]。
- 隱私侵犯：模型在經過過濾以移除 PII（個人可識別資訊）的資料上訓練。鼓勵開發者遵守隱私法規，並採用隱私保護技術。

### 優點

在發布時，與同等規模的模型相比，此模型家族提供從零開始為負責任 AI 開發而設計的高效能開放大型語言模型實作。

根據本文件所述的基準測試評測指標，這些模型的表現優於其他同等規模的開放模型替代方案。
