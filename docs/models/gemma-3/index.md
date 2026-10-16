---
title: Gemma 3 模型卡
description: "Gemma 3 多模態模型的模型卡：128K 上下文、140 種以上語言、輸入與輸出、訓練資料與評測。"
models: [gemma-3]
source_url: "https://ai.google.dev/gemma/docs/core/model_card_3.md.txt"
---
# 模型卡

**模型頁面**：[Gemma](index.md)

**資源與技術文件**：

- [Gemma 3 技術報告](https://arxiv.org/abs/2503.19786)
- [負責任生成式 AI 工具包][responsible-ai]
- [Kaggle 上的 Gemma](https://www.kaggle.com/models/google/gemma-3){ data-model="gemma-3" }
- [Vertex Model Garden 上的 Gemma](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/gemma3){ data-model="gemma-3" }

**使用條款**：[條款][gemma-terms]

**作者**：Google DeepMind

## 模型資訊

摘要說明，以及輸入與輸出的簡要定義。

### 說明

Gemma 是 Google 推出的輕量級、最先進的開放模型家族，採用與打造 Gemini 模型相同的技術與研究成果。Gemma 3 模型為多模態，可處理文字與圖像輸入並生成文字輸出，提供預訓練與指令微調版本的開放權重。Gemma 3 具備 128K 的大型上下文視窗、超過 140 種語言的多語言支援，並提供比先前版本更多的尺寸。Gemma 3 模型非常適合各種文字生成與圖像理解任務，包括問答、摘要與推理。相對較小的體積使其能夠部署在資源有限的環境中，例如筆記型電腦、桌機或您自己的雲端基礎設施，讓最先進的 AI 模型更加普及，並促進人人創新。

### 輸入與輸出

- **輸入：**
    - 文字字串，例如問題、提示或待摘要的文件
    - 圖像，正規化為 896 x 896 解析度，每張編碼為 256 個詞元
    - 4B、12B 與 27B 尺寸的總輸入上下文為 128K 詞元；1B 與 270M 尺寸為 32K 詞元。
- **輸出：**
    - 針對輸入生成的文字，例如問題的答案、圖像內容分析或文件摘要
    - 每次請求的總輸出上下文，4B、12B 與 27B 尺寸最高達 128K 詞元，1B 與 270M 尺寸為 32K 詞元，並扣除請求的輸入詞元

### 引用

```python
@article{gemma_2025,
    title={Gemma 3},
    url={https://arxiv.org/abs/2503.19786},
    publisher={Google DeepMind},
    author={Gemma Team},
    year={2025}
}
```

## 模型資料

用於模型訓練的資料，以及資料的處理方式。

### 訓練資料集

這些模型在包含各種來源的文字資料集上進行訓練。27B 模型以 14 兆個詞元訓練，12B 模型以 12 兆個詞元訓練，4B 模型以 4 兆個詞元訓練，1B 模型以 2 兆個詞元訓練，270M 模型以 6 兆個詞元訓練。訓練資料的知識截止日期為 2024 年 8 月。主要組成如下：

- 網路文件：多元的網路文字蒐集，讓模型接觸廣泛的語言風格、主題與詞彙。訓練資料集包含超過 140 種語言的內容。
- 程式碼：讓模型接觸程式碼有助於學習程式語言的語法與模式，進而提升生成程式碼與理解程式相關問題的能力。
- 數學：以數學文字訓練有助於模型學習邏輯推理、符號表示，並處理數學問題。
- 圖像：廣泛的圖像讓模型能執行圖像分析與視覺資料擷取任務。

這些多元資料來源的組合，對於訓練出能處理各種任務與資料格式的強大多模態模型至關重要。

### 資料預處理

以下為應用於訓練資料的主要資料清理與過濾方法：

- CSAM 過濾：在資料準備過程的多個階段實施嚴格的 CSAM（兒童性虐待內容）過濾，以確保排除有害與非法內容。
- 敏感資料過濾：為使 Gemma 預訓練模型安全可靠，使用自動化技術從訓練集中過濾掉部分個人資訊與其他敏感資料。
- 其他方法：依據內容品質與安全性進行過濾，符合[我們的政策](https://ai.google/static/documents/ai-responsibility-update-published-february-2025.pdf)。

## 實作資訊

模型內部機制的詳細資訊。

### 硬體

Gemma 使用[張量處理單元（Tensor Processing Unit, TPU）](https://cloud.google.com/tpu/docs/intro-to-tpu)硬體（TPUv4p、TPUv5p 與 TPUv5e）進行訓練。訓練視覺語言模型（VLM）需要龐大的運算能力。TPU 專為機器學習中常見的矩陣運算而設計，在此領域具有多項優勢：

- 效能：TPU 專為處理訓練 VLM 所涉及的大規模運算而設計。與 CPU 相比，可大幅加速訓練。
- 記憶體：TPU 通常配備大量高頻寬記憶體，可在訓練期間處理大型模型與批次大小，有助於提升模型品質。
- 可擴展性：TPU Pod（大型 TPU 叢集）為應對日益複雜的大型基礎模型提供了可擴展的解決方案。您可以將訓練分散到多個 TPU 裝置，以更快、更有效率地處理。
- 成本效益：在許多情境下，與基於 CPU 的基礎設施相比，TPU 為訓練大型模型提供了更具成本效益的方案，尤其是考量到更快的訓練所節省的時間與資源。
- 這些優勢與[Google 永續營運的承諾](https://sustainability.google/operating-sustainably/)相符。

### 軟體

訓練使用 [JAX](https://github.com/jax-ml/jax) 與 [ML Pathways](https://blog.google/technology/ai/introducing-pathways-next-generation-ai-architecture/) 進行。

JAX 讓研究人員得以利用最新一代硬體（包括 TPU），更快、更有效率地訓練大型模型。ML Pathways 是 Google 為打造能跨多個任務泛化的人工智慧系統所做的最新努力。這特別適合基礎模型，包括這類大型語言模型。

JAX 與 ML Pathways 的共同使用方式如[關於 Gemini 模型家族的論文](https://arxiv.org/abs/2312.11805)所述：*「Jax 與 Pathways 的『單一控制器』程式設計模型，讓單一 Python 行程即可編排整個訓練流程，大幅簡化了開發工作流程。」*

## 評測

模型評測指標與結果。

### 基準測試結果

這些模型使用大量不同的資料集與指標進行評測，涵蓋文字生成的各個面向。標示 **IT** 的評測結果為指令微調模型的結果；標示 **PT** 的為預訓練模型的結果。

#### Gemma 3 1B, 4B, 12B \& 27B

##### 推理與事實性

| 基準測試 | n-shot | Gemma 3 IT 1B | Gemma 3 IT 4B | Gemma 3 IT 12B | Gemma 3 IT 27B |
|---|---|---|---|---|---|
| [GPQA](https://arxiv.org/abs/2311.12022) Diamond | 0-shot | 19.2 | 30.8 | 40.9 | 42.4 |
| [SimpleQA](https://arxiv.org/abs/2411.04368) | 0-shot | 2.2 | 4.0 | 6.3 | 10.0 |
| [FACTS Grounding](https://goo.gle/FACTS_paper) | - | 36.4 | 70.1 | 75.8 | 74.9 |
| [BIG-Bench Hard](https://paperswithcode.com/dataset/bbh) | 0-shot | 39.1 | 72.2 | 85.7 | 87.6 |
| [BIG-Bench Extra Hard](https://github.com/google-deepmind/bbeh) | 0-shot | 7.2 | 11.0 | 16.3 | 19.3 |
| [IFEval](https://arxiv.org/abs/2311.07911) | 0-shot | 80.2 | 90.2 | 88.9 | 90.4 |

| 基準測試 | n-shot | Gemma 3 PT 1B | Gemma 3 PT 4B | Gemma 3 PT 12B | Gemma 3 PT 27B |
|---|---|---|---|---|---|
| [HellaSwag](https://arxiv.org/abs/1905.07830) | 10-shot | 62.3 | 77.2 | 84.2 | 85.6 |
| [BoolQ](https://arxiv.org/abs/1905.10044) | 0-shot | 63.2 | 72.3 | 78.8 | 82.4 |
| [PIQA](https://arxiv.org/abs/1911.11641) | 0-shot | 73.8 | 79.6 | 81.8 | 83.3 |
| [SocialIQA](https://arxiv.org/abs/1904.09728) | 0-shot | 48.9 | 51.9 | 53.4 | 54.9 |
| [TriviaQA](https://arxiv.org/abs/1705.03551) | 5-shot | 39.8 | 65.8 | 78.2 | 85.5 |
| [Natural Questions](https://github.com/google-research-datasets/natural-questions) | 5-shot | 9.48 | 20.0 | 31.4 | 36.1 |
| [ARC-c](https://arxiv.org/abs/1911.01547) | 25-shot | 38.4 | 56.2 | 68.9 | 70.6 |
| [ARC-e](https://arxiv.org/abs/1911.01547) | 0-shot | 73.0 | 82.4 | 88.3 | 89.0 |
| [WinoGrande](https://arxiv.org/abs/1907.10641) | 5-shot | 58.2 | 64.7 | 74.3 | 78.8 |
| [BIG-Bench Hard](https://paperswithcode.com/dataset/bbh) | few-shot | 28.4 | 50.9 | 72.6 | 77.7 |
| [DROP](https://arxiv.org/abs/1903.00161) | 1-shot | 42.4 | 60.1 | 72.2 | 77.2 |

##### STEM 與程式設計

| 基準測試 | n-shot | Gemma 3 IT 1B | Gemma 3 IT 4B | Gemma 3 IT 12B | Gemma 3 IT 27B |
|---|---|---|---|---|---|
| [MMLU](https://arxiv.org/abs/2009.03300) (Pro) | 0-shot | 14.7 | 43.6 | 60.6 | 67.5 |
| [LiveCodeBench](https://arxiv.org/abs/2403.07974) | 0-shot | 1.9 | 12.6 | 24.6 | 29.7 |
| [Bird-SQL](https://arxiv.org/abs/2305.03111) (dev) | - | 6.4 | 36.3 | 47.9 | 54.4 |
| [Math](https://arxiv.org/abs/2103.03874) | 0-shot | 48.0 | 75.6 | 83.8 | 89.0 |
| HiddenMath | 0-shot | 15.8 | 43.0 | 54.5 | 60.3 |
| [MBPP](https://arxiv.org/abs/2108.07732) | 3-shot | 35.2 | 63.2 | 73.0 | 74.4 |
| [HumanEval](https://arxiv.org/abs/2107.03374) | 0-shot | 41.5 | 71.3 | 85.4 | 87.8 |
| [Natural2Code](https://arxiv.org/abs/2405.04520) | 0-shot | 56.0 | 70.3 | 80.7 | 84.5 |
| [GSM8K](https://arxiv.org/abs/2110.14168) | 0-shot | 62.8 | 89.2 | 94.4 | 95.9 |

| 基準測試 | n-shot | Gemma 3 PT 4B | Gemma 3 PT 12B | Gemma 3 PT 27B |
|---|---|---|---|---|
| [MMLU](https://arxiv.org/abs/2009.03300) | 5-shot | 59.6 | 74.5 | 78.6 |
| [MMLU](https://arxiv.org/abs/2009.03300) (Pro COT) | 5-shot | 29.2 | 45.3 | 52.2 |
| [AGIEval](https://arxiv.org/abs/2304.06364) | 3-5-shot | 42.1 | 57.4 | 66.2 |
| [MATH](https://arxiv.org/abs/2103.03874) | 4-shot | 24.2 | 43.3 | 50.0 |
| [GSM8K](https://arxiv.org/abs/2110.14168) | 8-shot | 38.4 | 71.0 | 82.6 |
| [GPQA](https://arxiv.org/abs/2311.12022) | 5-shot | 15.0 | 25.4 | 24.3 |
| [MBPP](https://arxiv.org/abs/2108.07732) | 3-shot | 46.0 | 60.4 | 65.6 |
| [HumanEval](https://arxiv.org/abs/2107.03374) | 0-shot | 36.0 | 45.7 | 48.8 |

#### 多語言

| 基準測試 | n-shot | Gemma 3 IT 1B | Gemma 3 IT 4B | Gemma 3 IT 12B | Gemma 3 IT 27B |
|---|---|---|---|---|---|
| [Global-MMLU-Lite](https://huggingface.co/datasets/CohereForAI/Global-MMLU-Lite) | 0-shot | 34.2 | 54.5 | 69.5 | 75.1 |
| [ECLeKTic](https://arxiv.org/abs/2502.21228) | 0-shot | 1.4 | 4.6 | 10.3 | 16.7 |
| [WMT24++](https://arxiv.org/abs/2502.12404v1) | 0-shot | 35.9 | 46.8 | 51.6 | 53.4 |

| 基準測試 | Gemma 3 PT 1B | Gemma 3 PT 4B | Gemma 3 PT 12B | Gemma 3 PT 27B |
|---|---|---|---|---|
| [MGSM](https://arxiv.org/abs/2210.03057) | 2.04 | 34.7 | 64.3 | 74.3 |
| [Global-MMLU-Lite](https://huggingface.co/datasets/CohereForAI/Global-MMLU-Lite) | 24.9 | 57.0 | 69.4 | 75.7 |
| [WMT24++](https://arxiv.org/abs/2502.12404v1) (ChrF) | 36.7 | 48.4 | 53.9 | 55.7 |
| [FloRes](https://arxiv.org/abs/2106.03193) | 29.5 | 39.2 | 46.0 | 48.8 |
| [XQuAD](https://arxiv.org/abs/1910.11856v3) (all) | 43.9 | 68.0 | 74.5 | 76.8 |
| [ECLeKTic](https://arxiv.org/abs/2502.21228) | 4.69 | 11.0 | 17.2 | 24.4 |
| [IndicGenBench](https://arxiv.org/abs/2404.16816) | 41.4 | 57.2 | 61.7 | 63.4 |

##### 多模態

| 基準測試 | Gemma 3 IT 4B | Gemma 3 IT 12B | Gemma 3 IT 27B |
|---|---|---|---|
| [MMMU](https://arxiv.org/abs/2311.16502) (val) | 48.8 | 59.6 | 64.9 |
| [DocVQA](https://www.docvqa.org/) | 75.8 | 87.1 | 86.6 |
| [InfoVQA](https://arxiv.org/abs/2104.12756) | 50.0 | 64.9 | 70.6 |
| [TextVQA](https://textvqa.org/) | 57.8 | 67.7 | 65.1 |
| [AI2D](https://allenai.org/data/diagrams) | 74.8 | 84.2 | 84.5 |
| [ChartQA](https://arxiv.org/abs/2203.10244) | 68.8 | 75.7 | 78.0 |
| [VQAv2](https://visualqa.org/index.html) (val) | 62.4 | 71.6 | 71.0 |
| [MathVista](https://arxiv.org/abs/2310.02255) (testmini) | 50.0 | 62.9 | 67.6 |

| 基準測試 | Gemma 3 PT 4B | Gemma 3 PT 12B | Gemma 3 PT 27B |
|---|---|---|---|
| [COCOcap](https://cocodataset.org/#home) | 102 | 111 | 116 |
| [DocVQA](https://www.docvqa.org/) (val) | 72.8 | 82.3 | 85.6 |
| [InfoVQA](https://arxiv.org/abs/2104.12756) (val) | 44.1 | 54.8 | 59.4 |
| [MMMU](https://arxiv.org/abs/2311.16502) (pt) | 39.2 | 50.3 | 56.1 |
| [TextVQA](https://textvqa.org/) (val) | 58.9 | 66.5 | 68.6 |
| [RealWorldQA](https://paperswithcode.com/dataset/realworldqa) | 45.5 | 52.2 | 53.9 |
| [ReMI](https://arxiv.org/html/2406.09175v1) | 27.3 | 38.5 | 44.8 |
| [AI2D](https://allenai.org/data/diagrams) | 63.2 | 75.2 | 79.0 |
| [ChartQA](https://arxiv.org/abs/2203.10244) | 63.6 | 74.7 | 76.3 |
| [VQAv2](https://visualqa.org/index.html) | 63.9 | 71.2 | 72.9 |
| [BLINK](https://arxiv.org/abs/2404.12390) | 38.0 | 35.9 | 39.6 |
| [OKVQA](https://okvqa.allenai.org/) | 51.0 | 58.7 | 60.2 |
| [TallyQA](https://arxiv.org/abs/1810.12440) | 42.5 | 51.8 | 54.3 |
| [SpatialSense VQA](https://arxiv.org/abs/1908.02660) | 50.9 | 60.0 | 59.4 |
| [CountBenchQA](https://github.com/google-research/big_vision/blob/main/big_vision/datasets/countbenchqa/) | 26.1 | 17.8 | 68.0 |

#### Gemma 3 270M

| **基準測試** | **n-shot** | **Gemma 3 PT 270M** |
|---|---|---|
| [HellaSwag](https://arxiv.org/abs/1905.07830) | 10-shot | 40.9 |
| [BoolQ](https://arxiv.org/abs/1905.10044) | 0-shot | 61.4 |
| [PIQA](https://arxiv.org/abs/1911.11641) | 0-shot | 67.7 |
| [TriviaQA](https://arxiv.org/abs/1705.03551) | 5-shot | 15.4 |
| [ARC-c](https://arxiv.org/abs/1911.01547) | 25-shot | 29.0 |
| [ARC-e](https://arxiv.org/abs/1911.01547) | 0-shot | 57.7 |
| [WinoGrande](https://arxiv.org/abs/1907.10641) | 5-shot | 52.0 |

| **基準測試** | **n-shot** | **Gemma 3 IT 270M** |
|---|---|---|
| [HellaSwag](https://arxiv.org/abs/1905.07830) | 0-shot | 37.7 |
| [PIQA](https://arxiv.org/abs/1911.11641) | 0-shot | 66.2 |
| [ARC-c](https://arxiv.org/abs/1911.01547) | 0-shot | 28.2 |
| [WinoGrande](https://arxiv.org/abs/1907.10641) | 0-shot | 52.3 |
| [BIG-Bench Hard](https://paperswithcode.com/dataset/bbh) | few-shot | 26.7 |
| [IF Eval](https://arxiv.org/abs/2311.07911) | 0-shot | 51.2 |

## 倫理與安全

倫理與安全評測方法及結果。

### 評測方法

我們的評測方法包括結構化評測，以及針對相關內容政策的內部紅隊測試。紅隊測試由多個不同團隊執行，各自有不同的目標與人工評測指標。這些模型針對與倫理及安全相關的多個類別進行評測，包括：

- **兒童安全**：針對涵蓋兒童安全政策的文字轉文字與圖像轉文字提示進行評測，包括兒童性虐待與剝削。
- **內容安全：**針對涵蓋安全政策的文字轉文字與圖像轉文字提示進行評測，包括騷擾、暴力與血腥，以及仇恨言論。
- **表徵危害**：針對涵蓋安全政策的文字轉文字與圖像轉文字提示進行評測，包括偏見、刻板印象，以及有害關聯或不實內容。

除了開發階段的評測，我們還進行「保證評測（assurance evaluations）」——這是我們為負責任治理決策所做的「保持距離（arms-length）」內部評測。這些評測由模型開發團隊以外的單位獨立執行，為發布決策提供依據。高層級發現會回饋給模型團隊，但提示集合會保留不公開，以防止過擬合，並維持結果對決策的參考價值。保證評測結果會作為發布審查的一部分，呈報給我們的責任與安全委員會。

### 評測結果

在所有安全測試領域，相較於先前的 Gemma 模型，我們在兒童安全、內容安全與表徵危害等類別都看到大幅改善。所有測試均在未使用安全篩選器的情況下進行，以評測模型的能力與行為。在文字轉文字與圖像轉文字、以及所有模型尺寸上，模型僅產生極少的政策違規，並在無根據推論方面較先前 Gemma 模型的表現有顯著改善。我們評測的一個限制是僅包含英文提示。

## 使用方式與限制

這些模型存在使用者應注意的特定限制。

### 預期用途

開放視覺語言模型（VLM）在各產業與領域有廣泛應用。以下潛在用途清單並不詳盡，目的在於提供背景資訊，說明模型建立者在模型訓練與開發過程中所考量的可能使用情境。

- 內容創作與溝通
    - 文字生成：這些模型可用於生成詩歌、劇本、程式碼、行銷文案與電子郵件草稿等創意文字格式。
    - 聊天機器人與對話式 AI：為客戶服務、虛擬助理或互動式應用程式提供對話介面。
    - 文字摘要：為文字語料庫、研究論文或報告生成精簡摘要。
    - 圖像資料擷取：這些模型可用於擷取、解讀與摘要視覺資料，供文字溝通使用。
- 研究與教育
    - 自然語言處理（NLP）與 VLM 研究：這些模型可作為研究人員的基礎，用於實驗 VLM 與 NLP 技術、開發演算法，並推動該領域的發展。
    - 語言學習工具：支援互動式語言學習體驗，協助文法修正或提供寫作練習。
    - 知識探索：協助研究人員探索大量文字，透過生成摘要或回答特定主題的問題。

### 限制

- 訓練資料
    - 訓練資料的品質與多樣性會顯著影響模型的能力。訓練資料中的偏見或缺口可能導致模型回應受限。
    - 訓練資料集的範圍決定了模型能有效處理的主題領域。
- 上下文與任務複雜度
    - 模型在能以明確提示與指示框架化的任務上表現較好。開放式或高度複雜的任務可能較具挑戰性。
    - 模型的效能可能受提供的上下文量影響（上下文越長通常輸出越好，但有其上限）。
- 語言歧義與細微差別
    - 自然語言本質上很複雜。模型可能難以掌握細微差別、諷刺或比喻性語言。
- 事實準確性
    - 模型根據從訓練資料集學到的資訊生成回應，但它們並非知識庫，可能會生成錯誤或過時的事實陳述。
- 常識
    - 模型依賴語言中的統計模式，在某些情境下可能缺乏應用常識推理的能力。

### 倫理考量與風險

視覺語言模型（VLM）的開發引發了若干倫理議題。在打造開放模型時，我們審慎考量了以下事項：

- 偏見與公平性
    - 以大規模真實世界文字與圖像資料訓練的 VLM 可能反映訓練資料中蘊含的社會文化偏見。這些模型經過嚴格審查，輸入資料預處理如上所述，後續評測結果載於本卡。
- 錯誤資訊與濫用
    - VLM 可能被濫用來生成虛假、誤導或有害的文字。
    - 我們提供負責任使用模型的指南，詳見 [負責任生成式 AI 工具包][responsible-ai]。
- 透明度與問責制：
    - 本模型卡彙整了模型的架構、能力、限制與評測流程的詳細資訊。
    - 一個負責任地開發的開放模型，讓整個 AI 生態系的開發者與研究人員都能取用 VLM 技術，提供了共享創新的機會。

已識別的風險與緩解措施：

- **偏見的延續**：鼓勵在模型訓練、微調與其他使用情境中持續監控（使用評測指標、人工審查），並探索去偏見技術。
- **有害內容的生成**：內容安全機制與指南至關重要。鼓勵開發者謹慎行事，並依據其特定產品政策與應用情境實施適當的內容安全防護。
- **惡意用途的濫用**：技術限制，以及開發者與終端使用者的教育，有助於緩解 VLM 的惡意應用。我們提供教育資源與回報機制，供使用者舉報濫用。Gemma 模型的禁用用途載於 [Gemma 禁用用途政策][prohibited-use]。
- **隱私侵犯**：模型在經過過濾以移除部分個人資訊與其他敏感資料的資料上訓練。鼓勵開發者遵守隱私法規，並採用隱私保護技術。

### 優點

在發布時，與同等規模的模型相比，此模型家族提供從零開始為負責任 AI 開發而設計的高效能開放視覺語言模型實作。

根據本文件所述的基準測試評測指標，這些模型的表現優於其他同等規模的開放模型替代方案。

--8<-- "glossary.md"
