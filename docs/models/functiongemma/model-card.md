---
title: FunctionGemma 模型卡
description: "FunctionGemma 模型卡：函式呼叫的輸入與輸出、訓練資料、評估結果與預期用途。"
models: [functiongemma]
source_url: "https://ai.google.dev/gemma/docs/functiongemma/model_card.md.txt"
---
# 模型卡

**模型頁面** : [FunctionGemma](index.md)

**資源與技術文件**:

- [Responsible Generative AI Toolkit][responsible-ai]
- [Kaggle 上的 FunctionGemma](https://www.kaggle.com/models/google/functiongemma/){ data-model="functiongemma" }
- [Vertex Model Garden 上的 FunctionGemma](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/functiongemma){ data-model="functiongemma" }

**使用條款** : [Terms][gemma-terms]  

**作者**：Google DeepMind

## 模型資訊

輸入與輸出的摘要說明與簡要定義。

### 描述

!!! Note
    FunctionGemma 預期會針對您的特定函式呼叫任務進行微調，包括多回合的使用場景。


FunctionGemma 是來自 Google 的輕量級開放模型，作為打造您自己專門函式呼叫模型的基礎。FunctionGemma 不適合直接作為對話模型使用，並且像這個大小的模型一樣，經過進一步微調後效能會非常出色。FunctionGemma 基於 Gemma 3 270M 模型，並使用打造 Gemini 模型的相同研究與技術，專為函式呼叫訓練。模型的架構與 Gemma 3 相同，但使用不同的聊天格式。模型非常適合純文字的函式呼叫。獨特的小體積讓它能部署在資源有限的環境中，例如筆電、桌機或您自己的雲端基礎設施，讓尖端 AI 模型更容易取得、促進人人創新。此外，與基礎 Gemma 270M 類似，模型經過最佳化，極為靈活，在單回合情境下於多種硬體上效能出色，但應在單回合或多回合的任務專用資料上微調，才能在特定領域達到最佳準確度。
為了展示將 270M 參數模型專門化後能在特定代理式工作流程上達到高效能，我們在 [Google AI Edge Gallery app](https://play.google.com/store/apps/details?id=com.google.ai.edge.gallery&pcampaignid=web_share) 中凸顯了兩個使用場景。

- **Tiny Garden：**一個微調來驅動語音控制互動遊戲的模型。它處理遊戲邏輯來管理一塊虛擬田地，把「在最上面那一排種向日葵」、「給 1 號和 2 號田澆水」這類指令拆解成應用程式專用的函式（例如 plant_seed、water_plots）與座標目標。這展示了模型在無需伺服器連線的情況下驅動自訂應用程式機制的能力。

- **Mobile Actions：**為了讓開發者能打造自己的專家代理，我們發布了[資料集](https://huggingface.co/datasets/google/mobile-actions)與[微調配方](https://github.com/google-gemma/cookbook/blob/main/.archive/FunctionGemma/%5BFunctionGemma%5DFinetune_FunctionGemma_270M_for_Mobile_Actions_with_Hugging_Face.ipynb){ data-model="functiongemma" }
  來示範微調 FunctionGemma。它把使用者輸入（例如「建立一個午餐行事曆事件」、「打開手電筒」）翻譯成觸發 Android 系統工具的函式呼叫。這個互動式筆記本示範如何從基礎 FunctionGemma 模型出發，從零打造「Mobile Actions」微調版本，用於 [Google AI Edge gallery app](https://play.google.com/store/apps/details?id=com.google.ai.edge.gallery&pcampaignid=web_share)。
  這個使用場景展示了模型作為離線、隱私的個人裝置代理的能力。

### 輸入與輸出

- **輸入：**
    - 文字字串，例如問題、提示，或要摘要的文件
    - 總輸入上下文 32K 詞元
- **輸出：**
    - 針對輸入生成的文字，例如問題的答案，或文件的摘要
    - 每次請求的總輸出上下文最多 32K 詞元（需扣除請求輸入的詞元）

## 模型資料

模型訓練所用的資料與資料處理方式。

### 訓練資料集

這些模型在包含廣泛來源的文字資料資料集上訓練。模型以 6T 詞元訓練。訓練資料的知識截止日期為 2024 年 8 月。主要組成如下：

- Public Tool Definitions - 網路上常見的 API
- Tool Use Interactions - 混合了提示、函式呼叫、函式回覆，以及模型用來摘要函式呼叫回覆或在提示模糊或不完整時請求澄清的自然語言回覆。

### 資料預處理

以下是在訓練資料上套用的主要資料清理與篩選方法：

- CSAM 篩選：在資料準備流程的多個階段套用嚴格的 CSAM（兒童性虐待素材）篩選，確保排除有害與非法內容。
- 敏感資料篩選：為了讓 Gemma 預訓練模型安全可靠，使用自動化技術從訓練集中篩除特定個人資訊與其他敏感資料。
- 其他方法：依循[我們的政策](https://ai.google/static/documents/ai-responsibility-update-published-february-2025.pdf)基於內容品質與安全進行篩選。

## 實作資訊

關於模型內部的細節。

### 硬體

Gemma 使用 [Tensor Processing Unit (TPU)](https://cloud.google.com/tpu/docs/intro-to-tpu) 硬體（TPUv4p、TPUv5p 與 TPUv5e）訓練。訓練視覺語言模型（VLM）需要可觀的運算能力。TPU 專為機器學習中常見的矩陣運算設計，在這個領域有幾項優勢：

- 效能：TPU 專為處理訓練 VLM 所涉及的大規模運算設計。相較於 CPU，它們能大幅加速訓練。
- 記憶體：TPU 通常配備大量高頻寬記憶體，能在訓練期間處理大型模型與批次大小，有助於提升模型品質。
- 可擴展性：TPU Pod（大型 TPU 叢集）為處理日益複雜的大型基礎模型提供可擴展的解決方案。您可以把訓練分散到多個 TPU 裝置上，獲得更快、更有效率的處理。
- 成本效益：在許多情境下，考量到因更快訓練而節省的時間與資源，TPU 能提供比 CPU 基礎設施更具成本效益的大型模型訓練方案。
- 這些優勢符合 [Google 永續營運的承諾](https://sustainability.google/operating-sustainably/)。

### 軟體

訓練使用 [JAX](https://github.com/jax-ml/jax) 與 [ML Pathways](https://blog.google/technology/ai/introducing-pathways-next-generation-ai-architecture/) 進行。
JAX 讓研究人員能利用最新一代硬體（包括 TPU），更快、更有效率地訓練大型模型。ML Pathways 是 Google 最新打造能跨多任務泛化的人工智慧系統的努力。這特別適合基礎模型，包括這些大型語言模型。

JAX 與 ML Pathways 如 [Gemini 模型家族論文](https://goo.gle/gemma2report) 所述一起使用；*「Jax 與 Pathways 的『單一控制器』程式設計模型，讓單一 Python 程序能編排整個訓練流程，大幅簡化開發工作流程。」*

## 評估

模型評估指標與結果。

### 基準測試結果

| **基準測試** | **n-shot** | **Function Gemma 270m** |
|---|---|---|
| BFCL Simple | 0-shot | 61.6 |
| BFCL Multiple | 0-shot | 63.5 |
| BFCL Parallel | 0-shot | 39 |
| BFCL Parallel Multiple | 0-shot | 29.5 |
| BFCL Live Simple | 0-shot | 36.2 |
| BFCL Live Multiple | 0-shot | 25.7 |
| BFCL Live Parallel | 0-shot | 22.9 |
| BFCL Live Parallel Multiple | 0-shot | 20.8 |
| BFCL Relevance | 0-shot | 61.1 |
| BFCL Irrelevance | 0-shot | 73.7 |

**在 Mobile Actions 資料集上微調後的效能影響**   

為了展示專門化對小型語言模型的價值，我們使用「Mobile Actions」[配方](https://github.com/google-gemma/cookbook/blob/main/.archive/FunctionGemma/%5BFunctionGemma%5DFinetune_FunctionGemma_270M_for_Mobile_Actions_with_Hugging_Face.ipynb){ data-model="functiongemma" } 比較了基礎 FunctionGemma 模型與微調後的模型。微調大幅提升了基礎 FunctionGemma 模型正確識別與格式化行動系統呼叫的能力。

| 模型 | Mobile Actions 評估結果 |
|---|---|
| 基礎 FunctionGemma 模型 | 58% |
| Mobile Actions 微調 | 85% |

**Gemma 270m 微調使用場景的裝置端效能**   

我們在 Samsung S25 Ultra 上評估微調後的使用場景，測量裝置端延遲與記憶體佔用量。

- **情境：**512 個預填充詞元與 32 個解碼詞元。
- **硬體：**S25 Ultra CPU，使用 LiteRT XNNPACK delegate，4 執行緒。

Mobile Actions 裝置端效能

| 後端 | 量化方案 | 上下文長度 | 預填充（每秒詞元數） | 解碼（每秒詞元數） | 首詞元時間（秒） | 模型大小（MB） | 峰值 RSS 記憶體（MB） |
|---|---|---|---|---|---|---|---|
| CPU | dynamic_int8 | 1024 | 1718 | 125.9 | 0.3 | 288 | 551 |

Tiny Garden 裝置端效能

| 後端 | 量化方案 | 上下文長度 | 預填充（每秒詞元數） | 解碼（每秒詞元數） | 首詞元時間（秒） | 模型大小（MB） | 峰值 RSS 記憶體（MB） |
|---|---|---|---|---|---|---|---|
| CPU | dynamic_int8 | 1024 | 1743 | 125.7 | 0.3 | 288 | 549 |

## 倫理與安全

倫理與安全評估方法與結果。

### 評估方法

我們的評估方法包括結構化評估，以及針對相關內容政策的內部紅隊測試。紅隊測試由數個不同的團隊執行，每個團隊有不同的目標與人工評估指標。這些模型針對與倫理安全相關的多個類別進行評估，包括：

- **兒童安全**：評估涵蓋兒童安全政策的文字對文字與影像對文字提示，包括兒童性虐待與剝削。
- **內容安全：**評估涵蓋安全政策的文字對文字與影像對文字提示，包括騷擾、暴力與血腥、仇恨言論。
- **代表性傷害**：評估涵蓋安全政策的文字對文字與影像對文字提示，包括偏見、刻板印象、有害的關聯或不準確內容。

### 評估結果

在所有安全測試領域，相較於前幾代 Gemma 模型，兒童安全、內容安全與代表性傷害類別都有顯著改善。所有測試都是在無安全篩選器的情況下進行的，以評估模型能力與行為。模型產生的政策違規極少，並較前幾代 Gemma 模型在無根據推論方面有顯著改善。我們評估的限制是僅包含英文提示。

## 使用方式與限制

這些模型有一些使用者應注意的限制。

### 預期用途

此模型不適合直接作為對話模型使用。  

開放式大型語言模型（LLM）在各行各業與領域有廣泛的應用。以下潛在用途清單並非詳盡無遺，目的在於提供關於模型創建者在模型訓練與開發過程中考慮過的可能使用場景的背景資訊。

- 內容創作與溝通
    - 文字生成：這些模型可用來生成創意文字格式，如詩歌、腳本、程式碼、行銷文案與電子郵件草稿。
    - 聊天機器人與對話式 AI：為客服、虛擬助理或互動應用提供對話介面。
    - 文字摘要：為文字語料庫、研究論文或報告生成簡潔摘要。
- 研究與教育
    - 自然語言處理（NLP）研究：這些模型可作為研究人員的基礎，用於實驗 NLP 技術、開發演算法，為領域進展做出貢獻。
    - 語言學習工具：支援互動式語言學習體驗，協助文法修正或提供寫作練習。
    - 知識探索：透過生成摘要或回答特定主題的問題，協助研究人員探索大量文本。

### 限制

- 訓練資料
    - 訓練資料的品質與多樣性顯著影響模型能力。訓練資料中的偏見或缺口可能導致模型回覆的限制。
    - 訓練資料集的範疇決定了模型能有效處理的主題領域。
- 上下文與任務複雜度
    - 模型在能以清楚的提示與指令框架化的任務上表現較好。開放式或高度複雜的任務可能具有挑戰性。
    - 提供上下文的量會影響模型效能（上下文越長通常輸出越好，但有一定上限）。
- 語言歧義與細微差別
    - 自然語言本質上很複雜。模型可能難以掌握細微差別、諷刺或比喻。
- 事實正確性
    - 模型根據從訓練資料集學到的資訊生成回覆，但它們不是知識庫。可能產生不正確或過時的事實陳述。
- 常識
    - 模型仰賴語言中的統計模式。在某些情況下可能缺乏運用常識推理的能力。

### 倫理考量與風險

大型語言模型（LLM）的開發引發了幾項倫理關切。在創建開放模型時，我們審慎考慮了以下事項：

- 偏見與公平性
    - 在大規模真實世界文字資料上訓練的 LLM，可能反映訓練素材中嵌入的社會文化偏見。這些模型經過審慎檢視、本卡所述的輸入資料預處理與事後評估。
- 錯誤資訊與濫用
    - LLM 可能被濫用來產生虛假、誤導或有害的文字。
    - 本模型提供負責任使用的指引，請參閱 [Responsible Generative AI Toolkit][responsible-ai]。
- 透明度與可歸責性：
    - 本模型卡彙整了模型架構、能力、限制與評估流程的詳細資訊。
    - 一個負責任開發的開放模型，讓整個 AI 生態系的開發者與研究人員都能取得 LLM 技術，提供了分享創新的機會。

已識別的風險與緩解措施：

- 偏見的延續：鼓勵在模型訓練、微調及其他使用場景中持續監控（使用評估指標、人工審查）並探索去偏見技術。
- 產生有害內容：內容安全機制與指引至關重要。鼓勵開發者依其特定產品政策與應用場景，謹慎行事並實施適當的內容安全防護措施。
- 用於惡意目的的濫用：技術限制，以及對開發者與終端使用者的教育，可幫助減輕 LLM 的惡意應用。我們提供教育資源與通報機制，讓使用者能標記濫用行為。Gemma 模型的禁止用途列於 [Gemma Prohibited Use Policy][prohibited-use]。
- 侵犯隱私：模型在訓練資料上篩除了 PII（個人可識別資訊）。鼓勵開發者遵守隱私法規，並採用隱私保護技術。

### 優點

在發布時，這個模型家族提供了高效能的開放大型語言模型實作，相較於同級模型，從頭開始就為負責任的 AI 開發而設計。
