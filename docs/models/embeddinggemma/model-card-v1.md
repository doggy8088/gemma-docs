---
title: EmbeddingGemma 模型卡
description: "EmbeddingGemma 模型卡：嵌入輸入與輸出、訓練資料、評估結果與預期用途。"
models: [embeddinggemma]
source_url: "https://ai.google.dev/gemma/docs/embeddinggemma/model_card.md.txt"
---
# EmbeddingGemma 模型卡

**模型頁面**：[EmbeddingGemma](index.md)

**資源與技術文件**：

- [Responsible Generative AI Toolkit][responsible-ai]
- [Kaggle 上的 EmbeddingGemma](https://www.kaggle.com/models/google/embeddinggemma/){ data-model="embeddinggemma" }
- [Vertex Model Garden 上的 EmbeddingGemma](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/embeddinggemma){ data-model="embeddinggemma" }

**使用條款**：[條款][gemma-terms]

**作者**：Google DeepMind

## 模型資訊

輸入與輸出的摘要描述與簡要定義。

### 描述

EmbeddingGemma 是一個 3 億參數、以其規模而言最先進的開放嵌入模型，由 Google 推出，基於 Gemma 3（採用 T5Gemma 初始化）打造，並使用了開發 Gemini 模型的相同研究與技術。
EmbeddingGemma 產生文字的向量表示，非常適合搜尋與檢索任務，包括分類、分群與語義相似度搜尋。此模型以 100 多種口語語言的資料訓練。

其小巧的體積與裝置端導向，讓它能部署在資源有限的環境（例如手機、筆記型電腦或桌機），讓最先進的 AI 模型觸手可及，促進人人創新。

更多技術細節請參閱我們的論文：
[EmbeddingGemma: Powerful and Lightweight Text Representations](https://arxiv.org/abs/2509.20354)。

### 輸入與輸出

- **輸入：**
    - 文字字串，例如要嵌入的問題、提示或文件
    - 最大輸入上下文長度為 2K
- **輸出：**
    - 輸入文字資料的數值向量表示
    - 輸出嵌入維度為 768，並可透過 Matryoshka Representation Learning (MRL) 使用較小的選項（512、256 或 128）。MRL 讓使用者能將 768 維的輸出嵌入截斷為所需的維度，再重新正規化，以達到高效且準確的表示。

### 引用

```python
@article{embedding_gemma_2025,
    title={EmbeddingGemma: Powerful and Lightweight Text Representations},
    publisher={Google DeepMind},
    author={Schechter Vera, Henrique and Dua, Sahil and Team, EmbeddingGemma},
    year={2025},
    url={https://arxiv.org/abs/2509.20354}
}
```

## 模型資料

### 訓練資料集

此模型以文字資料資料集訓練，涵蓋種類繁多的來源，總計約 3,200 億個詞元。關鍵組成如下：

- **網頁文件**：多樣化的網頁文字集合，確保模型接觸廣泛的語言風格、主題與詞彙。訓練資料集包含 100 多種語言的內容。
- **程式碼與技術文件**：讓模型接觸程式碼與技術文件，有助於它學習程式語言的結構與模式，以及專門的科學內容，提升對程式碼與技術問題的理解。
- **合成與任務專用資料**：合成訓練資料有助於教會模型特定技能，包括針對資訊檢索、分類和情緒分析等任務的精選資料，有助於微調它在常見嵌入應用上的表現。

這些多樣資料來源的組合，對於訓練能處理各種不同任務與資料格式的強大 多語言嵌入模型至關重要。

### 資料預處理

以下是套用於訓練資料的關鍵資料清理與篩選方法：

- CSAM 篩選：在資料準備流程的多個階段套用嚴格的 CSAM（兒童性虐待素材）篩選，確保排除有害與非法內容。
- 敏感資料篩選：作為讓 Gemma 預訓練模型安全可靠的一部分，使用自動化技術從訓練集中篩除某些個人資訊與其他敏感資料。
- 其他方法：依[我們的政策](https://ai.google/static/documents/ai-responsibility-update-published-february-2025.pdf)進行內容品質與安全篩選。

## 模型開發

### 硬體

EmbeddingGemma 使用最新一代 [Tensor Processing Unit (TPU)](https://cloud.google.com/tpu/docs/intro-to-tpu) 硬體（TPUv5e）訓練，更多細節請參閱 [Gemma 3 模型卡](../gemma-3/index.md)。

### 軟體

訓練使用 [JAX](https://github.com/jax-ml/jax) 與 [ML Pathways](https://blog.google/technology/ai/introducing-pathways-next-generation-ai-architecture/)。更多細節請參閱 [Gemma 3 模型卡](../gemma-3/index.md)。

## 評估

### 基準測試結果

模型以大量不同的資料集與指標評估，涵蓋文字理解的不同面向。

#### 全精度檢查點

| **MTEB（多語言，v2）** |||
|---|---|---|
| **維度** | **平均值（任務）** | **平均值（任務類型）** |
| 768d | 61.15 | 54.31 |
| 512d | 60.71 | 53.89 |
| 256d | 59.68 | 53.01 |
| 128d | 58.23 | 51.77 |

| **MTEB（英文，v2）** |||
|---|---|---|
| **維度** | **平均值（任務）** | **平均值（任務類型）** |
| 768d | 69.67 | 65.11 |
| 512d | 69.18 | 64.59 |
| 256d | 68.37 | 64.02 |
| 128d | 66.66 | 62.70 |

| **MTEB（程式碼，v1）** |||
|---|---|---|
| **維度** | **平均值（任務）** | **平均值（任務類型）** |
| 768d | 68.76 | 68.76 |
| 512d | 68.48 | 68.48 |
| 256d | 66.74 | 66.74 |
| 128d | 62.96 | 62.96 |

#### QAT 檢查點

| **MTEB（多語言，v2）** |||
|---|---|---|
| **量化設定（維度）** | **平均值（任務）** | **平均值（任務類型）** |
| 混合精度\* (768d) | 60.69 | 53.82 |
| Q8_0 (768d) | 60.93 | 53.95 |
| Q4_0 (768d) | 60.62 | 53.61 |

| **MTEB（英文，v2）** |||
|---|---|---|
| **量化設定（維度）** | **平均值（任務）** | **平均值（任務類型）** |
| 混合精度\* (768d) | 69.32 | 64.82 |
| Q8_0 (768d) | 69.49 | 64.84 |
| Q4_0 (768d) | 69.31 | 64.65 |

| **MTEB（程式碼，v1）** |||
|---|---|---|
| **量化設定（維度）** | **平均值（任務）** | **平均值（任務類型）** |
| 混合精度\* (768d) | 68.03 | 68.03 |
| Q8_0 (768d) | 68.70 | 68.70 |
| Q4_0 (768d) | 67.99 | 67.99 |

!!! Note
    QAT 模型在量化後評估


\* 混合精度指逐通道量化，嵌入、前饋與投影層使用 int4，注意力使用 int8（e4_a8_f4_p4）。

### 提示指令

EmbeddingGemma 能使用加在輸入字串前的提示，為各種使用情境（例如文件檢索、問答和事實查核）或特定輸入類型（查詢或文件）產生最佳化的嵌入。

查詢提示的格式為 `task: {task description} | query:`，其中任務描述依使用情境而異，預設任務描述為 `search result`。文件式提示的格式為 `title: {title | "none"} | text:`，其中標題為 `none`（預設值）或文件的實際標題。請注意，若有標題，提供標題可提升文件提示的模型表現，但可能需要手動格式化。

請依您的使用情境與輸入資料類型使用以下提示。它們可能已包含在您所選建模框架的 EmbeddingGemma 設定中。

| **使用情境（任務類型列舉）** | **描述** | **建議提示** |
|---|---|---|
| 檢索（查詢） | 用於產生針對文件搜尋或資訊檢索最佳化的嵌入 | task: search result \| query: {content} |
| 檢索（文件） | 用於產生針對文件搜尋或資訊檢索最佳化的嵌入 | title: {title \| "none"} \| text: {content} |
| 問答 | 用於產生針對文件搜尋或資訊檢索最佳化的嵌入 | task: question answering \| query: {content} |
| 事實查核 | 用於產生針對文件搜尋或資訊檢索最佳化的嵌入 | task: fact checking \| query: {content} |
| 分類 | 用於產生針對依預設標籤分類文字最佳化的嵌入 | task: classification \| query: {content} |
| 分群 | 用於產生針對依相似度分群文字最佳化的嵌入 | task: clustering \| query: {content} |
| 語義相似度 | 用於產生針對評估文字相似度最佳化的嵌入，不適用於檢索使用情境。 | task: sentence similarity \| query: {content} |
| 程式碼檢索 | 用於根據自然語言查詢（例如 *sort an array* 或 *reverse a linked list*）檢索程式碼區塊。程式碼區塊的嵌入使用 retrieval_document 計算。 | task: code retrieval \| query: {content} |

## 使用與限制

這些模型有一些使用者應注意的限制。

### 預期用途

開放嵌入模型在各產業與領域有廣泛應用。以下潛在用途清單並不完整，其目的是提供模型建立者在訓練與開發時考慮的可能使用情境背景資訊。

- **語義相似度**：針對評估文字相似度最佳化的嵌入，例如推薦系統與重複內容偵測
- **分類**：針對依預設標籤分類文字最佳化的嵌入，例如情緒分析與垃圾郵件偵測
- **分群**：針對依相似度分群文字最佳化的嵌入，例如文件組織、市場研究與異常偵測
- **檢索**
    - **文件**：針對文件搜尋最佳化的嵌入，例如為文章、書籍或網頁建立搜尋索引
    - **查詢**：針對一般搜尋查詢最佳化的嵌入，例如自訂搜尋
    - **程式碼查詢**：針對依自然語言查詢檢索程式碼區塊最佳化的嵌入，例如程式碼建議與搜尋
- **問答**：用於問答系統中問題的嵌入，針對尋找能回答問題的文件最佳化，例如聊天機器人。

- **事實查核**：用於需要查證陳述的嵌入，針對檢索包含支援或反駁該陳述證據的文件最佳化，例如自動化事實查核系統。

### 限制

- 訓練資料
    - 訓練資料的品質與多樣性顯著影響模型能力。訓練資料中的偏見或缺口可能導致模型回應的限制。
    - 訓練資料集的範圍決定了模型能有效處理的主題領域。
- 語言歧義與細微差別
    - 自然語言本質上很複雜。模型可能難以掌握細微差別、諷刺或比喻語言。

### 倫理考量與風險

已識別的風險與緩解措施：

- **偏見的延續**：鼓勵在模型訓練、微調與其他使用情境中進行持續監控（使用評估指標、人工審查），並探索去偏見技術。
- **惡意目的的濫用**：技術限制，以及開發者與終端使用者的教育，有助於減緩嵌入模型的惡意應用。已提供教育資源與回報機制，讓使用者能標記濫用。Gemma 模型的禁止用途列於 [Gemma 禁止使用政策][prohibited-use]。
- **隱私侵犯**：模型以已篩除某些個人資訊與其他敏感資料的資料訓練。鼓勵開發者以隱私保護技術遵守隱私法規。

### 優點

截至發佈時，這個模型系列提供高效能的開放嵌入模型實作，從設計之初就面向負責任的 AI 開發，相較於同等規模的模型。根據本文件所述的基準評估指標，這些模型已展現出優於其他同等規模開放模型替代方案的表現。
