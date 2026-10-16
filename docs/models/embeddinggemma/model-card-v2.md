---
title: EmbeddingGemma 2 模型卡
description: "EmbeddingGemma 2 模型卡：架構、多模態能力、基準測試、最佳實務、訓練資料與預期用途。"
models: [embeddinggemma]
source_url: "https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2.md.txt"
---
# EmbeddingGemma 2 模型卡

![EmbeddingGemma 2 橫幅](../../assets/embeddinggemma/embeddinggemma2_banner.png)

[Hugging Face](https://huggingface.co/google/embeddinggemma-2) |
[GitHub](https://github.com/google-gemma) |
[發佈部落格](https://blog.google/innovation-and-ai/technology/developers-tools/embeddinggemma-2) |
[文件](index.md)

**授權**：[Apache 2.0](../../resources/apache-2.md) | **作者**：[Google DeepMind](https://deepmind.google/models/gemma/)

**EmbeddingGemma 2** 是由 Google DeepMind 打造的開放多模態嵌入模型，能將文字（含程式碼）、圖像、影片與音訊輸入（及其組合）映射到單一統一的 768 維向量空間。模型共有 740M 參數，由 270M 參數的文字模型結合模組化的視覺（170M）與音訊（300M）編碼器組成。

EmbeddingGemma 2 設計為可在手機與筆記型電腦等消費級硬體上執行，為裝置端應用（例如搜尋、檢索增強生成（RAG）、分類和分群）提供低延遲的語義表示。

EmbeddingGemma 2 承襲 Gemma 4 的架構與能力進展，提供以下核心特色：

- **原生多模態：** 在單一共享的 768 維嵌入空間中，統合 4 種模態（文字、圖像、影片與音訊）。
- **多語言與程式碼：** EmbeddingGemma 2 理解 100 多種語言，相較前一代在程式碼任務上提升約 14%。
- **彈性體積：** 將 270M 參數的文字主幹（130M transformer + 140M 嵌入器）與可選擇性載入的視覺（170M）及音訊（300M）編碼器結合，讓開發者只載入其使用情境所需的模態。
- **Matryoshka Representation Learning (MRL)：** 原生支援跨 128d、256d、512d 與 768d 的截斷嵌入，在品質影響極小的情況下，將向量儲存成本降低高達 **6 倍**。
- **上下文長度：** 8K 詞元上下文視窗，可處理數分鐘的音訊或影片。
- **任務導向表示：** 使用輕量文字指令前綴，針對不同任務（搜尋、分類、分群、語義相似度等）最佳化嵌入。

### 模型概覽

| 參數 | 總計 | 740M |
|---|---|---|
| | **主幹** | 130M |
| | **嵌入器** | 140M |
| | **模態編碼器** | *視覺：* 170M *音訊：* 300M |
| **架構** | **層數** | 24 |
| | **模型維度** | 512 |
| | **隱藏維度** | 2048 |
| | **滑動視窗** | 1024 詞元 |
| | **詞彙表大小** | 262,144 |
| | **注意力頭數** | 4 |
| | **KV 頭數（局部/全域）** | 2/1 |
| | **局部：全域** | 5:1 |
| | **注意力** | GQA/MQA |
| | **激活函數** | 帶 GELU 的 Gated FFN |
| | **池化** | Mean Pooling |
| | **投影層** | 512→768 |
| **輸入/輸出** | **支援模態** | 文字、圖像、影片、音訊 |
| | **上下文視窗** | 8,192 詞元 |
| | **原生輸出維度** | 768 |
| | **MRL 截斷維度** | 128, 256, 512 |

## 基準測試結果

EmbeddingGemma 2 在文字、程式碼、視覺、視覺文件、影片與音訊嵌入基準測試上進行了評估。以下所有結果均使用全精度檢查點。

### 整體評估結果 (768d)

| 模態 | 基準測試 | 指標 | EmbeddingGemma 2 | EmbeddingGemma 1 |
|---|---|---|---|---|
| *文字* | 大規模文字嵌入基準測試（MTEB，多語言，v2） | Mean(Task), Multiple | 61.36 | 61.15 |
| | 大規模文字嵌入基準測試（MTEB，程式碼，v1） | Mean(Task), NDCG@10 | 78.68 | 68.76 |
| *圖像* | 大規模圖像嵌入基準測試（MIEB，lite） | Mean(TaskType), Multiple | 64.64 | - |
| | 大規模多模態嵌入基準測試（MMEB v2 - 圖像） | Mean(Task), Hit@1 | 57.28 | - |
| | 大規模多模態嵌入基準測試（MMEB v2 - 視覺文件） | Mean(Task), NDCG@5 | 67.84 | - |
| *影片* | 大規模多模態嵌入基準測試（MMEB v2 - 影片） | Mean(Task), Hit@1 | 50.67 | - |
| *音訊* | 大規模聲音嵌入基準測試（MSEB，檢索） | Mean(Task), MRR@10 | 69.54 | - |
| | 大規模音訊嵌入基準測試（MAEB） *Hugging Face* | Mean(Task), Multiple | 49.39 | - |

### 向量截斷評估結果

透過 MRL，EmbeddingGemma 2 的表示可截斷至低於原生 768d 的 128d、256d、512d 表示並重新正規化。如此一來，模型使用者可降低儲存需求，在 256d 以上品質影響極小。128d 最適合純文字工作負載。

| 輸出維度 | 壓縮比 | MTEB *(多語言，v2) Mean(Task)* | MTEB *(英文，v2) Mean(Task)* | MTEB *(程式碼，v1) Mean(Task)* | MIEB *(lite) Mean(TaskType)* | MMEB (v2) *整體* | MSEB *(檢索）Mean(Task)* | MAEB *Mean(Task)* |
|---|---|---|---|---|---|---|---|---|
| 768d（完整） | 1:1 | 61.36 | 68.46 | 78.68 | 64.64 | 59.01 | 69.54 | 49.39 |
| 512d | 1:1.5 | 61.17 | 68.41 | 77.24 | 64.32 | 58.38 | 69.18 | 49.21 |
| 256d | 1:3 | 60.41 | 67.78 | 76.18 | 63.13 | 56.24 | 66.76 | 48.91 |
| 128d | 1:6 | 57.89 | 65.68 | 71.41 | 59.06 | 45.65 | 56.71 | 46.92 |

## 快速入門

安裝 `sentence-transformers` 函式庫：

```bash
pip install -U sentence-transformers transformers
```

產生文字嵌入：

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("google/embeddinggemma-2")

query = "What causes the northern lights?"
document = "The northern lights are caused by charged particles from the sun."

query_emb = model.encode(query, prompt_name="SearchQuery")
doc_emb = model.encode(document, prompt_name="Document")
print(model.similarity(query_emb, doc_emb))
```

## 最佳實務

為獲得最佳嵌入品質與執行效率，請遵循以下設定與最佳實務：

### 1. 任務指令前綴

EmbeddingGemma 2 以加在文字輸入前的短任務指令前綴訓練。使用正確的前綴可提升品質；省略仍可運作，但精確度會降低。前綴僅適用於文字。傳入圖像、影片與音訊時不加任何前綴。

有真實標題的文件應格式化為 `title: {title} | text: {content}`。無標題時使用 `title: none`。

**前綴表示法與用法**

根據嵌入在任務中的使用方式，我們提供兩種任務前綴。有兩種任務類型：

- **非對稱任務（例如檢索）：** 查詢使用查詢前綴，語料庫項目使用文件前綴。
- **對稱任務（例如分類、相似度）**：對所有被比較的輸入套用相同的前綴。

| 使用情境 | 任務類型 | 提示名稱 | 查詢任務指令 | 文件任務指令 *（無標題請用 `none`）* |
|---|---|---|---|---|
| 網頁/文件搜尋 | 非對稱 | SearchQuery | `task: search result | query: {query}` | `title: {title} | text: {content}` |
| 問答 | 非對稱 | QuestionAnswering | `task: question answering | query: {question}` | `title: {title} | text: {passage}` |
| 事實查核 | 非對稱 | FactChecking | `task: fact checking | query: {claim}` | `title: {title} | text: {evidence}` |
| 程式碼搜尋 | 非對稱 | CodeRetrieval | `task: code retrieval | query: {query}` | `title: {title or filename} | text: {code}` |
| 文字分類 | 對稱 | Classification | `task: classification | query: {content}` | N/A |
| 分群 | 對稱 | Clustering | `task: clustering | query: {content}` | N/A |
| 衡量相似度 | 對稱 | SentenceSimilarity | `task: sentence similarity | query: {content}` | N/A |

**請注意：** `prompt_name="Document"` 套用的是 `title: none`；有標題的文件仍需手動格式化，例如 `model.encode(f"title: {title} | text: {document}")`

### 2. 選擇性編碼器載入

視覺與音訊編碼器是獨立元件。在部署純文字或單模態管線時，若要降低記憶體消耗，可透過 SentenceTransformer 的 `config_kwargs` 停用未使用的模態編碼器：

**請注意：** 設定 EmbeddingGemma 2 省略部分編碼器的載入方式因模型函式庫而異；請參閱相應文件以取得此功能的更多資訊。

| 啟用模態 | `config_kwargs` | 有效大小 |
|---|---|---|
| 僅文字 | `{"vision_config": None, "audio_config": None}` | 270M |
| 文字與圖像 | `{"audio_config": None}` | 440M |
| 文字與音訊 | `{"vision_config": None}` | 570M |
| 完整多模態 | `{}` | 740M |

### 3. Matryoshka 維度截斷

EmbeddingGemma 2 以 Matryoshka Representation Learning 訓練，因此 768 維的輸出向量可以透過僅保留其前導維度來縮短。支援的維度為 768、512、256 與 128。較短的向量可降低儲存成本並加速相似度搜尋，但會犧牲部分品質。

在執行時期，請遵守以下指引：

- **截斷後重新正規化：** 對單位長度向量切片無法保持單位長度。縮短後的向量必須在用於餘弦相似度前先做 L2 正規化。跳過此步驟會無聲地降低排名品質——它產生的是看似合理的分數，而非錯誤。
- **查詢與文件必須共用同一維度。** 768 維的查詢無法與 128 維的語料庫評分。
- 為避免自行截斷與正規化，您應在呼叫 `model.encode()` 時傳入 `truncate_dim` 與 `normalize_embeddings` 欄位：

```python
query_emb = model.encode(
   query,
   truncate_dim=128, # or 512, 256
   normalize_embeddings=True,
)
```

各維度的模型品質列於上方**基準測試結果**一節的截斷表中。品質在 256 維以上幾乎無損。128 維會大幅降低多模態品質，採用前應先針對您的工作負載驗證。

### 4. 數值精度

以 `bfloat16` 或 `float32` 執行推論。不要使用 `float16`。

EmbeddingGemma 2 的激活範圍超出 `float16` 的動態範圍。在 `float16` 下，模型會回傳 NaN 或無聲劣化的嵌入，而非引發錯誤，因此這種失敗很容易被忽略。

`bfloat16` 是安全的，其 8 位元指數與 `float32` 相同。這是原生支援硬體上的建議預設值，且記憶體使用量為 `float32` 的一半。其他情況下請使用 `float32`，包括大多數 CPU。

在 SentenceTransformers 中，可透過 `model_kwargs` 設定，並以 `is_bf16_supported` 以程式判斷：

```python
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

model = SentenceTransformer("google/embeddinggemma-2", model_kwargs={"torch_dtype": dtype})
```

### 5. 多模態輸入

#### 交錯式

單一輸入可使用單一共享的 8,192 詞元上下文，混合文字與圖像、影片和音訊。

每個媒體項目在序列中的位置使用來自模型詞彙表的佔位符詞元在文字中標示：

- `<|image|>` 標示圖像輸入的位置
- `<|video|>` 標示影片輸入的位置
- `<|audio|>` 標示音訊輸入的位置

例如，為搜尋索引的產品列表可編碼為：

```python
emb_interleaved = model.encode({
    "text": "Waterproof running shoes. <|image|> Featuring a breathable mesh upper. <|image|> Grip test on wet rock: <|video|>",
    "image": ["shoe.jpg", "mesh.jpg"],
    "video": "demo.mp4",
})
```

文字中的每個佔位符依序從對應的鍵填入。此呼叫回傳單一嵌入，同時表示文字、圖像與影片。此嵌入可直接與任何其他 EmbeddingGemma 2 嵌入比較；例如純文字查詢 "waterproof shoes for trail running"。

#### 上下文限制

所有模態共用單一 8,192 詞元的上下文視窗。每種模態以固定速率消耗該預算：

| 模態 | 詞元成本 | 最大輸入 |
|---|---|---|
| *文字* | 每個子詞 1 詞元 | 8,192 詞元 |
| *圖像* | 每張圖像 280 詞元 *（預設）* | 約 29 張圖像 |
| *影片* | 每幀 140 詞元 *（預設）* | 約 58 幀 |
| *音訊* | 每秒 25 詞元 | 約 327 秒 |

請注意，使用較低的視覺詞元預算（如下所述）時，圖像與影片的最大輸入可高達約 114 張圖像或幀。

上述最大值假設單一模態且無伴隨文字。交錯式輸入從同一預算支出，因此混合模態會減少每種模態能容納的量。另請注意，使用較低的視覺詞元預算（如下所述）時，圖像與影片的最大輸入可高達約 114 張圖像或幀。

#### 視覺詞元預算

模型使用者可以放棄輸入圖像/影片幀的預設序列長度，以 70 到 1120 之間的「軟詞元」數量來表示圖像。提高輸入圖像的視覺預算會以延遲與詞元數換取品質。

一般而言，更高的輸入序列長度能捕捉更多資訊。擴大輸入序列長度意味著更具表現力的輸入表示；因此，提高輸入序列長度可改善細粒度視覺理解，提升嵌入品質與下游使用情境的效能。

#### 輸入取樣預設值

音訊與影片輸入的預設取樣率如下：

- 預設情況下，影片以 **每秒 1 幀**的取樣幀透過視覺編碼器處理；FPS 取樣率可設定。
- 音訊應提供為 **16 kHz 單聲道**。

## 模型資料

我們的預訓練資料集是涵蓋廣泛領域與模態的大規模多樣資料集合，包括網頁文件、程式碼、圖像、影片與音訊，截止日期為 2025 年 1 月。其中包括：

- **網頁文件：** 多樣化的網頁文字集合，確保模型接觸廣泛的語言風格、主題與詞彙。訓練資料集包含 140 多種語言的內容。
- **程式碼：** 讓模型接觸程式碼有助於它學習程式語言的語法與模式，提升其理解程式碼相關語義的能力。
- **圖像：** 廣泛的圖像，讓模型能執行圖像分析與視覺資料擷取任務。
- **影片：** 涵蓋多樣視覺場景、人類動作與時間動態的影片序列與片段。
- **音訊：** 多種語言的語音錄音、環境音與聲學事件。
- **跨模態樣本：** 配對的文字、圖像、影片與音訊資料，以對齊不同模態的表示。

### 資料處理

訓練資料套用了多種資料清理與篩選方法：

- **CSAM 篩選**：在資料準備流程的多個階段套用嚴格的 CSAM（兒童性虐待素材）篩選，確保排除有害與非法內容。
- **敏感資料篩選**：作為讓 Gemma 預訓練模型安全可靠的一部分，使用自動化技術從訓練集中篩除某些個人資訊與其他敏感資料。
- **其他方法**：依[我們的政策](https://ai.google/static/documents/ai-responsibility-update-published-february-2025.pdf)進行內容品質與安全篩選。

### 倫理與安全

EmbeddingGemma 2 是預訓練嵌入模型。與生成式模型不同，它不經過後訓練對齊、安全調校或輸出層審查。開發期間的安全緩解措施聚焦於預訓練資料篩選，以減少有害內容的暴露，並降低學習到的嵌入空間中的嚴重偏見，符合 [Google AI 原則](https://ai.google/principles/)。

由於嵌入模型產生的是表示而非面向使用者的文字，安全風險體現在這些表示的下游使用方式。

開發者與部署者負責評估並實施應用層防護措施，例如檢索篩選與公平性測試，以適合其特定生產使用情境。

部署必須遵守 [Gemma 禁止使用政策][prohibited-use]。

## 使用與限制

EmbeddingGemma 2 有一些使用者應注意的限制。

### 預期用途

EmbeddingGemma 2 從輸入內容產生嵌入，可用於多種下游應用。

以下潛在用途清單並不完整，其目的是提供模型建立者在訓練與開發時考慮的可能使用情境背景資訊。

- **檢索：** 用於跨文字、程式碼、圖像、影片與音訊的語義搜尋的嵌入，例如文件搜尋、企業知識庫 RAG、來自自然語言查詢的程式碼搜尋，或以語音查詢搜尋音訊典藏。
- **分類：** 用於依預設標籤分類輸入的嵌入，例如情緒分析、內容審查、音訊事件偵測或圖像分類。
- **分群：** 用於依語義相似度分組輸入的嵌入，例如組織文件集合、依主題分群客戶回饋，或發現跨模態的近似重複內容。
- **語義相似度：** 用於衡量輸入間成對相似度的嵌入，例如推薦系統、重複內容偵測、改寫識別或跨語言內容對齊。
- **事實查核：** 用於根據主張檢索證據文件的嵌入，例如自動化事實查核系統或來源歸屬管線。

### 限制

- **訓練資料：** 訓練資料的品質與多樣性顯著影響模型能力。訓練資料中的偏見或缺口可能導致模型回應的限制。
  - 訓練資料集的範圍決定了模型能有效處理的主題領域。
  - 例如，雖然 EmbeddingGemma 2 支援 100 多種語言，模型在各語言上的表現可能並不相等。
- **上下文與任務複雜度：** 模型在能以清晰提示與指令框架化的任務上表現良好。
  - 開放式或高度複雜的任務可能具有挑戰性。
  - 模型表現可能受提供上下文量的影響（在一定程度內，較長的上下文通常帶來更好的輸出）。
- **語言歧義與細微差別：** 自然語言本質上很複雜。模型可能難以掌握細微差別、諷刺或比喻語言。
- **任務指令使用：** 對於文字任務，省略建議的任務前綴可能導致次佳的嵌入品質。

### 倫理考量與風險

在建立開放嵌入模型時，我們審慎考慮了以下事項：

- **偏見與公平性**
  - 以大規模真實世界文字與圖像資料訓練的模型，可能反映訓練素材中內嵌的社會文化偏見。
  - EmbeddingGemma 2 的訓練資料經過安全篩選，以減緩這些偏見的風險。
- **錯誤資訊與濫用**
  - 嵌入表示可能被濫用，以錯誤、誤導或有害的方式檢索、分類或以其他方式組織嵌入內容。
  - 模型附有負責任使用的指引，請參閱 [Responsible Generative AI Toolkit][responsible-ai]。
- **透明度與問責**
  - 本模型卡彙整了模型架構、能力、限制與評估流程的細節。
  - 負責任開發的開放模型提供分享創新的機會，透過讓 AI 生態系中的開發者與研究者都能使用視覺語言模型（VLM）技術。

### 已識別的風險與緩解措施

- **惡意目的的濫用：** 技術限制，以及開發者與終端使用者的教育，有助於減緩嵌入模型的惡意應用。已提供教育資源與回報機制，讓使用者能標記濫用。
- **隱私侵犯：** 模型以已篩除某些個人資訊與其他敏感資料的資料訓練。鼓勵開發者以隱私保護技術遵守隱私法規。
- **偏見的延續：** 鼓勵進行持續監控（使用評估指標、人工審查），並在模型訓練、微調與其他使用情境中探索去偏見技術。

### 優點

EmbeddingGemma 2 是 10 億參數以下最強的多模態嵌入模型之一。我們很期待看到開發者如何為裝置端或邊緣 AI 應用使用與調適這個模型。

此模型從設計之初就面向負責任的 AI 開發，如同其他 Gemma 系列模型。
