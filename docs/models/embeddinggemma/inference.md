---
title: 使用 Sentence Transformers 產生嵌入
description: 使用 EmbeddingGemma 2 與 Sentence Transformers 產生文字嵌入、計算語義相似度、套用任務提示，並打造 RAG 與零樣本分類器。
models: [embeddinggemma]
frameworks: [sentence-transformers, transformers]
colab: https://colab.research.google.com/github/google-gemma/docs/blob/main/docs/notebooks/embeddinggemma-inference.ipynb
source_url: "https://ai.google.dev/gemma/docs/embeddinggemma/inference-embeddinggemma-with-sentence-transformers.md.txt"
---
# 使用 Sentence Transformers 產生嵌入

**EmbeddingGemma 2** 是一個輕量、開源的 740M 參數模型，專為統一的文字與多模態嵌入而設計。它基於 Gemma 4 解碼器架構，能將輸入映射到統一的 768 維向量空間。

得益於其精簡的體積，您可以在消費級 GPU 或 CPU 上，於本機離線執行要求嚴苛的語義工作流程（例如檢索增強生成（RAG）、語義搜尋和零樣本分類）。

**本指南涵蓋：**

- 安裝相依套件並載入 **EmbeddingGemma 2**
- 產生**文字嵌入**、計算語義相似度，並套用**任務專用提示**
- 打造下游應用（**RAG** 與**零樣本分類**）
- 使用 **Matryoshka Representation Learning (MRL)** 最佳化儲存與推論速度

!!! tip "想找圖像、音訊或影片嵌入？"
    請參閱 [EmbeddingGemma 2 多模態嵌入](multimodal.md)指南。

## 設定與安裝

安裝 `sentence-transformers` 與 `transformers`。

Sentence Transformers 是一個用於使用與訓練嵌入（embedding）及重排序（reranker）模型的 Python 框架。如需更多框架細節，請參閱 [Sentence Transformers 文件](https://www.sbert.net/)。

```bash { .notebook-cell }
pip install -U sentence-transformers transformers pillow soundfile torchcodec
```

## 模型初始化

直接使用 `SentenceTransformer` 類別實例化 EmbeddingGemma 2。

```python { .notebook-cell }
from sentence_transformers import SentenceTransformer

model_id = "google/embeddinggemma-2"
model = SentenceTransformer(model_id)

print(f"device: {model.device}")
print(f"dtype: {model.dtype}")
print(model)
print("Total number of parameters in the model:", sum(p.numel() for p in model.parameters()))
```

``` { .no-copy .output }
Loading weights:   0%|          | 0/1376 [00:00<?, ?it/s]
device: cuda:0
dtype: torch.bfloat16
SentenceTransformer(
  (0): Transformer({'transformer_task': 'feature-extraction', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'last_hidden_state'}, 'image': {'method': 'forward', 'method_output_name': 'last_hidden_state'}, 'audio': {'method': 'forward', 'method_output_name': 'last_hidden_state'}, 'video': {'method': 'forward', 'method_output_name': 'last_hidden_state'}, 'message': {'method': 'forward', 'method_output_name': 'last_hidden_state', 'format': 'structured'} }, 'module_output_name': 'token_embeddings', 'architecture': 'EmbeddingGemma2Model'})
  (1): Pooling({'embedding_dimension': 768, 'pooling_mode': 'mean', 'include_prompt': True})
  (2): Normalize({'module_input_name': 'sentence_embedding', 'module_output_name': 'sentence_embedding'})
)
Total number of parameters in the model: 744371488
```

!!! tip "純文字記憶體最佳化"
    預設情況下，`SentenceTransformer` 會載入所有模態編碼器（740M 參數）。如果您的工作流程只使用文字，可以傳入 `config_kwargs={"vision_config": None, "audio_config": None}` 來跳過視覺與音訊編碼器的載入（270M 參數）。

!!! note "精度說明"
    EmbeddingGemma 2 的活化值與 `float16` 不相容。請使用 `bfloat16`（支援的 GPU 上為預設值）或 `float32`。

## 文字嵌入與任務提示

了解如何將文字編碼為稠密向量表示、衡量輸入之間的語義相似度，並套用任務專用提示前綴以最佳化嵌入品質。

### 產生文字嵌入

嵌入（embedding）是捕捉輸入語義的稠密數值向量。讓我們編碼三個單字——`["king", "queen", "car"]`——來檢視它們的 768 維表示。

```python { .notebook-cell }
words = ["king", "queen", "car"]

# Generate embeddings using model.encode()
embeddings = model.encode(words)

print(embeddings)
for idx, embedding in enumerate(embeddings):
    print(f"Embedding {idx+1} (shape): {embedding.shape}")
```

``` { .no-copy .output }
[[ 0.03222656  0.00285339  0.02160645 ...  0.012146    0.03149414
  -0.0067749 ]
 [ 0.00787354  0.01361084  0.02502441 ... -0.01269531  0.02612305
   0.01361084]
 [-0.04052734 -0.00860596  0.04492188 ...  0.00964355 -0.01501465
  -0.00726318]]
Embedding 1 (shape): (768,)
Embedding 2 (shape): (768,)
Embedding 3 (shape): (768,)
```

嵌入的關鍵性質是**語義空間中向量之間的距離**。由於 `"king"` 與 `"queen"` 共享王室頭銜的語義特徵，它們的向量位置彼此靠近，而 `"car"` 則距離較遠。

### 判斷語義相似度

我們可以使用 `model.similarity()` 計算嵌入之間的餘弦相似度，來量化兩個輸入在語義上有多相似。

以下是三組示範高、中、低語義相似度的句子配對：

- **高相似度：**
  - *句子 A：* `"The chef prepared a delicious meal for the guests."`
  - *句子 B：* `"A tasty dinner was cooked by the chef for the visitors."`
  - *推論：* 兩個句子使用主動與被動語態描述完全相同的事件。
- **中相似度：**
  - *句子 A：* `"She is an expert in machine learning."`
  - *句子 B：* `"He has a deep interest in artificial intelligence."`
  - *推論：* 相關領域（ML 是 AI 的子領域），但主體與專業程度不同。
- **低相似度：**
  - *句子 A：* `"The weather in Tokyo is sunny today."`
  - *句子 B：* `"I need to buy groceries for the week."`
  - *推論：* 完全無關的主題，沒有語義重疊。

```python { .notebook-cell }
sentence_high = [
    "The chef prepared a delicious meal for the guests.",
    "A tasty dinner was cooked by the chef for the visitors.",
]
sentence_medium = [
    "She is an expert in machine learning.",
    "He has a deep interest in artificial intelligence.",
]
sentence_low = [
    "The weather in Tokyo is sunny today.",
    "I need to buy groceries for the week.",
]

for pair in [sentence_high, sentence_medium, sentence_low]:
    print("🙋‍♂️", pair)
    pair_embeddings = model.encode(pair)
    score = model.similarity(pair_embeddings[0], pair_embeddings[1]).item()
    print(f" `-> 🤖 similarity score: {score:.4f}")
```

``` { .no-copy .output }
🙋‍♂️ ['The chef prepared a delicious meal for the guests.', 'A tasty dinner was cooked by the chef for the visitors.']
 `-> 🤖 similarity score: 0.9184
🙋‍♂️ ['She is an expert in machine learning.', 'He has a deep interest in artificial intelligence.']
 `-> 🤖 similarity score: 0.7835
🙋‍♂️ ['The weather in Tokyo is sunny today.', 'I need to buy groceries for the week.']
 `-> 🤖 similarity score: 0.6880
```

!!! note "分數解讀說明"
    文字嵌入的餘弦相似度值往往比人類直覺預期偏高，實務上很少低於 0.5。例如，即使是完全無關的句子（像上方「低相似度」範例），相似度分數也可能高達約 0.6880。評估語義重疊時，請記住這個較高的基準。

### 使用任務提示最佳化嵌入

要在文字任務上發揮 EmbeddingGemma 2 的最佳效能，請在輸入前加上**指示性提示**（任務前綴）。任務提示能針對特定目標（例如句子相似度、非對稱搜尋檢索或分類）調節模型，並幫助區分查詢與文件。

**前綴表示法與用法**

根據嵌入在任務中的使用方式，我們提供兩種任務前綴。有兩種任務類型：

- **非對稱任務（例如檢索）：** 查詢使用查詢前綴，語料庫項目使用文件前綴。
- **對稱任務（例如分類、相似度）：** 對所有被比較的輸入套用相同的前綴。

| 使用情境 | 任務類型 | 查詢任務指令 | 文件任務指令 *（無標題請用 `none`）* |
|---|---|---|---|
| 網頁/文件搜尋 | 非對稱 | `task: search result | query: {query}` | `title: {title} | text: {content}` |
| 問答 | 非對稱 | `task: question answering | query: {question}` | `title: {title} | text: {passage}` |
| 事實查核 | 非對稱 | `task: fact checking | query: {claim}` | `title: {title} | text: {evidence}` |
| 程式碼搜尋 | 非對稱 | `task: code retrieval | query: {query}` | `title: {title or filename} | text: {code}` |
| 文字分類 | 對稱 | `task: classification | query: {content}` | N/A |
| 分群 | 對稱 | `task: clustering | query: {content}` | N/A |
| 衡量相似度 | 對稱 | `task: sentence similarity | query: {content}` | N/A |

**請注意：** 前綴僅用於文字輸入；編碼圖像、影片或音訊時，請直接傳入。

您可以在推論期間以三種方式套用提示：

1. **使用 `prompt_name`（建議）：** 選擇註冊在 `model.prompts` 中的內建任務提示：
   ```python
   embeddings = model.encode(sentences, prompt_name="STS")
   ```
2. **使用 `prompt`：** 直接將自訂前綴字串傳給 `model.encode()`：
   ```python
   embeddings = model.encode(sentences, prompt="task: sentence similarity | query: ")
   ```
3. **使用預設提示：** 若模型已設定 `default_prompt_name`，呼叫 `model.encode(sentences)` 會自動套用。

讓我們檢視所有內建任務提示，並使用 `prompt_name="STS"` 重新評估句子相似度配對：

```python { .notebook-cell }
print("Available task prompts:")
for name, prefix in model.prompts.items():
    print(f"  {name}: {prefix!r}")
print("-" * 80)

for pair in [sentence_high, sentence_medium, sentence_low]:
    print("🙋‍♂️", pair)
    pair_embeddings = model.encode(pair, prompt_name="STS")
    score = model.similarity(pair_embeddings[0], pair_embeddings[1]).item()
    print(f" `-> 🤖 STS score: {score:.4f}")
```

``` { .no-copy .output }
Available task prompts:
  query: 'task: search result | query: '
  document: 'title: none | text: '
  BitextMining: 'task: search result | query: '
  Classification: 'task: classification | query: '
  Clustering: 'task: clustering | query: '
  CodeRetrieval: 'task: code retrieval | query: '
  Document: 'title: none | text: '
  FactChecking: 'task: fact checking | query: '
  InstructionRetrieval: 'task: code retrieval | query: '
  MultilabelClassification: 'task: classification | query: '
  PairClassification: 'task: sentence similarity | query: '
  QuestionAnswering: 'task: question answering | query: '
  Reranking: 'task: search result | query: '
  Retrieval: 'task: search result | query: '
  Retrieval-document: 'title: none | text: '
  Retrieval-query: 'task: search result | query: '
  STS: 'task: sentence similarity | query: '
  SearchQuery: 'task: search result | query: '
  SentenceSimilarity: 'task: sentence similarity | query: '
  Summarization: 'task: sentence similarity | query: '
---
🙋‍♂️ ['The chef prepared a delicious meal for the guests.', 'A tasty dinner was cooked by the chef for the visitors.']
 `-> 🤖 STS score: 0.9555
🙋‍♂️ ['She is an expert in machine learning.', 'He has a deep interest in artificial intelligence.']
 `-> 🤖 STS score: 0.8628
🙋‍♂️ ['The weather in Tokyo is sunny today.', 'I need to buy groceries for the week.']
 `-> 🤖 STS score: 0.7340
```

## 下游應用

將 EmbeddingGemma 2 嵌入實際應用於常見的真實工作流程，例如檢索增強生成（RAG）的文件檢索與零樣本文字分類。

### 檢索增強生成（RAG）

在非對稱檢索與 RAG 管線中，查詢通常是簡短的問題，而文件是較長的段落。使用專用的檢索提示來最佳化搜尋索引的兩端：

- **搜尋查詢：** 使用 `prompt_name="Retrieval-query"`（套用 `"task: search result | query: "`）。
- **文件：** 使用 `prompt_name="Retrieval-document"`（套用 `"title: none | text: "`），或傳入自訂 `prompt` 字串以包含文件標題（`"title: <Document Title> | text: "`）。加入標題可顯著提升檢索準確度。

```python { .notebook-cell }
query = "How do I reduce memory usage when running EmbeddingGemma 2?"

documents = [
    {
        "title": "Model Initialization and Memory Optimization",
        "text": "You can disable the vision or audio encoders by passing vision_config=None or audio_config=None inside config_kwargs.",
    },
    {
        "title": "Matryoshka Representation Learning",
        "text": "MRL allows truncating 768-dimensional embeddings to 512 or 256 dimensions for faster vector search and smaller indexes.",
    },
    {
        "title": "Audio Processing Specifications",
        "text": "Audio waveforms are automatically resampled to 16 kHz before feature extraction.",
    },
]

# Encode query with Retrieval-query prompt
query_emb = model.encode(query, prompt_name="Retrieval-query")

# Encode documents with their titles embedded in the prompt
doc_embeddings = [
    model.encode(doc["text"], prompt=f"title: {doc['title']} | text: ")
    for doc in documents
]

scores = model.similarity(query_emb, doc_embeddings)[0]
for doc, score in zip(documents, scores):
    print(f"[Score: {score:.4f}] {doc['title']}")
```

``` { .no-copy .output }
[Score: 0.6855] Model Initialization and Memory Optimization
[Score: 0.6386] Matryoshka Representation Learning
[Score: 0.5653] Audio Processing Specifications
```

如需完整的端對端 RAG 教學，請參閱 Gemma Cookbook 中的 [RAG with EmbeddingGemma notebook](https://github.com/google-gemma/cookbook/blob/main/tutorials/RAG_with_EmbeddingGemma.ipynb){ data-model="embeddinggemma-2" }。

### 零樣本分類

零樣本文字分類不需任務專用訓練，透過選擇與輸入文字嵌入相似度最高的候選標籤，將輸入分配到預定義類別。

為零樣本分類選擇任務提示時，請考慮目標標籤是否符合標準分類資料集：

- **自訂或任意類別（`prompt_name="Clustering"`）：** 對於領域特定或臨時標籤（如 `"Billing Issue"`、`"Technical Support"`、`"Sales Inquiry"`），`"Classification"` 提示效果不佳，因為訓練資料集不包含這些特定標籤。請改用 `prompt_name="Clustering"`，依語義主題相似度分組輸入與候選標籤。
- **標準分類任務（`prompt_name="Classification"`）：** 常見分類提示（例如情緒分析：`"Positive"`、`"Neutral"`、`"Negative"`）適合使用 `prompt_name="Classification"`，因為它們與訓練期間見過的分類任務一致。

#### 使用 `Clustering` 做自訂類別路由

使用 `prompt_name="Clustering"` 將收到的支援工單路由到自訂部門類別：

```python { .notebook-cell }
labels = ["Billing Issue", "Technical Support", "Sales Inquiry"]

tickets = [
    "Excuse me, the app freezes on the login screen. It won't work even when I try to reset my password.",
    "I would like to inquire about your enterprise plan pricing and features for a team of 50 people.",
]

# Encode labels and incoming tickets using the Clustering prompt
label_embeddings = model.encode(labels, prompt_name="Clustering")
ticket_embeddings = model.encode(tickets, prompt_name="Clustering")

# Compute similarity matrix and find the best-matching label for each ticket
similarities = model.similarity(ticket_embeddings, label_embeddings)
print("Similarity matrix:\n", similarities)

predicted_indices = similarities.argmax(dim=1)

for ticket, pred_idx in zip(tickets, predicted_indices):
    print(f"\n🙋‍♂️ Ticket: {ticket}")
    print(f" `-> 🤖 Predicted Category: {labels[pred_idx]}")
```

``` { .no-copy .output }
Similarity matrix:
 tensor([[0.8704, 0.8904, 0.8394],
        [0.8426, 0.8566, 0.8758]])

🙋‍♂️ Ticket: Excuse me, the app freezes on the login screen. It won't work even when I try to reset my password.
 `-> 🤖 Predicted Category: Technical Support

🙋‍♂️ Ticket: I would like to inquire about your enterprise plan pricing and features for a team of 50 people.
 `-> 🤖 Predicted Category: Sales Inquiry
```

#### 使用 `Classification` 做情緒分析

對於標準情緒分析，使用 `prompt_name="Classification"` 編碼常見情緒標籤與輸入句子：

```python { .notebook-cell }
labels = ["Positive", "Neutral", "Negative"]

sentences = [
    # Positive
    "The sunset over the ridge was absolutely breathtaking.",
    "This is hands down the best coffee I have ever tasted.",
    # Neutral
    "The package arrived on Tuesday afternoon.",
    "The meeting is scheduled for 10:00 AM in Conference Room B.",
    # Negative
    "The battery dies after barely two hours of use.",
    "It felt like an utter waste of time and money.",
]

# Encode labels and incoming sentences using the Classification prompt
label_embeddings = model.encode(labels, prompt_name="Classification")
sentence_embeddings = model.encode(sentences, prompt_name="Classification")

# Compute similarity matrix and find the best-matching label for each sentence
similarities = model.similarity(sentence_embeddings, label_embeddings)
print("Similarity matrix:\n", similarities)

predicted_indices = similarities.argmax(dim=1)

for sentence, pred_idx in zip(sentences, predicted_indices):
    print(f"\n🙋‍♂️ Sentence: {sentence}")
    print(f" `-> 🤖 Predicted Sentiment: {labels[pred_idx]}")
```

``` { .no-copy .output }
Similarity matrix:
 tensor([[0.8177, 0.6642, 0.6423],
        [0.8731, 0.6198, 0.6375],
        [0.6581, 0.8198, 0.6425],
        [0.6115, 0.7675, 0.6093],
        [0.6273, 0.6680, 0.7666],
        [0.6103, 0.6236, 0.8660]])

🙋‍♂️ Sentence: The sunset over the ridge was absolutely breathtaking.
 `-> 🤖 Predicted Sentiment: Positive

🙋‍♂️ Sentence: This is hands down the best coffee I have ever tasted.
 `-> 🤖 Predicted Sentiment: Positive

🙋‍♂️ Sentence: The package arrived on Tuesday afternoon.
 `-> 🤖 Predicted Sentiment: Neutral

🙋‍♂️ Sentence: The meeting is scheduled for 10:00 AM in Conference Room B.
 `-> 🤖 Predicted Sentiment: Neutral

🙋‍♂️ Sentence: The battery dies after barely two hours of use.
 `-> 🤖 Predicted Sentiment: Negative

🙋‍♂️ Sentence: It felt like an utter waste of time and money.
 `-> 🤖 Predicted Sentiment: Negative
```

## Matryoshka Representation Learning (MRL)

EmbeddingGemma 2 以 **Matryoshka Representation Learning (MRL)** 訓練，將最重要的語義資訊集中在輸出向量的前幾個維度。

您無需為不同的延遲或儲存預算部署多個模型——只需將完整的 768 維嵌入**截斷**為較小的前綴（例如 **512**、**256** 或 **128** 維），同時保持強勁的語義品質。截斷維度時，務必設定 `normalize_embeddings=True`，讓截斷後的向量重新正規化為單位長度。

讓我們比較 **768**、**512** 與 **128** 維的單字相似度分數：

```python { .notebook-cell }
words = ["king", "queen", "car"]

def evaluate_mrl(embeddings, sim_fn_name):
    print(f"Embedding shape: {embeddings.shape} | Similarity function: {sim_fn_name}")
    sims = model.similarity(embeddings[0], embeddings[1:])
    for idx, word in enumerate(words[1:]):
        print(f"  🙋‍♂️ 'king' vs. '{word}' -> 🤖 score: {sims[0][idx].item():.4f}")

# 1. Full 768-dimensional embeddings
emb_768 = model.encode(words, prompt_name="STS", normalize_embeddings=True)
evaluate_mrl(emb_768, model.similarity_fn_name)
```

``` { .no-copy .output }
Embedding shape: (3, 768) | Similarity function: cosine
  🙋‍♂️ 'king' vs. 'queen' -> 🤖 score: 0.9128
  🙋‍♂️ 'king' vs. 'car' -> 🤖 score: 0.8354
```

### 截斷至 512 維

直接將 `truncate_dim=512` 傳給 `model.encode()`，即可即時切片並重新正規化向量：

```python { .notebook-cell }
# 2. Truncate to 512 dimensions at inference time
emb_512 = model.encode(
    words,
    prompt_name="STS",
    truncate_dim=512,
    normalize_embeddings=True,
)
evaluate_mrl(emb_512, model.similarity_fn_name)
```

``` { .no-copy .output }
Embedding shape: (3, 512) | Similarity function: cosine
  🙋‍♂️ 'king' vs. 'queen' -> 🤖 score: 0.9124
  🙋‍♂️ 'king' vs. 'car' -> 🤖 score: 0.8334
```

### 截斷至 128 維並使用點積相似度

要在高吞吐量向量資料庫中追求最高速度，可以在模型初始化時設定 `truncate_dim=128` 與 `similarity_fn_name="dot"`。由於正規化後的嵌入具有單位長度，點積在數學上等價於餘弦相似度，但計算更快：

```python { .notebook-cell }
# 3. Initialize model with default 128-dim truncation and dot-product similarity
fast_model = SentenceTransformer(
    model_id,
    truncate_dim=128,
    similarity_fn_name="dot",
    device=model.device,
)

emb_128 = fast_model.encode(words, prompt_name="STS", normalize_embeddings=True)
evaluate_mrl(emb_128, fast_model.similarity_fn_name)
```

``` { .no-copy .output }
Loading weights:   0%|          | 0/1376 [00:00<?, ?it/s]
Embedding shape: (3, 128) | Similarity function: dot
  🙋‍♂️ 'king' vs. 'queen' -> 🤖 score: 0.9470
  🙋‍♂️ 'king' vs. 'car' -> 🤖 score: 0.8957
```

!!! note
    EmbeddingGemma 2 支援 128 到 768 之間的任意輸出維度。不過，我們建議使用 768、512、256 或 128 以獲得最佳效能。

## 摘要與後續步驟

您現在已經知道如何使用 **EmbeddingGemma 2** 與 `sentence-transformers` 函式庫產生文字嵌入，也探索了如何套用任務專用提示、打造零樣本分類器與 RAG 檢索器，以及使用 Matryoshka Representation Learning (MRL) 壓縮嵌入。

### 延伸閱讀

- [EmbeddingGemma 2 模型卡](model-card-v2.md)
- [EmbeddingGemma 2 多模態嵌入](multimodal.md)
- [使用 Sentence Transformers 微調 EmbeddingGemma](fine-tuning.md)
- [Sentence Transformers 提示模板指南](https://sbert.net/examples/sentence_transformer/applications/computing-embeddings/README.html#prompt-templates)
- [Gemma Cookbook 中的 EmbeddingGemma 簡易 RAG](https://github.com/google-gemma/cookbook/blob/main/tutorials/RAG_with_EmbeddingGemma.ipynb){ data-model="embeddinggemma-2" }

--8<-- "glossary.md"
