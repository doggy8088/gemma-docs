---
title: 使用 Sentence Transformers 微調 EmbeddingGemma
description: 使用 Sentence Transformers 的對比式三元組微調（triplet fine-tuning），讓 EmbeddingGemma 2 適應特定領域術語。
models: [embeddinggemma]
frameworks: [sentence-transformers, transformers]
colab: https://colab.research.google.com/github/google-gemma/docs/blob/main/docs/notebooks/embeddinggemma-fine-tuning.ipynb
source_url: "https://ai.google.dev/gemma/docs/embeddinggemma/fine-tuning-embeddinggemma-with-sentence-transformers.md.txt"
---
# 使用 Sentence Transformers 微調 EmbeddingGemma

微調有助於彌補模型的通用理解與應用所需的特定領域高精度表現之間的落差。雖然 **EmbeddingGemma 2** 在文字與多模態輸入上提供了強大的開箱即用語義表示，但特定領域的術語通常能從有針對性的調適中獲益。

### 範例情境：金融知識庫搜尋

想像您的公司「Shibuya Financial」提供各種複雜的金融產品，例如投資信託、**NISA 帳戶**（日本免稅儲蓄帳戶）和房屋貸款。客服人員依賴內部語義搜尋引擎，為客戶諮詢檢索相關的政策文件。

未經微調的通用嵌入可能難以區分密切相關的金融概念，例如將關於*免稅投資帳戶*（NISA）的查詢與*一般儲蓄帳戶*混淆。

**本指南涵蓋：**

- 安裝相依套件並載入 **EmbeddingGemma 2** (`google/embeddinggemma-2`)
- 在微調前評估基準檢索效能
- 準備對比式 `(anchor, positive, negative)` 訓練三元組
- 使用 `SentenceTransformerTrainer` 與 `MultipleNegativesRankingLoss` 微調模型
- 評估訓練後的改善情況，並儲存您調適後的模型

## 設定與安裝

安裝 `sentence-transformers`、`transformers`，以及處理圖像、音訊和影片所需的多媒體函式庫（`pillow`、`soundfile`、`torchcodec`）。

Sentence Transformers 是一個用於使用與訓練嵌入（embedding）及重排序（reranker）模型的 Python 框架。如需更多框架細節，請參閱 [Sentence Transformers 文件](https://www.sbert.net/)。

```bash { .notebook-cell }
pip install -U sentence-transformers transformers pillow soundfile torchcodec
```

## （選用）Hugging Face 驗證

您需要有效的 Hugging Face Token 才能發佈模型。如果您在 Google Colab 內執行，可以使用 Colab 密鑰安全地使用您的 Hugging Face Token；否則可以直接在登入方法中設定 Token。請確保 Token 也具有寫入權限，因為訓練期間您會將模型推送到 Hub。

```python { .notebook-cell }
# Login into Hugging Face Hub
from huggingface_hub import login
login()
```

## 載入基礎模型

使用 `SentenceTransformer` 載入 `google/embeddinggemma-2`。

!!! tip
    如果您只在文字資料上微調，且想盡量減少 GPU 記憶體使用量，可以在初始化時傳入 `config_kwargs={"vision_config": None, "audio_config": None}` 來省略視覺與音訊編碼器。

```python { .notebook-cell }
from sentence_transformers import SentenceTransformer

model_id = "google/embeddinggemma-2"
model = SentenceTransformer(
    model_id,
    config_kwargs={"vision_config": None, "audio_config": None},
)

print(f"Device: {model.device}")
print(f"dtype: {model.dtype}")
print(model)
print("Total number of parameters in the model:", sum([p.numel() for _, p in model.named_parameters()]))
```

``` { .no-copy .output }
Loading weights:   0%|          | 0/413 [00:00<?, ?it/s]
Device: cuda:0
dtype: torch.bfloat16
SentenceTransformer(
  (0): Transformer({'transformer_task': 'feature-extraction', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'last_hidden_state'}, 'image': {'method': 'forward', 'method_output_name': 'last_hidden_state'}, 'audio': {'method': 'forward', 'method_output_name': 'last_hidden_state'}, 'video': {'method': 'forward', 'method_output_name': 'last_hidden_state'}, 'message': {'method': 'forward', 'method_output_name': 'last_hidden_state', 'format': 'structured'} }, 'module_output_name': 'token_embeddings', 'architecture': 'EmbeddingGemma2Model'})
  (1): Pooling({'embedding_dimension': 768, 'pooling_mode': 'mean', 'include_prompt': True})
  (2): Normalize({'module_input_name': 'sentence_embedding', 'module_output_name': 'sentence_embedding'})
)
Total number of parameters in the model: 271002624
```

## 基準評估（微調前）

首先，當客戶詢問以下問題時，測試基礎模型如何對三份候選知識庫文件進行排名：
> *「我想開始免稅的定期定額投資，該怎麼做？」*

正確的匹配是 **Opening a NISA Account**，因為 NISA 專指免稅投資帳戶。

!!! note "任務提示說明"
    使用 EmbeddingGemma 2 編碼文字時，務必包含任務專用提示。在此我們使用 `prompt_name="STS"`（句子相似度）。如需所有可用提示的細節，請參閱 [EmbeddingGemma 2 模型卡](model-card-v2.md#最佳實務)。

```python { .notebook-cell }
task_name = "STS"

query = "I want to start a tax-free installment investment, what should I do?"
documents = [
    "Opening a NISA Account",
    "Opening a Regular Savings Account",
    "Home Loan Application Guide",
]

def evaluate_retrieval(query_text, docs):
    query_emb = model.encode(query_text, prompt_name=task_name)
    doc_embs = model.encode(docs, prompt_name=task_name)
    similarities = model.similarity(query_emb, doc_embs)[0]

    for doc, score in zip(docs, similarities):
        print(f"Document: {doc:<35} -> 🤖 Score: {score.item():.4f}")

print("Baseline Scores (Before Fine-Tuning):")
evaluate_retrieval(query, documents)
```

``` { .no-copy .output }
Baseline Scores (Before Fine-Tuning):
Document: Opening a NISA Account              -> 🤖 Score: 0.8552
Document: Opening a Regular Savings Account   -> 🤖 Score: 0.8463
Document: Home Loan Application Guide         -> 🤖 Score: 0.8078
```

## 準備微調資料集

為了教會模型特定領域的區別，我們將訓練資料組織為對比三元組：**`(anchor, positive, negative)`**：

- **`anchor`**：使用者查詢或參考句。
- **`positive`**：具有相同領域意圖的句子（例如將 "NISA account" 與 "tax-free investment account" 配對）。
- **`negative`**：來自同一領域但語義不同的困難負樣本句子（例如 "regular savings account" 或 "investment trust fee"）。

!!! note
    為示範起見，我們在下方建立一個僅含 3 個三元組的玩具資料集。在實際應用中，請使用數百或數千個特定領域三元組，以達到穩健的泛化能力。

```python { .notebook-cell }
from datasets import Dataset

triplets = [
    [
        "How do I open a NISA account?",
        "What is the procedure for starting a new tax-free investment account?",
        "I want to check the balance of my regular savings account.",
    ],
    [
        "Are there fees for making an early repayment on a home loan?",
        "If I pay back my house loan early, will there be any costs?",
        "What is the management fee for this investment trust?",
    ],
    [
        "What is the coverage for medical insurance?",
        "Tell me about the benefits of the health insurance plan.",
        "What is the cancellation policy for my life insurance?",
    ],
]

# Convert into a Hugging Face Dataset with 'anchor', 'positive', and 'negative' columns
data_as_dicts = [
    {"anchor": row[0], "positive": row[1], "negative": row[2]}
    for row in triplets
]
train_dataset = Dataset.from_list(data_as_dicts)
print(train_dataset)
```

``` { .no-copy .output }
Dataset({
    features: ['anchor', 'positive', 'negative'],
    num_rows: 3
})
```

## 微調模型

我們使用 `SentenceTransformerTrainer` 與 **`MultipleNegativesRankingLoss`** 微調模型；此損失函數會將 `(anchor, positive)` 配對在向量空間中拉近，同時將 `(anchor, negative)` 配對（以及所有其他批次內負樣本）推遠。

關鍵是：我們在 `SentenceTransformerTrainingArguments` 中傳入 `prompts=model.prompts[task_name]`，讓模型在訓練時使用與推論時完全相同的任務前綴。

```python { .notebook-cell }
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from transformers import TrainerCallback

# Initialize contrastive ranking loss
loss = MultipleNegativesRankingLoss(model)

# Configure training arguments
args = SentenceTransformerTrainingArguments(
    output_dir="my-embeddinggemma-2",
    prompts=model.prompts[task_name],  # Train with the target task prompt prefix
    num_train_epochs=5,
    per_device_train_batch_size=1,
    learning_rate=2e-5,
    logging_steps=train_dataset.num_rows,
    report_to="none",
)

class EvaluationCallback(TrainerCallback):
    """Callback that evaluates retrieval scores at each logging step."""

    def __init__(self, eval_fn):
        self.eval_fn = eval_fn

    def on_log(self, args, state, control, **kwargs):
        print(f"\n--- Step {state.global_step} Evaluation ---")
        self.eval_fn()

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=loss,
    callbacks=[EvaluationCallback(lambda: evaluate_retrieval(query, documents))],
)

trainer.train()
```

``` { .no-copy .output }
 [15/15 01:01, Epoch 5/5]
Step	Training Loss
3	0.145459
6	0.013492
9	0.003812
12	0.001927
15	0.001546

--- Step 3 Evaluation ---
Document: Opening a NISA Account              -> 🤖 Score: 0.8542
Document: Opening a Regular Savings Account   -> 🤖 Score: 0.8270
Document: Home Loan Application Guide         -> 🤖 Score: 0.7811

--- Step 6 Evaluation ---
Document: Opening a NISA Account              -> 🤖 Score: 0.8382
Document: Opening a Regular Savings Account   -> 🤖 Score: 0.8064
Document: Home Loan Application Guide         -> 🤖 Score: 0.7507

--- Step 9 Evaluation ---
Document: Opening a NISA Account              -> 🤖 Score: 0.8292
Document: Opening a Regular Savings Account   -> 🤖 Score: 0.7952
Document: Home Loan Application Guide         -> 🤖 Score: 0.7337

--- Step 12 Evaluation ---
Document: Opening a NISA Account              -> 🤖 Score: 0.8253
Document: Opening a Regular Savings Account   -> 🤖 Score: 0.7904
Document: Home Loan Application Guide         -> 🤖 Score: 0.7269

--- Step 15 Evaluation ---
Document: Opening a NISA Account              -> 🤖 Score: 0.8254
Document: Opening a Regular Savings Account   -> 🤖 Score: 0.7902
Document: Home Loan Application Guide         -> 🤖 Score: 0.7273
Writing model shards: 100%
 1/1 [00:13<00:00, 13.14s/it]

--- Step 15 Evaluation ---
Document: Opening a NISA Account              -> 🤖 Score: 0.8254
Document: Opening a Regular Savings Account   -> 🤖 Score: 0.7902
Document: Home Loan Application Guide         -> 🤖 Score: 0.7273
TrainOutput(global_step=15, training_loss=0.033247121175130206, metrics={'train_runtime': 61.8294, 'train_samples_per_second': 0.243, 'train_steps_per_second': 0.243, 'total_flos': 0.0, 'train_loss': 0.033247121175130206, 'epoch': 5.0})
```

## 微調後評估

讓我們在微調後的模型上再執行一次測試查詢，並將結果與基準比較：

```python { .notebook-cell }
print("Post-Training Scores (After Fine-Tuning):")
evaluate_retrieval(query, documents)
```

``` { .no-copy .output }
Post-Training Scores (After Fine-Tuning):
Document: Opening a NISA Account              -> 🤖 Score: 0.8254
Document: Opening a Regular Savings Account   -> 🤖 Score: 0.7902
Document: Home Loan Application Guide         -> 🤖 Score: 0.7273
```

### 結果比較

| 候選文件 | 相關性 | 分數（微調前） | 分數（微調後） | 與榜首的區隔幅度 |
| :--- | :--- | :---: | :---: | :---: |
| **Opening a NISA Account** | **正確匹配** | `0.8552` | `0.8254` | — |
| **Opening a Regular Savings Account** | 困難負樣本 | `0.8463` | `0.7902` | **+0.0352** *（為 0.0091 的 3.9 倍寬）* |
| **Home Loan Application Guide** | 無關干擾項 | `0.8078` | `0.7273` | **+0.0981** *（為 0.0443 的 2.2 倍寬）* |

即使只經過少數訓練步驟，微調仍顯著拉大了真實匹配（**Opening a NISA Account**）與競爭金融文件之間的差距，讓下游搜尋與 RAG 管線更加可靠。

## 儲存與分享您的模型

將微調後的模型儲存在本機，以便直接用 `SentenceTransformer` 重新載入：

```python { .notebook-cell }
# Save locally
model.save_pretrained("my-embeddinggemma-2-finetuned")
```

若要與團隊分享調適後的模型，或從 Hugging Face Hub 部署，請使用 `push_to_hub`：

```python { .notebook-cell }
# Optional: Push your fine-tuned model to the Hugging Face Hub
model.push_to_hub("my-embeddinggemma-2-finetuned")
```

## 摘要與後續步驟

您已學會如何使用對比式三元組微調與 Sentence Transformers 函式庫，將 **EmbeddingGemma 2** 調適到特定領域術語。

探索 EmbeddingGemma 的更多用法：

- [Sentence Transformers 訓練概覽](https://sbert.net/docs/sentence_transformer/training_overview.html)
- [EmbeddingGemma 2 多模態嵌入](multimodal.md)
- [Gemma Cookbook 中的 EmbeddingGemma 簡易 RAG](https://github.com/google-gemma/cookbook/blob/main/tutorials/RAG_with_EmbeddingGemma.ipynb){ data-model="embeddinggemma-2" }

--8<-- "glossary.md"
