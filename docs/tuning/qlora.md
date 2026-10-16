---
title: 使用 Hugging Face Transformers 與 QLoRA 微調 Gemma
description: 本指南將引導您使用 Hugging Face Transformers 與 TRL 在自訂 text-to-sql 資料集上微調 Gemma。
models: [gemma-4]
frameworks: [transformers, peft, trl]
source_url: "https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora.md.txt"
colab: https://colab.research.google.com/github/google-gemma/docs/blob/main/docs/notebooks/qlora-gemma-4-e2b.ipynb
---
# 使用 Hugging Face Transformers 與 QLoRA 微調 Gemma


本指南將引導您使用 Hugging Face [Transformers](https://huggingface.co/docs/transformers/index) 與 [TRL](https://huggingface.co/docs/trl/index) 在自訂 text-to-sql 資料集上微調 Gemma。您將學到：

- 什麼是量化低秩適配（QLoRA）
- 設定開發環境
- 建立並準備微調資料集
- 使用 TRL 與 SFTTrainer 微調 Gemma
- 測試模型推論並產生 SQL 查詢

!!! Note
    本指南是為在配備 NVIDIA T4 GPU（16GB）與 Gemma 1B 的 Google colaboratory 帳戶上執行而設計，但也可以調整為在更大的 GPU 與更大的模型上執行。


## 什麼是量化低秩適配（QLoRA）

本指南展示了[量化低秩適配（Quantized Low-Rank Adaptation, QLoRA）](https://arxiv.org/abs/2305.14314)的用法。它成為高效微調 LLM 的熱門方法，因為它降低了運算資源需求，同時維持高效能。在 QLoRA 中，預訓練模型會被量化為 4-bit 並凍結權重，接著附加可訓練的轉接器層（LoRA），只訓練這些轉接器層。之後可以將轉接器權重合併到基底模型，或保留為獨立的轉接器。

## 設定開發環境

第一步是安裝 Hugging Face 函式庫（包括 TRL）與 datasets，以微調開放模型，包括不同的 RLHF 與對齊技術。

```python { .notebook-cell }
# Install Pytorch & other libraries
%pip install torch tensorboard
%pip install -U torchao

# Install Transformers
%pip install "transformers>=5.10.1"

# Install Hugging Face libraries
%pip install datasets accelerate evaluate bitsandbytes trl "peft>=0.19.0" protobuf sentencepiece

# COMMENT IN: if you are running on a GPU that supports BF16 data type and flash attn, such as NVIDIA L4 or NVIDIA A100
#%pip install flash-attn
```

*注意：如果您使用的 GPU 採用 Ampere 架構（如 NVIDIA L4）或更新版本，可以使用 Flash Attention。Flash Attention 是一種顯著加速運算並將記憶體使用量從序列長度的二次方降為線性的作法，最多可加速訓練 3 倍。更多資訊請參閱 [FlashAttention](https://github.com/Dao-AILab/flash-attention/tree/main)。*

您需要有效的 Hugging Face Token 才能發布模型。如果您在 Google Colab 中執行，可以使用 Colab 密鑰安全地使用 Hugging Face Token；否則可以直接在 `login` 方法中設定 Token。請確保您的 Token 具有寫入權限，因為訓練期間會將模型推送到 Hub。

```python { .notebook-cell }
# Login into Hugging Face Hub
from huggingface_hub import login
login()
```

## 建立並準備微調資料集

微調 LLM 時，瞭解您的用例與要解決的任務很重要。這有助於建立用來微調模型的資料集。如果您還沒定義用例，可能要先回頭重新規劃。

例如，本指南聚焦於以下用例：

- 將自然語言轉 SQL 的模型微調後，無縫整合到資料分析工具中。目標是大幅減少產生 SQL 查詢所需的時間與專業知識，讓非技術使用者也能從資料中提取有意義的洞察。

Text-to-SQL 是微調 LLM 的好用例，因為它是複雜任務，需要大量關於資料與 SQL 語言的（內部）知識。

一旦確定微調是正確的解法，就需要一個微調用的資料集。資料集應該是您要解決的任務的多樣化示範集合。建立這種資料集有幾種方式，包括：

- 使用現有的開源資料集，例如 [Spider](https://huggingface.co/datasets/spider)
- 使用 LLM 產生的合成資料集，例如 [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)
- 使用人工建立的資料集，例如 [Dolly](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
- 組合多種方法，例如 [Orca](https://huggingface.co/datasets/Open-Orca/OpenOrca)

每種方法各有優缺點，取決於預算、時間與品質需求。例如，使用現有資料集最簡單，但可能不適合您的特定用例；請領域專家建立可能最準確，但耗時又昂貴。也可以組合多種方法來建立指令資料集，如 [Orca: Progressive Learning from Complex Explanation Traces of GPT-4.](https://arxiv.org/abs/2306.02707) 所示。

本指南使用現有資料集（[philschmid/gretel-synthetic-text-to-sql](https://huggingface.co/datasets/philschmid/gretel-synthetic-text-to-sql)），這是高品質的合成 Text-to-SQL 資料集，包含自然語言指令、結構描述定義、推理過程與對應的 SQL 查詢。

[Hugging Face TRL](https://huggingface.co/docs/trl/en/index) 支援對話資料集格式的自動套版。這表示您只需要把資料集轉換成正確的 JSON 物件，`trl` 就會負責套版並轉為正確格式。

```text
{"messages": [{"role": "system", "content": "You are..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"messages": [{"role": "system", "content": "You are..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"messages": [{"role": "system", "content": "You are..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

[philschmid/gretel-synthetic-text-to-sql](https://huggingface.co/datasets/philschmid/gretel-synthetic-text-to-sql) 包含超過 10 萬個樣本。為了讓指南精簡，已下採樣至僅使用 10,000 個樣本。

您現在可以使用 Hugging Face Datasets 函式庫載入資料集，並建立提示範本，將自然語言指令、結構描述定義結合起來，同時為助手加上系統訊息。

```python { .notebook-cell }
from datasets import load_dataset

# System message for the assistant
system_message = """You are a text to SQL query translator. Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA."""

# User prompt that combines the user query and the schema
user_prompt = """Given the <USER_QUERY> and the <SCHEMA>, generate the corresponding SQL command to retrieve the desired data, considering the query's syntax, semantics, and schema constraints.

<SCHEMA>
{context}
</SCHEMA>

<USER_QUERY>
{question}
</USER_QUERY>
"""
def create_conversation(sample, idx):
  return {
    "messages": [
      {"role": "system", "content": system_message},
      {"role": "user", "content": user_prompt.format(question=sample["sql_prompt"], context=sample["sql_context"])},
      {"role": "assistant", "content": sample["sql"]}
    ]
  }

# Load dataset from the hub
dataset = load_dataset("philschmid/gretel-synthetic-text-to-sql", split="train")
dataset = dataset.select(range(1250))

# Convert dataset to OAI messages
dataset = dataset.map(create_conversation, with_indices=True, remove_columns=dataset.features)
# split dataset into 80% training samples and 20% test samples
dataset = dataset.train_test_split(test_size=0.2, shuffle=False)

# Print formatted user prompt
for item in dataset["train"][0]["messages"]:
  print(item)
```

```{ .no-copy .output }
{'role': 'system', 'content': 'You are a text to SQL query translator. Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA.'}
{'role': 'user', 'content': "Given the <USER_QUERY> and the <SCHEMA>, generate the corresponding SQL command to retrieve the desired data, considering the query's syntax, semantics, and schema constraints.\n\n<SCHEMA>\nCREATE TABLE salesperson (salesperson_id INT, name TEXT, region TEXT); INSERT INTO salesperson (salesperson_id, name, region) VALUES (1, 'John Doe', 'North'), (2, 'Jane Smith', 'South'); CREATE TABLE timber_sales (sales_id INT, salesperson_id INT, volume REAL, sale_date DATE); INSERT INTO timber_sales (sales_id, salesperson_id, volume, sale_date) VALUES (1, 1, 120, '2021-01-01'), (2, 1, 150, '2021-02-01'), (3, 2, 180, '2021-01-01');\n</SCHEMA>\n\n<USER_QUERY>\nWhat is the total volume of timber sold by each salesperson, sorted by salesperson?\n</USER_QUERY>\n"}
{'role': 'assistant', 'content': 'SELECT salesperson_id, name, SUM(volume) as total_volume FROM timber_sales JOIN salesperson ON timber_sales.salesperson_id = salesperson.salesperson_id GROUP BY salesperson_id, name ORDER BY total_volume DESC;'}
```

## 使用 TRL 與 SFTTrainer 微調 Gemma

您現在已可開始微調模型。Hugging Face TRL 的 [SFTTrainer](https://huggingface.co/docs/trl/sft_trainer) 讓監督式微調開放 LLM 變得簡單。`SFTTrainer` 是 `transformers` 函式庫中 `Trainer` 的子類別，支援所有相同功能，包括日誌記錄、評估與檢查點，但還額外提供多項便利功能，包括：

- 資料集格式設定，包括對話與指令格式
- 只在補全上訓練，忽略提示
- 打包資料集以提升訓練效率
- 參數效率型微調（PEFT）支援，包括 QLoRA
- 為對話微調準備模型與分詞器（例如新增特殊詞元）

以下程式碼從 Hugging Face 載入 Gemma 模型與分詞器，並初始化量化設定。

```python { .notebook-cell .model-swap }
import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

# Hugging Face model id
model_id = "google/gemma-4-E2B"

# Check if GPU supports bfloat16
if torch.cuda.is_bf16_supported():
    torch_dtype = torch.bfloat16
else:
    torch_dtype = torch.float16

# Define model init arguments
model_kwargs = dict(
    dtype=torch_dtype, # What torch dtype to use
    device_map="auto", # Let torch decide how to load the model
)

# BitsAndBytesConfig: Enables 4-bit quantization to reduce model size/memory usage
model_kwargs["quantization_config"] = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_quant_storage=torch_dtype,
)

# Load model and processor
model = AutoModelForMultimodalLM.from_pretrained(model_id, **model_kwargs)
processor = AutoProcessor.from_pretrained("google/gemma-4-E2B-it") # Load the Instruction Processor to use the official Gemma template

# NOTE: You should call the prepare_model_for_kbit_training() function to preprocess the quantized model for training.
# On T4, we are skipping this step purely due to VRAM limitation and for a quick demonstration.
if (torch.cuda.get_device_properties(0).total_memory/1024**3) > 16:
    model = prepare_model_for_kbit_training(model)
```

`SFTTrainer` 內建與 `peft` 的整合，讓使用 QLoRA 高效微調 LLM 變得簡單。您只需要建立一個 `LoraConfig` 並傳給訓練器。

```python { .notebook-cell }
from peft import LoraConfig

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    # no target_modules --- PEFT's Gemma 4 defaults scope to the LM layers
    task_type="CAUSAL_LM",
    modules_to_save=["lm_head", "embed_tokens"], # make sure to save the lm_head and embed_tokens as you train the special tokens
    ensure_weight_tying=True,
)
```

在開始訓練之前，您需要在 `SFTConfig` 實例中定義要使用的超參數。

```python { .notebook-cell }
import torch
from trl import SFTConfig

args = SFTConfig(
    output_dir="gemma-text-to-sql",         # directory to save and repository id
    max_length=512,                         # max length for model and packing of the dataset
    num_train_epochs=3,                     # number of training epochs
    per_device_train_batch_size=1,          # batch size per device during training
    per_device_eval_batch_size=1,           # batch size per device during evaluation
    optim="adamw_torch_fused",              # use fused adamw optimizer
    logging_steps=10,                       # log every 10 steps
    save_strategy="epoch",                  # save checkpoint every epoch
    eval_strategy="epoch",                  # evaluate checkpoint every epoch
    learning_rate=2e-4,                     # learning rate
    fp16=True if torch_dtype == torch.float16 else False,  # use float16 precision
    bf16=True if torch_dtype == torch.bfloat16 else False, # use bfloat16 precision
    lr_scheduler_type="constant",           # use constant learning rate scheduler
    push_to_hub=True,                       # push model to hub
    report_to="tensorboard",                # report metrics to tensorboard
    dataset_kwargs={"skip_prepare_dataset": True}, # important for collator
    remove_unused_columns = False,                 # important for collator
)

# Data collator
def collate_fn(examples):
    texts = []

    for example in examples:
        full_text = processor.apply_chat_template(
            example["messages"], add_generation_prompt=False, tokenize=False
        )
        texts.append(full_text.strip())

    # Tokenize the texts and process the audios
    batch = processor(text=texts, return_tensors="pt", padding=True)

    # The labels are the input_ids, and we mask the padding tokens and audio tokens in the loss computation
    labels = batch["input_ids"].clone()

    target_tokens = [
        processor.tokenizer.convert_tokens_to_ids("<|turn>"),
        processor.tokenizer.convert_tokens_to_ids("model"),
        processor.tokenizer.convert_tokens_to_ids("\n")
    ]
    target_len = len(target_tokens)

    for i in range(labels.size(0)):
        row_tokens = batch["input_ids"][i].tolist()

        # Find where the assistant block begins
        assistant_start_idx = None
        for idx in range(len(row_tokens) - target_len + 1):
            if row_tokens[idx : idx + target_len] == target_tokens:
                # We want to keep loss calculation on the assistant transcription tokens,
                # so we move the index right past the assistant header ('<|turn>\nmodel\n')
                assistant_start_idx = idx + target_len
                break

        if assistant_start_idx is not None:
            # Mask everything from index 0 up to the start of the actual Japanese text response
            labels[i, :assistant_start_idx] = -100
        else:
            # Fallback safety: if template matching fails for an anomalous row, mask padding anyway
            print("WARNING: maybe the sample is too long, try to increase `token_limit` value.")
            labels[i, labels[i] == processor.tokenizer.pad_token_id] = -100

        """
        # --- DEBUG PRINT CODE ---
        print(f"\n--- Example {i} (Split index: {assistant_start_idx}) ---")
        debug_string = []
        for token_id, label_id in zip(row_tokens, labels[i].tolist()):
            # Decode token by token so we can see exactly what is masked
            decoded_token = processor.tokenizer.decode([token_id])

            if label_id == -100:
                # Red text for masked tokens (ANSI Escape Code)
                debug_string.append(f"\033[91m{decoded_token}\033[0m")
            else:
                # Green text for active loss tokens
                debug_string.append(f"\033[92m{decoded_token}\033[0m")

        print("".join(debug_string))
        # ---
        """


    # Mask tokens for not being used in the loss computation
    labels[labels == processor.tokenizer.pad_token_id] = -100

    batch["labels"] = labels
    return batch
```

您現在已具備建立 `SFTTrainer` 所需的所有組件，可以開始訓練模型了。

```python { .notebook-cell }
from trl import SFTTrainer

# Create Trainer object
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
    processing_class=processor,
    data_collator=collate_fn,
)
```

呼叫 `train()` 方法開始訓練。

```python { .notebook-cell }
# Start training, the model will be automatically saved to the Hub and the output directory
trainer.train()

# Save the final model again to the Hugging Face Hub
trainer.save_model()
```

```{ .no-copy .output }
[3000/3000 1:51:30, Epoch 3/3]

| Epoch | Training Loss | Validation Loss | Entropy | Mean Token Accuracy | Num Tokens |
|---|---|---|---|---|---|
| 1 | 0.315478 | 0.267880 | 0.204330 | 0.927141 | 246435.000000 |
| 2 | 0.139383 | 0.297303 | 0.131811 | 0.929568 | 492870.000000 |
| 3 | 0.150697 | 0.315715 | 0.110690 | 0.927032 | 739305.000000 |
```

在測試模型之前，請確保先釋放記憶體。

```python { .notebook-cell }
# free the memory again
del model
del trainer
torch.cuda.empty_cache()
```

使用 QLoRA 時，您只訓練轉接器，而非完整模型。這表示訓練期間儲存模型時，只會儲存轉接器權重，而非完整模型。如果您想儲存完整模型（這會讓搭配 vLLM 或 TGI 等服務堆疊使用更方便），可以使用 `merge_and_unload` 方法將轉接器權重合併到模型權重中，再用 `save_pretrained` 方法儲存模型。這樣會儲存一個可直接用於推論的預設模型。

!!! Note
    注意：將轉接器合併到模型中需要超過 30GB 的 CPU 記憶體。您可以跳過此步驟，直接繼續「測試模型推論」。


```python { .notebook-cell .model-swap }
from transformers import AutoModelForMultimodalLM, AutoProcessor
from peft import PeftModel

# Load Model base model
model = AutoModelForMultimodalLM.from_pretrained(model_id, low_cpu_mem_usage=True)

# Merge LoRA and base model and save
peft_model = PeftModel.from_pretrained(model, args.output_dir)
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained("merged_model", safe_serialization=True, max_shard_size="2GB")

processor = AutoProcessor.from_pretrained("google/gemma-4-E2B-it")
processor.save_pretrained("merged_model")
```

```{ .no-copy .output }
Loading weights:   0%|          | 0/1951 [00:00<?, ?it/s]
Writing model shards:   0%|          | 0/4 [00:00<?, ?it/s]
['merged_model/processor_config.json']
```

## 測試模型推論並產生 SQL 查詢

訓練完成後，您會想評估並測試模型。可以從測試資料集載入不同的樣本，並在這些樣本上評估模型。

!!! Note
    注意：評估生成式 AI 模型並非簡單任務，因為一個輸入可能有多個正確輸出。本指南僅聚焦於人工評估與主觀檢驗。


```python { .notebook-cell }
from transformers import AutoModelForMultimodalLM, AutoProcessor

model_id = "merged_model"

# Load Model with PEFT adapter
model = AutoModelForMultimodalLM.from_pretrained(
  model_id,
  device_map="auto",
  dtype="auto",
)
processor = AutoProcessor.from_pretrained(model_id)
```

```{ .no-copy .output }
Loading weights:   0%|          | 0/1951 [00:00<?, ?it/s]
```

讓我們從測試資料集載入一個隨機樣本，並產生一條 SQL 指令。

```python { .notebook-cell }
from random import randint
import re
from transformers import pipeline, GenerationConfig, pipeline

config = GenerationConfig.from_pretrained(model_id)
config.max_new_tokens = 256
config.eos_token_id = [processor.tokenizer.convert_tokens_to_ids("<turn|>")]

# Load the model and tokenizer into the pipeline
pipe = pipeline("text-generation", model=model, tokenizer=processor.tokenizer)

# Load a random sample from the test dataset
rand_idx = randint(0, len(dataset["test"]))
test_sample = dataset["test"][rand_idx]

# Convert as test example into a prompt with the Gemma template
prompt = processor.tokenizer.apply_chat_template(test_sample["messages"][:2], tokenize=False, add_generation_prompt=True)
print(prompt)

# Generate our SQL query.
outputs = pipe(text_inputs=prompt, generation_config=config)

# Extract the user query and original answer
print(f"Context:\n", re.search(r'<SCHEMA>\n(.*?)\n</SCHEMA>', test_sample['messages'][1]['content'], re.DOTALL).group(1).strip())
print(f"Query:\n", re.search(r'<USER_QUERY>\n(.*?)\n</USER_QUERY>', test_sample['messages'][1]['content'], re.DOTALL).group(1).strip())
print(f"Original Answer:\n{test_sample['messages'][2]['content']}")
print(f"Generated Answer:\n{outputs[0]['generated_text'][len(prompt):].strip().removesuffix("<turn|>")}")
```

```{ .no-copy .output }
<bos><|turn>system
You are a text to SQL query translator. Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA.<turn|>
<|turn>user
Given the <USER_QUERY> and the <SCHEMA>, generate the corresponding SQL command to retrieve the desired data, considering the query's syntax, semantics, and schema constraints.

<SCHEMA>
CREATE TABLE Auto_Shows (id INT, show_name VARCHAR(255), show_year INT, location VARCHAR(255)); INSERT INTO Auto_Shows (id, show_name, show_year INT, location VARCHAR(255)); INSERT INTO Auto_Shows (id, show_name, show_year, location) VALUES (1, 'New York International Auto Show', 2019, 'United States'); INSERT INTO Auto_Shows (id, show_name, show_year, location) VALUES (2, 'Chicago Auto Show', 2019, 'United States'); INSERT INTO Auto_Shows (id, show_name, show_year, location) VALUES (3, 'North American International Auto Show', 2018, 'United States');
</SCHEMA>

<USER_QUERY>
How many auto shows were held in the United States in the year 2019?
</USER_QUERY><turn|>
<|turn>model

Context:
 CREATE TABLE Auto_Shows (id INT, show_name VARCHAR(255), show_year INT, location VARCHAR(255)); INSERT INTO Auto_Shows (id, show_name, show_year, location) VALUES (1, 'New York International Auto Show', 2019, 'United States'); INSERT INTO Auto_Shows (id, show_name, show_year, location) VALUES (2, 'Chicago Auto Show', 2019, 'United States'); INSERT INTO Auto_Shows (id, show_name, show_year, location) VALUES (3, 'North American International Auto Show', 2018, 'United States');
Query:
 How many auto shows were held in the United States in the year 2019?
Original Answer:
SELECT COUNT(*) FROM Auto_Shows WHERE show_year = 2019 AND location = 'United States';
Generated Answer:
SELECT COUNT(*) FROM Auto_Shows WHERE location = 'United States' AND show_year = 2019;
```

## 摘要與後續步驟

本教學介紹了如何使用 TRL 與 QLoRA 微調 Gemma 模型。接下來請參閱以下文件：

- 瞭解如何[用 Gemma 模型產生文字](../get-started/index.md)。
- 瞭解如何[使用 Hugging Face Transformers 針對視覺任務微調 Gemma](vision-qlora.md)。
- 瞭解如何[在 Vertex AI 上使用 Gemma 開放模型][vertex-use-gemma]。
- 瞭解如何[使用 KerasNLP 微調 Gemma 並部署到 Vertex AI](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/model_garden/model_garden_gemma_kerasnlp_to_vertexai.ipynb)。
