---
title: 使用 Hugging Face Transformers 與 QLoRA 針對視覺任務微調 Gemma
description: 本指南將引導您使用 Hugging Face Transformers 與 TRL 在自訂圖片與文字資料集上微調 Gemma，以執行視覺任務（產生商品描述）。
models: [gemma-4]
frameworks: [transformers, peft, trl]
source_url: "https://ai.google.dev/gemma/docs/core/huggingface_vision_finetune_qlora.md.txt"
colab: https://colab.research.google.com/github/google-gemma/docs/blob/main/docs/notebooks/vision-qlora-gemma-4-e2b.ipynb
---
# 使用 Hugging Face Transformers 與 QLoRA 針對視覺任務微調 Gemma


本指南將引導您使用 Hugging Face [Transformers](https://huggingface.co/docs/transformers/index) 與 [TRL](https://huggingface.co/docs/trl/index) 在自訂圖片與文字資料集上微調 Gemma，以執行視覺任務（產生商品描述）。您將學到：

- 什麼是量化低秩適配（QLoRA）
- 設定開發環境
- 建立並準備微調資料集
- 使用 TRL 與 SFTTrainer 微調 Gemma
- 測試模型推論，從圖片與文字產生商品描述

!!! Note
    本指南需要支援 bfloat16 資料型別的 GPU（如 NVIDIA L4 或 NVIDIA A100），且記憶體需超過 16GB。


## 什麼是量化低秩適配（QLoRA）

本指南展示了[量化低秩適配（Quantized Low-Rank Adaptation, QLoRA）](https://arxiv.org/abs/2305.14314)的用法。它成為高效微調 LLM 的熱門方法，因為它降低了運算資源需求，同時維持高效能。在 QLoRA 中，預訓練模型會被量化為 4-bit 並凍結權重，接著附加可訓練的轉接器層（LoRA），只訓練這些轉接器層。之後可以將轉接器權重合併到基底模型，或保留為獨立的轉接器。

## 設定開發環境

第一步是安裝 Hugging Face 函式庫（包括 TRL）與 datasets，以微調開放模型。

```python { .notebook-cell }
# Install Pytorch & other libraries
%pip install torch tensorboard torchvision
%pip install -U torchao

# Install Transformers
%pip install "transformers>=5.10.1"

# Install Hugging Face libraries
%pip install datasets accelerate evaluate bitsandbytes trl "peft>=0.19.0" protobuf pillow sentencepiece

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

- 將 Gemma 模型微調為能為電商平台產生簡潔、SEO 最佳化的商品描述，特別針對手機搜尋進行調校。

本指南使用 [philschmid/amazon-product-descriptions-vlm](https://huggingface.co/datasets/philschmid/amazon-product-descriptions-vlm) 資料集，這是 Amazon 商品描述資料集，包含商品圖片與分類。

Hugging Face TRL 支援多模態對話。關鍵是「image」角色，它告訴處理類別應該載入圖片。結構應如下所示：

```text
{"messages": [{"role": "system", "content": [{"type": "text", "text":"You are..."}]}, {"role": "user", "content": [{"type": "text", "text": "..."}, {"type": "image"}]}, {"role": "assistant", "content": [{"type": "text", "text": "..."}]}]}
{"messages": [{"role": "system", "content": [{"type": "text", "text":"You are..."}]}, {"role": "user", "content": [{"type": "text", "text": "..."}, {"type": "image"}]}, {"role": "assistant", "content": [{"type": "text", "text": "..."}]}]}
{"messages": [{"role": "system", "content": [{"type": "text", "text":"You are..."}]}, {"role": "user", "content": [{"type": "text", "text": "..."}, {"type": "image"}]}, {"role": "assistant", "content": [{"type": "text", "text": "..."}]}]}
```

您現在可以使用 Hugging Face Datasets 函式庫載入資料集，並建立提示範本，將圖片、商品名稱與分類結合起來，同時加上系統訊息。資料集中的圖片是 `Pil.Image` 物件。

```python { .notebook-cell }
from datasets import load_dataset
from PIL import Image

# System message for the assistant
system_message = "You are an expert product description writer for Amazon."

# User prompt that combines the user query and the schema
user_prompt = """Create a Short Product description based on the provided <PRODUCT> and <CATEGORY> and image.
Only return description. The description should be SEO optimized and for a better mobile search experience.

<PRODUCT>
{product}
</PRODUCT>

<CATEGORY>
{category}
</CATEGORY>
"""

# Convert dataset to OAI messages
def format_data(sample):
    return {
        "messages": [
            {
                "role": "system",
                #"content": [{"type": "text", "text": system_message}],
                "content": system_message,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt.format(
                            product=sample["Product Name"],
                            category=sample["Category"],
                        ),
                    },
                    {
                        "type": "image",
                        "image": sample["image"],
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["description"]}],
            },
        ],
    }

def process_vision_info(messages: list[dict]) -> list[Image.Image]:
    image_inputs = []
    # Iterate through each conversation
    for msg in messages:
        # Get content (ensure it's a list)
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]

        # Check each content element for images
        for element in content:
            if isinstance(element, dict) and (
                "image" in element or element.get("type") == "image"
            ):
                # Get the image and convert to RGB
                if "image" in element:
                    image = element["image"]
                else:
                    image = element
                image_inputs.append(image.convert("RGB"))
    return image_inputs

# Load dataset from the hub
dataset = load_dataset("philschmid/amazon-product-descriptions-vlm", split="train")
dataset = dataset.train_test_split(test_size=0.1)

# Convert dataset to OAI messages
# need to use list comprehension to keep Pil.Image type, .mape convert image to bytes
dataset_train = [format_data(sample) for sample in dataset["train"]]
dataset_test = [format_data(sample) for sample in dataset["test"]]

print(dataset_train[345]["messages"])
```

```{ .no-copy .output }
[{'role': 'system', 'content': 'You are an expert product description writer for Amazon.'}, {'role': 'user', 'content': [{'type': 'text', 'text': 'Create a Short Product description based on the provided <PRODUCT> and <CATEGORY> and image.\nOnly return description. The description should be SEO optimized and for a better mobile search experience.\n\n<PRODUCT>\nBachmann Trains inches 6 inches Pine Trees with Snow 6 Per Box\n</PRODUCT>\n\n<CATEGORY>\nToys & Games | Hobbies | Trains & Accessories | Accessories | Scenery\n</CATEGORY>\n'}, {'type': 'image', 'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=173x500 at 0x7ED5E4422C00>}]}, {'role': 'assistant', 'content': [{'type': 'text', 'text': 'Enhance your model train landscape with these realistic Bachmann Trains 6-inch pine trees, flocked with snow.  Six highly detailed trees per box, perfect for adding winter charm to your HO scale layout.  Create a captivating snowy scene!'}]}]
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
if not torch.cuda.is_bf16_supported():
    raise ValueError("GPU does not support bfloat16, please use a GPU that supports bfloat16.")

torch_dtype = torch.bfloat16

# Define model init arguments
model_kwargs = dict(
    dtype=torch_dtype, # What torch dtype to use
    device_map="auto", # Let torch decide how to load the model
)

# BitsAndBytesConfig int-4 config
model_kwargs["quantization_config"] = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_quant_storage=torch_dtype,
)

# Load model and processor
model = AutoModelForMultimodalLM.from_pretrained(model_id, **model_kwargs)
processor = AutoProcessor.from_pretrained("google/gemma-4-E2B-it") # Load the Instruction Tokenizer to use the official Gemma template

# Prepare model for QLoRA training
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

在開始訓練之前，您需要在 `SFTConfig` 實例中定義要使用的超參數，以及一個自訂的 `collate_fn` 來處理視覺資料。`collate_fn` 會將包含文字與圖片的訊息轉換為模型能理解的格式。

```python { .notebook-cell }
from trl import SFTConfig

args = SFTConfig(
    output_dir="gemma-product-description",     # directory to save and repository id
    num_train_epochs=3,                         # number of training epochs
    per_device_train_batch_size=1,              # batch size per device during training
    per_device_eval_batch_size=1,               # batch size per device during evaluation
    optim="adamw_torch_fused",                  # use fused adamw optimizer
    logging_steps=5,                            # log every 5 steps
    save_strategy="epoch",                      # save checkpoint every epoch
    eval_strategy="epoch",                      # evaluate checkpoint every epoch
    learning_rate=2e-4,                         # learning rate, based on QLoRA paper
    bf16=True,                                  # use bfloat16 precision
    max_grad_norm=0.3,                          # max gradient norm based on QLoRA paper
    lr_scheduler_type="constant",               # use constant learning rate scheduler
    push_to_hub=True,                           # push model to hub
    report_to="tensorboard",                    # report metrics to tensorboard
    dataset_text_field="",                      # need a dummy field for collator
    dataset_kwargs={"skip_prepare_dataset": True}, # important for collator
    remove_unused_columns = False               # important for collator
)

# Create a data collator to encode text and image pairs
def collate_fn(examples):
    texts = []
    images = []
    for example in examples:
        image_inputs = process_vision_info(example["messages"])
        text = processor.apply_chat_template(
            example["messages"], add_generation_prompt=False, tokenize=False
        )
        texts.append(text.strip())
        images.append(image_inputs)

    # Tokenize the texts and process the images
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    # The labels are the input_ids, and we mask the padding tokens and image tokens in the loss computation
    labels = batch["input_ids"].clone()

    # Mask tokens for not being used in the loss computation
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == processor.tokenizer.boi_token_id] = -100
    labels[labels == processor.tokenizer.image_token_id] = -100
    labels[labels == processor.tokenizer.eoi_token_id] = -100

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
    train_dataset=dataset_train,
    eval_dataset=dataset_test,
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
[3630/3630 57:38, Epoch 3/3]

| Epoch | Training Loss | Validation Loss | Entropy | Mean Token Accuracy | Num Tokens |
|---|---|---|---|---|---|
| 1 | 0.835934 | 0.844098 | 0.735220 | 0.818571 | 521741.000000 |
| 2 | 0.411868 | 0.886929 | 0.548703 | 0.822062 | 1043482.000000 |
| 3 | 0.488929 | 1.007192 | 0.417825 | 0.822075 | 1565223.000000 |
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


```python { .notebook-cell }
from transformers import AutoModelForMultimodalLM, AutoProcessor
from peft import PeftModel

# Load Model base model
model = AutoModelForMultimodalLM.from_pretrained(model_id, low_cpu_mem_usage=True)

# Merge LoRA and base model and save
peft_model = PeftModel.from_pretrained(model, args.output_dir)
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained("merged_model", safe_serialization=True, max_shard_size="2GB")

processor = AutoProcessor.from_pretrained(args.output_dir)
processor.save_pretrained("merged_model")
```

```{ .no-copy .output }
Loading weights:   0%|          | 0/1951 [00:00<?, ?it/s]
Writing model shards:   0%|          | 0/4 [00:00<?, ?it/s]
['merged_model/processor_config.json']
```

## 測試模型推論並產生商品描述

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

您可以提供商品名稱、分類與圖片來測試推論。`sample` 包含一個漫威動作人偶。

```python { .notebook-cell }
import requests
from PIL import Image

# Test sample with Product Name, Category and Image
sample = {
  "product_name": "Hasbro Marvel Avengers-Serie Marvel Assemble Titan-Held, Iron Man, 30,5 cm Actionfigur",
  "category": "Toys & Games | Toy Figures & Playsets | Action Figures",
  "image": Image.open(requests.get("https://m.media-amazon.com/images/I/81+7Up7IWyL._AC_SY300_SX300_.jpg", stream=True).raw).convert("RGB")
}

def generate_description(sample, model, processor):
    # Convert sample into messages and then apply the chat template
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": [
            {"type": "image","image": sample["image"]},
            {"type": "text", "text": user_prompt.format(product=sample["product_name"], category=sample["category"])},
        ]},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print(text)
    # Process the image and text
    image_inputs = process_vision_info(messages)
    # Tokenize the text and process the images
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    # Move the inputs to the device
    inputs = inputs.to(model.device)

    # Generate the output
    stop_token_ids = [processor.tokenizer.eos_token_id, processor.tokenizer.convert_tokens_to_ids("<turn|>")]
    generated_ids = model.generate(**inputs, max_new_tokens=256, top_p=1.0, do_sample=True, temperature=0.8, eos_token_id=stop_token_ids, disable_compile=True)
    # Trim the generation and decode the output to text
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

# generate the description
description = generate_description(sample, model, processor)
print("MODEL OUTPUT>> \n")
print(description)
```

```{ .no-copy .output }
<bos><|turn>system
You are an expert product description writer for Amazon.<turn|>
<|turn>user
<|image|>Create a Short Product description based on the provided <PRODUCT> and <CATEGORY> and image.
Only return description. The description should be SEO optimized and for a better mobile search experience.

<PRODUCT>
Hasbro Marvel Avengers-Serie Marvel Assemble Titan-Held, Iron Man, 30,5 cm Actionfigur
</PRODUCT>

<CATEGORY>
Toys & Games | Toy Figures & Playsets | Action Figures
</CATEGORY><turn|>
<|turn>model

MODEL OUTPUT>> 

Bring home the action with the Hasbro Marvel Avengers Assemble Titan-held Iron Man Action Figure! This highly detailed 30.5-centimeter figure is perfect for recreating epic battles.  Collect the entire Marvel Avengers Assemble Titan-held series for an authentic experience.  Shop now!
```

## 摘要與後續步驟

本教學介紹了如何使用 TRL 與 QLoRA 針對視覺任務微調 Gemma 模型，特別是用於產生商品描述。接下來請參閱以下文件：

- 瞭解如何[用 Gemma 模型產生文字](../get-started/index.md)。
- 瞭解如何[使用 Hugging Face Transformers 針對文字任務微調 Gemma](qlora.md)。
- 瞭解如何[使用 Hugging Face Transformers 進行完整模型微調](full-finetune.md)。
- 瞭解如何[在 Vertex AI 上使用 Gemma 開放模型][vertex-use-gemma]。
- 瞭解如何[使用 KerasNLP 微調 Gemma 並部署到 Vertex AI](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/model_garden/model_garden_gemma_kerasnlp_to_vertexai.ipynb)。
