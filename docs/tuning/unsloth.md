---
title: Unsloth
seo_title: 使用 Unsloth 微調 Gemma 4
description: 在任何單 GPU Gemma 4 教學中，把 Unsloth 的 FastVisionModel 或 FastModel 換入，以加速 LoRA、QLoRA 並匯出 GGUF。
models: [gemma-4]
frameworks: [unsloth, trl]
---

# Unsloth

本節的每個實作指南都使用 Hugging Face `transformers`、`trl` 與 `peft`。我們這樣做，是為了讓同一套 Python 環境能同時適用於單 GPU 與多 GPU 執行。不過，如果你只使用單張 GPU，可以把 [Unsloth](https://unsloth.ai/docs/models/gemma-4){ data-model="gemma-4" } 放入任何教學中，只需替換模型載入與 LoRA 設定步驟即可。其他部分，例如資料集格式、collator 與 TRL 訓練器，都保持不變。

---

## 1. 安裝 Unsloth

首先，安裝 `unsloth` 以及 Hugging Face 訓練套件：

```python
# Install Unsloth alongside the standard post-training libraries
%pip install -q -U unsloth "transformers>=5.10.1" "trl>=0.16.0" "peft>=0.19.0" datasets accelerate
```

---

## 2. 替換模型載入與 LoRA 設定

在任何只使用 1 張 GPU 的教學（大多數都是）中，請把以下項目替換掉：

* `BitsAndBytesConfig`、`AutoModelForMultimodalLM.from_pretrained` 與 `LoraConfig`

替換為：

* `FastVisionModel.from_pretrained` 與 `FastVisionModel.get_peft_model`

```python { .model-swap }
import unsloth  # Import unsloth before transformers, trl, or peft
from unsloth import FastVisionModel

# Load the Gemma 4 checkpoint in 4-bit precision with Unsloth gradient checkpointing
model, processor = FastVisionModel.from_pretrained(
    "google/gemma-4-E2B-it",
    load_in_4bit=True,                  # Set load_in_4bit=False, load_in_16bit=True for 16-bit LoRA
    use_gradient_checkpointing="unsloth",
)

# Attach LoRA adapters to the attention and MLP projections
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=False,       # Set True when adapting the vision encoder
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=32,
    lora_dropout=0.0,
    bias="none",
)
```

針對特定教學，還需要以下其他小幅調整：

| 教學 | 載入器類別 | 關鍵旗標 |
| :--- | :--- | :--- |
| [(Q)LoRA](qlora.md)、[持續預訓練](continued-pretraining.md)、[偏好最佳化](preference-alignment.md)、[可驗證獎勵](reinforcement-learning.md) | `FastVisionModel` | `finetune_vision_layers=False, finetune_language_layers=True` |
| [MoE 微調](moe-finetune.md)（26B A4B） | `FastVisionModel` | `finetune_mlp_modules=True` 會調整共用專家與全部 128 個路由專家，同時保持路由器凍結 |
| [視覺微調](vision-qlora.md) | `FastVisionModel` | 設定 `processor.image_processor.max_soft_tokens = 560`；傳入 `UnslothVisionDataCollator(model, processor)` 或該教學的 `collate_fn` |
| [音訊微調](audio-qlora.md) | `FastModel` | 在匯入 `unsloth` 之前設定 `os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"`，讓長度不一的音訊批次不會觸發重新編譯 |

---

## 3. 訓練模型

Unsloth 非常方便，已經幫你設定好量化，你只需要執行以下程式碼來啟用訓練：

```python
from trl import SFTConfig, SFTTrainer

# Enable training mode before running the trainer
FastVisionModel.for_training(model)

# Define and run the trainer with the exact same dataset and SFTConfig
trainer = SFTTrainer(
    model=model,
    args=SFTConfig(output_dir="gemma-4-unsloth-lora", max_length=2048, learning_rate=2e-4, bf16=True),
    train_dataset=dataset["train"],
    processing_class=processor.tokenizer,
)
trainer.train()
```

---

## 4. 匯出為已合併的 16 位元或 GGUF

Unsloth 內建輔助方法，可以把 LoRA 轉接器摺疊為獨立的 16 位元 SafeTensors，或直接匯出為量化的 GGUF 檔案，供 [llama.cpp](../deployment/llamacpp.md) 與 [Ollama](../deployment/ollama.md) 使用：

```python
# Save a standalone 16-bit merged checkpoint for vLLM or LiteRT-LM conversion
model.save_pretrained_merged("merged_model", processor, save_method="merged_16bit")

# Or export directly to a 4-bit Q4_K_M GGUF file
model.save_pretrained_gguf("gemma-4-gguf", processor.tokenizer, quantization_method="q4_k_m")
```

* 多 GPU 分片與量化感知訓練，請見[擴展規模](scaling.md)。
* vLLM 與 LiteRT-LM 行動匯出，請見[匯出](export.md)。
* 在本機執行 Unsloth Studio 與動態量化 GGUF，請見[使用 Unsloth 執行 Gemma](../deployment/unsloth.md)。

---

## 接下來

<div class="card-grid">
<a class="card" href="qlora.md">
  <span class="card-icon">{{ icon('lucide/zap') }}</span>
  <div class="card-title">(Q)LoRA</div>
  <p class="card-desc">從資料集到已合併模型，走完一次完整的單 GPU QLoRA 微調。</p>
</a>
<a class="card" href="moe-finetune.md">
  <span class="card-icon">{{ icon('lucide/split') }}</span>
  <div class="card-title">MoE 微調</div>
  <p class="card-desc">調整 Gemma 4 26B A4B 的全部 128 個路由專家，同時保持路由器凍結。</p>
</a>
<a class="card" href="vision-qlora.md">
  <span class="card-icon">{{ icon('lucide/eye') }}</span>
  <div class="card-title">視覺微調</div>
  <p class="card-desc">使用可選擇的詞元預算，在結構化文件與影像資料集上微調 Gemma 4。</p>
</a>
<a class="card" href="export.md">
  <span class="card-icon">{{ icon('lucide/rocket') }}</span>
  <div class="card-title">匯出</div>
  <p class="card-desc">將微調後的檢查點部署到 vLLM、GGUF、Ollama 或 LiteRT-LM。</p>
</a>
</div>

--8<-- "glossary.md"
