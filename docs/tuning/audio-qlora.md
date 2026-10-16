---
title: 使用 Unsloth 與 QLoRA 針對音訊任務微調 Gemma
description: 本指南將引導您使用 Unsloth 在自訂音訊資料集上微調 Gemma，以執行音訊任務（產生熊本方言日語的轉錄）。
models: [gemma-4]
requires: audio
frameworks: [unsloth, peft]
colab: https://colab.research.google.com/github/google-gemma/docs/blob/main/docs/notebooks/audio-qlora-gemma-4-e2b.ipynb
---
# 使用 Unsloth 與 QLoRA 針對音訊任務微調 Gemma

本指南將引導您使用 Unsloth 在自訂音訊資料集上微調 Gemma，以執行音訊任務（產生[熊本方言日語](https://en.wikipedia.org/wiki/Kumamoto_dialect)的轉錄）。

## 概觀與目標
- **基底模型**：Gemma 4 PT 多模態基礎模型
- **資料集**：Hugging Face 上的 [`federerjiang/dialect.kumamoto`](https://huggingface.co/datasets/federerjiang/dialect.kumamoto)（1,300 段熊本方言日語的語音錄音）
- **方法**：透過 Unsloth 與 PEFT 進行 **4-bit QLoRA**，大幅降低記憶體需求，同時微調語言注意力層與 MLP 投影層
- **目標指標**：**文字相似度 > 80.0%**

## 1. 環境設定與相依套件安裝

您將安裝：

- `unsloth`：用於加速的 4-bit QLoRA 與記憶體最佳化
- 來自 Hugging Face 的 `transformers`、`trl`、`peft`、`accelerate`
- `soundfile`、`librosa`、`jiwer`：用於音訊處理與字元錯誤率（CER）評估



```python { .notebook-cell }
# Install Unsloth and required dependencies
!pip install unsloth[colab-new] unsloth_zoo
!pip install -U torch torchvision torchaudio transformers trl peft accelerate datasets bitsandbytes soundfile librosa jiwer
```

## 2.（選用）Hugging Face 驗證

您需要有效的 Hugging Face Token 才能發布模型。如果您在 Google Colab 中執行，可以使用 Colab 密鑰安全地使用 Hugging Face Token；否則可以直接在 `login` 方法中設定 Token。請確保您的 Token 具有寫入權限，因為訓練期間會將模型推送到 Hub。


```python { .notebook-cell }
# Login into Hugging Face Hub
from huggingface_hub import login
login()
```

## 3. 硬體與 GPU 精度驗證



```python { .notebook-cell }
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    supports_bf16 = torch.cuda.is_bf16_supported()
    print(f"GPU: {gpu_name} ({vram_gb:.2f} GB VRAM)")
    print(f"Bfloat16 Supported: {supports_bf16}")
    dtype = torch.bfloat16 if supports_bf16 else torch.float16
else:
    raise RuntimeError("GPU acceleration is required. In Colab: Runtime -> Change runtime type -> T4 GPU / A100.")

```

``` { .no-copy .output }
Device: cuda
GPU: NVIDIA L4 (22.03 GB VRAM)
Bfloat16 Supported: True
```

## 4. 資料集載入與準備 (`federerjiang/dialect.kumamoto`)

您將直接從 Hugging Face 下載 `federerjiang/dialect.kumamoto` 資料集。

資料集包含：

- `audio`：音訊波形與取樣率（16 kHz）
- `sentence`：熊本方言（*Kumamoto-ben*）的標準轉錄文字

您將把 1,300 個樣本分為：

- **訓練集**：1,170 個樣本（90%）
- **測試集**：130 個樣本（10%），使用固定隨機種子（`seed=42`）



```python { .notebook-cell }
import os
import soundfile as sf
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

print("Downloading dataset from Hugging Face...")
raw_dataset = load_dataset("federerjiang/dialect.kumamoto", split="train")
print(f"Loaded {len(raw_dataset)} total samples.")

# Create local audio storage directory
audio_dir = "./audios"
os.makedirs(audio_dir, exist_ok=True)

# Split into 90% train (1170) and 10% test (130) with fixed seed
split_ds = raw_dataset.train_test_split(test_size=0.1, seed=42)
train_raw = split_ds["train"]
test_raw = split_ds["test"]

print(f"Train samples: {len(train_raw)}, Test samples: {len(test_raw)}")

def process_and_cache(raw_split, split_name):
    items = []
    instruction_text = "音声を聞いて、文字起こしをしてください。"
    for idx, sample in enumerate(tqdm(raw_split, desc=f"Processing {split_name}")):
        sample_id = f"kumamoto_{split_name}_{idx:04d}"
        wav_path = os.path.join(audio_dir, f"{sample_id}.wav")

        # Extract audio array and sampling rate
        audio_info = sample["audio"]
        audio_array = np.array(audio_info["array"], dtype=np.float32)
        sr = audio_info["sampling_rate"]

        # Resample to 16kHz if not already
        if sr != 16000:
            import librosa
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
            sr = 16000

        # Save audio file
        sf.write(wav_path, audio_array, sr)

        text = sample["sentence"].strip()

        # Format into standard multi-turn conversation
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "url": wav_path},
                    {"type": "text", "text": instruction_text}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": text}
                ]
            }
        ]

        items.append({
            "id": sample_id,
            "audio_path": wav_path,
            "sentence": text,
            "messages": messages
        })
    return items

train_items = process_and_cache(train_raw, "train")
test_items = process_and_cache(test_raw, "test")

import json
os.makedirs("./data", exist_ok=True)
with open("./data/train.json", "w", encoding="utf-8") as f:
    json.dump(train_items, f, ensure_ascii=False, indent=2)
with open("./data/test.json", "w", encoding="utf-8") as f:
    json.dump(test_items, f, ensure_ascii=False, indent=2)

print("Dataset prepared and saved to ./data/train.json and ./data/test.json!")

```

```{ .no-copy .output }
Downloading dataset from Hugging Face...
Loaded 1300 total samples.
Train samples: 1170, Test samples: 130
Processing train: 100%|██████████| 1170/1170 [00:23<00:00, 49.88it/s]
Processing test: 100%|██████████| 130/130 [00:02<00:00, 53.72it/s]
Dataset prepared and saved to ./data/train.json and ./data/test.json!
```

## 5. 使用 4-bit QLoRA 初始化模型

以下程式碼使用 Unsloth 以 4-bit 正規浮點數（NF4）量化載入 Gemma 模型與處理器。

主要調整：

1. **音訊遮罩散佈修補**：修補 PyTorch 的 `masked_scatter`，以在量化權重旁嵌入音訊表示時處理張量 dtype 不一致的問題。
2. **LoRA 轉接器設定**：
    - 秩（$r$）：**32**
    - Alpha（$\alpha$）：**64**（$2 \times r$）
    - 目標模組：語言模型的注意力層（`q_proj`、`k_proj`、`v_proj`、`o_proj`）與前饋 MLP 層（`gate_proj`、`up_proj`、`down_proj`）。
    - 可訓練參數：**約 48.3M**（僅佔基底模型權重的 **約 0.94%**）。



```python { .notebook-cell .model-swap }
# Patch torch masked_scatter for multimodal audio tensor compatibility in QLoRA
_original_masked_scatter = torch.Tensor.masked_scatter
def _patched_masked_scatter(self, mask, source):
    if self.dtype != source.dtype:
        source = source.to(self.dtype)
    return _original_masked_scatter(self, mask, source)
torch.Tensor.masked_scatter = _patched_masked_scatter

import unsloth
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template

model_name = "google/gemma-4-E2B"
max_seq_length = 2048

print(f"Loading {model_name} in 4-bit NF4 (QLoRA)...")
model, processor = FastModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=True,
    full_finetuning=False,
    use_gradient_checkpointing="unsloth",
)

processor = get_chat_template(processor, chat_template="gemma-4")

print("Attaching LoRA adapters...")
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=32,
    lora_alpha=64,
    lora_dropout=0,
    bias="none",
)

model.print_trainable_parameters()

```

``` { .no-copy .output }
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
🦥 Unsloth Zoo will now patch everything to make training faster!
Loading google/gemma-4-E2B in 4-bit NF4 (QLoRA)...
==((====))==  Unsloth 2026.9.4: Fast Gemma4 patching. Transformers: 5.17.0.
   \\   /|    NVIDIA L4. Num GPUs = 1. Max memory: 22.034 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.14.0+cu130. CUDA: 8.9. CUDA Toolkit: 13.0. Triton: 3.8.0
\        /    Bfloat16 = TRUE. FA [Xformers = None. FA2 = False]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
Loading weights:   0%|          | 0/1951 [00:00<?, ?it/s]
Attaching LoRA adapters...
trainable params: 48,316,416 || all params: 5,152,613,920 || trainable%: 0.9377
```

## 6. 音訊轉錄 Collator 與損失遮罩

為確保模型只學習轉錄音訊，而非背誦提示詞元：

- 以下程式碼會在每個序列中解析 <code><span class="token-turn"><|turn></span><span class="token-role">model</span>\n</code>。
- 在 <code><span class="token-turn"><|turn></span><span class="token-role">model</span>\n</code> 之前的所有詞元（包括音訊前綴標記與使用者指令）都會被遮罩為 `-100`。
- 只有助手回應的詞元會計入交叉熵損失。



```python { .notebook-cell }
def get_collate_fn(processor):
    def collate_fn(examples):
        texts = []
        audios = []
        for example in examples:
            messages = example["messages"]
            full_text = processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
            texts.append(full_text.strip())

            audio_path = example.get("audio_path")
            if not audio_path:
                for m in messages:
                    for b in m.get("content", []):
                        if isinstance(b, dict) and b.get("type") == "audio":
                            audio_path = b.get("audio") or b.get("url")

            audio_arr, sr = sf.read(audio_path, dtype="float32")
            if audio_arr.ndim > 1:
                audio_arr = audio_arr.mean(axis=1)
            audios.append(audio_arr)

        batch = processor(text=texts, audio=audios, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()

        target_phrase = "<|turn>model\n"
        target_tokens = processor.tokenizer.encode(target_phrase, add_special_tokens=False)
        target_len = len(target_tokens)

        for i in range(labels.size(0)):
            row_tokens = batch["input_ids"][i].tolist()
            assistant_start_idx = None
            for idx in range(len(row_tokens) - target_len + 1):
                if row_tokens[idx : idx + target_len] == target_tokens:
                    assistant_start_idx = idx + target_len
                    break

            if assistant_start_idx is not None:
                labels[i, :assistant_start_idx] = -100
            else:
                labels[i, labels[i] == processor.tokenizer.pad_token_id] = -100

        labels[labels == processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch

    return collate_fn

collate_fn = get_collate_fn(processor)
print("Collate function defined successfully!")

```

```{ .no-copy .output }
Collate function defined successfully!
```

## 7. 使用 TRL `SFTTrainer` 微調

### 訓練超參數
| 參數 | 值 |
| :--- | :--- |
| **學習率** | `2.0e-4`，搭配餘弦衰減 |
| **暖機比例** | 總步數的 3%（約 18 步） |
| **訓練輪數** | 4 輪（588 個優化器步數） |
| **批次大小** | 每裝置 4 $\times$ 2 梯度累積 = 有效批次 8 |
| **精度** | `bfloat16`（或退回 `float16`） |
| **檢查點策略** | 每輪儲存，並依 `eval_loss` 使用 `load_best_model_at_end=True` |



```python { .notebook-cell }
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset as hf_load_dataset

train_dataset = hf_load_dataset("json", data_files="./data/train.json", split="train")
test_dataset = hf_load_dataset("json", data_files="./data/test.json", split="train")

output_dir = "./checkpoints"
os.makedirs(output_dir, exist_ok=True)

batch_size = 4
grad_accum = 2
epochs = 4
learning_rate = 2e-4
total_steps = (len(train_dataset) // (batch_size * grad_accum)) * epochs
warmup_steps = max(1, int(total_steps * 0.03))

sft_config = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=grad_accum,
    num_train_epochs=epochs,
    learning_rate=learning_rate,
    lr_scheduler_type="cosine",
    warmup_steps=warmup_steps,
    bf16=supports_bf16,
    fp16=not supports_bf16,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    remove_unused_columns=False,
    dataset_kwargs={"skip_prepare_dataset": True},
    max_length=max_seq_length,
    logging_steps=10,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    processing_class=processor,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=collate_fn,
    args=sft_config,
)

print(f"Starting QLoRA training ({total_steps} steps across {epochs} epochs)...")
train_result = trainer.train()
print(f"Training completed! Stats: {train_result}")

# Save final best model and processor
final_model_path = os.path.join(output_dir, "final_model")
trainer.save_model(final_model_path)
processor.save_pretrained(final_model_path)
print(f"Fine-tuned QLoRA adapter saved to {final_model_path}!")

```

```{ .no-copy .output }
Generating train split: 0 examples [00:00, ? examples/s]
Generating train split: 0 examples [00:00, ? examples/s]
The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'bos_token_id': 2}.
Starting QLoRA training (584 steps across 4 epochs)...
==((====))==  Unsloth - 2x faster free finetuning | Num GPUs used = 1
   \\   /|    Num examples = 1,170 | Num Epochs = 4 | Total steps = 588
O^O/ \_/ \    Batch size per device = 4 | Gradient accumulation steps = 2
\        /    Data Parallel GPUs = 1 | Total batch size (4 x 2 x 1) = 8
 "-____-"     Trainable parameters = 48,316,416 of 5,152,613,920 (0.94% trained)
Unsloth: Not an error, but Gemma4ForConditionalGeneration does not accept `num_items_in_batch`.
Using gradient accumulation will be very slightly less accurate.
Read more on gradient accumulation issues here: https://unsloth.ai/blog/gradient

[588/588 18:18, Epoch 4/4]

| Epoch | Training Loss | Validation Loss |
|---|---|---|
| 1 | 0.635206 | 0.630127 |
| 2 | 0.279609 | 0.644313 |
| 3 | 0.069598 | 0.765895 |
| 4 | 0.014357 | 0.905066 |

Unsloth: Restored added_tokens_decoder metadata in ./checkpoints/checkpoint-147/tokenizer_config.json.
Unsloth: Restored added_tokens_decoder metadata in ./checkpoints/checkpoint-294/tokenizer_config.json.
Unsloth: Restored added_tokens_decoder metadata in ./checkpoints/checkpoint-441/tokenizer_config.json.
Unsloth: Restored added_tokens_decoder metadata in ./checkpoints/checkpoint-588/tokenizer_config.json.
Training completed! Stats: TrainOutput(global_step=588, training_loss=0.3260592963217067, metrics={'train_runtime': 1196.9343, 'train_samples_per_second': 3.91, 'train_steps_per_second': 0.491, 'total_flos': 1.816203342288845e+16, 'train_loss': 0.3260592963217067, 'epoch': 4.0})
Unsloth: Restored added_tokens_decoder metadata in ./checkpoints/final_model/tokenizer_config.json.
Fine-tuned QLoRA adapter saved to ./checkpoints/final_model!
```

## 8. 在未見過的測試集（130 個樣本）上評估

您將使用貪心解碼，在全部 **130 段未見過的測試音訊錄音** 上評估微調後的模型。

計算的指標：

1. **SequenceMatcher 相似度**（`difflib.SequenceMatcher`）：字元子序列對齊比例。
2. **Levenshtein 相似度**：正規化編輯距離比例（$1 - \frac{\text{dist}}{\max(L_1, L_2)}$）。
3. **字元正確率**：$1 - \text{CER}$，使用 `jiwer.cer`。
4. **完全比對率**：字元完全相符的百分比。

> **評估目標**：達到 > 80.0% 的文字相似度。



```python { .notebook-cell }
import difflib
import jiwer
from tqdm import tqdm

def compute_levenshtein(s1, s2):
    if len(s1) < len(s2):
        return compute_levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            ins = prev[j + 1] + 1
            dels = curr[j] + 1
            subs = prev[j] + (c1 != c2)
            curr.append(min(ins, dels, subs))
        prev = curr
    return prev[-1]

# Load fine-tuned adapter for inference
print(f"Loading fine-tuned model for inference from {final_model_path}...")
eval_model, eval_processor = FastModel.from_pretrained(
    model_name=final_model_path,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=True,
)
eval_processor = get_chat_template(eval_processor, chat_template="gemma-4")
FastModel.for_inference(eval_model)
eval_device = next(eval_model.parameters()).device

similarities = []
lev_sims = []
char_accs = []
cers = []
exact_matches = 0
results = []

print(f"Evaluating {len(test_items)} test samples...")
for item in tqdm(test_items, desc="Evaluating"):
    ref = item["sentence"].strip()
    audio_path = item["audio_path"]
    prompt_text = "音声を聞いて、文字起こしをしてください。"

    audio_arr, sr = sf.read(audio_path, dtype="float32")
    if audio_arr.ndim > 1:
        audio_arr = audio_arr.mean(axis=1)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "url": audio_path},
                {"type": "text", "text": prompt_text}
            ]
        }
    ]

    prompt = eval_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = eval_processor(text=[prompt], audio=[audio_arr], return_tensors="pt", padding=True)
    inputs = {k: v.to(eval_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = eval_model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=eval_processor.tokenizer.pad_token_id,
            eos_token_id=[1, 106],
        )

    input_len = inputs["input_ids"].shape[1]
    pred = eval_processor.tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
    pred = pred.split("<turn|>")[0].split("\n")[0].strip()

    # Metrics
    sim = difflib.SequenceMatcher(None, ref, pred).ratio() * 100.0
    max_l = max(len(ref), len(pred))
    dist = compute_levenshtein(ref, pred)
    lev_sim = ((max_l - dist) / max_l * 100.0) if max_l > 0 else 100.0
    try:
        cer = jiwer.cer(ref, pred)
    except Exception:
        cer = 1.0
    char_acc = max(0.0, 1.0 - cer) * 100.0
    is_exact = (ref == pred)

    if is_exact:
        exact_matches += 1

    similarities.append(sim)
    lev_sims.append(lev_sim)
    char_accs.append(char_acc)
    cers.append(cer * 100.0)

    results.append({
        "id": item["id"],
        "reference": ref,
        "prediction": pred,
        "similarity": round(sim, 2),
        "exact_match": is_exact
    })

mean_sim = sum(similarities) / len(similarities)
mean_lev = sum(lev_sims) / len(lev_sims)
mean_acc = sum(char_accs) / len(char_accs)
mean_cer = sum(cers) / len(cers)
exact_rate = (exact_matches / len(test_items)) * 100.0

print("\n" + "="*50)
print("QLoRA ASR EVALUATION SUMMARY")
print("="*50)
print(f"Total Test Samples:        {len(test_items)}")
print(f"Mean Text Similarity:      {mean_sim:.2f}%")
print(f"Mean Levenshtein Sim:      {mean_lev:.2f}%")
print(f"Mean Character Accuracy:   {mean_acc:.2f}%")
print(f"Mean CER:                  {mean_cer:.2f}%")
print(f"Exact Match Rate:          {exact_rate:.2f}% ({exact_matches}/{len(test_items)})")
print(f"Target (> 80%) Achieved:   {'YES 🎉' if mean_sim > 80.0 else 'NO'}")
print("="*50 + "\n")

```

```{ .no-copy .output }
Loading fine-tuned model for inference from ./checkpoints/final_model...
==((====))==  Unsloth 2026.9.4: Fast Gemma4 patching. Transformers: 5.17.0.
    \\   /|    NVIDIA L4. Num GPUs = 1. Max memory: 22.034 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.14.0+cu130. CUDA: 8.9. CUDA Toolkit: 13.0. Triton: 3.8.0
\        /    Bfloat16 = TRUE. FA [Xformers = None. FA2 = False]
    "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!

Loading weights:   0%|          | 0/1951 [00:00<?, ?it/s]

Evaluating 130 test samples...
Evaluating: 100%|██████████| 130/130 [08:33<00:00,  3.95s/it]    
==================================================
QLoRA ASR EVALUATION SUMMARY
==================================================
Total Test Samples:        130
Mean Text Similarity:      84.75%
Mean Levenshtein Sim:      81.83%
Mean Character Accuracy:   80.89%
Mean CER:                  19.11%
Exact Match Rate:          9.23% (12/130)
Target (> 80%) Achieved:   YES 🎉
==================================================
```

## 9. 互動式檢視與方言音訊示範

讓我們對照標準熊本方言語句檢視預測範例，並直接在 Colab 播放一段音訊範例。



```python { .notebook-cell }
from IPython.display import Audio, display
import pandas as pd

# Display top predictions in a clean table
df_results = pd.DataFrame(results[:15])
display(df_results[["id", "reference", "prediction", "similarity", "exact_match"]])

# Play an example audio
sample_to_play = test_items[0]
print(f"Playing sample {sample_to_play['id']}:")
print(f"Ground Truth: {sample_to_play['sentence']}")
matched_pred = next(r['prediction'] for r in results if r['id'] == sample_to_play['id'])
print(f"QLoRA Pred  : {matched_pred}")

display(Audio(sample_to_play["audio_path"]))

```

```{ .no-copy .output }
| | id | reference | prediction | similarity | exact_match |
|--|--|--|--|--|--|
| 0 | kumamoto_test_0000 | カピバラに触らるっ場所もあっばい。親子で歩きよる姿に癒さるっけん。 | カピバラに触らるっ場所もあっばい。親子で歩きよる姿に癒さるっけん。 | 100.00 | True |
| 1 | kumamoto_test_0001 | 舗装もぴしゃっとしとらすけんあんま揺れんし，風ば受けながり土手ん上ば走るとは気持ちんよか（爽... | ホソーもぴしゃっとしとらすけんあんまよれんし、風邪ば受けながらどっでん上ば走るとは気持ちんよ... | 77.23 | False |
| 2 | kumamoto_test_0002 | だけん，祇園界隈な観光客でわんさかしとって，人気の店ん前には長蛇ん列のでくる． | だけん、気取らない観光客でわん探しとって、人気の店ん前にはちょうだん列のでくる。 | 70.89 | False |
| 3 | kumamoto_test_0003 | 飛行機の着陸する寸前に窓から下ば見っと、格子状ん畑の広がっとって帰ってきたなっちゅう感じのすっ。 | 飛行機の着陸する瞬間に窓から下ばみっと、合志城ん畑の広がっとって、帰ってきたなっちゅう感じのすっ。 | 86.60 | False |
| 4 | kumamoto_test_0004 | 次，料理ん本番ばい． | 次，両親ん本番ばい． | 80.00 | False |
| 5 | kumamoto_test_0005 | 透き通っとる白湯スープのラーメンで，たいぎゃあっさりとした味わいばい． | 好きとっとるパイタンスープのラーメンで、たいぎゃあっさりとした味ばい。 | 77.14 | False |
| 6 | kumamoto_test_0006 | 東京まぢ戻ってきたばってん，１１時オープンてだけん，時間のちと早かごたる． | 東京マジ戻ってきたばってん、１１時オープンてだけん、時間の落ちと早かごたる。 | 85.33 | False |
| 7 | kumamoto_test_0007 | 近くにゃレンゲ畑のあって、きれいかねって思いよりながら自転車ん乗っとったら、いがわさんつこけた。 | 近くにゃレンギバタケのあって、きれいかねって思いよりながら自転車ん乗っとったら、飯川さんつこけだ。 | 86.60 | False |
| 8 | kumamoto_test_0008 | 大人んなってくっと一人で過ごしたか時間の出てくるもんたい． | 大人になってくっと一人で過ごしたか時間の出てくるもんだい． | 93.10 | False |
| 9 | kumamoto_test_0009 | 近所ん豚舎んもんの友達におって、たまあにうまか豚バラん串焼きば食わせてもらいよった。 | キンジョン豚しゃんもんの友達におって、たまーにうまか豚バラん串焼きば作ってもらいよった。 | 79.07 | False |
| 10 | kumamoto_test_0010 | ばってん夏休みな結局ずっと部活しかしとらん． | ばってん夏休みは結局ずっと部活しとらん． | 90.48 | False |
| 11 | kumamoto_test_0011 | そぎゃん親切ばしてもらうと、旅の何倍でん楽しゅうなったい。 | そぎゃん新設ばしてもらうと、旅の何倍でん楽ちゅうなったい。 | 89.66 | False |
| 12 | kumamoto_test_0012 | またいろいろあると思ちょるばってんが，今ん状態じゃ，そんぐらいしか見つけきらんかった． | まあたいろいろあると思ちょるばってんが，今ん状態じゃそんくらいしか見つけきらんかった． | 95.35 | False |
| 13 | kumamoto_test_0013 | 飲み放題もついとったばってん，だっもいっちょん飲まんで，食ぶる食ぶる肉ばひたすら食べよらす． | 飲み放題もついとったばってん、だんもいっちょん飲まんで、食べぶる食べぶる、肉ばひたすら食べよらす。 | 88.42 | False |
| 14 | kumamoto_test_0014 | 桜んイメージのなかかもしれんばってん、一心行ん大桜は有名ばい。 | 桜んイメージのなかかも試練ばってん、一芯ぎょうん大桜は夢え。 | 72.13 | False |

Playing sample kumamoto_test_0000:
Ground Truth: カピバラに触らるっ場所もあっばい。親子で歩きよる姿に癒さるっけん。
QLoRA Pred  : カピバラに触らるっ場所もあっばい。親子で歩きよる姿に癒さるっけん。
```

<audio class="output" controls="controls" >
    <source src="/assets/tuning/output_kumamoto_test.wav" type="audio/x-wav" />
    您的瀏覽器不支援音訊元素。
</audio>

## 10.（選用）儲存並將 LoRA 轉接器推送到 Hugging Face Hub

您可以將微調後的 LoRA 權重直接推送到 Hugging Face Hub，方便分享或後續部署：



```python { .notebook-cell }
from huggingface_hub import whoami

hf_repo_id = whoami()["name"] + "/gemma-4-kumamoto-asr-qlora"
eval_model.push_to_hub(hf_repo_id, token=hf_token)
eval_processor.push_to_hub(hf_repo_id, token=hf_token)
print(f"Adapter successfully pushed to https://huggingface.co/{hf_repo_id}")

```

```{ .no-copy .output }
Processing Files (0 / 0)      : |          |  0.00B /  0.00B            
New Data Upload               : |          |  0.00B /  0.00B            
...adapter_model.safetensors:   8%|8         | 16.0MB /  193MB            
Saved model to https://huggingface.co/bebechien/gemma-4-kumamoto-asr-qlora
Unsloth: Restored added_tokens_decoder metadata in /tmp/tmpx47_aabs/tokenizer_config.json.
Processing Files (0 / 0)      : |          |  0.00B /  0.00B            
New Data Upload               : |          |  0.00B /  0.00B            
...mpx47_aabs/tokenizer.json:  50%|####9     | 16.0MB / 32.2MB            
Adapter successfully pushed to https://huggingface.co/bebechien/gemma-4-kumamoto-asr-qlora
```

## 摘要與後續步驟

本教學介紹了如何使用 Unsloth 與 QLoRA 微調 Gemma 模型。接下來請參閱以下文件：

- 瞭解如何[使用 Hugging Face Transformers 針對文字任務微調 Gemma](qlora.md)。
- 瞭解如何[使用 Hugging Face Transformers 針對視覺任務微調 Gemma](vision-qlora.md)。
- 瞭解如何[在 Vertex AI 上使用 Gemma 開放模型](https://cloud.google.com/vertex-ai/docs/generative-ai/open-models/use-gemma)。
