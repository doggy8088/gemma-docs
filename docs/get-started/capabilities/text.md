---
title: 推論
description: "在 Hugging Face Transformers 中使用 Gemma 4 進行基本文字推論：單輪提示、多輪對話與系統指令。"
models: [gemma-4]
frameworks: [transformers]
colab: https://colab.research.google.com/github/google-gemma/docs/blob/main/docs/notebooks/text-gemma-4-e2b.ipynb
---
# 推論

Gemma 是一個輕量級的頂尖開放模型家族，採用與創建 [Gemini](https://deepmind.google/technologies/gemini/#introduction) 模型相同的研發與技術打造而成。Gemma 4 旨在成為全球最具效率的開放權重模型家族。

本文件提供使用 Hugging Face [`transformers`](https://huggingface.co/docs/transformers/installation) 函式庫，以 Gemma 4 進行基本文字推論的指南。內容涵蓋環境設定、模型載入，以及各種文字生成情境，包括單輪提示、結構化多輪對話，以及套用系統指令。

## 安裝 Python 套件

安裝執行 Gemma 模型與發送請求所需的 Hugging Face 函式庫。

```bash { .notebook-cell }
# Install PyTorch & other libraries
pip install torch accelerate

# Install the transformers library
pip install "transformers>=5.10.1"
```

[`Dialog`](https://github.com/google-deepmind/dialog) 是一個用於操作與顯示對話的函式庫。

```bash { .notebook-cell }
pip install dialog
```

## 載入模型

使用 [`transformers`](https://huggingface.co/docs/transformers/installation) 函式庫載入管線（pipeline）

```python { .notebook-cell .model-swap }
MODEL_ID = "google/gemma-4-E2B-it"

from transformers import pipeline

txt_pipe = pipeline(
    task="text-generation",
    model=MODEL_ID,
    device_map="auto",
    dtype="auto"
)
```


## 執行文字生成

一旦您已將 Gemma 模型載入並設定為 `pipeline` 物件，就可以向模型發送提示。以下範例程式碼展示了使用 `text_inputs` 參數的基本請求：

```python { .notebook-cell }
output = txt_pipe(text_inputs="<|turn>user\nRoses are..<turn|>\n<|turn>model\n")
print(output[0]['generated_text'].removesuffix("<turn|>"))
```

``` { .no-copy .output }
<|turn>user
Roses are..<turn|>
<|turn>model
Here are a few ways to complete the phrase "Roses are...":

**Focusing on the color:**

* **Roses are red.** (A classic, though slightly contradictory!)
* **Roses are beautiful.**
* **Roses are pink.**

**Focusing on the feeling/meaning:**

* **Roses are lovely.**
* **Roses are sweet.**
* **Roses are a sign of affection.**

**A slightly more poetic answer:**

* **Roses are a memory.**

**Which one feels right to you? 😊**
```

## 使用 Dialog 函式庫

```python { .notebook-cell }
import dialog
from transformers import GenerationConfig
config = GenerationConfig.from_pretrained(MODEL_ID)
config.max_new_tokens = 512

conv = dialog.Conversation(
    dialog.User("Roses are...")
)
output = txt_pipe(text_inputs=conv.as_text(), return_full_text=False, generation_config=config)
conv += dialog.Model(output[0]['generated_text'].removesuffix("<turn|>"))

print(conv.as_text())
conv.show()
```

``` { .no-copy .output }
<|turn>user
Roses are...<turn|>
<|turn>model
Here are a few ways to complete the phrase "Roses are...":

**Focusing on the scent:**

* **...fragrant.**
* **...scented.**

**Focusing on the visual:**

* **...beautiful.**
* **...vibrant.**
* **...red.**

**Focusing on the emotion (the most classic completion):**

* **...a symbol of love.**
* **...a declaration.**
* **...perfect.**

**If you want a simple, classic answer, I recommend:**

**"Roses are beautiful."** or **"Roses are a symbol of love."**
```

## 使用提示範本

在使用更複雜的提示生成內容時，請使用提示範本來建構您的請求。提示範本讓您能指定來自特定角色的輸入，例如 `user`{.token-role} 或 `model`{.token-role}，是管理與 Gemma 模型多輪聊天互動所需的格式。以下範例程式碼展示如何為 Gemma 建構提示範本：

```python { .notebook-cell }
from transformers import GenerationConfig
config = GenerationConfig.from_pretrained(MODEL_ID)
config.max_new_tokens = 512

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Write a short poem about the Kraken."},
        ]
    }
]

output = txt_pipe(messages, return_full_text=False, generation_config=config)
print(output[0]['generated_text'].removesuffix("<turn|>"))
```

``` { .no-copy .output }
Beneath the waves, where sunlight dies,
A shadow stirs, with ancient sighs.
The Kraken wakes, a monstrous might,
With tentacles of endless night.

A crushing grip, a salty dread,
Where ships are lost and hope is dead.
A legend spun of ink and brine,
A primal fear, a dark design.
```

## 多輪對話

在多輪設定中，對話記錄會保留為交替出現的 `user`{.token-role} 與 `model`{.token-role} 角色序列。這個累積的清單充當模型的記憶，確保每次新的輸出都以前述對話為依據。

```python { .notebook-cell }
import dialog
from transformers import GenerationConfig
config = GenerationConfig.from_pretrained(MODEL_ID)
config.max_new_tokens = 512

# User turn #1
conv = dialog.Conversation(
    dialog.User("Write a short poem about the Kraken.")
)

# Model response #1
output = txt_pipe(text_inputs=conv.as_text(), return_full_text=False, generation_config=config)
conv += dialog.Model(output[0]['generated_text'].removesuffix("<turn|>"))

# User turn #2
conv += dialog.User("Now with the Siren.")

# Model response #2
output = txt_pipe(text_inputs=conv.as_text(), return_full_text=False, generation_config=config)
conv += dialog.Model(output[0]['generated_text'].removesuffix("<turn|>"))

print(conv.as_text())
conv.show()
```

``` { .no-copy .output }
<|turn>user
Write a short poem about the Kraken.<turn|>
<|turn>model
In depths where sunlight cease,
A shadow vast and deep,
The Kraken wakes with might,
A terror to the sleep.
With tentacles of ink,
It pulls the ocean's brink.<turn|>
<|turn>user
Now with the Siren.<turn|>
<|turn>model
Where coral sleeps in silent grace,
A melody floats from the sea,
The Siren calls with silver thread,
A siren song for all to see.
With eyes of emerald, deep and wide,
She lures the sailor to the tide.
A siren's kiss, a deadly art,
That breaks the sailor's guarded heart.
```

以下是以文字匯出的對話。

```python { .notebook-cell }
chat_history = conv.as_text(training=True)
print(chat_history)
print("-"*80)

# display as Conversation widget
chat_history
```

``` { .no-copy .output }
<|turn>user
Write a short poem about the Kraken.<turn|>
<|turn>model
In depths where sunlight cease,
A shadow vast and deep,
The Kraken wakes with might,
A terror to the sleep.
With tentacles of ink,
It pulls the ocean's brink.<turn|>
<|turn>user
Now with the Siren.<turn|>
<|turn>model
Where coral sleeps in silent grace,
A melody floats from the sea,
The Siren calls with silver thread,
A siren song for all to see.
With eyes of emerald, deep and wide,
She lures the sailor to the tide.
A siren's kiss, a deadly art,
That breaks the sailor's guarded heart.<turn|>
--------------------------------------------------------------------------------
```


!!! Note
    如果您設定 `training=True`，該對話會被視為完整範例，結尾一定是 `<turn|>`


## 系統指令

使用 `system`{.token-role} 角色來提供系統層級的指令。

```python { .notebook-cell }
import dialog
from transformers import GenerationConfig
config = GenerationConfig.from_pretrained(MODEL_ID)
config.max_new_tokens = 512

conv = dialog.Conversation(
    dialog.System("Speak like a pirate."),
    dialog.User("Why is the sky blue?")
)

output = txt_pipe(text_inputs=conv.as_text(), return_full_text=False, generation_config=config)
conv += dialog.Model(output[0]['generated_text'].removesuffix("<turn|>"))

print(conv.as_text())
conv.show()
```

``` { .no-copy .output }
<|turn>system
Speak like a pirate.<turn|>
<|turn>user
Why is the sky blue?<turn|>
<|turn>model
Ahoy there! Why is the sky blue, ye ask? It be down to the **sunlight** and the **air** itself!

Imagine the sunlight be a big crew of tiny, invisible particles—like a whole fleet of little pirates! When the sunlight be crew of tiny particles, these particles go through the air and bump into the gas molecules that make up our sky (mostly nitrogen and oxygen).

When the sunlight hits these molecules, something magical happens! The light gets **scattered** in all directions, just like when a beam of light hits a big, dusty mirror and gets scattered everywhere!

Of all the colors in the sunlight—red, orange, yellow, green, blue, indigo—the **blue light gets scattered the most**! It gets bounced and spread out across the entire sky, making our beautiful daytime sky appear blue to our eyes!

So, next time ye look up, ye can tell the secret: it be the **sunlight** bein' **scattered** by the **air**!

**Shiver me timbers!** That's the pirate explanation!
```

## 摘要
在本指南中，您學會了如何使用 Hugging Face `transformers` 函式庫以 Gemma 4 進行基本文字推論，內容包括：

1. **環境**：設定環境並安裝相依套件。
2. **載入模型**：使用 `pipeline` 抽象層。
3. **推論**：執行基本文字生成。
4. **對話**：使用 [`dialog`](https://github.com/google-deepmind/dialog) 函式庫追蹤對話。
5. **多輪對話**：多輪對話與套用系統指令。

## 後續步驟

<div class="card-grid">
<a class="card" href="../prompt-formatting.md">
  <span class="card-icon">{{ icon('lucide/braces') }}</span>
  <div class="card-title">提示格式</div>
  <p class="card-desc">建構 Gemma 提示的控制詞元。</p>
</a>
<a class="card" href="function-calling.md">
  <span class="card-icon">{{ icon('lucide/wrench') }}</span>
  <div class="card-title">工具呼叫</div>
  <p class="card-desc">賦予 Gemma 可呼叫的函式，並將結果回饋。</p>
</a>
<a class="card" href="../../deployment/index.md">
  <span class="card-icon">{{ icon('lucide/play') }}</span>
  <div class="card-title">執行 Gemma</div>
  <p class="card-desc">從筆電到叢集，所有能執行 Gemma 的推論引擎。</p>
</a>
</div>
