---
title: vLLM
description: 使用 vLLM 服務 Gemma 4 —— 安裝、vllm serve、模型儲存庫，以及 OpenAI 相容 API。
models: [gemma-4]
frameworks: [vllm]
---

# 使用 vLLM 服務 Gemma

[vLLM](https://docs.vllm.ai) 是高吞吐量的服務引擎。當您有多個
並發使用者時請使用它；對於單使用者本機工作，[Ollama](ollama.md) 或
[llama.cpp](llamacpp.md) 更簡單。

## 安裝

```bash
uv pip install vllm --torch-backend=auto
```

## 服務

```bash { .model-swap }
vllm serve google/gemma-4-12B-it
```

在 `http://localhost:8000` 上提供 OpenAI 相容 API。

## 模型儲存庫

| 尺寸 | 儲存庫 |
|---|---|
| E2B | `google/gemma-4-E2B-it` |
| E4B | `google/gemma-4-E4B-it` |
| 12B | `google/gemma-4-12B-it` |
| 26B A4B | `google/gemma-4-26B-A4B-it` |
| 31B | `google/gemma-4-31B-it` |

去掉 `-it` 即為基礎權重，僅在進一步訓練時需要。

對於 4 位元服務，請使用 [QAT](../get-started/qat.md) compressed-tensors 建置版本——4 位元權重搭配 16 位元
激活：

```bash { .model-swap }
vllm serve google/gemma-4-12B-it-qat-w4a16-ct
```

發布為 `google/gemma-4-{E2B,E4B,12B,31B}-it-qat-w4a16-ct`。目前沒有 26B A4B 的 compressed-tensors
檢查點，因此本頁的選取器提供其他四種尺寸；請從上方的全精度儲存庫服務 26B A4B。

## 呼叫它

```python { .model-swap }
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

response = client.chat.completions.create(
    model="google/gemma-4-12B-it",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."},
    ],
)
print(response.choices[0].message.content)
```
