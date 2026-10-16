---
title: LM Studio
description: 使用 LM Studio 執行 Gemma 4 —— 桌面應用程式、lms CLI 與 OpenAI 相容的本機伺服器。
models: [gemma-4]
frameworks: [lmstudio]
---

# 使用 LM Studio 執行 Gemma

[LM Studio](https://lmstudio.ai) 是執行本機模型的桌面應用程式，具備 GUI、CLI
與本機伺服器。它以 GGUF（llama.cpp）與 MLX 兩種格式支援 Gemma。

## 安裝

從
[lmstudio.ai/download](https://lmstudio.ai/download) 下載 macOS、Windows 或 Linux 版本。

## 取得模型

在應用程式中，按下 ++cmd+shift+m++（macOS）或 ++ctrl+shift+m++（Windows/Linux），搜尋
"Gemma"，LM Studio 會建議適合您硬體的變體。

從終端機：

```bash { .model-swap }
lms get google/gemma-4-12b
```

列出您擁有的模型，然後載入一個：

```bash
lms ls
lms load <model_key>
```

`lms load` 接受 `--gpu=max|auto|0.0-1.0` 與 `--context-length=1-N`。

## 伺服器

```bash
lms server start
```

OpenAI 相容 API 位於 `http://localhost:1234/v1`，提供 `/v1/chat/completions`、
`/v1/completions`、`/v1/embeddings`、`/v1/models` 與 `/v1/responses`。

```python { .model-swap }
from openai import OpenAI

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

response = client.chat.completions.create(
    model="google/gemma-4-12b",
    messages=[{"role": "user", "content": "Say this is a test"}],
)
print(response.choices[0].message.content)
```

## 自己的 GGUF

```bash
lms import <path/to/model.gguf>
```

模型會出現在**我的模型**下。如果未被偵測到，請參閱
[匯入模型](https://lmstudio.ai/docs/app/advanced/import-model)。
