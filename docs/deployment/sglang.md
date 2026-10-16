---
title: SGLang
description: 使用 SGLang 服務 Gemma 4 —— 安裝、launch_server、模型儲存庫，以及 OpenAI 相容 API。
models: [gemma-4]
frameworks: [sglang]
---

# 使用 SGLang 服務 Gemma

[SGLang](https://docs.sglang.io) 是為高吞吐量與結構化
生成打造的服務引擎。如同 [vLLM](vllm.md)，它針對多使用者生產環境，而非本機
單使用者工作。

## 安裝

```bash
uv pip install --prerelease=allow sglang
```

## 服務

```bash { .model-swap }
python3 -m sglang.launch_server \
  --model-path google/gemma-4-12B-it \
  --host 0.0.0.0 \
  --port 30000
```

## 模型儲存庫

| 尺寸 | 儲存庫 |
|---|---|
| E2B | `google/gemma-4-E2B-it` |
| E4B | `google/gemma-4-E4B-it` |
| 12B | `google/gemma-4-12B-it` |
| 26B A4B | `google/gemma-4-26B-A4B-it` |
| 31B | `google/gemma-4-31B-it` |

對於 4 位元服務，請使用 [QAT](../get-started/qat.md) compressed-tensors 建置版本，
`google/gemma-4-{E2B,E4B,12B,31B}-it-qat-w4a16-ct`。

## 呼叫它

```python { .model-swap }
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:30000/v1")

response = client.chat.completions.create(
    model="google/gemma-4-12B-it",
    messages=[{"role": "user", "content": "Say this is a test"}],
)
print(response.choices[0].message.content)
```
