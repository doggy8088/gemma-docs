---
title: Apple MLX
description: 在 Apple 晶片上使用 MLX 執行 Gemma 4 —— 文本與視覺生成、模型儲存庫，以及 OpenAI 相容伺服器。
models: [gemma-4]
frameworks: [mlx]
---

# 使用 MLX 執行 Gemma

[MLX](https://github.com/ml-explore/mlx) 是 Apple 用於在 Apple 晶片上進行機器學習的陣列框架。
它在統一記憶體架構上原生執行 Gemma，無需 GGUF
轉換步驟。

## 安裝

```bash
pip install mlx mlx-lm mlx-vlm
```

## 執行

=== "文本"

    ```bash { .model-swap }
    mlx_lm.generate --model mlx-community/gemma-4-E2B-it-qat-4bit --prompt "Who are you?"
    ```

=== "視覺"

    ```bash { .model-swap }
    mlx_vlm.generate --model mlx-community/gemma-4-E2B-it-qat-4bit \
      --prompt "Describe this image." --image <path_to_image>
    ```

## 模型儲存庫

社群轉換的權重位於
[mlx-community](https://huggingface.co/mlx-community)：

| 尺寸 | 儲存庫 |
|---|---|
| E2B | `mlx-community/gemma-4-E2B-it-qat-4bit` |
| E4B | `mlx-community/gemma-4-E4B-it-qat-4bit` |
| 12B | `mlx-community/gemma-4-12B-it-qat-4bit` |
| 26B A4B | `mlx-community/gemma-4-26B-A4B-it-qat-4bit` |
| 31B | `mlx-community/gemma-4-31B-it-qat-4bit` |

!!! note
    涵蓋範圍與命名不一——這些是社群轉換，而非官方發佈。
    請搜尋 [mlx-community 上的 Gemma 4](https://huggingface.co/mlx-community?search=gemma-4){ data-model="gemma-4" }
    查看最新清單。如果您不想挑選儲存庫，[Ollama](ollama.md) 也在其 `-mlx`
    標籤下為每個尺寸提供 MLX 建置版本。

## 伺服器

```bash { .model-swap }
mlx_vlm.server --model mlx-community/gemma-4-E2B-it-qat-4bit --port 8080
```

OpenAI 相容 API 位於 `http://localhost:8080/v1`。

```python { .model-swap }
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="mlx-community/gemma-4-E2B-it-qat-4bit",
    messages=[{"role": "user", "content": "Say this is a test"}],
)
print(response.choices[0].message.content)
```
