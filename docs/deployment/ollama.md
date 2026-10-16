---
title: Ollama
description: 使用 Ollama 在本機執行 Gemma 4 —— 模型標籤、圖像輸入，以及 OpenAI 相容伺服器。
models: [gemma-4]
frameworks: [ollama]
---

# 使用 Ollama 執行 Gemma

Ollama 會為您下載量化權重，並提供 CLI 與本機 HTTP API。無需 GPU，
但有會更好。

## 安裝

從 [ollama.com/download](https://ollama.com/download) 下載，然後確認安裝成功：

```bash
ollama --version
```

## 執行

```bash { .model-swap }
ollama run gemma4:12b "Explain quantum computing simply"
```

首次使用時會下載模型。若只想下載而不執行：

```bash { .model-swap }
ollama pull gemma4:12b
```

## 模型標籤

| 標籤 | 下載大小 | 脈絡 |
|---|---|---|
| `gemma4:e2b` | 7.2 GB | 128K |
| `gemma4:e4b` | 9.6 GB | 128K |
| `gemma4:12b` | 7.6 GB | 256K |
| `gemma4:26b` | 18 GB | 256K |
| `gemma4:31b` | 20 GB | 256K |

`gemma4:latest` 是 `e4b` 的別名。在 Apple Silicon 上，`-mlx` 標籤
（`gemma4:12b-mlx`）使用 MLX 執行環境。完整清單：
[ollama.com/library/gemma4/tags](https://ollama.com/library/gemma4/tags){ data-model="gemma-4" }。

## 圖像

```bash { .model-swap }
ollama run gemma4:12b ./image.png "What is in this image?"
```

!!! note
    Ollama 發布的 Gemma 4 僅支援**文本與圖像**。若要使用音訊輸入，請使用
    [Transformers](huggingface.md) 或 [llama.cpp](llamacpp.md)。

## 伺服器

Ollama 在連接埠 `11434` 上提供服務。

=== "OpenAI 相容"

    ```python { .model-swap }
    from openai import OpenAI

    client = OpenAI(
        base_url="http://localhost:11434/v1/",
        api_key="ollama",  # required but ignored
    )

    response = client.chat.completions.create(
        model="gemma4:12b",
        messages=[{"role": "user", "content": "Say this is a test"}],
    )
    print(response.choices[0].message.content)
    ```

=== "原生 API"

    ```bash { .model-swap }
    curl http://localhost:11434/api/generate -d '{
      "model": "gemma4:12b",
      "prompt": "roses are red"
    }'
    ```

=== "透過 API 傳送圖像"

    ```bash { .model-swap }
    IMG=$(base64 < test.jpg | tr -d '\n')

    curl -X POST http://localhost:11434/api/chat \
      -H "Content-Type: application/json" \
      -d '{
        "model": "gemma4:12b",
        "messages": [{
          "role": "user",
          "content": "What is in this image?",
          "images": ["'"$IMG"'"]
        }],
        "stream": false
      }'
    ```

!!! tip "在 24 GiB VRAM 以下，脈絡預設為 4K"
    Ollama 依可用 VRAM（而非模型）決定脈絡大小。需要時請調高：
    `OLLAMA_CONTEXT_LENGTH=64000 ollama serve`，或以
    `"options": {"num_ctx": 64000}` 針對單一請求設定。Agent 與程式設計工具至少需要 64K——
    請參閱 [Ollama：脈絡長度](https://docs.ollama.com/context-length)。
