---
title: llama.cpp
description: 使用 llama.cpp 執行 Gemma 4 —— 各尺寸的 GGUF 儲存庫、圖像/音訊/視訊輸入，以及 OpenAI 相容伺服器。
models: [gemma-4]
frameworks: [llamacpp]
---

# 使用 llama.cpp 執行 Gemma

[llama.cpp](https://github.com/ggml-org/llama.cpp) 是大多數本機 LLM
工具底層的引擎。直接使用它可完全掌控量化與 GPU 卸載，它也是 Gemma 4
功能最強的本機執行環境——**圖像、音訊與視訊輸入都支援**。

## 安裝

=== "macOS / Linux"

    ```bash
    brew install llama.cpp
    ```

=== "Windows"

    ```bash
    winget install llama.cpp
    ```

=== "conda"

    ```bash
    conda install -c conda-forge llama.cpp
    ```

=== "Nix"

    ```bash
    nix profile install nixpkgs#llama-cpp
    ```

## 執行

```bash { .model-swap }
llama-cli -hf ggml-org/gemma-4-E2B-it-GGUF --prompt "Write a poem about the Kraken."
```

`-hf` 會在首次使用時下載權重，包括多模態投影器。搭配系統
提示：

```bash { .model-swap }
llama-cli -hf ggml-org/gemma-4-E2B-it-GGUF -sys "You are Hong Gildong." -p "Who are you?"
```

## 模型儲存庫

| 尺寸 | 儲存庫 |
|---|---|
| E2B | `ggml-org/gemma-4-E2B-it-GGUF` |
| E4B | `ggml-org/gemma-4-E4B-it-GGUF` |
| 12B | `ggml-org/gemma-4-12B-it-GGUF` |
| 26B A4B | `ggml-org/gemma-4-26B-A4B-it-GGUF` |
| 31B | `ggml-org/gemma-4-31B-it-GGUF` |

對於 4 位元，請優先使用 Google 的 [QAT](../get-started/qat.md) 權重——以模擬量化的方式訓練，而非
事後壓縮：

```bash { .model-swap }
llama-cli -hf google/gemma-4-12B-it-qat-q4_0-gguf -p "Explain PLE in one paragraph"
```

提供 `google/gemma-4-{E2B,E4B,12B,26B-A4B,31B}-it-qat-q4_0-gguf`。

## 多模態

```bash { .model-swap }
llama-mtmd-cli -hf ggml-org/gemma-4-E2B-it-GGUF
```

使用本機檔案時，請明確傳入投影器：

```bash
llama-server -m gemma-4-12B-it-Q4_K_M.gguf --mmproj mmproj-gemma-4-12B-it.gguf
```

!!! note
    音訊適用於 E2B、E4B 與 12B。26B A4B 與 31B 模型接受文本與圖像，並以影格
    方式讀取視訊——它們沒有音訊編碼器。

## 伺服器

```bash { .model-swap }
llama-server -hf ggml-org/gemma-4-E2B-it-GGUF
```

瀏覽器 UI 位於 `http://localhost:8080`，OpenAI 相容 API 位於 `http://localhost:8080/v1`。

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="gemma-4",  # llama-server serves whichever model it was started with
    messages=[{"role": "user", "content": "Say this is a test"}],
)
print(response.choices[0].message.content)
```

多模態輸入會快速消耗脈絡；請用 `-c` 設定，例如 `-c 32768`。
