---
title: Unsloth
description: 使用 Unsloth 執行 Gemma 4 —— 桌面 Studio 應用程式、適用於 llama.cpp 的動態量化 GGUF，以及快 2 倍的微調。
models: [gemma-4]
frameworks: [unsloth, llamacpp]
---

# 使用 Unsloth 執行 Gemma

[Unsloth](https://unsloth.ai/docs) 發布 Gemma 4 的動態量化 GGUF 建置版本，並提供桌面應用程式 Unsloth Studio，
可在本地下載並執行它們。它也是在單張消費級 GPU 上微調 Gemma 的最快途徑。

在這裡執行 Gemma 4 有兩種方式：Studio 應用程式，或直接在
[llama.cpp](llamacpp.md) 中使用 Unsloth 的 GGUF。

## 硬體需求

Unsloth 自家的建議總量——RAM + VRAM 合計，或統一記憶體：

| 尺寸 | 4 位元 | 8 位元 | BF16 / FP16 |
|---|---:|---:|---:|
| E2B | 4 GB | 5–8 GB | 10 GB |
| E4B | 5.5–6 GB | 9–12 GB | 16 GB |
| 12B | 7–8 GB | 13–14 GB | 25 GB |
| 26B A4B | 16–18 GB | 28–30 GB | 52 GB |
| 31B | 17–20 GB | 34–38 GB | 62 GB |

!!! note
    這些是 Unsloth 針對其 GGUF 建置版本的數字，包含執行期開銷，因此略高於
    [Gemma 4 總覽](../models/gemma-4/index.md)中的靜態權重大小。脈絡（KV 快取）還需額外計算。

Unsloth 建議 12B、26B A4B 與 31B 使用**動態 4 位元**（`UD-Q4_K_XL`），E2B 與 E4B 使用
**8 位元**（`Q8_0`）。

## 建議設定

使用 Google 預設的 Gemma 4 取樣參數——與
[最佳實務](../get-started/best-practices.md)中所述相同：

```text
temperature = 1.0
top_p = 0.95
top_k = 64
```

## Unsloth Studio

<ol class="steps" markdown>
<li markdown>
**安裝**

下載 [Unsloth 應用程式](https://unsloth.ai/download) 的 macOS、Windows 或 Linux 版本，或手動安裝。

macOS、Linux、WSL：

```bash
curl -fsSL https://unsloth.ai/install.sh | sh
```

Windows PowerShell：

```powershell
irm https://unsloth.ai/install.ps1 | iex
```
</li>
<li markdown>
**啟動**

```bash
unsloth studio -H 0.0.0.0 -p 8888
```

然後開啟 `http://127.0.0.1:8888`。若要改為透過免費的 Cloudflare 隧道以 HTTPS 提供服務，
請使用 `unsloth studio --secure`。
</li>
<li markdown>
**下載並執行 Gemma 4**

在聊天分頁中搜尋 Gemma 4 並選擇一個量化版本。推論參數會自動設定，
GGUF 與 MLX 檔案都支援。
</li>
</ol>

## llama.cpp

先建置 `llama.cpp`——CPU 機器與 Apple 晶片（Metal 預設開啟）請設定 `-DGGML_CUDA=OFF`：

```bash
git clone https://github.com/ggml-org/llama.cpp
cmake llama.cpp -B llama.cpp/build \
    -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON
cmake --build llama.cpp/build --config Release -j --clean-first \
    --target llama-cli llama-mtmd-cli llama-server llama-gguf-split
cp llama.cpp/build/bin/llama-* llama.cpp
```

然後直接從 Hugging Face 拉取並執行模型。無需設定脈絡長度——
llama.cpp 會使用正好所需的量。

=== "E2B"

    ```bash
    export LLAMA_CACHE="unsloth/gemma-4-E2B-it-GGUF"
    ./llama.cpp/llama-cli \
        -hf unsloth/gemma-4-E2B-it-GGUF:Q8_0 \
        --temp 1.0 --top-p 0.95 --top-k 64
    ```

=== "E4B"

    ```bash
    export LLAMA_CACHE="unsloth/gemma-4-E4B-it-GGUF"
    ./llama.cpp/llama-cli \
        -hf unsloth/gemma-4-E4B-it-GGUF:Q8_0 \
        --temp 1.0 --top-p 0.95 --top-k 64
    ```

=== "12B"

    ```bash
    export LLAMA_CACHE="unsloth/gemma-4-12B-it-GGUF"
    ./llama.cpp/llama-cli \
        -hf unsloth/gemma-4-12b-it-GGUF:UD-Q4_K_XL \
        --temp 1.0 --top-p 0.95 --top-k 64
    ```

=== "26B A4B"

    ```bash
    export LLAMA_CACHE="unsloth/gemma-4-26B-A4B-it-GGUF"
    ./llama.cpp/llama-cli \
        -hf unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL \
        --temp 1.0 --top-p 0.95 --top-k 64
    ```

=== "31B"

    ```bash
    export LLAMA_CACHE="unsloth/gemma-4-31B-it-GGUF"
    ./llama.cpp/llama-cli \
        -hf unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL \
        --temp 1.0 --top-p 0.95 --top-k 64
    ```

### 視覺

將權重與多模態投影器一起下載，然後同時傳入兩者：

```bash
hf download unsloth/gemma-4-26B-A4B-it-GGUF \
    --local-dir unsloth/gemma-4-26B-A4B-it-GGUF \
    --include "*mmproj-BF16*" \
    --include "*UD-Q4_K_XL*"
```

```bash
./llama.cpp/llama-cli \
    --model unsloth/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf \
    --mmproj unsloth/gemma-4-26B-A4B-it-GGUF/mmproj-BF16.gguf \
    --temp 1.0 --top-p 0.95 --top-k 64
```

### 伺服器

```bash
./llama.cpp/llama-server \
    --model unsloth/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf \
    --mmproj unsloth/gemma-4-26B-A4B-it-GGUF/mmproj-BF16.gguf \
    --temp 1.0 --top-p 0.95 --top-k 64 \
    --alias "unsloth/gemma-4-26B-A4B-it-GGUF" \
    --port 8001 \
    --chat-template-kwargs '{"enable_thinking":true}'
```

## 思考模式

思考模式的啟用方式是在系統提示開頭加上 `<|think|>`。若要改為透過 llama.cpp 關閉它，請傳入：

=== "macOS, Linux"

    ```bash
    --chat-template-kwargs '{"enable_thinking":false}'
    ```

=== "Windows PowerShell"

    ```powershell
    --chat-template-kwargs "{\"enable_thinking\":false}"
    ```

關於思考輸出的結構以及對話歷史中應保留的內容，請參閱[最佳實務](../get-started/best-practices.md)。

## 微調

Unsloth 訓練 Gemma 所需的記憶體遠少於原生的 Transformers 迴圈，這使得
單 GPU 與免費 Colab 微調成為可能。請參閱 Unsloth 的
[Gemma 4 微調指南](https://unsloth.ai/docs/models/gemma-4){ data-model="gemma-4" }，以及我們自己的
[微調總覽](../tuning/index.md)，了解 QLoRA 與完整微調路徑。

## 資源

- [Gemma 4 GGUF 集合](https://huggingface.co/collections/unsloth/gemma-4){ data-model="gemma-4" }
- [Unsloth 文件](https://unsloth.ai/docs/models/gemma-4){ data-model="gemma-4" }
