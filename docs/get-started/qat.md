---
title: 量化感知訓練（Quantization-Aware Training）
description: 說明各引擎應下載哪一組 Gemma 4 量化權重，以及在 4 位元下 QAT 為何優於訓練後量化。
models: [gemma-4]
frameworks: [llamacpp, vllm]
---

# 量化感知訓練（QAT）

Google 與全精度權重一併發布**量化感知訓練**（QAT）檢查點。QAT 在訓練*期間*就模擬量化，讓模型學會補償精度損失——這與訓練後量化不同，後者壓縮的是已訓練完成的模型，會在壓縮過程中損失品質。

**在 4 位元下，請選擇 QAT 版本，而非同尺寸的通用 Q4 版本。**

## 該下載哪一個？

| 執行環境 | 後綴 |
|---|---|
| llama.cpp, LM Studio, Ollama | `-qat-q4_0-gguf` |
| vLLM, SGLang | `-qat-w4a16-ct` |
| 行動裝置（Transformers） | `-qat-mobile-transformers` |
| 行動裝置（compressed tensors） | `-qat-mobile-ct` |
| 推測性解碼 | `-qat-q4_0-unquantized` + `-qat-q4_0-unquantized-assistant` |
| 轉換為其他格式（例如 MLX） | `-qat-q4_0-unquantized` |

命名遵循 `google/gemma-4-{SIZE}-it{SUFFIX}`，例如：

```bash { .model-swap }
# Local, via llama.cpp
llama-cli -hf google/gemma-4-12B-it-qat-q4_0-gguf
```

```bash { .model-fixed }
# Served, via vLLM
vllm serve google/gemma-4-12B-it-qat-w4a16-ct
```

GGUF 範例會跟隨模型選擇器；compressed-tensors 範例固定為 12B，因為目前未發布 26B A4B 的 `-qat-w4a16-ct` 檢查點——詳見下方「版本供應」。

## 版本供應

| 變體 | 尺寸 |
|---|---|
| `-gguf`, `-unquantized`, `-assistant` | E2B, E4B, 12B, 26B A4B, 31B |
| `-w4a16-ct` | E2B, E4B, 12B, 31B |
| `-mobile-transformers`, `-mobile-ct` | E2B, E4B |

集合：
[gemma-4-qat-q4-0][hf-gemma4-qat-q40] ·
[gemma-4-qat-mobile][hf-gemma4-qat-mobile]。
所有 QAT 檢查點也可在
[Kaggle](https://www.kaggle.com/models/google/gemma-4/transformers){ data-model="gemma-4" } 取得。

!!! note "行動裝置版本採用不同架構"
    `-mobile-*` 權重建構於自訂的 `wNa8o8` 架構，具備 2 位元解碼層與最佳化的 KV 快取，針對裝置端 RAM 限制調校，而非追求原始品質。

## 記憶體

靜態權重的概略記憶體占用，取自
[Gemma 4 總覽](../models/gemma-4/index.md)：

| 尺寸 | BF16 | SFP8 | Q4_0 |
|---|---|---|---|
| E2B | 11.4 GB | 5.7 GB | 2.9 GB |
| E4B | 17.9 GB | 8.9 GB | 4.5 GB |
| 12B | 26.7 GB | 13.4 GB | 6.7 GB |
| 26B A4B | 57.7 GB | 28.8 GB | 14.4 GB |
| 31B | 69.9 GB | 34.9 GB | 17.5 GB |

上下文（KV 快取）在此之外另行占用，且隨提示與生成長度增長。
## 接下來

<div class="card-grid">
<a class="card" href="../models/index.md">
  <span class="card-icon">{{ icon('lucide/boxes') }}</span>
  <div class="card-title">模型家族</div>
  <p class="card-desc">每一代與每個變體，以及各自的來源。</p>
</a>
<a class="card" href="../deployment/index.md">
  <span class="card-icon">{{ icon('lucide/play') }}</span>
  <div class="card-title">執行 Gemma</div>
  <p class="card-desc">從筆記型電腦到叢集，涵蓋所有可執行 Gemma 的引擎。</p>
</a>
</div>

--8<-- "glossary.md"
