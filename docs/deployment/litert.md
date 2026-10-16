---
title: LiteRT-LM
description: 使用 LiteRT-LM 在裝置端執行 Gemma 4 —— CLI 安裝、預建的 .litertlm 權重，以及匯出您自己的微調模型。
models: [gemma-4]
frameworks: [litert]
source_url: "https://developers.google.com/edge/litert-lm/models/gemma-4"
---

# 使用 LiteRT-LM 在裝置端執行 Gemma

[LiteRT-LM](https://developers.google.com/edge/litert-lm) 是 Google AI Edge 的執行環境，用於
直接在裝置上執行語言模型——Android、iOS、macOS、Windows、Linux 與
Raspberry Pi——支援 CPU、GPU 與 NPU 加速，無需網路往返。

!!! note "目前支援的尺寸"
    LiteRT-LM 目前提供 **E2B** 與 **E4B**，更大的模型即將推出。上方的選取器提供
    已發布 `.litertlm` 權重的兩種尺寸。

## 安裝

```bash
uv tool install litert-lm
```

## 執行

權重直接從 Hugging Face LiteRT Community 取得：

```bash { .model-swap }
litert-lm run \
  --from-huggingface-repo=litert-community/gemma-4-E2B-it-litert-lm \
  gemma-4-E2B-it.litertlm \
  --prompt="What is the capital of France?"
```

## 多詞元預測

LiteRT-LM 支援 [MTP](../get-started/capabilities/mtp.md)，以無品質損失的方式加速解碼：在行動 GPU 上最高達
**2.2 倍**，在行動 CPU 上達 **1.5 倍**。

Google 建議在 GPU 後端的所有任務都使用 MTP，CPU 上的 E4B 也是。在 CPU 的 E2B 上，它
對改寫、摘要與程式設計最有幫助，但可能會讓自由形式的生成略微變慢——
因此請在該處斟酌啟用。

## 部署您自己的微調模型

將 Safetensors（包括您已微調的模型）匯出為 `.litertlm` 格式：

```bash { .model-swap }
uv tool install litert-torch-nightly

litert-torch export_hf \
  --model=google/gemma-4-E2B-it \
  --output_dir=/tmp/gemma4 \
  --externalize_embedder \
  --jinja_chat_template_override=litert-community/gemma-4-E2B-it-litert-lm
```

然後以相同方式執行匯出的套件：

```bash
litert-lm run /tmp/gemma4/model.litertlm --prompt="What is the capital of France?"
```

## 模型尺寸

| 尺寸 | `.litertlm` 套件 |
|---|---|
| E2B | 2.58 GB |
| E4B | 3.65 GB |

Google 在 [Gemma 4 LiteRT-LM 頁面](https://developers.google.com/edge/litert-lm/models/gemma-4){ data-model="gemma-4" } 上發布了按裝置與
後端測量的預填、解碼與首詞元時間數據。

## 平台指南

LiteRT-LM 為 CLI、Python、Android（Kotlin）、iOS（Swift）與 C++ 提供 API。各平台請參閱
[LiteRT-LM 文件](https://developers.google.com/edge/litert-lm)。

要在手機上免寫程式碼試用 Gemma，請使用
[AI Edge Gallery](ai-edge-gallery.md)。
