---
title: AI Edge Gallery
description: 在 Android 或 iOS 手機上免寫程式碼執行 Gemma 4 —— 聊天、視覺、音訊轉錄與基準測試，完全離線。
models: [gemma-4]
frameworks: [litert, android, ios]
---

# 在手機上用 AI Edge Gallery 試試 Gemma

[AI Edge Gallery](https://github.com/google-ai-edge/gallery) 是 Google AI Edge 用於直接在手機或筆電上執行
開放模型的應用程式。所有推論都在裝置上進行——無需網路、無需帳號，資料不會離開硬體——這是
體驗 Gemma 4 在您隨身裝置上實際感受的最快方式。

它是 [LiteRT-LM](litert.md) 的免程式碼對應版本：底層是相同的執行環境，包裝在
可以交給不會寫 Python 的人使用的介面中。

## 安裝

| 平台 | 需求 |
|---|---|
| [Google Play](https://play.google.com/store/apps/details?id=com.google.aiedge.gallery) | Android 12 及以上 |
| [App Store](https://apps.apple.com/app/google-ai-edge-gallery/id6737687734) | iOS 17 及以上 |
| macOS | 從[發佈頁面](https://github.com/google-ai-edge/gallery/releases)下載 |

若無法使用 Google Play，請從
[最新發佈版本](https://github.com/google-ai-edge/gallery/releases)安裝 APK。

## 它能做什麼

- **具思考模式的 AI 聊天** —— 多輪對話，並有開關可顯示模型的
  逐步推理。思考模式適用於 Gemma 4 家族。
- **圖片問答** —— 將相機對準或選取照片並提問。
- **音訊速記** —— 即時轉錄並翻譯語音錄音。
- **提示實驗室** —— 單輪提示的工作區，可直接控制溫度（temperature）與
  top-k，適合在將設定寫入程式碼前進行調校。
- **Agent 技能** —— 以維基百科接地（grounding）、地圖與視覺
  摘要卡等工具增強模型。技能可從 URL 載入。
- **Mobile Actions** 與 **Tiny Garden** —— 離線裝置控制與自然語言花園
  遊戲，兩者皆由 [FunctionGemma](../models/functiongemma/index.md) 270M 的微調版本驅動。
- **模型管理與基準測試** —— 下載模型、側載自己的模型，並在實際硬體上對每個
  模型進行基準測試，而不是相信別人的數字。

!!! tip "自備權重"
    Gallery 可載入自訂模型，因此您以
    [LiteRT-LM](litert.md) 微調並匯出的 Gemma，可以直接在裝置上測試，無需為它建置應用程式。

## 各層如何協作

| 層 | 功能 |
|---|---|
| [Google AI Edge](https://ai.google.dev/edge) | 核心的裝置端 ML API 與工具 |
| [LiteRT](litert.md) | 執行模型的執行環境 |
| [Hugging Face](https://huggingface.co/litert-community){ data-model="gemma-4" } | `.litertlm` 權重的來源 |

## 後續步驟

- 使用 [LiteRT-LM](litert.md) 從終端機或自己的應用程式執行相同的模型。
- 先微調再匯出：[微調總覽](../tuning/index.md)。
- 閱讀[專案 wiki](https://github.com/google-ai-edge/gallery/wiki) 了解詳細設定，
  包括企業管理的裝置。
