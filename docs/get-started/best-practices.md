---
title: 最佳做法
description: 能讓 Gemma 4 發揮最佳效果的取樣參數、思考模式詞元、模態順序與圖像詞元預算。
models: [gemma-4]
---

# 最佳做法

使用 [Gemma 4](../models/gemma-4/index.md) 時，請採用以下設定與最佳做法以獲得最佳效能。

## 1. 取樣參數

在所有使用情境下，一律採用以下標準化取樣設定：

| 參數 | 值 |
|---|---|
| `temperature` | `1.0` |
| `top_p` | `0.95` |
| `top_k` | `64` |

## 2. 思考模式設定

相較於 Gemma 3，這些模型使用標準的 `system`{.token-role}、`assistant`{.token-role} 與
`user`{.token-role} 角色。要正確管理思考流程，請使用以下控制詞元：

- **觸發思考：** 在系統提示開頭加入 `<|think|>`{.token-think} 詞元即可啟用思考模式；移除該詞元則停用思考。
- **標準生成：** 啟用思考時，模型會先輸出內部推理，再以下列結構輸出最終答案：

<pre><code><span class="token-think">&lt;|channel&gt;thought</span>
<strong>[內部推理]</strong><span class="token-think">&lt;channel|&gt;</span></code></pre>

- **停用思考的行為：** 除 E2B 與 E4B 變體外的所有模型，若停用思考，模型仍會生成標籤，但思考區塊為空：

<pre><code><span class="token-think">&lt;|channel&gt;thought</span>
<span class="token-think">&lt;channel|&gt;</span><strong>[最終答案]</strong></code></pre>

!!! note
    許多函式庫（例如 Transformers 與 llama.cpp）會替您處理聊天範本的複雜細節。

## 3. 多輪對話

**歷史紀錄不含思考內容**：多輪對話中，歷史模型輸出應只包含最終回應。前幾輪模型的思考內容，在下一輪使用者輸入開始前*不得加入*，唯有工具呼叫輪次除外（該輪次應保留思考內容）。

## 4. 模態順序

為讓多模態輸入發揮最佳效能，請按以下順序放置：

- 將圖像內容放在提示中文字**之前**。
- 將音訊內容放在提示中文字**之後**。

## 5. 可變圖像解析度

除了可變長寬比之外，Gemma 4 還可透過可設定的視覺詞元預算支援可變圖像解析度；詞元預算決定以多少個詞元來表示一張圖像。較高的預算能以額外運算為代價保留更多視覺細節，而較低的預算則能為不需要精細理解的任務提供更快的推論。

支援的詞元預算為 **70**、**140**、**280**、**560** 與 **1120**。

- 對分類、圖說生成或影片理解使用*較低預算*：這類任務中更快的推論與處理大量畫格，比精細細節更重要。
- 對 OCR、文件解析或閱讀小字等任務使用*較高預算*。

!!! tip
    [視覺詞元預算示範](../resources/vision-token-budget.md)展示了各預算下的品質與速度權衡。

## 6. 音訊

處理音訊時請使用以下提示結構。

**音訊語音辨識（ASR）**

```text
Transcribe the following speech segment in {LANGUAGE} into {LANGUAGE} text.

Follow these specific instructions for formatting the answer:
* Only output the transcription, with no newlines.
* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three.
```

**自動語音翻譯（AST）**

```text
Transcribe the following speech segment in {SOURCE_LANGUAGE}, then translate it into {TARGET_LANGUAGE}.
When formatting the answer, first output the transcription in {SOURCE_LANGUAGE}, then one newline, then output the string '{TARGET_LANGUAGE}: ', then the translation in {TARGET_LANGUAGE}.
```

## 7. 音訊與影片長度

所有模型都支援圖像輸入，並能將影片以畫格方式處理；其中 E2B、E4B 與 12B 模型還支援音訊輸入。音訊最長支援 30 秒；若以每秒一畫格處理圖像，影片最長支援 60 秒。

## 接下來

<div class="card-grid">
<a class="card" href="capabilities/thinking.md">
  <span class="card-icon">{{ icon('lucide/brain') }}</span>
  <div class="card-title">推理</div>
  <p class="card-desc">讓 Gemma 在回答前先推理，並將兩者解析分離。</p>
</a>
<a class="card" href="prompt-formatting.md">
  <span class="card-icon">{{ icon('lucide/braces') }}</span>
  <div class="card-title">提示格式</div>
  <p class="card-desc">組織 Gemma 提示結構的控制詞元。</p>
</a>
<a class="card" href="capabilities/vision.md">
  <span class="card-icon">{{ icon('lucide/eye') }}</span>
  <div class="card-title">視覺總覽</div>
  <p class="card-desc">偵測、OCR、視覺問答與圖說生成。</p>
</a>
</div>

--8<-- "glossary.md"
