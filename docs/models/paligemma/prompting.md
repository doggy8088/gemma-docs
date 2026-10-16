---
title: PaliGemma 提示與系統指示
description: 本頁說明 PaliGemma 視覺語言模型的提示格式、任務語法與系統指示的撰寫方式，並提供實用範例。
models: [paligemma]
source_url: "https://ai.google.dev/gemma/docs/paligemma/prompt-system-instructions.md.txt"
---
# PaliGemma 提示與系統指示

本頁說明 PaliGemma 模型的提示格式與系統指示。這些 Gemma 模型變體使用與
Gemma 基礎模型相同的大致格式，並支援針對特定影像相關任務的特殊語法。

## 提示格式

PaliGemma 模型使用與其所基於的 Gemma 基礎模型相同的提示格式。不過，
PaliGemma 模型也支援特殊的[任務語法](prompting.md#提示任務語法)，下一節
將加以說明。有關 Gemma 提示格式的更多資訊，請參閱
[Gemma 提示與系統指示](../../get-started/prompt-formatting.md)。

!!! Note
    雖然預訓練檢查點支援此提示結構，但仍需要針對您的特定使用情境進行微調。


### 影像與文字資料的順序

以文字與影像資料提示 PaliGemma 模型時，必須*先*提供影像資料，再提供文字
提示資料。顛倒影像與文字提示資料的順序，或混合影像與文字資料，通常會產生
無法使用的回應。

### 提示任務語法

PaliGemma 模型以特定提示模式與語法進行訓練，適用於物件識別與影像圖說等任務。
您可以如下使用此提示任務語法，要求 PaliGemma 模型執行特定行為：

- **`"cap {lang}\n"`:**：非常簡短的原始圖說（僅 PT 支援）
- **`"caption {lang}\n"`:**：簡短圖說
- **`"describe {lang}\n"`:**：稍長、更具描述性的圖說（僅 PT 支援）
- **`"ocr"`:**：光學字元辨識（僅 PT 支援）
- **`"answer {lang} {question}\n"`:**：針對影像內容的問答
- **`"question {lang} {answer}\n"`:**：針對給定答案產生問題（僅 PT 支援）
- **`"detect {object} ; {object}\n"`:**：在影像中定位所列物件，並回傳其邊界框
- **`"segment {object} ; {object}\n"`:**：在影像中定位所列物件所佔的區域，為該物件建立影像分割

!!! Note
    用於偵測與分割的「物件」也可以是表達式，例如「左側的企鵝」。

`{lang}` 選項是語言代碼。PaliGemma 支援以此選項辨識 34 種不同語言的任務提示。
支援的語言清單可參閱
[GitHub](https://github.com/google/crossmodal-3600/blob/main/web-data/README.md)。

!!! Important
    [PaliGemma FT](index.md#模型類別) 類別的模型是為特定任務訓練的，*僅*支援目標任務的語法。

有關展示如何使用此語法的詳細程式碼範例，請參閱教學
[使用 Keras 產生 PaliGemma 輸出](inference.md)。

### 以自然語言提示

雖然建議使用上一節的語法，但混合模型也支援以自然語言執行許多任務。例如，
"describe this image briefly" 或 "what is this text" 即使未使用精確的語法提示，
仍然有效。

### 批次提示指令

您可以在單一提示中提供多個提示指令，作為一批指令。每個提示指令都必須以 `\n`
字元結尾。以下範例示範如何組織提示文字以提供多個指令。

```python
prompts = [
    'answer en where is the cow standing?\n',
    'answer en what color is the cow?\n',
    'describe en\n',
    'detect cow\n',
    'segment cow\n',
]
images = [cow_image, cow_image, cow_image, cow_image, cow_image]
outputs = paligemma.generate(
    inputs={
        "images": images,
        "prompts": prompts,
    }
)
for output in outputs:
    print(output)
```

## 系統指示

PaliGemma 模型除了其所基於的基礎模型的
[Gemma 系統指示](../../get-started/prompt-formatting.md#系統指令)
之外，不支援任何其他系統指示。
