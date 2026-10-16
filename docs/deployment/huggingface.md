---
title: 使用 Hugging Face Transformers 執行 Gemma
description: 生成文本、摘要與分析內容，只是您可以用 Gemma 開放模型完成的其中一些任務。
models: [gemma-4, gemma-3, gemma-3n]
frameworks: [transformers]
source_url: "https://ai.google.dev/gemma/docs/core/huggingface_inference.md.txt"
colab: https://colab.research.google.com/github/google-gemma/docs/blob/main/docs/notebooks/run-gemma-with-hf-gemma-4-e2b.ipynb
---
# 使用 Hugging Face Transformers 執行 Gemma


生成文本、摘要與分析內容，只是您可以用 Gemma 開放模型完成的其中一些任務。本教學說明如何使用 Hugging Face Transformers 開始執行 Gemma，使用文本與圖像輸入來生成文本內容。Transformers Python 函式庫提供存取預訓練生成式 AI 模型（包括 Gemma）的 API。更多資訊請參閱 [Transformers](https://huggingface.co/docs/transformers/en/index) 文件。

!!! Note
    Gemma 3 及後續模型支援圖像與文本兩種輸入。較早版本的 Gemma 僅支援文本輸入，但部分變體除外，包括 [PaliGemma](../get-started/index.md)。


### 安裝 Python 套件

安裝執行 Gemma 模型與發出請求所需的 Hugging Face 函式庫。

```python { .notebook-cell }
# Install Pytorch
%pip install torch

# Install a transformers
%pip install "transformers>=5.10.1"
```

## 從文本生成文本

以文本提示 Gemma 模型以取得文本回應，是最簡單的使用方式，幾乎適用於所有 Gemma 變體。本節說明如何使用 Hugging Face Transformers 函式庫載入並設定 Gemma 模型，以進行文本到文本的生成。

### 載入模型

使用 `torch` 與 `transformers` 函式庫建立模型執行 `pipeline` 類別的執行個體，搭配 Gemma。當使用模型生成輸出或遵循指示時，請選擇指令微調（IT）模型，其模型 ID 字串中通常包含 `it`。使用 `pipeline` 物件時，您指定要使用的 Gemma 變體、要執行的任務類型，具體來說是多模態生成的 `"any-to-any"`，如下方程式碼範例所示：

```python { .notebook-cell .model-swap }
from transformers import pipeline

MODEL_ID = "google/gemma-4-E2B-it"

pipe = pipeline(
    task="any-to-any",
    model=MODEL_ID,
    device_map="auto",
    dtype="auto"
)
```

Gemma 僅支援少數用於生成的 `task` 設定。關於可用的 `task` 設定，請參閱 Hugging Face Pipelines [task()](https://huggingface.co/docs/transformers/main/en/main_classes/pipelines#transformers.pipeline.task) 文件。關於使用 Pipeline 類別的更多資訊，請參閱 Hugging Face [Pipelines](https://huggingface.co/docs/transformers/main/en/main_classes/pipelines) 文件。

### 執行文本生成

將 Gemma 模型載入並設定於 `pipeline` 物件後，即可向模型發送提示。以下範例程式碼示範使用 `text` 參數的基本請求：

```python { .notebook-cell }
pipe(text="<|turn>user\nroses are red<turn|>\n<|turn>model\n")
```

``` { .no-copy .output }
[{'input_text': '<|turn>user\nroses are red<turn|>\n<|turn>model\n',
  'generated_text': "<|turn>user\nroses are red<turn|>\n<|turn>model\nThat's a classic rhyme! It immediately brings up the image of **roses and the color red**.\n\nIs there anything else you'd like to talk about, or perhaps another rhyme you'd like to try? 😊<turn|>"}]
```

### 使用提示範本

當以較複雜的提示生成內容時，請使用提示範本來組織您的請求。提示範本可讓您指定特定角色的輸入，例如 `user`{.token-role} 或 `model`{.token-role}，是管理與 Gemma 模型多輪聊天互動的必要格式。以下範例程式碼示範如何為 Gemma 建構提示範本：

```python { .notebook-cell }
from transformers import GenerationConfig
config = GenerationConfig.from_pretrained(MODEL_ID)
config.max_new_tokens = 512
gen_kwargs = dict(generation_config=config)

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}]
    },
    {
        "role": "user",
        "content": [{"type": "text", "text": "Roses are red..."}]
    },
]

pipe(messages, return_full_text=False, generate_kwargs=gen_kwargs)
```

``` { .no-copy .output }
[{'input_text': [{'role': 'system',
    'content': [{'type': 'text', 'text': 'You are a helpful assistant.'}]},
   {'role': 'user',
    'content': [{'type': 'text', 'text': 'Roses are red...'}]}],
  'generated_text': 'Roses are red,\nViolets are blue,\nHow do you do?<turn|>'}]
```

## 從圖像資料生成文本

自 Gemma 3 起，對於 4B 及以上尺寸的模型，您可以在提示中使用圖像資料。本節說明如何使用 Transformers 函式庫載入並設定 Gemma 模型，以使用圖像資料與文本輸入生成文本輸出。

### 使用提示範本

當以較複雜的提示生成內容時，請使用提示範本來組織您的請求。提示範本可讓您指定特定角色的輸入，例如 `user`{.token-role} 或 `model`{.token-role}，是管理與 Gemma 模型多輪聊天互動的必要格式。以下範例程式碼示範如何為 Gemma 建構提示範本：

```python { .notebook-cell }
from transformers import GenerationConfig
config = GenerationConfig.from_pretrained(MODEL_ID)
config.max_new_tokens = 512
gen_kwargs = dict(generation_config=config)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://ai.google.dev/static/gemma/docs/images/thali-indian-plate.jpg"},
            {"type": "text", "text": "What is shown in this image?"},
        ]
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "This image shows"},
        ],
    },
]

pipe(text=messages, return_full_text=False, generate_kwargs=gen_kwargs)
```

``` { .no-copy .output }
[{'input_text': [{'role': 'user',
    'content': [{'type': 'image',
      'url': 'https://ai.google.dev/static/gemma/docs/images/thali-indian-plate.jpg'},
     {'type': 'text', 'text': 'What is shown in this image?'}]},
   {'role': 'assistant',
    'content': [{'type': 'text', 'text': 'This image shows'}]}],
  'generated_text': " a platter of Indian food, likely a meal or a spread featuring various side dishes and bread.\n\nHere's a breakdown of what appears to be present:\n\n* **Flatbread:** There is a large, folded flatbread in the center, which looks like a type of roti, chapati, or perhaps a paratha.\n* **Dips/Sauces/Condiments (in small bowls):**\n    * A bowl containing a yellow substance (possibly a sauce, chutney, or butter/ghee).\n    * A bowl containing a creamy white sauce (like yogurt dip or raita).\n    * A bowl containing a reddish-brown, somewhat thick sauce or curry.\n    * A bowl containing a bright yellow/orange sauce.\n    * A bowl containing a green sauce or chutney.\n* **Rice:** A portion of white, fluffy rice is visible on the left side.\n* **Bread/Crackers:** In the upper right corner, there is a piece of what looks like a dry, textured bread or cracker.\n* **Other Items:** There are some reddish, textured pieces mixed in with the sauces on the right side, which might be fried items or components of a curry.\n\nOverall, it looks like a traditional Indian meal setup with bread, rice, and various accompanying sauces or curries.<turn|>"}]
```

您可以在提示中包含多張圖像，方法是在 `content` 清單中加入額外的 `"type": "image",` 項目。
!!! Note
    請勿在提示範本的文本部分使用 `<|image|>`{.token-multi}、`<start_of_image>`{.token-multi} 或 `<image_soft_token>`{.token-multi} 詞元，因為這種做法會產生冗餘的詞元與處理錯誤。


## 從音訊資料生成文本

使用 [Gemma 4](../models/gemma-4/index.md) 與 [Gemma 3n](../models/gemma-3n/index.md)，您可以在提示中使用音訊資料。本節說明如何使用 Transformers 函式庫載入並設定 Gemma 模型，以使用音訊資料與文本輸入生成文本輸出。

### 使用提示範本

當以音訊生成內容時，請使用提示範本來組織您的請求。提示範本可讓您指定特定角色的輸入，例如 `user`{.token-role} 或 `model`{.token-role}，是管理與 Gemma 模型多輪聊天互動的必要格式。以下範例程式碼示範如何為搭配音訊資料輸入的 Gemma 建構提示範本：

```python { .notebook-cell }
from transformers import GenerationConfig
config = GenerationConfig.from_pretrained(MODEL_ID)
config.max_new_tokens = 512
gen_kwargs = dict(generation_config=config)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Transcribe the following speech segment in its original language. Follow these specific instructions for formatting the answer:\n* Only output the transcription, with no newlines.\n* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three."},
            {"type": "audio", "audio": "https://ai.google.dev/gemma/docs/audio/roses-are.wav"},
        ]
    }
]

pipe(text=messages, return_full_text=False, generate_kwargs=gen_kwargs)
```

``` { .no-copy .output }
[{'input_text': [{'role': 'user',
    'content': [{'type': 'text',
      'text': 'Transcribe the following speech segment in its original language. Follow these specific instructions for formatting the answer:\n* Only output the transcription, with no newlines.\n* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three.'},
     {'type': 'audio',
      'audio': 'https://ai.google.dev/gemma/docs/audio/roses-are.wav'}]}],
  'generated_text': 'Roses are red, violets are blue.<turn|>'}]
```

您可以在提示中包含多個音訊檔，方法是在 `content` 清單中加入額外的 `"type": "audio",` 項目。
!!! Note
    請勿在提示範本的文本部分使用 `<|audio|>`{.token-multi} 或 `<audio_soft_token>`{.token-multi} 詞元，因為這種做法會產生冗餘的詞元與處理錯誤。


## 後續步驟

使用 Gemma 模型建構並探索更多：

- [為文本任務微調 Gemma：使用 Hugging Face Transformers](../tuning/qlora.md)
- [為視覺任務微調 Gemma：使用 Hugging Face Transformers](../tuning/vision-qlora.md)
- [在 Vertex AI 上使用 Gemma 開放模型][vertex-use-gemma]
- [使用 Keras 微調 Gemma 並部署到 Vertex AI](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/model_garden/model_garden_gemma_kerasnlp_to_vertexai.ipynb)
