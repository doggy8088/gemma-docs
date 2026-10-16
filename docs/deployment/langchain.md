---
title: 開始使用 Gemma 與 LangChain
description: 本教學說明如何在 Google Cloud 或 Colab 環境中開始使用 Gemma 與 LangChain。
models: [gemma-4, gemma-1]
frameworks: [langchain, transformers, google-cloud, keras, jax]
source_url: "https://ai.google.dev/gemma/docs/integrations/langchain.md.txt"
---
# 開始使用 Gemma 與 LangChain


本教學說明如何在 Google Cloud 或 Colab 環境中開始使用 [Gemma](../index.md) 與 [LangChain](https://python.langchain.com/docs/get_started/introduction)。Gemma 是輕量、最先進的開放模型家族，以建立 Gemini 模型的相同研究與技術打造。LangChain 是用於建構與部署由語言模型驅動的情境感知應用程式的框架。

!!! Note
    本教學在 Google Colab 的 A100 GPU 上執行。免費的 Colab 硬體加速*不足以*執行所有程式碼。


## 使用 LangChain 執行 Gemma 4

目前的路徑是 [`langchain-huggingface`](https://python.langchain.com/docs/integrations/chat/huggingface/)，
適用於 Hugging Face 上的任何 Gemma 儲存庫。以下範例遵循本頁的
模型選取器。

```bash
pip install -qU langchain-huggingface transformers accelerate
```

=== "本機管線"

    在本機硬體上執行權重。

    ```python { .model-swap }
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

    llm = HuggingFacePipeline.from_model_id(
        model_id="google/gemma-4-12B-it",
        task="text-generation",
        pipeline_kwargs=dict(
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.03,
        ),
    )

    chat_model = ChatHuggingFace(llm=llm)
    ```

=== "託管端點"

    改為呼叫託管的供應商，因此本機不執行任何東西。

    ```python { .model-swap }
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

    llm = HuggingFaceEndpoint(
        repo_id="google/gemma-4-12B-it",
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
        provider="auto",
    )

    chat_model = ChatHuggingFace(llm=llm)
    ```

然後像其他 LangChain 聊天模型一樣呼叫它：

```python
messages = [
    ("system", "You are a helpful assistant."),
    ("human", "What is the capital of France?"),
]
print(chat_model.invoke(messages).content)
```

!!! note "以下各節涵蓋較早的 Gemma 版本"
    `GemmaLocalKaggle`、`GemmaChatLocalHF` 與 Vertex AI 一鍵部署是為 Gemma 1 撰寫的，
    仍引用其 `gemma_2b_en` 與 `google/gemma-2b` 檢查點。它們被保留下來是因為該部署路徑仍然存在，
    而非因為它是執行 Gemma 4 的建議方式。

## 在 Google Cloud 上執行 Gemma

[`langchain-google-vertexai`](https://pypi.org/project/langchain-google-vertexai/) 套件提供 LangChain 與 Google Cloud 模型的整合。

### 安裝依賴

```bash
pip install --upgrade -q langchain langchain-google-vertexai
```

### 驗證

除非您使用 Colab Enterprise，否則需要進行驗證。

```python
from google.colab import auth
auth.authenticate_user()
```

### 部署模型

Vertex AI 是用於訓練與部署 AI 模型及應用程式的平台。Model Garden 是精選的模型集合，可在 Google Cloud 主控台中探索。

要部署 Gemma，請在 Vertex AI 的 Model Garden 中[開啟模型](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/335)並完成以下步驟：

1. 選取**部署**。
2. 視需要變更部署表單欄位，或保留預設值。記下以下欄位，稍後會用到：
    - **端點名稱**（例如 `google_gemma-7b-it-mg-one-click-deploy`）
    - **區域**（例如 `us-west1`）
3. 選取**部署**以將模型部署到 Vertex AI。部署需要幾分鐘才能完成。

端點就緒後，複製其專案 ID、端點 ID 與位置，並將它們作為參數輸入。

```python
# @title Basic parameters
project: str = ""  # @param {type:"string"}
endpoint_id: str = ""  # @param {type:"string"}
location: str = "" # @param {type:"string"}
```

### 執行模型

```python
from langchain_google_vertexai import GemmaVertexAIModelGarden, GemmaChatVertexAIModelGarden

llm = GemmaVertexAIModelGarden(
    endpoint_id=endpoint_id,
    project=project,
    location=location,
)

output = llm.invoke("What is the meaning of life?")
print(output)
```

```{ .no-copy .output }
Prompt:
What is the meaning of life?
Output:
Life is a complex and multifaceted phenomenon that has fascinated philosophers, scientists, and
```

您也可以將 Gemma 用於多輪聊天：

```python
from langchain_core.messages import (
    HumanMessage
)

llm = GemmaChatVertexAIModelGarden(
    endpoint_id=endpoint_id,
    project=project,
    location=location,
)

message1 = HumanMessage(content="How much is 2+2?")
answer1 = llm.invoke([message1])
print(answer1)

message2 = HumanMessage(content="How much is 3+3?")
answer2 = llm.invoke([message1, answer1, message2])

print(answer2)
```

```{ .no-copy .output }
content='Prompt:\n<start_of_turn>user\nHow much is 2+2?<end_of_turn>\n<start_of_turn>model\nOutput:\nSure, the answer is 4.\n\n2 + 2 = 4'
content='Prompt:\n<start_of_turn>user\nHow much is 2+2?<end_of_turn>\n<start_of_turn>model\nOutput:\nSure, the answer is 4.\n\n2 + 2 = 4<end_of_turn>\n<start_of_turn>user\nHow much is 3+3?<end_of_turn>\n<start_of_turn>model\nOutput:\nSure, the answer is 6.\n\n3 + 3 = 6'
```

您可以對回應進行後處理以避免重複：

```python
answer1 = llm.invoke([message1], parse_response=True)
print(answer1)

answer2 = llm.invoke([message1, answer1, message2], parse_response=True)

print(answer2)
```

```{ .no-copy .output }
content='Output:\nSure, here is the answer:\n\n2 + 2 = 4'
content='Output:\nSure, here is the answer:\n\n3 + 3 = 6<'
```

## 從 Kaggle 下載執行 Gemma

本節說明如何從 Kaggle 下載 Gemma 然後執行模型。

要完成本節，您首先需要完成 [Gemma 設定][legacy-gemma-setup] 的設定說明。

然後繼續下一節，在 Colab 環境中設定環境變數。

!!! Note
    本節的教學在 Google Colab 的 A100 GPU 上執行。


### 設定環境變數

設定 `KAGGLE_USERNAME` 與 `KAGGLE_KEY` 的環境變數。

```python
import os
from google.colab import userdata

# Note: `userdata.get` is a Colab API. If you're not using Colab, set the env
# vars as appropriate for your system.
os.environ["KAGGLE_USERNAME"] = userdata.get('KAGGLE_USERNAME')
os.environ["KAGGLE_KEY"] = userdata.get('KAGGLE_KEY')
```

### 安裝依賴

```bash
# Install Keras 3 last. See https://keras.io/getting_started/ for more details.
pip install -q -U keras-nlp
pip install -q -U keras>=3
```

### 執行模型

```python
from langchain_google_vertexai import GemmaLocalKaggle
```

您可以指定 Keras 後端（預設為 `tensorflow`，但可以改為 `jax` 或 `torch`）。

```python
# @title Basic parameters
keras_backend: str = "jax"  # @param {type:"string"}
model_name: str = "gemma_2b_en" # @param {type:"string"}

llm = GemmaLocalKaggle(model_name=model_name, keras_backend=keras_backend)
```

```python
output = llm.invoke("What is the meaning of life?", max_tokens=30)
print(output)
```

```{ .no-copy .output }
What is the meaning of life?

The question is one of the most important questions in the world.

It’s the question that has
```

### 執行聊天模型

如同上方的 Google Cloud 範例，您可以使用 Gemma 的本機部署進行多輪聊天。您可能需要重新啟動筆記本並清理 GPU 記憶體，以避免 OOM 錯誤：

```python
from langchain_google_vertexai import GemmaChatLocalKaggle

# @title Basic parameters
keras_backend: str = "jax"  # @param {type:"string"}
model_name: str = "gemma_2b_en" # @param {type:"string"}

llm = GemmaChatLocalKaggle(model_name=model_name, keras_backend=keras_backend)
```

```python
from langchain_core.messages import (
    HumanMessage
)

message1 = HumanMessage(content="Hi! Who are you?")
answer1 = llm.invoke([message1], max_tokens=30)
print(answer1)
```

```python
message2 = HumanMessage(content="What can you help me with?")
answer2 = llm.invoke([message1, answer1, message2], max_tokens=60)

print(answer2)
```

如果想避免多輪陳述，可以對回應進行後處理：

```python
answer1 = llm.invoke([message1], max_tokens=30, parse_response=True)
print(answer1)

answer2 = llm.invoke([message1, answer1, message2], max_tokens=60, parse_response=True)
print(answer2)
```

## 從 Hugging Face 下載執行 Gemma

### 設定

如同 Kaggle，Hugging Face 要求您先接受 Gemma 的條款與條件才能存取模型。要透過 Hugging Face 取得 Gemma 存取權，請前往 [Gemma 模型卡](https://huggingface.co/google/gemma-2b)。

您還需要取得具有讀取權限的[使用者存取權杖](https://huggingface.co/docs/hub/en/security-tokens)，可在下方輸入。

!!! Note
    本節的教學在 Google Colab 的 A100 GPU 上執行。


```python
# @title Basic parameters
hf_access_token: str = ""  # @param {type:"string"}
model_name: str = "google/gemma-2b" # @param {type:"string"}
```

### 執行模型

```python
from langchain_google_vertexai import GemmaLocalHF, GemmaChatLocalHF

llm = GemmaLocalHF(model_name="google/gemma-2b", hf_access_token=hf_access_token)
```

```python
output = llm.invoke("What is the meaning of life?", max_tokens=50)
print(output)
```

```{ .no-copy .output }
What is the meaning of life?

The question is one of the most important questions in the world.

It’s the question that has been asked by philosophers, theologians, and scientists for centuries.

And it’s the question that
```

如同上方的範例，您可以使用 Gemma 的本機部署進行多輪聊天。您可能需要重新啟動筆記本並清理 GPU 記憶體，以避免 OOM 錯誤：

### 執行聊天模型

```python
llm = GemmaChatLocalHF(model_name=model_name, hf_access_token=hf_access_token)
```

```python
from langchain_core.messages import (
    HumanMessage
)

message1 = HumanMessage(content="Hi! Who are you?")
answer1 = llm.invoke([message1], max_tokens=60)
print(answer1)
```

```python
message2 = HumanMessage(content="What can you help me with?")
answer2 = llm.invoke([message1, answer1, message2], max_tokens=140)

print(answer2)
```

如同先前的範例，您可以對回應進行後處理：

```python
answer1 = llm.invoke([message1], max_tokens=60, parse_response=True)
print(answer1)

answer2 = llm.invoke([message1, answer1, message2], max_tokens=120, parse_response=True)
print(answer2)
```

## 下一步

- 了解如何[微調 Gemma 模型](../tuning/qlora.md)。
- 了解如何[在 Vertex AI 上使用 Gemma 模型][vertex-use-gemma]。
