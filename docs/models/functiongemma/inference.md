---
title: 使用 Hugging Face Transformers 進行函式呼叫
description: 使用 Hugging Face Transformers 搭配 FunctionGemma 呼叫函式，以手動 JSON schema 或原始 Python 函式傳遞工具。
models: [functiongemma]
frameworks: [transformers]
colab: https://colab.research.google.com/github/google-gemma/docs/blob/main/docs/notebooks/functiongemma-inference.ipynb
source_url: "https://ai.google.dev/gemma/docs/functiongemma/function-calling-with-hf.md.txt"
---
# 使用 Hugging Face Transformers 進行函式呼叫

FunctionGemma 是 Gemma 3 270M 模型的專門版本，專為函式呼叫的改進而訓練。它的架構與 Gemma 相同，但使用不同的聊天格式與分詞器。

本指南示範在 Hugging Face 生態系中使用 FunctionGemma 的流程。涵蓋必要的設定步驟，例如安裝 `torch` 與 `transformers` 函式庫，以及使用 `AutoProcessor` 與 `AutoModelForCausalLM` 載入模型。此外，本指南說明如何透過手動 JSON schema 或原始 Python 函式將工具傳遞給模型，並建議何時使用手動 schema 來有效處理複雜的自訂物件。

## 設定

開始本教學之前，請完成以下步驟：

- 登入 [Hugging Face](https://huggingface.co/google/functiongemma-270m-it){ data-model="functiongemma" } 並為 FunctionGemma 模型選擇 **Acknowledge license** 以取得存取權。
- 生成 Hugging Face [Access Token](https://huggingface.co/docs/hub/en/security-tokens#how-to-manage-user-access-token) 並加入您的 Colab 環境。


## 安裝 Python 套件

安裝執行 FunctionGemma 模型與發送請求所需的 Hugging Face 函式庫。

```bash { .notebook-cell }
# Install PyTorch & other libraries
pip install torch

# Install the transformers library
pip install transformers
```

接受授權後，您需要有效的 Hugging Face 權杖才能存取模型。

```python { .notebook-cell }
# Login into Hugging Face Hub
from huggingface_hub import login
login()
```

## 載入模型

使用 `torch` 與 `transformers` 函式庫，透過 `AutoProcessor` 與 `AutoModelForCausalLM` 類別建立 `processor` 與 `model` 實例，如以下程式碼範例所示：

```python { .notebook-cell }
from transformers import AutoProcessor, AutoModelForCausalLM

GEMMA_MODEL_ID = "google/functiongemma-270m-it"

processor = AutoProcessor.from_pretrained(GEMMA_MODEL_ID, device_map="auto")
model = AutoModelForCausalLM.from_pretrained(GEMMA_MODEL_ID, dtype="auto", device_map="auto")
```

## 傳遞工具

您可以透過 `apply_chat_template()` 函式的 `tools` 引數將工具傳遞給模型。定義工具的方法有兩種：

- **JSON schema**：您可以手動建構 JSON 字典，定義函式名稱、描述與參數（包括型別與必填欄位）。
- **原始 Python 函式**：您可以直接傳遞實際的 Python 函式。系統會自動解析函式的型別提示、引數與 docstring 來生成所需的 JSON schema。為獲得最佳結果，docstring 應遵循 [Google Python 風格指南](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)。

以下是使用 JSON schema 的範例。

```python { .notebook-cell }
weather_function_schema = {
    "type": "function",
    "function": {
        "name": "get_current_temperature",
        "description": "Gets the current temperature for a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city name, e.g. San Francisco",
                },
            },
            "required": ["location"],
        },
    }
}

message = [
    # ESSENTIAL SYSTEM PROMPT:
    # This line activates the model's function calling logic.
    {
        "role": "developer", "content": "You are a model that can do function calling with the following functions"
    },
    {
        "role": "user", "content": "What's the temperature in London?"
    }
]

inputs = processor.apply_chat_template(message, tools=[weather_function_schema], add_generation_prompt=True, return_dict=True, return_tensors="pt")

out = model.generate(**inputs.to(model.device), pad_token_id=processor.eos_token_id, max_new_tokens=128)
output = processor.decode(out[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)

print(f"Output: {output}")
```

``` { .no-copy .output }
Output: <start_function_call>call:get_current_temperature{location:<escape>London<escape>}<end_function_call>
```
!!! Note
    為確保 FunctionGemma 正確解讀可用工具並生成結構化呼叫而非純文字，**developer** 訊息至關重要。這個特定的系統提示指示模型它有權限與能力執行函式呼叫。
    

```python
message = [
        # ESSENTIAL SYSTEM PROMPT:
        # This line activates the model's function calling logic.
        {"role": "developer", "content": "You are a model that can do function calling with the following functions"},
        {"role": "user", "content": prompt},
]
```

以下是同一個範例使用原始 Python 函式的版本。

```python { .notebook-cell }
def get_current_temperature(location: str):
    """
    Gets the current temperature for a given location.

    Args:
        location: The city name, e.g. San Francisco
    """
    return "15°C"

message = [
    {
        "role": "user", "content": "What's the temperature in London?"
    }
]

inputs = processor.apply_chat_template(message, tools=[weather_function_schema], add_generation_prompt=True, return_dict=True, return_tensors="pt")

out = model.generate(**inputs.to(model.device), pad_token_id=processor.eos_token_id, max_new_tokens=128)
output = processor.decode(out[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)

print(f"Output: {output}")
```

``` { .no-copy .output }
Output: <start_function_call>call:get_current_temperature{location:<escape>London<escape>}<end_function_call>
```

## 重要提醒：自動與手動 Schema

仰賴從 Python 函式自動轉換為 JSON schema 時，生成的輸出在複雜參數方面可能不完全符合預期。

如果函式使用自訂物件（例如 Config 類別）作為引數，自動轉換器可能只會把它描述為通用的「object」，而不說明其內部屬性。

在這些情況下，建議手動定義 JSON schema，確保巢狀屬性（例如 config 物件內的 theme 或 font_size）為模型明確定義。

```python { .notebook-cell }
import json
from transformers.utils import get_json_schema

class Config:
    def __init__(self):
        self.theme = "light"
        self.font_size = 14

def update_config(config: Config):
    """
    Updates the configuration of the system.

    Args:
        config: A Config object

    Returns:
        True if the configuration was successfully updated, False otherwise.
    """

update_config_schema = {
    "type": "function",
    "function": {
        "name": "update_config",
        "description": "Updates the configuration of the system.",
        "parameters": {
            "type": "object",
            "properties": {
                "config": {
                    "type": "object",
                    "description": "A Config object",
                    "properties": {"theme": {"type": "string"}, "font_size": {"type": "number"} },
                    },
                },
            "required": ["config"],
            },
        },
    }

print(f"--- [Automatic] ---")
print(json.dumps(get_json_schema(update_config), indent=2))

print(f"\n--- [Manual Schemas] ---")
print(json.dumps(update_config_schema, indent=2))
```

``` { .no-copy .output }
--- [Automatic] ---
{
  "type": "function",
  "function": {
    "name": "update_config",
    "description": "Updates the configuration of the system.",
    "parameters": {
      "type": "object",
      "properties": {
        "config": {
          "type": "object",
          "description": "A Config object"
        }
      },
      "required": [
        "config"
      ]
    }
  }
}

--- [Manual Schemas] ---
{
  "type": "function",
  "function": {
    "name": "update_config",
    "description": "Updates the configuration of the system.",
    "parameters": {
      "type": "object",
      "properties": {
        "config": {
          "type": "object",
          "description": "A Config object",
          "properties": {
            "theme": {
              "type": "string"
            },
            "font_size": {
              "type": "number"
            }
          }
        }
      },
      "required": [
        "config"
      ]
    }
  }
}
```

## 摘要

現在您已經了解如何使用 FunctionGemma 進行函式呼叫。重點包括：

- **定義工具**：可以用兩種方法定義工具：建立手動 JSON schema，或傳遞原始 Python 函式（系統會解析型別提示與 docstring）。
- **Schema 提醒**：雖然自動轉換對簡單型別有效，但處理複雜的自訂物件時會遇到困難。在這些情況下，需要手動定義 JSON schema，確保巢狀屬性對模型可見。

## 接下來

<div class="card-grid">
<a class="card" href="full-sequence.md">
  <span class="card-icon">{{ icon('lucide/split') }}</span>
  <div class="card-title">完整呼叫序列</div>
  <p class="card-desc">從模型的呼叫到最終答案的完整循環。</p>
</a>
<a class="card" href="fine-tuning.md">
  <span class="card-icon">{{ icon('lucide/graduation-cap') }}</span>
  <div class="card-title">微調 FunctionGemma</div>
  <p class="card-desc">教它依您的政策挑選正確的工具。</p>
</a>
</div>
