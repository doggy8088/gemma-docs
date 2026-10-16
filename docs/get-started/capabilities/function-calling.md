---
title: 工具呼叫
description: 當您使用像 Gemma 這樣的生成式人工智慧（AI）模型時，您可能會希望使用模型來操作程式設計介面，以完成任務或回答問題。
models: [gemma-4]
frameworks: [transformers]
colab: https://colab.research.google.com/github/google-gemma/docs/blob/main/docs/notebooks/function-calling-gemma-4-e2b.ipynb
---
<!-- TODO: Rename this to "tool calling"? -->
# 工具呼叫

當您使用像 Gemma 這樣的生成式人工智慧（AI）模型時，您可能會希望使用模型來操作程式設計介面，以完成任務或回答問題。透過定義程式設計介面來指示模型，再提出使用該介面的請求，這種做法稱為*函式呼叫*（function calling）或*工具呼叫*（tool calling）。

!!! Tip "重要"
    *Gemma 模型無法自行執行程式碼。*使用函式呼叫產生程式碼後，您必須自行執行產生的程式碼，或將其作為應用程式的一部分來執行。在執行前，請一律採取保護措施，驗證所有產生的程式碼。

本指南說明在 Hugging Face 生態系中使用 Gemma 4 的流程。


## 安裝 Python 套件

安裝執行 Gemma 模型與發出請求所需的 Hugging Face 函式庫。

```bash { .notebook-cell }
# Install PyTorch & other libraries
pip install torch accelerate

# Install the transformers library
pip install "transformers>=5.10.1"
```

## 載入模型

使用以下程式碼範例中的 `AutoProcessor` 與 `AutoModelForImageTextToText` 類別，透過 `transformers` 函式庫建立 `processor` 與 `model` 的執行個體：

```python { .notebook-cell .model-swap }
MODEL_ID = "google/gemma-4-E2B-it"

from transformers import AutoProcessor, AutoModelForMultimodalLM

model = AutoModelForMultimodalLM.from_pretrained(MODEL_ID, dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained(MODEL_ID)
```

## 傳入工具

您可以使用 `apply_chat_template()` 函式，透過 `tools` 參數將工具傳入模型。定義這些工具有兩種方法：

- **JSON schema**：您可以手動建構 JSON 字典，定義函式名稱、說明與參數（包含型別與必填欄位）。
- **原始 Python 函式（Raw Python Functions）**：您可以直接傳入實際的 Python 函式。系統會自動解析函式的型別提示、引數與文件字串，產生所需的 JSON schema。為了獲得最佳效果，文件字串應遵循 [Google Python 風格指南](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)。

以下是使用 JSON schema 的範例。

```python { .notebook-cell }
from transformers import TextStreamer

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
    {
        "role": "system", "content": "You are a helpful assistant."
    },
    {
        "role": "user", "content": "What's the temperature in London?"
    }
]

text = processor.apply_chat_template(message, tools=[weather_function_schema], tokenize=False, add_generation_prompt=True)
inputs = processor(text=text, return_tensors="pt").to(model.device)
streamer = TextStreamer(processor)
outputs = model.generate(**inputs, streamer=streamer, max_new_tokens=64)
```

``` { .no-copy .output }
<bos><|turn>system
You are a helpful assistant.<|tool>declaration:get_current_temperature{description:<|"|>Gets the current temperature for a given location.<|"|>,parameters:{properties:{location:{description:<|"|>The city name, e.g. San Francisco<|"|>,type:<|"|>STRING<|"|>} },required:[<|"|>location<|"|>],type:<|"|>OBJECT<|"|>} }<tool|><turn|>
<|turn>user
What's the temperature in London?<turn|>
<|turn>model
<|tool_call>call:get_current_temperature{location:<|"|>London<|"|>}<tool_call|><|tool_response>
```

以下是使用原始 Python 函式的相同範例。

```python { .notebook-cell }
from transformers.utils import get_json_schema

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

text = processor.apply_chat_template(message, tools=[get_json_schema(get_current_temperature)], tokenize=False, add_generation_prompt=True)
inputs = processor(text=text, return_tensors="pt").to(model.device)
streamer = TextStreamer(processor)
outputs = model.generate(**inputs, streamer=streamer, max_new_tokens=256)
```

``` { .no-copy .output }
<bos><|turn>system
<|tool>declaration:get_current_temperature{description:<|"|>Gets the current temperature for a given location.<|"|>,parameters:{properties:{location:{description:<|"|>The city name, e.g. San Francisco<|"|>,type:<|"|>STRING<|"|>} },required:[<|"|>location<|"|>],type:<|"|>OBJECT<|"|>} }<tool|><turn|>
<|turn>user
What's the temperature in London?<turn|>
<|turn>model
<|tool_call>call:get_current_temperature{location:<|"|>London<|"|>}<tool_call|><|tool_response>
```

## 完整的函式呼叫流程

本節示範將模型連接至外部工具的三階段循環：**模型的回合**（Model's Turn）產生函式呼叫物件，**開發者的回合**（Developer's Turn）解析並執行程式碼（例如天氣 API），以及**最終回應**（Final Response）——模型使用工具的輸出回答使用者。

### 模型的回合

以下是使用者提示 `"Hey, what's the weather in Tokyo right now?"` 與工具 `[get_current_weather]`。Gemma 會產生如下的函式呼叫物件。

```python { .notebook-cell }
# Define a function that our model can use.
def get_current_weather(location: str, unit: str = "celsius"):
    """
    Gets the current weather in a given location.

    Args:
        location: The city and state, e.g. "San Francisco, CA" or "Tokyo, JP"
        unit: The unit to return the temperature in. (choices: ["celsius", "fahrenheit"])

    Returns:
        temperature: The current temperature in the given location
        weather: The current weather in the given location
    """
    return {"temperature": 15, "weather": "sunny"}

prompt = "Hey, what's the weather in Tokyo right now?"
tools = [get_current_weather]

message = [
    {
        "role": "system", "content": "You are a helpful assistant."
    },
    {
        "role": "user", "content": prompt
    },
]

text = processor.apply_chat_template(message, tools=tools, tokenize=False, add_generation_prompt=True)
inputs = processor(text=text, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=128)
generated_tokens = out[0][len(inputs["input_ids"][0]):]
output = processor.decode(generated_tokens, skip_special_tokens=False)

print(f"Prompt: {prompt}")
print(f"Tools: {tools}")
print(f"Output: {output}")
```

``` { .no-copy .output }
Prompt: Hey, what's the weather in Tokyo right now?
Tools: [<function get_current_weather at 0x7bbe91d18180>]
Output: <|tool_call>call:get_current_weather{location:<|"|>Tokyo, JP<|"|>}<tool_call|><|tool_response>
```
### 開發者的回合

您的應用程式應解析模型的回應，擷取函式名稱與引數，並以 `assistant`{.token-role} 角色附加 `tool_calls`{.token-tool} 與 `tool_responses`{.token-tool}。
!!! Note
    執行前請一律驗證函式名稱與引數。

```python { .notebook-cell }
import re
import json

def extract_tool_calls(text):
    def cast(v):
        try: return int(v)
        except:
            try: return float(v)
            except: return {'true': True, 'false': False}.get(v.lower(), v.strip("'\""))

    return [{
        "name": name,
        "arguments": {
            k: cast((v1 or v2).strip())
            for k, v1, v2 in re.findall(r'(\w+):(?:<\|"\|>(.*?)<\|"\|>|([^,}]*))', args)
        }
    } for name, args in re.findall(r"<\|tool_call>call:(\w+)\{(.*?)\}<tool_call\|>", text, re.DOTALL)]

calls = extract_tool_calls(output)
if calls:
    # Call the function and get the result
    #####################################
    # WARNING: This is a demonstration. #
    #####################################
    # Using globals() to call functions dynamically can be dangerous in
    # production. In a real application, you should implement a secure way to
    # map function names to actual function calls, such as a predefined
    # dictionary of allowed tools and their implementations.
    results = [
        {"name": c['name'], "response": globals()[c['name']](**c['arguments'])}
        for c in calls
    ]

    message.append({
        "role": "assistant",
        "tool_calls": [
            {"function": call} for call in calls
        ],
        "tool_responses": results
    })
    print(json.dumps(message[-1], indent=2))
```

``` { .no-copy .output }
{
  "role": "assistant",
  "tool_calls": [
    {
      "function": {
        "name": "get_current_weather",
        "arguments": {
          "location": "Tokyo, JP"
        }
      }
    }
  ],
  "tool_responses": [
    {
      "name": "get_current_weather",
      "response": {
        "temperature": 15,
        "weather": "sunny"
      }
    }
  ]
}
```
!!! Note
    為了獲得最佳效果，請使用以下特定格式，將工具執行結果附加到您的訊息歷史記錄中。這可確保聊天範本正確產生所需的詞元結構（例如 `response:get_current_weather{temperature:15,weather:<|"|>sunny<|"|>}`）。

```python { .notebook-cell }
"tool_responses": [
  {
    "name": function_name,
    "response": function_response
  }
]
```

如有多個獨立請求：

```python { .notebook-cell }
"tool_responses": [
  {
    "name": function_name_1,
    "response": function_response_1
  },
  {
    "name": function_name_2,
    "response": function_response_2
  }
]
```

### 最終回應

最後，Gemma 會讀取工具回應並回覆使用者。

```python { .notebook-cell }
text = processor.apply_chat_template(message, tools=tools, tokenize=False, add_generation_prompt=True)
inputs = processor(text=text, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=128)
generated_tokens = out[0][len(inputs["input_ids"][0]):]
output = processor.decode(generated_tokens, skip_special_tokens=True)
print(f"Output: {output}")
message[-1]["content"] = output
```

``` { .no-copy .output }
Output: The current weather in Tokyo is 15 degrees Celsius and sunny.
```

您可以在下方看到完整的聊天歷史記錄。

```python { .notebook-cell }
# full history
print(json.dumps(message, indent=2))

print("-"*80)
output = processor.decode(out[0], skip_special_tokens=False)
print(f"Output: {output}")
```

``` { .no-copy .output }
[
  {
    "role": "system",
    "content": "You are a helpful assistant."
  },
  {
    "role": "user",
    "content": "Hey, what's the weather in Tokyo right now?"
  },
  {
    "role": "assistant",
    "tool_calls": [
      {
        "function": {
          "name": "get_current_weather",
          "arguments": {
            "location": "Tokyo, JP"
          }
        }
      }
    ],
    "tool_responses": [
      {
        "name": "get_current_weather",
        "response": {
          "temperature": 15,
          "weather": "sunny"
        }
      }
    ],
    "content": "The current weather in Tokyo is 15 degrees Celsius and sunny."
  }
]
--------------------------------------------------------------------------------
Output: <bos><|turn>system
You are a helpful assistant.<|tool>declaration:get_current_weather{description:<|"|>Gets the current weather in a given location.<|"|>,parameters:{properties:{location:{description:<|"|>The city and state, e.g. "San Francisco, CA" or "Tokyo, JP"<|"|>,type:<|"|>STRING<|"|>},unit:{description:<|"|>The unit to return the temperature in.<|"|>,enum:[<|"|>celsius<|"|>,<|"|>fahrenheit<|"|>],type:<|"|>STRING<|"|>} },required:[<|"|>location<|"|>],type:<|"|>OBJECT<|"|>} }<tool|><turn|>
<|turn>user
Hey, what's the weather in Tokyo right now?<turn|>
<|turn>model
<|tool_call>call:get_current_weather{location:<|"|>Tokyo, JP<|"|>}<tool_call|><|tool_response>response:get_current_weather{temperature:15,weather:<|"|>sunny<|"|>}<tool_response|>The current weather in Tokyo is 15 degrees Celsius and sunny.<turn|>
```

### 函式呼叫搭配思考模式

透過利用內部推理流程，模型可大幅提升函式呼叫的準確度，讓模型在何時觸發工具、如何定義參數方面做出更精確的決策。

```python { .notebook-cell }
prompt = "Hey, I'm in Seoul. Is it good for running now?"
message = [
    {
        "role": "system", "content": "You are a helpful assistant."
    },
    {
        "role": "user", "content": prompt
    },
]

text = processor.apply_chat_template(message, tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=True)
inputs = processor(text=text, return_tensors="pt").to(model.device)
input_len = inputs["input_ids"].shape[-1]

out = model.generate(**inputs, max_new_tokens=1024)
output = processor.decode(out[0][input_len:], skip_special_tokens=False)
result = processor.parse_response(output)

for key, value in result.items():
  if key == "role":
    print(f"Role: {value}")
  elif key == "thinking":
    print(f"\n=== Thoughts ===\n{value}")
  elif key == "content":
    print(f"\n=== Answer ===\n{value}")
  elif key == "tool_calls":
    print(f"\n=== Tool Calls ===\n{value}")
  else:
    print(f"\n{key}: {value}...\n")
```

``` { .no-copy .output }
Role: assistant

=== Thoughts ===

1. **Analyze the Request:** The user is asking if it's good for running in Seoul right now.

2. **Identify Necessary Information:** To answer this question, I need current weather information for Seoul.

3. **Examine Available Tools:** The only tool available is `get_current_weather(location: str, unit: str = None)`.

4. **Determine Tool Usage:**
    * The request specifies the location: "Seoul".
    * The request implies needing current weather conditions to assess if it's suitable for running.
    * The `get_current_weather` tool is appropriate for this.

5. **Construct the Tool Call:**
    * `location` should be "Seoul".
    * `unit` is optional, but it's good practice to decide if a specific unit is needed or if the default is fine. Since the user didn't specify a unit, I can omit it or choose a default (though the tool definition doesn't specify a default, just that it's optional). Let's just call it with the location.

6. **Formulate the Response (Tool Call):** Call `get_current_weather` with `location="Seoul"`.

7. **Self-Correction/Refinement:** The tool call is the necessary next step to gather the information needed to answer the user's question. I cannot answer the question without the weather data.

8. **Final Output Generation:** Generate the JSON tool call.

=== Answer ===
<eos>

=== Tool Calls ===
[{'type': 'function', 'function': {'name': 'get_current_weather', 'arguments': {'location': 'Seoul'} } }]
```

處理工具呼叫並取得最終答案。

```python { .notebook-cell }
calls = extract_tool_calls(output)
if calls:
    # Call the function and get the result
    #####################################
    # WARNING: This is a demonstration. #
    #####################################
    # Using globals() to call functions dynamically can be dangerous in
    # production. In a real application, you should implement a secure way to
    # map function names to actual function calls, such as a predefined
    # dictionary of allowed tools and their implementations.
    results = [
        {"name": c['name'], "response": globals()[c['name']](**c['arguments'])}
        for c in calls
    ]

    message.append({
        "role": "assistant",
        "tool_calls": [
            {"function": call} for call in calls
        ],
        "tool_responses": results
    })

text = processor.apply_chat_template(message, tools=tools, tokenize=False, add_generation_prompt=True)
inputs = processor(text=text, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=128)
generated_tokens = out[0][len(inputs["input_ids"][0]):]
output = processor.decode(generated_tokens, skip_special_tokens=True)
print(f"Output: {output}")
message[-1]["content"] = output

print("-"*80)
print("Full History")
print("-"*80)
print(json.dumps(message, indent=2))
```

``` { .no-copy .output }
Output: The current weather in Seoul is 15 degrees Celsius and sunny. That sounds like great weather for running!
--------------------------------------------------------------------------------
Full History
--------------------------------------------------------------------------------
[
  {
    "role": "system",
    "content": "You are a helpful assistant."
  },
  {
    "role": "user",
    "content": "Hey, I'm in Seoul. Is it good for running now?"
  },
  {
    "role": "assistant",
    "tool_calls": [
      {
        "function": {
          "name": "get_current_weather",
          "arguments": {
            "location": "Seoul"
          }
        }
      }
    ],
    "tool_responses": [
      {
        "name": "get_current_weather",
        "response": {
          "temperature": 15,
          "weather": "sunny"
        }
      }
    ],
    "content": "The current weather in Seoul is 15 degrees Celsius and sunny. That sounds like great weather for running!"
  }
]
```

## 重要注意事項：自動 Schema 與手動 Schema

仰賴從 Python 函式自動轉換為 JSON schema 時，產生的輸出可能不完全符合對複雜參數的特定期望。

如果函式使用自訂物件（例如 Config 類別）作為引數，自動轉換器可能只會將其描述為一般的 "object"，而不說明其內部屬性。

在這些情況下，建議手動定義 JSON schema，以確保巢狀屬性（例如設定物件中的 theme 或 font_size）能明確地為模型定義。

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

您已學會如何使用 Gemma 4 建構可呼叫函式的應用程式。工作流程由以下四個階段組成：

1. **定義工具（Define Tools）**：建立模型可使用的函式，並指定引數與說明（例如天氣查詢函式）。
2. **模型的回合（Model's Turn）**：模型接收使用者的提示與可用工具清單，回傳結構化的函式呼叫物件而非純文字。
3. **開發者的回合（Developer's Turn）**：開發者使用正規表示式解析此輸出，擷取函式名稱與引數，執行實際的 Python 程式碼，並使用特定的工具角色將結果附加到聊天歷史記錄中。
4. **最終回應（Final Response）**：模型處理工具的執行結果，為使用者產生最終的自然語言答案。
## 後續步驟

<div class="card-grid">
<a class="card" href="vision.md">
  <span class="card-icon">{{ icon('lucide/eye') }}</span>
  <div class="card-title">視覺總覽</div>
  <p class="card-desc">偵測、OCR、視覺問答與圖像描述。</p>
</a>
<a class="card" href="audio.md">
  <span class="card-icon">{{ icon('lucide/audio-lines') }}</span>
  <div class="card-title">音訊理解</div>
  <p class="card-desc">轉錄、翻譯語音並進行推理。</p>
</a>
<a class="card" href="../../deployment/index.md">
  <span class="card-icon">{{ icon('lucide/play') }}</span>
  <div class="card-title">執行 Gemma</div>
  <p class="card-desc">從筆記型電腦到叢集，執行 Gemma 的各種引擎。</p>
</a>
</div>
