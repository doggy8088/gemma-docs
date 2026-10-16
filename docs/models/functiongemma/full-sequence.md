---
title: FunctionGemma 完整函式呼叫序列
description: "FunctionGemma 完整的工具呼叫循環：模型的呼叫、開發者的執行，以及用工具輸出建構的最終回覆。"
models: [functiongemma]
frameworks: [transformers]
colab: https://colab.research.google.com/github/google-gemma/docs/blob/main/docs/notebooks/functiongemma-full-sequence.ipynb
source_url: "https://ai.google.dev/gemma/docs/functiongemma/full-function-calling-sequence-with-functiongemma.md.txt"
---
# FunctionGemma 完整函式呼叫序列


FunctionGemma 是 Gemma 3 270M 模型的專門版本，專為函式呼叫的改進而訓練。它的架構與 Gemma 相同，但使用不同的聊天格式與分詞器。

本指南示範在 Hugging Face 生態系中使用 FunctionGemma 的完整工作流程。涵蓋必要的設定步驟，包括安裝 `torch` 與 `transformers` 等 Python 套件，以及透過 Hugging Face Hub 載入模型。本教學的核心示範了連接模型與外部工具的三階段循環：**模型的回合**生成函式呼叫物件、**開發者的回合**解析並執行程式碼（例如天氣 API），以及**最終回覆**——模型用工具的輸出回答使用者。

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

After you have accepted the license, you need a valid Hugging Face Token to access the model.

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

## 範例使用場景

函式呼叫把 Gemma 的生成能力與外部資料及服務連接起來。以下是一些常見應用：

- **用即時資料回答問題：**使用搜尋引擎或天氣 API 回答「東京的天氣如何？」或「最近的 F1 比賽誰贏了？」這類問題。
- **控制外部系統：**把 Gemma 連接到其他應用程式來執行動作，例如發送電子郵件（「提醒團隊開會」）、管理行事曆，或控制智慧家庭裝置。
- **建立複雜工作流程**：把多個工具呼叫串連起來完成多步驟任務，例如找航班、訂飯店、建立行事曆事件來規劃旅行。

## 使用工具

函式呼叫的核心是一個四步驟流程：

1. **定義工具**：建立模型可以使用的函式，指定引數與描述（例如天氣查詢函式）。
2. **模型的回合**：FunctionGemma 收到使用者的提示與可用工具清單。它生成一個特殊的物件，指出要呼叫哪個函式以及用什麼引數，而不是純文字回覆。
3. **開發者的回合**：您的程式碼收到這個物件，執行指定的函式與提供的引數，並把結果格式化後送回模型。
4. **最終回覆**：FunctionGemma 用函式的輸出生成最終、面向使用者的回覆。

讓我們模擬這個流程。

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
```

### 模型的回合

以下是使用者提示 `"Hey, what's the weather in Tokyo right now?"` 與工具 `[get_current_weather]`。FunctionGemma 生成的函式呼叫物件如下。

```python { .notebook-cell }
prompt = "Hey, what's the weather in Tokyo right now?"
tools = [get_current_weather]

message = [
        # ESSENTIAL SYSTEM PROMPT:
        # This line activates the model's function calling logic.
        {"role": "developer", "content": "You are a model that can do function calling with the following functions"},
        {"role": "user", "content": prompt},
]

inputs = processor.apply_chat_template(message, tools=tools, add_generation_prompt=True, return_dict=True, return_tensors="pt")
output = processor.decode(inputs["input_ids"][0], skip_special_tokens=False)

out = model.generate(**inputs.to(model.device), pad_token_id=processor.eos_token_id, max_new_tokens=128)
generated_tokens = out[0][len(inputs["input_ids"][0]):]
output = processor.decode(generated_tokens, skip_special_tokens=True)

print(f"Prompt: {prompt}")
print(f"Tools: {tools}")
print(f"Output: {output}")
```

``` { .no-copy .output }
Prompt: Hey, what's the weather in Tokyo right now?
Tools: [<function get_current_weather at 0x79b7e0f52e80>]
Output: <start_function_call>call:get_current_weather{location:<escape>Tokyo, Japan<escape>}<end_function_call>
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

### 開發者的回合

您的應用程式應解析模型的回覆來擷取函式名稱與引數，並以 `tool` 角色附加函式呼叫結果。
!!! Note
    執行前務必驗證函式名稱與引數。
    

```python { .notebook-cell }
import re

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
            for k, v1, v2 in re.findall(r"(\w+):(?:<escape>(.*?)<escape>|([^,}]*))", args)
        }
    } for name, args in re.findall(r"<start_function_call>call:(\w+)\{(.*?)\}<end_function_call>", text, re.DOTALL)]

calls = extract_tool_calls(output)
if calls:
    message.append({
        "role": "assistant",
        "tool_calls": [{"type": "function", "function": call} for call in calls]
    })
    print(message[-1])

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
        "role": "tool",
        "content": results
    })
    print(message[-1])
```

``` { .no-copy .output }
{'role': 'assistant', 'tool_calls': [{'type': 'function', 'function': {'name': 'get_current_weather', 'arguments': {'location': 'Tokyo, Japan'} } }]}
{'role': 'tool', 'content': [{'name': 'get_current_weather', 'response': {'temperature': 15, 'weather': 'sunny'} }]}
```
!!! Note
    為獲得最佳結果，請使用以下特定格式將工具執行結果附加到訊息歷史中。這能確保聊天範本正確生成所需的詞元結構（例如 `response:get_current_weather{temperature:15,weather:<escape>sunny<escape>}`）。
    

```python
message.append({
    "role": "tool",
    "content": {
        "name": function_name,
        "response": function_response
    }
})
```

如果是多個獨立請求的情況：

```python
message.append({
    "role": "tool",
    "content": [
        {
            "name": function_name_1,
            "response": function_response_1
        },
        {
            "name": function_name_2,
            "response": function_response_2
        }
    ]
})
```

### 最終回覆

最後，FunctionGemma 讀取工具回覆並回覆使用者。

```python { .notebook-cell }
inputs = processor.apply_chat_template(message, tools=tools, add_generation_prompt=True, return_dict=True, return_tensors="pt")
out = model.generate(**inputs.to(model.device), pad_token_id=processor.eos_token_id, max_new_tokens=128)
generated_tokens = out[0][len(inputs["input_ids"][0]):]
output = processor.decode(generated_tokens, skip_special_tokens=True)
print(f"Output: {output}")
message.append({"role": "assistant", "content": output})
```

``` { .no-copy .output }
Output: The current weather in Tokyo is sunny with a temperature of 15 degrees Celsius.
```

您可以在下方看到完整的聊天歷史。

```python { .notebook-cell }
# full history
for item in message:
  print(item)

print("-"*80)
output = processor.decode(out[0], skip_special_tokens=False)
print(f"Output: {output}")
```

``` { .no-copy .output }
{'role': 'developer', 'content': 'You are a model that can do function calling with the following functions'}
{'role': 'user', 'content': "Hey, what's the weather in Tokyo right now?"}
{'role': 'assistant', 'tool_calls': [{'type': 'function', 'function': {'name': 'get_current_weather', 'arguments': {'location': 'Tokyo, Japan'} } }]}
{'role': 'tool', 'content': [{'name': 'get_current_weather', 'response': {'temperature': 15, 'weather': 'sunny'} }]}
{'role': 'assistant', 'content': 'The current weather in Tokyo is sunny with a temperature of 15 degrees Celsius.'}
--------------------------------------------------------------------------------
Output: <bos><start_of_turn>developer
You are a model that can do function calling with the following functions<start_function_declaration>declaration:get_current_weather{description:<escape>Gets the current weather in a given location.<escape>,parameters:{properties:{location:{description:<escape>The city and state, e.g. "San Francisco, CA" or "Tokyo, JP"<escape>,type:<escape>STRING<escape>},unit:{description:<escape>The unit to return the temperature in.<escape>,enum:[<escape>celsius<escape>,<escape>fahrenheit<escape>],type:<escape>STRING<escape>} },required:[<escape>location<escape>],type:<escape>OBJECT<escape>} }<end_function_declaration><end_of_turn>
<start_of_turn>user
Hey, what's the weather in Tokyo right now?<end_of_turn>
<start_of_turn>model
<start_function_call>call:get_current_weather{location:<escape>Tokyo, Japan<escape>}<end_function_call><start_function_response>response:get_current_weather{temperature:15,weather:<escape>sunny<escape>}<end_function_response>The current weather in Tokyo is sunny with a temperature of 15 degrees Celsius.<end_of_turn>
```

## 摘要

您已經建立了如何用 FunctionGemma 打造能呼叫函式的應用程式。這個工作流程透過四階段循環建立：

1. **定義工具**：建立模型可以使用的函式，指定引數與描述（例如天氣查詢函式）。
2. **模型的回合**：模型收到使用者的提示與可用工具清單，回傳結構化的函式呼叫物件而非純文字。
3. **開發者的回合**：開發者用正規表示式解析這個輸出，擷取函式名稱與引數，執行實際的 Python 程式碼，並以特定的 tool 角色把結果附加到聊天歷史中。
4. **最終回覆**：模型處理工具的執行結果，為使用者生成最終的自然語言答案。
## 接下來

<div class="card-grid">
<a class="card" href="fine-tuning.md">
  <span class="card-icon">{{ icon('lucide/graduation-cap') }}</span>
  <div class="card-title">微調 FunctionGemma</div>
  <p class="card-desc">教它依您的政策挑選正確的工具。</p>
</a>
<a class="card" href="../../deployment/index.md">
  <span class="card-icon">{{ icon('lucide/play') }}</span>
  <div class="card-title">執行 Gemma</div>
  <p class="card-desc">從筆電到叢集，每種能跑 Gemma 的引擎。</p>
</a>
</div>
