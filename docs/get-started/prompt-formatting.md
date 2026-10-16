---
title: 提示格式
description: Gemma 4 用於建構提示的控制詞元，以及 Gemma 1 至 3 使用的舊版 start_of_turn 格式。
models: [gemma-4]
---


# 提示格式

Gemma 4 引入了一組新的控制詞元。Gemma 1 至 3 則使用較舊的
`<start_of_turn>`{.token-turn} 格式，因為這些模型仍在使用中，相關說明保留於下方。

## Gemma 4

以下章節說明 Gemma 4 使用的控制詞元及其使用情境。請注意，這些控制詞元是保留給我們的
分詞器專用的。



- 表示系統指令的詞元：`system`{.token-role}
- 表示使用者輪次的詞元：`user`{.token-role}
- 表示模型輪次的詞元：`model`{.token-role}
- 表示對話輪次開始的詞元：`<|turn>`{.token-turn}
- 表示對話輪次結束的詞元：`<turn|>`{.token-turn}

以下是一個對話範例：

<pre><code><span class="token-turn">&lt;|turn&gt;</span><span class="token-role">system</span>
You are a helpful assistant.<span class="token-turn">&lt;turn|&gt;</span>
<span class="token-turn">&lt;|turn&gt;</span><span class="token-role">user</span>
Hello.<span class="token-turn">&lt;turn|&gt;</span></code></pre>


### 多模態

| 多模態詞元 | 用途 |
|---|---|
| `<|image>`{.token-multi} `<image|>`{.token-multi} | 表示圖像嵌入 |
| `<|audio>`{.token-multi} `<audio|>`{.token-multi} | 表示音訊嵌入 |
| `<|image|>`{.token-multi} `<|audio|>`{.token-multi} | 特殊佔位符詞元 |

我們使用兩個特殊佔位符詞元（`<|image|>`{.token-multi} 與 `<|audio|>`{.token-multi}）來指定
圖像與音訊詞元的插入位置。在分詞（tokenization）之後，這些詞元會在模型內部被實際的軟嵌入（soft embeddings）取代。

以下是一個對話範例：

<pre><code>prompt = """<span class="token-turn">&lt;|turn&gt;</span><span class="token-role">user</span>
Describe this image: <span class="token-multi">&lt;|image|&gt;</span>

And translate these audio:

a. <span class="token-multi">&lt;|audio|&gt;</span>
b. <span class="token-multi">&lt;|audio|&gt;</span><span class="token-turn">&lt;turn|&gt;</span>
<span class="token-turn">&lt;|turn&gt;</span><span class="token-role">model</span>"""</code></pre>

#### 音訊

若想獲得最佳結果，請使用以下提示結構。您也可以使用基本的 `transcribe the audio` 提示。

- **音訊語音辨識（ASR）**

```text
Transcribe the following speech segment in {LANGUAGE} into {LANGUAGE} text.

Follow these specific instructions for formatting the answer:
*   Only output the transcription, with no newlines.
*   When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three.
```

- **自動語音翻譯（AST）**

```text
Transcribe the following speech segment in {SOURCE_LANGUAGE}, then translate it into {TARGET_LANGUAGE}.
When formatting the answer, first output the transcription in {SOURCE_LANGUAGE}, then one newline, then output the string '{TARGET_LANGUAGE}: ', then the translation in {TARGET_LANGUAGE}.
```

但只要使用 `transcribe the audio` 也應該能運作。

### 智慧代理與推理控制詞元

為了支援智慧代理（agentic）工作流程，Gemma 使用了專門的控制詞元，將內部推理（thinking）與外部動作（函式呼叫）區分開來。這些詞元讓模型能在提供最終回應或與外部工具互動之前，先處理複雜的邏輯。

#### 函式呼叫

Gemma 4 接受過六個特殊詞元的訓練，用於管理「工具使用」生命週期。

| 詞元對 | 用途 |
|---|---|
| `<|tool|>`{.token-tool} `<tool|>`{.token-tool} | 定義工具 |
| `<|tool_call>`{.token-tool} `<tool_call|>`{.token-tool} | 表示模型請求使用工具。 |
| `<|tool_response>`{.token-tool} `<tool_response|>`{.token-tool} | 將工具的執行結果傳回模型。 |

!!! Note
    `<|tool_response>`{.token-tool} 可作為推論引擎額外的停止序列。

**字串值的分隔符：`<|"|>`{.token-tool}**

單一詞元 `<|"|>`{.token-tool} 作為結構化資料區塊內**所有字串值**的分隔符。

- **用途：** 此詞元可確保字串中的任何特殊字元（例如 `{`、`}`、`,` 或引號）都被視為字面文字，而非資料結構底層語法的一部分。
- **用法：** 函式宣告、呼叫與回應中的所有字串字面量都必須使用此詞元包覆（例如：<code>key:<span class="token-tool">&lt;|"|&gt;</span>string value<span class="token-tool">&lt;|"|&gt;</span></code>）。

#### 思考模式

若要啟用思考模式，請在系統指令中加入 `<|think|>`{.token-think} 控制詞元。

| 控制詞元 | 用途 |
|---|---|
| `<|think|>`{.token-think} | 啟用思考模式 |
| `<|channel>`{.token-think} `<channel|>`{.token-think} | 表示模型的內部處理過程。 |

!!! Note
    當思考模式啟用時，`<|channel>`{.token-think} 之後一律接著單字 `thought`{.token-think}。

以下是一個對話範例：

<pre><code><span class="token-turn">&lt;|turn&gt;</span><span class="token-role">system</span>
<span class="token-think">&lt;|think|&gt;</span><span class="token-turn">&lt;turn|&gt;</span>
<span class="token-turn">&lt;|turn&gt;</span><span class="token-role">user</span>
What is the water formula?<span class="token-turn">&lt;turn|&gt;</span>
<span class="token-turn">&lt;|turn&gt;</span><span class="token-role">model</span>
<span class="token-think">&lt;|channel&gt;thought</span>
...
<span class="token-think">&lt;channel|&gt;</span>The most common interpretation of "the water formula" refers...<span class="token-turn">&lt;turn|&gt;</span></code></pre>

思考模式的設計是在對話層級啟用。它應該與其他系統指令（例如工具定義）合併在同一個系統輪次中。

#### 推理與函式呼叫範例

在智慧代理輪次中，模型可能在決定呼叫函式之前先私下「思考」。生命週期遵循以下順序：

1. **使用者提問**：使用者提出一個問題。
2. **內部推理**：模型在思考通道中私下思考。
3. **工具請求**：模型停止生成以請求呼叫工具。
4. **執行與注入**：應用程式執行工具並附加回應。
5. **最終回應**：模型讀取回應並產生最終答案。

以下範例展示模型使用天氣工具：

<pre><code><span class="token-turn">&lt;|turn&gt;</span><span class="token-role">system</span>
<span class="token-think">&lt;|think|&gt;</span>You are a helpful assistant.<span class="token-tool">&lt;|tool&gt;</span>declaration:get_current_temperature{...}<span class="token-tool">&lt;tool|&gt;</span><span class="token-turn">&lt;turn|&gt;</span>
<span class="token-turn">&lt;|turn&gt;</span><span class="token-role">user</span>
What's the temperature in London?<span class="token-turn">&lt;turn|&gt;</span>
<span class="token-turn">&lt;|turn&gt;</span><span class="token-role">model</span>
<span class="token-think">&lt;|channel&gt;thought</span>
...
<span class="token-think">&lt;channel|&gt;</span><span class="token-tool">&lt;|tool_call&gt;</span>call:get_current_temperature{location:<span class="token-tool">&lt;|"|&gt;</span>London<span class="token-tool">&lt;|"|&gt;</span>}<span class="token-tool">&lt;tool_call|&gt;</span><span class="token-tool">&lt;|tool_response&gt;</span></code></pre>

您的應用程式應解析模型的回應以提取函式名稱與引數、執行該函式，然後將 `tool_calls`{.token-tool} 與
`tool_responses`{.token-tool} 以 `assistant`{.token-role} 角色附加到聊天記錄中。

<pre><code><span class="token-turn">&lt;|turn&gt;</span><span class="token-role">model</span>
<span class="token-tool">&lt;|tool_call&gt;</span>call:get_current_weather{location:<span class="token-tool">&lt;|"|&gt;</span>London<span class="token-tool">&lt;|"|&gt;</span>}<span class="token-tool">&lt;tool_call|&gt;</span><span class="token-tool">&lt;|tool_response&gt;</span>response:get_current_weather{temperature:15,weather:<span class="token-tool">&lt;|"|&gt;</span>sunny<span class="token-tool">&lt;|"|&gt;</span>}<span class="token-tool">&lt;tool_response|&gt;</span></code></pre>

最後，Gemma 讀取工具回應並回覆使用者。

<pre><code>The temperature in London is 15 degrees and it is sunny.<span class="token-turn">&lt;turn|&gt;</span></code></pre>

以下是此範例完整的 JSON 聊天記錄：

    [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "What's the temperature in London?"
      },
      {
        "role": "assistant",
        "tool_calls": [
          {
            "function": {
              "name": "get_current_weather",
              "arguments": {
                "location": "London"
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
        "content": "The temperature in London is 15 degrees and it is sunny."
      }
    ]

#### 在輪次之間管理思考上下文

妥善管理模型生成的思考內容，對於在多輪對話中維持效能至關重要。

- **標準多輪對話：** 在將對話記錄傳回模型以進行下一輪之前，必須移除（strip）前一輪中模型生成的思考內容。如果想在對話中途停用思考模式，可以在移除先前思考內容的同時，一併移除 `<|think|>`{.token-think} 詞元。
- **函式呼叫（例外）：** 如果單一模型輪次涉及函式或工具呼叫，則在函式呼叫之間**不得**移除思考內容。

**智慧代理工作流程與長時間執行的任務**

由於原始思考內容會在標準輪次之間被移除，開發長時間執行代理的開發者可能希望保留推理上下文，以避免模型陷入循環推理迴圈。

- **彙總思考內容：** 一種高度推薦的推論技巧是將模型先前的思考內容擷取、彙總後，以標準文字的形式回填到上下文視窗中。
- **格式限制：** 由於 Gemma 4 並未在明確包含原始思考內容的提示下受訓練（上述特定的工具呼叫情境除外），模型對這些注入的思考內容並沒有嚴格或特定的格式要求。您可以彈性地以最適合自身智慧代理架構的方式來格式化彙總後的推理內容。

### 整合注意事項

- **內部狀態：** `<|channel>`{.token-think} 與 `<channel|>`{.token-think} 詞元通常用於思維鏈（Chain-of-Thought, CoT）處理。在標準的面向使用者應用程式中，這些內容通常會對終端使用者隱藏。
- **工具迴圈：** `tool_call`{.token-tool} 與 `tool_response`{.token-tool} 詞元促成模型與您的應用程式環境之間的「握手」。應用程式會攔截 `tool_call`{.token-tool}、執行底層程式碼，並將結果以 `tool_response`{.token-tool} 詞元的形式回饋給模型。
- **模型行為：** 較大的模型（例如 gemma-4-26B-A4B-it、gemma-4-31B-it）即使在明確關閉思考模式的情況下，偶爾仍會產生思考通道。若要在這些邊緣案例中穩定模型行為，請考慮在提示中加入一個空的思考詞元。

### 提示：以不含思考的資料集微調大型模型

使用不含思考內容的資料集微調（fine-tuning）gemma-4-26B-A4B-it 與 gemma-4-31B-it 時，可在訓練提示中加入空的通道以獲得更好的結果：

<pre><code><span class="token-turn">&lt;|turn&gt;</span><span class="token-role">model</span>
<span class="token-think">&lt;|channel&gt;thought</span>
<span class="token-think">&lt;channel|&gt;</span></code></pre>

### 提示：使用系統指令達成適應性思考效率

雖然 Gemma 4 中的「思考」功能官方僅支援開啟或關閉的布林參數，但該模型具有極強的指令遵循能力，讓您可以動態調整其思考行為。

您無需依賴寫死的框架參數來設定「高」或「低」思考量，而是可以使用系統指令（System Instructions, SI）引導模型進入精簡的思考模式。明確指示模型以更有效率或較淺層次的方式思考（我們將此概念稱為「LOW」思考指令），即可達成適應性思考效率。

- **降低成本：** 測試顯示，套用「LOW」思考系統指令可將生成的思考詞元數量減少約 20%。
- **概念驗證：** 由於這種行為是模型可指導性的副產品，而非經專門訓練的行為，因此沒有唯一的「完美」提示。「LOW」指令只是一個概念驗證。
- **客製化：** 我們非常鼓勵開發者嘗試自己的客製化系統指令。您可以微調模型思考過程的深度、長度與風格，以完美平衡特定使用案例的延遲、成本與輸出品質。

## Gemma 1-3

Gemma 指令微調（instruction-tuned, IT）模型是使用特定的 *formatter* 訓練的，會在訓練與推論時為所有指令微調範例加上額外資訊的標註。此格式化器有兩個用途：

1. 標示對話中的角色，例如 *system*、*user* 或 *assistant* 角色。
2. 劃分對話中的輪次，特別是在多輪對話中。

以下說明 Gemma 使用的控制詞元及其使用情境。請注意，這些控制詞元是保留給我們的分詞器專用的。

- 表示使用者輪次的詞元：`user`{.token-role}
- 表示模型輪次的詞元：`model`{.token-role}
- 表示對話輪次開始的詞元：`<start_of_turn>`{.token-turn}
- 表示對話輪次結束的詞元：`<end_of_turn>`{.token-turn}

以下是一個對話範例：

<pre><code><span class="token-turn">&lt;start_of_turn&gt;</span><span class="token-role">user</span>
knock knock<span class="token-turn">&lt;end_of_turn&gt;</span>
<span class="token-turn">&lt;start_of_turn&gt;</span><span class="token-role">model</span>
who is there<span class="token-turn">&lt;end_of_turn&gt;</span>
<span class="token-turn">&lt;start_of_turn&gt;</span><span class="token-role">user</span>
Gemma<span class="token-turn">&lt;end_of_turn&gt;</span>
<span class="token-turn">&lt;start_of_turn&gt;</span><span class="token-role">model</span>
Gemma who?<span class="token-turn">&lt;end_of_turn&gt;</span></code></pre>

詞元 `"<end_of_turn>\n"`{.token-turn} 是輪次分隔符，提示前綴則是
`"<start_of_turn>model\n"`{.token-turn}。這表示，如果您想用「What is Cramer's Rule?」這類問題提示模型，應改為以下列方式餵給模型：

<pre><code><span class="token-turn">&lt;start_of_turn&gt;</span><span class="token-role">user</span>
What is Cramer's Rule?<span class="token-turn">&lt;end_of_turn&gt;</span>
<span class="token-turn">&lt;start_of_turn&gt;</span><span class="token-role">model</span></code></pre>

請注意，如果您想用自己的資料微調預訓練的 Gemma 模型，只要訓練與推論使用情境之間保持一致，您可以使用任何這類控制詞元架構。

### 系統指令

Gemma 的指令微調模型僅設計用於兩種角色：
`user`{.token-role} 與 `model`{.token-role}。因此不支援 `system`{.token-role} 角色或系統輪次。

請不要使用獨立的系統角色，而是直接在最初的使用者提示中提供系統層級的指令。模型的指令遵循能力讓 Gemma 能有效解讀這些指令。例如：

<pre><code><span class="token-turn">&lt;start_of_turn&gt;</span><span class="token-role">user</span>
Only reply like a pirate.

What is the answer to life the universe and everything?<span class="token-turn">&lt;end_of_turn&gt;</span>
<span class="token-turn">&lt;start_of_turn&gt;</span><span class="token-role">model</span>
Arrr, 'tis 42,<span class="token-turn">&lt;end_of_turn&gt;</span></code></pre>
## 後續步驟

<div class="card-grid">
<a class="card" href="capabilities/mtp.md">
  <span class="card-icon">{{ icon('lucide/zap') }}</span>
  <div class="card-title">多詞元預測</div>
  <p class="card-desc">在品質完全相同的前提下，以推測式解碼加快推論。</p>
</a>
<a class="card" href="qat.md">
  <span class="card-icon">{{ icon('lucide/shrink') }}</span>
  <div class="card-title">量化感知訓練</div>
  <p class="card-desc">該下載哪些量化權重，以及原因。</p>
</a>
<a class="card" href="../deployment/index.md">
  <span class="card-icon">{{ icon('lucide/play') }}</span>
  <div class="card-title">執行 Gemma</div>
  <p class="card-desc">從筆電到叢集，所有能執行 Gemma 的推論引擎。</p>
</a>
</div>
