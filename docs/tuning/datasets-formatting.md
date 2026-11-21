---
title: 資料集格式
seo_title: Gemma 4 微調資料集格式
description: 為 Gemma 4 格式化單回合、多回合、思考與工具呼叫資料集，並使用僅計算完成部分的損失遮罩與思考內容剔除。
models: [gemma-4]
---

# 資料集格式

要微調模型，讓它遵循指令、逐步推理或呼叫工具，必須用與推論時相同的控制詞元來格式化訓練範例。同時也必須遮罩提示詞元，讓模型只對自己的回應計算損失。

本指南說明 Gemma 4 如何為監督式微調建構對話、推理軌跡、多回合歷史記錄與工具呼叫。

---

## 1. Gemma 4 聊天範本

在 Gemma 4 中，對話裡的每則訊息（系統、使用者與助理）會在詞元化之前包在回合詞元中：

![結構化對話訊息如何對應到 Gemma 4 控制詞元](../assets/tuning/data_01_chat_turn_anatomy.svg)

每個回合都以 `<|turn>`{.token-turn} 開頭，緊接著是角色與換行，並以 `<turn|>`{.token-turn} 結尾。請注意，標準對話資料集中的助理角色會對應到 `model`{.token-role} 詞元。

以下是 Gemma 4 使用的主要控制詞元：

| 控制詞元 | 用途 |
| :--- | :--- |
| <code class="token-turn">&lt;&#124;turn&gt;</code> ... <code class="token-turn">&lt;turn&#124;&gt;</code> | 開啟與關閉每個對話回合。 |
| `system`{.token-role}、`user`{.token-role}、`model`{.token-role} | 緊接在 <code class="token-turn">&lt;&#124;turn&gt;</code> 之後宣告說話者角色。 |
| <code class="token-think">&lt;&#124;think&#124;&gt;</code> | 啟用推理模式。 |
| <code class="token-think">&lt;&#124;channel&gt;thought</code> ... <code class="token-think">&lt;channel&#124;&gt;</code> | 包住模型的內部推理軌跡。 |
| <code class="token-tool">&lt;&#124;tool&gt;</code> ... <code class="token-tool">&lt;tool&#124;&gt;</code> | 包住提供給模型的函式定義。 |
| <code class="token-tool">&lt;&#124;tool_call&gt;</code> ... <code class="token-tool">&lt;tool_call&#124;&gt;</code> | 包住模型產生的函式呼叫。 |
| <code class="token-tool">&lt;&#124;tool_response&gt;</code> ... <code class="token-tool">&lt;tool_response&#124;&gt;</code> | 將工具的輸出回傳給模型。 |
| <code class="token-tool">&lt;&#124;"&#124;&gt;</code> | 分隔結構化工具區塊內的字串值。 |

關於這些控制詞元的更多資訊，請見[提示詞格式](../get-started/prompt-formatting.md)。

---

## 2. 遮罩提示詞元 <a id="2-遮罩提示詞元"></a>


在語言建模中，損失是對序列中的每個詞元計算的。這對使用原始文件的持續預訓練很有效。但在監督式微調中，每個範例都以系統提示、工具定義與使用者問題開頭，情況就不同了。

如果不做遮罩就對整個序列計算損失，模型會把大量訓練花在記住提示上：

![全序列損失會對提示詞元更新權重](../assets/tuning/data_02a_naive_full_sequence_loss.svg)

在上述範例中，大多數更新都在訓練模型預測使用者的問題，而非 SQL 答案。當資料集包含很長的系統提示或檢索到的文件時，未遮罩的提示詞元可能占損失的 90% 以上。

修正方法是使用**僅計算完成部分的損失遮罩**，它會遮罩答案詞元以外的所有詞元：

![僅計算完成部分的損失遮罩會在計算損失時忽略提示詞元](../assets/tuning/data_02b_completion_only_masking.svg)

提示詞元仍會經過前向傳播，讓模型能關注完整的問題，但在計算損失時會被忽略。現在 100% 的權重更新都集中在模型的答案與結尾回合詞元上。在 Hugging Face TRL 中，把文字範例整理成 `prompt` 與 `completion` 鍵，就會自動遮罩提示；而視覺與音訊任務則使用小型自訂 collator，額外遮罩影像與音訊軟詞元。

---

## 3. 使用思考詞元訓練 <a id="3-使用思考詞元訓練"></a>

Gemma 4 模型可以在寫出最終答案之前，於 `<|channel>thought`{.token-think} ... `<channel|>`{.token-think} 區塊內進行推理。準備 SFT 資料集時，你可以把範例格式化為**思考模式**或**非思考模式**：

![比較思考模式與非思考模式的詞元配置](../assets/tuning/data_03_thinking_channel_anatomy.svg)

兩種模式的運作方式如下：

1. **思考模式**：系統回合包含 `<|think|>`{.token-think}，模型回合則以包含推理步驟的思考區塊開頭，再寫出最終答案。
2. **非思考模式**：當系統回合省略 `<|think|>`{.token-think} 時，**E2B** 與 **E4B** 會省略所有與思考相關的詞元，而 **12B**、**26B A4B** 與 **31B** 會在最終答案之前放一個空的思考區塊。

如果你只用沒有思考區塊的直接答案進行微調，即使提示模型推理，模型也會逐漸停止使用思考通道。

要在監督式微調期間保持推理能力：

- **混合推理與直接範例**：在訓練資料集中，同時納入帶有思考區塊的範例與直接作答的範例。
- **當資料集只有最終答案時，產生推理軌跡**：如果資料集只有問題與標準答案，請用開啟思考功能的 **Gemma 4 31B** 對訓練問題產生回應，保留最終答案與標準答案標籤相符的回應，並把這些產生的思考區塊納入訓練資料。

---

## 4. 多回合對話

在多回合對話中，模型可能在**第 1 回合**寫下數百個推理詞元，之後才給出簡短的最終答案。

當**第 2 回合**開始時，Gemma 4 會捨棄第 1 回合的思考區塊，在對話歷史記錄中只保留最終答案：

![在多回合對話中，第 1 回合與第 2 回合之間捨棄先前的思考區塊](../assets/tuning/data_04_multiturn_thought_stripping.svg)

流程如下：

1. **第 1 回合**：使用者問「列出所有訂單」，模型產生一個思考區塊，接著給出最終答案（`SELECT * FROM orders;`）。
2. **第 2 回合**：只有使用者提示與第 1 回合的最終答案會帶入第 2 回合的提示歷史記錄（並從損失中遮罩）。第 1 回合的思考區塊會被捨棄，而第 2 回合的新模型回應則保留其思考區塊以供訓練。

捨棄先前的思考內容可以讓多回合上下文視窗保持簡短，也符合推論引擎在執行期間格式化對話歷史記錄的方式。


!!! note "呼叫工具時的思考內容"
    有一個例外值得注意，那就是工具呼叫。在同一個多步驟回合內，工具呼叫前後產生的思考內容都應該保留。這有助於模型追蹤呼叫各個工具的原因。

---

## 5. 工具呼叫資料集

Gemma 4 使用專用的控制詞元，在對話內格式化函式定義、工具呼叫與工具輸出。

以下是工具呼叫軌跡在訓練範例內的結構：

<pre><code><span class="token-turn">&lt;|turn&gt;</span><span class="token-role">system</span>
<span class="token-think">&lt;|think|&gt;</span>
<span class="token-tool">&lt;|tool&gt;</span>declaration:get_stock{description:<span class="token-tool">&lt;|"|&gt;</span>Look up available units for a product SKU.<span class="token-tool">&lt;|"|&gt;</span>,parameters:{properties:{sku:{description:<span class="token-tool">&lt;|"|&gt;</span>Product SKU<span class="token-tool">&lt;|"|&gt;</span>,type:<span class="token-tool">&lt;|"|&gt;</span>STRING<span class="token-tool">&lt;|"|&gt;</span>}},required:[<span class="token-tool">&lt;|"|&gt;</span>sku<span class="token-tool">&lt;|"|&gt;</span>],type:<span class="token-tool">&lt;|"|&gt;</span>OBJECT<span class="token-tool">&lt;|"|&gt;</span>}}<span class="token-tool">&lt;tool|&gt;</span><span class="token-turn">&lt;turn|&gt;</span>
<span class="token-turn">&lt;|turn&gt;</span><span class="token-role">user</span>
How many units of SKU-9042 are in stock?<span class="token-turn">&lt;turn|&gt;</span>
<span class="token-turn">&lt;|turn&gt;</span><span class="token-role">model</span>
<span class="token-think">&lt;|channel&gt;thought</span>
I need to check the inventory for SKU-9042.
<span class="token-think">&lt;channel|&gt;</span><span class="token-tool">&lt;|tool_call&gt;</span>call:get_stock{sku:<span class="token-tool">&lt;|"|&gt;</span>SKU-9042<span class="token-tool">&lt;|"|&gt;</span>}<span class="token-tool">&lt;tool_call|&gt;</span><span class="token-tool">&lt;|tool_response&gt;</span>response:get_stock{value:<span class="token-tool">&lt;|"|&gt;</span>{"available_units": 142}<span class="token-tool">&lt;|"|&gt;</span>}<span class="token-tool">&lt;tool_response|&gt;</span><span class="token-think">&lt;|channel&gt;thought</span>
The tool returned 142 units.
<span class="token-think">&lt;channel|&gt;</span>There are 142 units of SKU-9042 in stock.<span class="token-turn">&lt;turn|&gt;</span></code></pre>

請注意，工具呼叫、工具回應與後續答案會合併為單一模型回合。為監督式微調遮罩工具呼叫軌跡時，請遮罩系統回合、使用者回合與工具回應，讓模型只對自己的思考內容、工具呼叫與最終答案計算損失。

## 接下來

<div class="card-grid">
<a class="card" href="qlora.md">
  <span class="card-icon">{{ icon('lucide/zap') }}</span>
  <div class="card-title">(Q)LoRA</div>
  <p class="card-desc">在單張 GPU 上使用 TRL 與 PEFT，於格式化好的資料集上訓練轉接器。</p>
</a>
<a class="card" href="full-finetune.md">
  <span class="card-icon">{{ icon('lucide/cpu') }}</span>
  <div class="card-title">完整微調</div>
  <p class="card-desc">以 bfloat16 更新 100% 的 Gemma 4 權重，並使用僅計算完成部分的損失遮罩。</p>
</a>
<a class="card" href="preference-alignment.md">
  <span class="card-icon">{{ icon('lucide/sparkles') }}</span>
  <div class="card-title">偏好最佳化</div>
  <p class="card-desc">為 DPO、ORPO 與 KTO 格式化選定與拒絕的回應配對。</p>
</a>
<a class="card" href="reinforcement-learning.md">
  <span class="card-icon">{{ icon('lucide/brain') }}</span>
  <div class="card-title">可驗證獎勵</div>
  <p class="card-desc">使用 GRPO 與可驗證獎勵函式訓練思考鏈。</p>
</a>
</div>

--8<-- "glossary.md"
