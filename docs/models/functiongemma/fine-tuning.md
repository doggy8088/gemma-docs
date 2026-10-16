---
title: 使用 FunctionGemma 微調
description: 本指南示範如何使用 TRL 與 SFTTrainer，針對工具呼叫與函式呼叫任務微調 FunctionGemma 模型。
models: [functiongemma]
frameworks: [transformers, peft, trl]
colab: https://colab.research.google.com/github/google-gemma/docs/blob/main/docs/notebooks/functiongemma-fine-tuning.ipynb
source_url: "https://ai.google.dev/gemma/docs/functiongemma/finetuning-with-functiongemma.md.txt"
---
# 使用 FunctionGemma 微調


本指南示範如何針對工具呼叫微調 FunctionGemma。

雖然 FunctionGemma 原生就具備呼叫工具的能力，但真正的實力來自兩種不同的技能：使用工具的機械性知識（語法），以及解讀*為何*與*何時*使用它的認知能力（意圖）。

模型（尤其是較小的模型）可用來保留複雜意圖理解的參數較少。這就是為什麼我們需要微調它們。

微調工具呼叫的常見使用場景包括：

- **模型蒸餾（Model Distillation）**：用較大的模型生成合成訓練資料，並微調較小的模型來有效率地複製特定工作流程。
- **處理非標準 Schema**：克服基礎模型在處理公開資料中沒有的遺留、高度複雜資料結構或專有格式時的困難，例如處理[領域專用的行動裝置操作](https://huggingface.co/datasets/google/mobile-actions)。
- **最佳化上下文使用**：把工具定義「烤」進模型權重中。這讓您可以在提示中使用簡寫描述，把上下文視窗留給實際的對話。
- **解決選擇歧義**：讓模型偏向特定企業政策，例如優先使用內部知識庫而非外部搜尋引擎。

在本範例中，我們將專注於管理工具選擇歧義。

## 設定開發環境

第一步是安裝 Hugging Face 函式庫，包括 TRL 與 datasets，用來微調開放模型，並使用不同的 RLHF 與對齊技術。

```python { .notebook-cell }
# Install Pytorch & other libraries
%pip install torch tensorboard

# Install Hugging Face libraries
%pip install transformers datasets accelerate evaluate trl protobuf sentencepiece

# COMMENT IN: if you are running on a GPU that supports BF16 data type and flash attn, such as NVIDIA L4 or NVIDIA A100
#% pip install flash-attn
```

!!! Note
    如果您使用的 GPU 具備 Ampere 架構（例如 NVIDIA L4）或更新的架構，就可以使用 Flash attention。Flash Attention 是一種能大幅加速運算、並將序列長度的記憶體使用量從二次方降為線性的方法，可加速訓練達 3 倍。詳情請參閱 [FlashAttention](https://github.com/Dao-AILab/flash-attention/tree/main)。


在開始訓練之前，必須先接受 Gemma 的使用條款。您可以在 [Hugging Face](http://huggingface.co/google/functiongemma-270m-it){ data-model="functiongemma" } 點擊模型頁面上的 **Agree** 並存取儲存庫按鈕來接受授權，模型頁面網址：<http://huggingface.co/google/functiongemma-270m-it>

接受授權後，您需要有效的 Hugging Face 權杖才能存取模型。如果您在 Google Colab 中執行，可以使用 Colab 密鑰安全地使用 Hugging Face 權杖，否則也可以直接在 `login` 方法中設定權杖。請確保您的權杖具有寫入權限，因為微調後會將模型推送到 Hugging Face Hub。

```python { .notebook-cell }
# Login into Hugging Face Hub
from huggingface_hub import login
login()
```

您可以把結果保留在 Colab 的本機虛擬機器上。不過，強烈建議將中間結果存到您的 Google Drive。這能確保訓練結果安全，也讓您能輕鬆比較並選擇最佳模型。

此外，請調整檢查點目錄與學習率。

```python { .notebook-cell }
from google.colab import drive

mount_google_drive = False
checkpoint_dir = "functiongemma-270m-it-simple-tool-calling"

if mount_google_drive:
    drive.mount('/content/drive')
    checkpoint_dir = f"/content/drive/MyDrive/{checkpoint_dir}"

print(f"Checkpoints will be saved to {checkpoint_dir}")

base_model = "google/functiongemma-270m-it"
learning_rate = 5e-5
```

``` { .no-copy .output }
Checkpoints will be saved to functiongemma-270m-it-simple-tool-calling
```

## 準備微調資料集

您將使用以下範例資料集，其中包含需要在兩個工具之間選擇的範例對話：`search_knowledge_base` 與 `search_google`。

### 簡單工具呼叫資料集

```python { .notebook-cell }
simple_tool_calling = [
  {"user_content":"What is the reimbursement limit for travel meals?","tool_name":"search_knowledge_base","tool_arguments":"{\"query\": \"travel meal reimbursement limit policy\"}"},
  {"user_content":"What is the current stock price of Google?","tool_name":"search_google","tool_arguments":"{\"query\": \"current Google stock price\"}"},
  {"user_content":"How do I configure the VPN for the New York office?","tool_name":"search_knowledge_base","tool_arguments":"{\"query\": \"VPN configuration guide New York office\"}"},
  {"user_content":"Explain the difference between REST and GraphQL.","tool_name":"search_google","tool_arguments":"{\"query\": \"difference between REST and GraphQL\"}"},
  {"user_content":"Who is the product owner for Project Chimera?","tool_name":"search_knowledge_base","tool_arguments":"{\"query\": \"Project Chimera product owner\"}"},
  {"user_content":"Find the documentation for the 'requests' library in Python.","tool_name":"search_google","tool_arguments":"{\"query\": \"Python requests library documentation\"}"},
  {"user_content":"What are the core values listed in our employee handbook?","tool_name":"search_knowledge_base","tool_arguments":"{\"query\": \"employee handbook core values\"}"},
  {"user_content":"What is the weather forecast for the company retreat in Bali?","tool_name":"search_google","tool_arguments":"{\"query\": \"weather forecast Bali\"}"},
  {"user_content":"I need to reset my Okta password. How do I do that?","tool_name":"search_knowledge_base","tool_arguments":"{\"query\": \"Okta password reset procedure\"}"},
  {"user_content":"Who won the World Series last year?","tool_name":"search_google","tool_arguments":"{\"query\": \"MLB World Series winner last year\"}"},
  {"user_content":"What is the guest Wi-Fi password for the 4th floor?","tool_name":"search_knowledge_base","tool_arguments":"{\"query\": \"guest wifi password 4th floor\"}"},
  {"user_content":"Comparison of AWS vs GCP pricing.","tool_name":"search_google","tool_arguments":"{\"query\": \"AWS vs GCP pricing comparison\"}"},
  {"user_content":"How do I install our internal 'utils-core' package?","tool_name":"search_knowledge_base","tool_arguments":"{\"query\": \"install utils-core package internal registry\"}"},
  {"user_content":"What are the dates for the upcoming federal holidays?","tool_name":"search_google","tool_arguments":"{\"query\": \"upcoming federal holidays dates\"}"},
  {"user_content":"Does the office insurance cover dental implants?","tool_name":"search_knowledge_base","tool_arguments":"{\"query\": \"dental insurance coverage implants\"}"},
  {"user_content":"What is the latest version of Node.js?","tool_name":"search_google","tool_arguments":"{\"query\": \"latest Node.js version\"}"},
  {"user_content":"Find the meeting minutes from last week's All-Hands.","tool_name":"search_knowledge_base","tool_arguments":"{\"query\": \"All-Hands meeting minutes last week\"}"},
  {"user_content":"What did our competitor, ABC Corp, announce at CES today?","tool_name":"search_google","tool_arguments":"{\"query\": \"ABC Corp announcements CES today\"}"},
  {"user_content":"Who is the emergency contact for the London data center?","tool_name":"search_knowledge_base","tool_arguments":"{\"query\": \"emergency contact London data center\"}"},
  {"user_content":"Convert 100 USD to JPY.","tool_name":"search_google","tool_arguments":"{\"query\": \"100 USD to JPY exchange rate\"}"},
  {"user_content":"How do I access my paystubs on the ADP portal?","tool_name":"search_knowledge_base","tool_arguments":"{\"query\": \"access paystubs ADP portal guide\"}"},
  {"user_content":"What is the syntax for Python list comprehensions?","tool_name":"search_google","tool_arguments":"{\"query\": \"python list comprehension syntax examples\"}"},
  {"user_content":"Where can I find the floor plan for Building B?","tool_name":"search_knowledge_base","tool_arguments":"{\"query\": \"floor plan Building B conference rooms\"}"},
  {"user_content":"Check the latest stock price for Apple.","tool_name":"search_google","tool_arguments":"{\"query\": \"Apple stock price today\"}"},
  {"user_content":"What is the procedure for reporting a phishing email?","tool_name":"search_knowledge_base","tool_arguments":"{\"query\": \"report phishing email security procedure\"}"},
  {"user_content":"Show me examples of using the useEffect hook in React.","tool_name":"search_google","tool_arguments":"{\"query\": \"React useEffect hook code examples\"}"},
  {"user_content":"Who are the direct reports for the VP of Engineering?","tool_name":"search_knowledge_base","tool_arguments":"{\"query\": \"VP of Engineering org chart direct reports\"}"},
  {"user_content":"How do I list open ports on a Linux server?","tool_name":"search_google","tool_arguments":"{\"query\": \"linux command check open ports\"}"},
  {"user_content":"What is our Slack message retention policy?","tool_name":"search_knowledge_base","tool_arguments":"{\"query\": \"Slack public channel data retention policy\"}"},
  {"user_content":"Compare the features of iPhone 15 vs Samsung S24.","tool_name":"search_google","tool_arguments":"{\"query\": \"iPhone 15 vs Samsung S24 feature comparison\"}"},
  {"user_content":"I need the expense code for team building events.","tool_name":"search_knowledge_base","tool_arguments":"{\"query\": \"finance expense code team building\"}"},
  {"user_content":"Best practices for writing a Dockerfile for Node.js.","tool_name":"search_google","tool_arguments":"{\"query\": \"Dockerfile best practices Node.js application\"}"},
  {"user_content":"How do I request a new monitor setup?","tool_name":"search_knowledge_base","tool_arguments":"{\"query\": \"IT hardware request monitor setup\"}"},
  {"user_content":"What is the difference between VLOOKUP and XLOOKUP in Google Sheets?","tool_name":"search_google","tool_arguments":"{\"query\": \"Google Sheets VLOOKUP vs XLOOKUP difference\"}"},
  {"user_content":"Find the onboarding checklist for new engineering hires.","tool_name":"search_knowledge_base","tool_arguments":"{\"query\": \"new hire onboarding checklist engineering\"}"},
  {"user_content":"What are the latest release notes for the OpenAI API?","tool_name":"search_google","tool_arguments":"{\"query\": \"OpenAI API latest release notes\"}"},
  {"user_content":"Do we have preferred hotel partners in Paris?","tool_name":"search_knowledge_base","tool_arguments":"{\"query\": \"corporate travel preferred hotels Paris\"}"},
  {"user_content":"How to undo the last git commit but keep the changes?","tool_name":"search_google","tool_arguments":"{\"query\": \"git reset soft undo last commit\"}"},
  {"user_content":"What is the process for creating a new Jira project?","tool_name":"search_knowledge_base","tool_arguments":"{\"query\": \"create new Jira project process\"}"},
  {"user_content":"Tutorial on SQL window functions.","tool_name":"search_google","tool_arguments":"{\"query\": \"SQL window functions tutorial\"}"},
]
```


假設有這個查詢：「What are the best practices for writing a simple recursive function in Python?」

該用哪個工具完全取決於您的特定政策。一般模型自然會預設使用 `search_google`，但企業應用通常需要先檢查 `search_knowledge_base`。

**資料切分說明**：在本示範中，您將使用 50/50 的訓練/測試切分。雖然正式工作流程的標準是 80/20，但這裡刻意選擇均等切分，來凸顯模型在未見過資料上的效能改善。

```python { .notebook-cell }
import json
from datasets import Dataset
from transformers.utils import get_json_schema

# --- Tool Definitions ---
def search_knowledge_base(query: str) -> str:
    """
    Search internal company documents, policies and project data.

    Args:
        query: query string
    """
    return "Internal Result"

def search_google(query: str) -> str:
    """
    Search public information.

    Args:
        query: query string
    """
    return "Public Result"


TOOLS = [get_json_schema(search_knowledge_base), get_json_schema(search_google)]

DEFAULT_SYSTEM_MSG = "You are a model that can do function calling with the following functions"

def create_conversation(sample):
  return {
      "messages": [
          {"role": "developer", "content": DEFAULT_SYSTEM_MSG},
          {"role": "user", "content": sample["user_content"]},
          {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": sample["tool_name"], "arguments": json.loads(sample["tool_arguments"])} }]},
      ],
      "tools": TOOLS
  }

dataset = Dataset.from_list(simple_tool_calling)
# You can also load the dataset from Hugging Face Hub
# dataset = load_dataset("bebechien/SimpleToolCalling", split="train")

# Convert dataset to conversational format
dataset = dataset.map(create_conversation, remove_columns=dataset.features, batched=False)

# Split dataset into 50% training samples and 50% test samples
dataset = dataset.train_test_split(test_size=0.5, shuffle=True)
```

``` { .no-copy .output }
Map:   0%|          | 0/40 [00:00<?, ? examples/s]
```

**資料分布的重要說明**

在您自己的自訂資料集上使用 `shuffle=False` 時，請確保原始資料已預先混合。如果資料分布未知或已排序，您應使用 `shuffle=True`，確保模型在訓練期間能均衡地學習所有工具的表示。

## 使用 TRL 與 SFTTrainer 微調 FunctionGemma

現在您已準備好微調模型。Hugging Face TRL 的 [SFTTrainer](https://huggingface.co/docs/trl/sft_trainer) 讓監督式微調開放式 LLM 變得簡單。`SFTTrainer` 是 `transformers` 函式庫中 `Trainer` 的子類別，支援所有相同的功能。

以下程式碼從 Hugging Face 載入 FunctionGemma 模型與分詞器。

```python { .notebook-cell }
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    dtype="auto",
    device_map="auto",
    attn_implementation="eager"
)
tokenizer = AutoTokenizer.from_pretrained(base_model)

print(f"Device: {model.device}")
print(f"DType: {model.dtype}")

# Print formatted user prompt
print("--- dataset input ---")
print(json.dumps(dataset["train"][0], indent=2))
debug_msg = tokenizer.apply_chat_template(dataset["train"][0]["messages"], tools=dataset["train"][0]["tools"], add_generation_prompt=False, tokenize=False)
print("--- Formatted prompt ---")
print(debug_msg)
```

``` { .no-copy .output }
Device: cuda:0
DType: torch.bfloat16
--- dataset input ---
{
  "messages": [
    {
      "content": "You are a model that can do function calling with the following functions",
      "role": "developer",
      "tool_calls": null
    },
    {
      "content": "What is the reimbursement limit for travel meals?",
      "role": "user",
      "tool_calls": null
    },
    {
      "content": null,
      "role": "assistant",
      "tool_calls": [
        {
          "function": {
            "arguments": {
              "query": "travel meal reimbursement limit policy"
            },
            "name": "search_knowledge_base"
          },
          "type": "function"
        }
      ]
    }
  ],
  "tools": [
    {
      "function": {
        "description": "Search internal company documents, policies and project data.",
        "name": "search_knowledge_base",
        "parameters": {
          "properties": {
            "query": {
              "description": "query string",
              "type": "string"
            }
          },
          "required": [
            "query"
          ],
          "type": "object"
        },
        "return": {
          "type": "string"
        }
      },
      "type": "function"
    },
    {
      "function": {
        "description": "Search public information.",
        "name": "search_google",
        "parameters": {
          "properties": {
            "query": {
              "description": "query string",
              "type": "string"
            }
          },
          "required": [
            "query"
          ],
          "type": "object"
        },
        "return": {
          "type": "string"
        }
      },
      "type": "function"
    }
  ]
}
--- Formatted prompt ---
<bos><start_of_turn>developer
You are a model that can do function calling with the following functions<start_function_declaration>declaration:search_knowledge_base{description:<escape>Search internal company documents, policies and project data.<escape>,parameters:{properties:{query:{description:<escape>query string<escape>,type:<escape>STRING<escape>} },required:[<escape>query<escape>],type:<escape>OBJECT<escape>} }<end_function_declaration><start_function_declaration>declaration:search_google{description:<escape>Search public information.<escape>,parameters:{properties:{query:{description:<escape>query string<escape>,type:<escape>STRING<escape>} },required:[<escape>query<escape>],type:<escape>OBJECT<escape>} }<end_function_declaration><end_of_turn>
<start_of_turn>user
What is the reimbursement limit for travel meals?<end_of_turn>
<start_of_turn>model
<start_function_call>call:search_knowledge_base{query:<escape>travel meal reimbursement limit policy<escape>}<end_function_call><start_function_response>
```

## 微調前

以下輸出顯示，開箱即用的能力對這個使用場景可能不夠好。

```python { .notebook-cell }
def check_success_rate():
  success_count = 0
  for idx, item in enumerate(dataset['test']):
    messages = [
        item["messages"][0],
        item["messages"][1],
    ]

    inputs = tokenizer.apply_chat_template(messages, tools=TOOLS, add_generation_prompt=True, return_dict=True, return_tensors="pt")

    out = model.generate(**inputs.to(model.device), pad_token_id=tokenizer.eos_token_id, max_new_tokens=128)
    output = tokenizer.decode(out[0][len(inputs["input_ids"][0]) :], skip_special_tokens=False)

    print(f"{idx+1} Prompt: {item['messages'][1]['content']}")
    print(f"  Output: {output}")

    expected_tool = item['messages'][2]['tool_calls'][0]['function']['name']
    other_tool = "search_knowledge_base" if expected_tool == "search_google" else "search_google"

    if expected_tool in output and other_tool not in output:
      print("  `-> ✅ correct!")
      success_count += 1
    elif expected_tool not in output:
      print(f"  -> ❌ wrong (expected '{expected_tool}' missing)")
    else:
      if output.startswith(f"<start_function_call>call:{expected_tool}"):
        print(f"  -> ⚠️ tool is correct {expected_tool}, but other_tool exists in output")
      else:
        print(f"  -> ❌ wrong (hallucinated '{other_tool}')")

  print(f"Success : {success_count} / {len(dataset['test'])}")

check_success_rate()
```

``` { .no-copy .output }
1 Prompt: How do I access my paystubs on the ADP portal?
  Output: I cannot assist with accessing or retrieving paystubs or other company documents on the ADP portal. My current capabilities are limited to assisting with searching internal company documents and knowledge base queries.<end_of_turn>
  -> ❌ wrong (expected 'search_knowledge_base' missing)
2 Prompt: What is the syntax for Python list comprehensions?
  Output: I cannot assist with programming or providing programming syntax information. My current capabilities are focused on searching internal company documents and project data.<end_of_turn>
  -> ❌ wrong (expected 'search_google' missing)
3 Prompt: Where can I find the floor plan for Building B?
  Output: <start_function_call>call:search_knowledge_base{query:<escape>Floor plan for Building B<escape>}<end_function_call><start_function_call>call:search_google{query:<escape>Floor plan for Building B<escape>}<end_function_call><start_function_response>
  -> ⚠️ tool is correct search_knowledge_base, but other_tool exists in output
4 Prompt: Check the latest stock price for Apple.
  Output: <start_function_call>call:search_google{query:<escape>AAPL stock price<escape>}<end_function_call><start_function_response>
  `-> ✅ correct!
5 Prompt: What is the procedure for reporting a phishing email?
  Output: I cannot assist with providing instructions or procedures for reporting phishing emails. My capabilities are limited to assisting with specific search and document management functions.<end_of_turn>
  -> ❌ wrong (expected 'search_knowledge_base' missing)
6 Prompt: Show me examples of using the useEffect hook in React.
  Output: I am sorry, but I cannot assist with providing examples of using the `useEffect` hook in React. My current capabilities are focused on assisting with searching and retrieving internal company documents and project data using the specified tools. I cannot generate or explain code examples related to React hooks.<end_of_turn>
  -> ❌ wrong (expected 'search_google' missing)
7 Prompt: Who are the direct reports for the VP of Engineering?
  Output: <start_function_call>call:search_knowledge_base{query:<escape>VP of Engineering<escape>}<end_function_call><start_function_call>call:search_google{query:<escape>VP of Engineering<escape>}<end_function_call><start_function_response>
  -> ⚠️ tool is correct search_knowledge_base, but other_tool exists in output
8 Prompt: How do I list open ports on a Linux server?
  Output: I cannot assist with listing or querying open ports on Linux servers. My current capabilities are limited to assisting with searching internal company documents and project data using the specified tools.<end_of_turn>
  -> ❌ wrong (expected 'search_google' missing)
9 Prompt: What is our Slack message retention policy?
  Output: I cannot assist with finding or recommending company policies or terms of service regarding Slack messaging retention. My current capabilities are focused on searching internal company documents and project data.<end_of_turn>
  -> ❌ wrong (expected 'search_knowledge_base' missing)
10 Prompt: Compare the features of iPhone 15 vs Samsung S24.
  Output: I cannot assist with comparing device features or specifications. My current capabilities are focused on assisting with searching and managing company knowledge. I cannot browse or compare external product information.<end_of_turn>
  -> ❌ wrong (expected 'search_google' missing)
11 Prompt: I need the expense code for team building events.
  Output: I can certainly assist with searching for expense codes. Could you please specify the exact query you would like to use for searching the company expense database?<end_of_turn>
  -> ❌ wrong (expected 'search_knowledge_base' missing)
12 Prompt: Best practices for writing a Dockerfile for Node.js.
  Output: I am sorry, but I cannot assist with recommending best practices for writing Dockerfiles for Node.js. My current capabilities are focused on searching company documents and project data using specific tools. I cannot recommend or provide expert advice on software development best practices.<end_of_turn>
  -> ❌ wrong (expected 'search_google' missing)
13 Prompt: How do I request a new monitor setup?
  Output: I cannot assist with recommending or requesting hardware setup or configurations. My current capabilities are limited to assisting with searching company documents and project data using the specified tools. I cannot provide technical advice or recommendations for hardware setup.<end_of_turn>
  -> ❌ wrong (expected 'search_knowledge_base' missing)
14 Prompt: What is the difference between VLOOKUP and XLOOKUP in Google Sheets?
  Output: I cannot assist with comparing or contrasting Google Sheets and VLOOKUP/XLOOKUP in specific technical terms. My current capabilities are limited to assisting with specific data management functions like searching internal company documents and searching public knowledge base. I cannot provide expert comparisons or explanations of Google Sheets and VLOOKUP/XLOOKUP.<end_of_turn>
  -> ❌ wrong (expected 'search_google' missing)
15 Prompt: Find the onboarding checklist for new engineering hires.
  Output: <start_function_call>call:search_knowledge_base{query:<escape>onboarding checklist for new engineering hires<escape>}<end_function_call><start_function_response>
  `-> ✅ correct!
16 Prompt: What are the latest release notes for the OpenAI API?
  Output: <start_function_call>call:search_knowledge_base{query:<escape>OpenAI API release notes latest<escape>}<end_function_call><start_function_response>
  -> ❌ wrong (expected 'search_google' missing)
17 Prompt: Do we have preferred hotel partners in Paris?
  Output: I apologize, but I cannot assist with finding hotel partner information. My current capabilities are focused on searching company documents and project data using specific keywords. I cannot connect with or query business databases for hotel partnerships.<end_of_turn>
  -> ❌ wrong (expected 'search_knowledge_base' missing)
18 Prompt: How to undo the last git commit but keep the changes?
  Output: I cannot assist with managing or undoing Git repositories. My current capabilities are limited to assisting with searching and managing organizational knowledge bases. I cannot provide specific instructions or assistance with technical issues related to software development or Git operations.<end_of_turn>
  -> ❌ wrong (expected 'search_google' missing)
19 Prompt: What is the process for creating a new Jira project?
  Output: I cannot assist with creating or managing Jira project processes. My current capabilities are limited to assisting with searching company documents and project data using the specified tools.<end_of_turn>
  -> ❌ wrong (expected 'search_knowledge_base' missing)
20 Prompt: Tutorial on SQL window functions.
  Output: I cannot assist with tutorials or programming advice regarding SQL window functions. My capabilities are limited to assisting with searching company documents and knowledge base information.<end_of_turn>
  -> ❌ wrong (expected 'search_google' missing)
Success : 2 / 20
```

## 訓練

在開始訓練之前，您需要在 `SFTConfig` 實例中定義要使用的超參數。

```python { .notebook-cell }
from trl import SFTConfig

torch_dtype = model.dtype

args = SFTConfig(
    output_dir=checkpoint_dir,              # directory to save and repository id
    max_length=512,                         # max sequence length for model and packing of the dataset
    packing=False,                          # Groups multiple samples in the dataset into a single sequence
    num_train_epochs=8,                     # number of training epochs
    per_device_train_batch_size=4,          # batch size per device during training
    gradient_checkpointing=False,           # Caching is incompatible with gradient checkpointing
    optim="adamw_torch_fused",              # use fused adamw optimizer
    logging_steps=1,                        # log every step
    #save_strategy="epoch",                  # save checkpoint every epoch
    eval_strategy="epoch",                  # evaluate checkpoint every epoch
    learning_rate=learning_rate,            # learning rate
    fp16=True if torch_dtype == torch.float16 else False,   # use float16 precision
    bf16=True if torch_dtype == torch.bfloat16 else False,  # use bfloat16 precision
    lr_scheduler_type="constant",            # use constant learning rate scheduler
    push_to_hub=True,                        # push model to hub
    report_to="tensorboard",                 # report metrics to tensorboard
)
```

現在您已經擁有建立 `SFTTrainer` 並開始訓練模型所需的所有元件。

```python { .notebook-cell }
from trl import SFTTrainer

# Create Trainer object
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    processing_class=tokenizer,
)
```

``` { .no-copy .output }
Tokenizing train dataset:   0%|          | 0/20 [00:00<?, ? examples/s]
Truncating train dataset:   0%|          | 0/20 [00:00<?, ? examples/s]
Tokenizing eval dataset:   0%|          | 0/20 [00:00<?, ? examples/s]
Truncating eval dataset:   0%|          | 0/20 [00:00<?, ? examples/s]
The model is already on multiple devices. Skipping the move to device specified in `args`.
```

呼叫 `train()` 方法開始訓練。

```python { .notebook-cell }
# Start training, the model will be automatically saved to the Hub and the output directory
trainer.train()

# Save the final model again to the Hugging Face Hub
trainer.save_model()
```

``` { .no-copy .output }
The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'bos_token_id': 2, 'pad_token_id': 0}.

[40/40 01:02, Epoch 8/8]

| Epoch | Training Loss | Validation Loss | Entropy | Num Tokens | Mean Token Accuracy |
|---|---|---|---|---|---|
| 1 | 0.800300 | 0.648739 | 0.590625 | 3241.000000 | 0.888866 |
| 2 | 0.269600 | 0.430649 | 0.310939 | 6482.000000 | 0.937447 |
| 3 | 0.179600 | 0.442410 | 0.217868 | 9723.000000 | 0.933030 |
| 4 | 0.104800 | 0.469016 | 0.196222 | 12964.000000 | 0.933815 |
| 5 | 0.064500 | 0.436476 | 0.183889 | 16205.000000 | 0.937461 |
| 6 | 0.054400 | 0.489788 | 0.119700 | 19446.000000 | 0.940793 |
| 7 | 0.035100 | 0.498824 | 0.083078 | 22687.000000 | 0.944682 |
| 8 | 0.036200 | 0.502200 | 0.093177 | 25928.000000 | 0.944957 |
```

要繪製訓練與驗證損失曲線，通常會從 `TrainerState` 物件或訓練期間產生的日誌中擷取這些值。

接著可以用 Matplotlib 這類函式庫將這些值繪成圖表，x 軸為訓練步驟或 epoch，y 軸為對應的損失值。

```python { .notebook-cell }
import matplotlib.pyplot as plt

# Access the log history
log_history = trainer.state.log_history

# Extract training / validation loss
train_losses = [log["loss"] for log in log_history if "loss" in log]
epoch_train = [log["epoch"] for log in log_history if "loss" in log]
eval_losses = [log["eval_loss"] for log in log_history if "eval_loss" in log]
epoch_eval = [log["epoch"] for log in log_history if "eval_loss" in log]

# Plot the training loss
plt.plot(epoch_train, train_losses, label="Training Loss")
plt.plot(epoch_eval, eval_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss per Epoch")
plt.legend()
plt.grid(True)
plt.show()
```

![png](../../assets/functiongemma/output_6JBty2rq05Jz_0.png)

## 測試模型推論

訓練完成後，您會想評估並測試模型。您可以從測試資料集載入不同的樣本來評估模型。

```python { .notebook-cell }
check_success_rate()
```

``` { .no-copy .output }
1 Prompt: How do I access my paystubs on the ADP portal?
  Output: <start_function_call>call:search_knowledge_base{query:<escape>paystubs API portal access codes<escape>}<end_function_call><start_function_response>
  `-> ✅ correct!
2 Prompt: What is the syntax for Python list comprehensions?
  Output: <start_function_call>call:search_google{query:<escape>Python list comprehensions syntax<escape>}<end_function_call><start_function_response>
  `-> ✅ correct!
3 Prompt: Where can I find the floor plan for Building B?
  Output: <start_function_call>call:search_knowledge_base{query:<escape>floor plan Building B floor plan<escape>}<end_function_call><start_function_response>
  `-> ✅ correct!
4 Prompt: Check the latest stock price for Apple.
  Output: <start_function_call>call:search_google{query:<escape>latest stock price Apple<escape>}<end_function_call><start_function_response>
  `-> ✅ correct!
5 Prompt: What is the procedure for reporting a phishing email?
  Output: <start_function_call>call:search_knowledge_base{query:<escape>phishing email procedure reporting policy<escape>}<end_function_call><start_function_response>
  `-> ✅ correct!
6 Prompt: Show me examples of using the useEffect hook in React.
  Output: <start_function_call>call:search_knowledge_base{query:<escape>useEffect hook examples React<escape>}<end_function_call><start_function_response>
  -> ❌ wrong (expected 'search_google' missing)
7 Prompt: Who are the direct reports for the VP of Engineering?
  Output: <start_function_call>call:search_knowledge_base{query:<escape>VP of Engineering direct reports<escape>}<end_function_call><start_function_response>
  `-> ✅ correct!
8 Prompt: How do I list open ports on a Linux server?
  Output: <start_function_call>call:search_google{query:<escape>open ports Linux server equivalents<escape>}<end_function_call><start_function_response>
  `-> ✅ correct!
9 Prompt: What is our Slack message retention policy?
  Output: <start_function_call>call:search_knowledge_base{query:<escape>slack message retention policy policy excerpt<escape>}<end_function_call><start_function_response>
  `-> ✅ correct!
10 Prompt: Compare the features of iPhone 15 vs Samsung S24.
  Output: <start_function_call>call:search_google{query:<escape>iPhone 15 vs Samsung S24 feature comparison<escape>}<end_function_call><start_function_response>
  `-> ✅ correct!
11 Prompt: I need the expense code for team building events.
  Output: <start_function_call>call:search_knowledge_base{query:<escape>expense code team building events<escape>}<end_function_call><start_function_response>
  `-> ✅ correct!
12 Prompt: Best practices for writing a Dockerfile for Node.js.
  Output: <start_function_call>call:search_knowledge_base{query:<escape>Docker file best practices Node.js<escape>}<end_function_call><start_function_response>
  -> ❌ wrong (expected 'search_google' missing)
13 Prompt: How do I request a new monitor setup?
  Output: <start_function_call>call:search_knowledge_base{query:<escape>new monitor setup request procedure<escape>}<end_function_call><start_function_response>
  `-> ✅ correct!
14 Prompt: What is the difference between VLOOKUP and XLOOKUP in Google Sheets?
  Output: <start_function_call>call:search_google{query:<escape>VLOOKUP vs XLOOKUP difference Google Sheets中<escape>}<end_function_call><start_function_response>
  `-> ✅ correct!
15 Prompt: Find the onboarding checklist for new engineering hires.
  Output: <start_function_call>call:search_knowledge_base{query:<escape>engineering hire onboarding checklist New hires.<escape>}<end_function_call><start_function_response>
  `-> ✅ correct!
16 Prompt: What are the latest release notes for the OpenAI API?
  Output: <start_function_call>call:search_google{query:<escape>latest OpenAI API release notes latest version<escape>}<end_function_call><start_function_response>
  `-> ✅ correct!
17 Prompt: Do we have preferred hotel partners in Paris?
  Output: <start_function_call>call:search_knowledge_base{query:<escape>preferred hotel partners in Paris<escape>}<end_function_call><start_function_response>
  `-> ✅ correct!
18 Prompt: How to undo the last git commit but keep the changes?
  Output: <start_function_call>call:search_knowledge_base{query:<escape>undo git commit last commit<escape>}<end_function_call><start_function_response>
  -> ❌ wrong (expected 'search_google' missing)
19 Prompt: What is the process for creating a new Jira project?
  Output: <start_function_call>call:search_knowledge_base{query:<escape>Jira project creation process<escape>}<end_function_call><start_function_response>
  `-> ✅ correct!
20 Prompt: Tutorial on SQL window functions.
  Output: <start_function_call>call:search_knowledge_base{query:<escape>SQL window functions tutorial<escape>}<end_function_call><start_function_response>
  -> ❌ wrong (expected 'search_google' missing)
Success : 16 / 20
```

## 摘要

您學會了如何微調 FunctionGemma 來解決**工具選擇歧義**——模型必須根據特定企業政策，在重疊的工具（例如內部與外部搜尋）之間選擇。透過使用 **Hugging Face TRL 函式庫**與 `SFTTrainer`，本教學帶您走過準備資料集、設定超參數，並執行監督式微調迴圈的流程。

結果清楚呈現了「能力足夠」的基礎模型與「可投入生產」的微調模型之間的關鍵差異：

- **微調前**：基礎模型難以遵守特定政策，經常無法呼叫工具或選錯工具，成功率偏低（例如 2/20）。
- **微調後**：訓練 8 個 epoch 後，模型學會正確區分需要 `search_knowledge_base` 與 `search_google` 的查詢，成功率提升（例如 16/20）。

現在您已擁有微調好的模型，接下來可以考慮以下步驟邁向生產環境：

- **擴充資料集**：目前的資料集只是示範用的小型合成切分（50/50）。對於穩健的企業應用，請整理更大、更多樣化的資料集，涵蓋邊界案例與罕見的政策例外。
- **搭配 RAG 評估**：將微調後的模型整合進檢索增強生成（RAG）管線，驗證 `search_knowledge_base` 工具呼叫是否真的能檢索到相關文件並產出正確的最終答案。
## 接下來

<div class="card-grid">
<a class="card" href="../../deployment/index.md">
  <span class="card-icon">{{ icon('lucide/play') }}</span>
  <div class="card-title">執行 Gemma</div>
  <p class="card-desc">使用任何支援的引擎部署您微調好的模型。</p>
</a>
<a class="card" href="../../tuning/index.md">
  <span class="card-icon">{{ icon('lucide/graduation-cap') }}</span>
  <div class="card-title">微調 Gemma</div>
  <p class="card-desc">QLoRA、完整微調與分散式訓練。</p>
</a>
<a class="card" href="https://github.com/google-gemma/cookbook/blob/main/FunctionGemma/%5BFunctionGemma%5DFinetune_FunctionGemma_270M_for_Mobile_Actions_with_Hugging_Face.ipynb" target="_blank" rel="noopener" data-model="functiongemma">
  <div class="card-title">行動操作筆記本 <span class="card-ext">↗</span></div>
  <p class="card-desc">針對裝置端操作微調 FunctionGemma 270M。</p>
</a>
</div>
