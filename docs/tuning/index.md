---
title: Gemma 模型微調
description: 微調 Gemma 這類生成式 AI 模型會改變模型的行為：了解各種微調方法，選擇最適合您需求的微調路徑。
models: [gemma-4]
source_url: "https://ai.google.dev/gemma/docs/tune.md.txt"
---
# Gemma 模型微調

微調 Gemma 這類生成式人工智慧（AI）模型，會改變模型的行為。您通常會為了提升模型在特定任務或領域的表現，或為了讓它更好地扮演某個角色（例如客服）而微調 Gemma。Gemma 模型以開放權重發布，這表示您可以修改這些權重，進而改變模型的行為。微調 Gemma 模型的一般步驟如下：

<ol class="steps" markdown>
<li markdown>
**[選擇框架](index.md#1-選擇框架)**

每個框架都與模型格式及您擁有的硬體綁定。請在準備資料之前先選好框架，因為資料格式會由框架決定。
</li>
<li markdown>
**[收集資料](index.md#2-收集資料)**

輸入與預期回應的配對。從小開始——少至 20 組配對就能改變行為。
</li>
<li markdown>
**[微調與測試模型](index.md#3-微調與測試模型)**

在完整微調與 LoRA 這類參數效率型方法之間選擇，然後用成功、失敗與邊界案例來測試。
</li>
<li markdown>
**[部署模型](index.md#4-部署模型)**

發布微調後的權重——如果是 LoRA，基底模型與轉接器一起發布。
</li>
</ol>

## 1. 選擇框架

Gemma 模型相容於多種 AI 微調框架。每個框架都與特定的模型格式及所需的硬體綁定，這個決定會影響後續一切。請確認您打算使用的框架支援您預期的部署格式——Keras、Safetensors 或 GGUF——作為輸出。

<!-- Generated from data/frameworks.yml -- edit that file, not these grids. -->
{% set fwmap = {} %}
{%- for fw in frameworkdata.frameworks %}{% set _ = fwmap.update({fw.id: fw}) %}{% endfor %}
{% macro paths(ids) %}
<ul class="engine-grid engine-grid--detailed">
{% for id in ids %}{% for path in frameworkdata.tuning_paths if path.id == id %}
{% set fw = fwmap.get(path.framework) if path.get('framework') else none %}
{% set page = path.get('page') %}
  <li><a class="engine" href="{% if page %}../{{ page }}{% else %}{{ path.url }}{% endif %}"
         {% if not page %}target="_blank" rel="noopener"{% endif %}
         {%- if path.get('pinned') %} data-model="{{ path.pinned }}"{% endif %}>
    <span class="engine-mark" aria-hidden="true">
      {%- if fw and fw.get('icon') %}{{ icon(fw.icon, 'engine-logo') }}{% else %}{{ path.get('mark') or fw.mark }}{% endif -%}
    </span>
    <span class="engine-body">
      <span class="engine-name">{{ path.label }}{% if not page %} ↗{% endif %}</span>
      <span class="engine-blurb">{{ path.blurb }}</span>
    </span>
  </a></li>
{% endfor %}{% endfor %}
</ul>
{% endmacro %}

### 用 Python 微調

當微調是您掌控的訓練管線的一部分時，您會選用的函式庫。

{{ paths(["keras-lora", "transformers-peft", "gemma-jax"]) }}

### 從 Notebook 開始

開箱即用的食譜式 Notebook，可端到端執行。這些是為較早世代的 Gemma 編寫的——方法仍可沿用，但請預期需要更新模型名稱與相依套件。

{{ paths(["unsloth-notebook", "axolotl", "llama-factory", "xtuner"]) }}

### 大規模微調

適合超出單機規模的訓練，或您不想自己操作的情境。

{{ paths(["keras-distributed", "gke", "vertex-ai"]) }}

!!! note
    JAX 微調指南已被 [Gemma 函式庫微調](https://gemma-llm.readthedocs.io/en/latest/colab_finetuning.html) 指南取代。

## 2. 收集資料

模型微調需要資料。微調資料通常由輸入資料與預期回應的配對組成。網路上有許多公開資料集，可用於各種任務或輸出的訓練。例如，如果您想訓練 Gemma 模型將汽車零件描述翻譯成零件編號，您的資料集可能長這樣：

```python
training_data = [
  {"input_text": "Part number for A4 brake caliper", "output_text": "4M0615107BS"},
  {"input_text": "Part number for Beetle fuel pump", "output_text": "6A127026H"},
  {"input_text": "Part number for Camaro cylinder head", "output_text": "12711770"},
]
```

如果您想讓 Gemma 模型執行一組特定任務或扮演某個角色，通常需要整理一個展示該任務多種變化的資料集。微調模型需要多少資料取決於您的目標，特別是您希望模型產生多大的行為改變，以及您希望模型在該任務上的表現水準和輸入資料的變化程度。

一般來說，您應該先用少量資料進行任務微調，調整訓練參數，再逐步增加資料，直到達到符合需求的任務表現。我們的一些範例應用顯示，僅用 20 組提示與回應配對，就能影響 Gemma 模型的行為。更多詳情請參閱[使用 Hugging Face Transformers 進行完整模型微調](full-finetune.md)。

## 3. 微調與測試模型

備齊微調框架與微調資料後，就可以開始 Gemma 模型微調流程。微調方式有幾種選擇，會影響完成所需的資源。您也應該為微調後的模型準備測試計畫，以評估它是否如預期運作。

### 參數效率型微調

在微調 Gemma 這類開放權重模型時，您可以選擇[調整模型的所有參數](full-finetune.md)，也可以使用資源需求較低的參數效率型微調技術，只更新其中一部分參數。*完整*微調方法是指，在套用微調資料時，為模型的所有參數計算新權重。這種方法運算與記憶體需求都很高，因為您要為數十億個參數執行這些計算。使用資源需求較低的方法——稱為*參數效率型微調（parameter efficient fine-tuning, PEFT)*，包括低秩轉接器（LoRA）微調等技術——可以用更少的運算資源達到相近的效果。如需瞭解如何用 LoRA 以較少資源進行微調，請參閱[使用 Hugging Face Transformers 與 QLoRA 微調 Gemma](qlora.md) 以及 [Fine-Tuning Gemma Models in Hugging Face](https://huggingface.co/blog/gemma-peft)。

### 測試微調後的模型

針對特定任務微調完模型後，您應該用它要執行的那組任務來測試其表現。應該用模型未曾專門訓練過的任務或請求來測試。測試方式取決於您要它執行的任務，以及您對模型輸入輸出的管理程度。管理生成式模型測試的常見做法是使用成功、失敗與邊界案例：

- **成功測試**：微調後模型應該永遠能成功執行的請求。
- **失敗測試**：微調後模型應該永遠不能執行、或被要求時應明確拒絕執行的請求。
- **邊界測試**：落在可接受輸出行為的定義邊界（或一組邊界）內時，微調後模型應該能執行的請求。

在為您的生成式 AI 應用測試失敗或邊界條件時，也應該套用 [Responsible Generative AI Toolkit][responsible-ai] 中描述的生成式 AI 安全方法、技術與工具。

## 4. 部署模型

完成微調並通過測試後，就該部署模型了。部署方式通常可以參考您[所選框架](index.md#1-選擇框架)的文件。

如果部署的是以 LoRA 微調權重的模型，請注意，這種技術通常會同時部署*原始模型及其權重*，並將 LoRA 權重作為模型的額外計算層。

--8<-- "glossary.md"
