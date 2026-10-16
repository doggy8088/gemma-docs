---
title: 總覽
description: 依您的硬體選擇合適的執行框架與 Gemma 模型變體，逐步完成環境設定並執行您的第一個文字生成與推論請求。
models: [gemma-4]
source_url: "https://ai.google.dev/gemma/docs/run.md.txt"
---
當您想執行 Gemma 模型時，有兩個關鍵決策：1）要執行哪個 Gemma 變體，2）要用哪個 AI 執行框架來執行？
這兩個決策的關鍵都與您和您的使用者可用的硬體有關。

本總覽協助您做出這些決策並開始使用 Gemma
模型。執行 Gemma 模型的一般步驟如下：

<ol class="steps" markdown>
<li markdown>
**[選擇框架](index.md#1-選擇框架)**

挑選符合您硬體的執行環境——託管端點、桌面應用程式、Python
函式庫或推論伺服器。
</li>
<li markdown>
**[選擇 Gemma 變體](index.md#2-選擇-gemma-變體)**

決定家族、尺寸與量化等級。先從可能適用的最小
指令微調核心模型開始。
</li>
<li markdown>
**[執行生成與推論請求](index.md#3-執行生成與推論請求)**

依照您框架的指南操作，並以變體預期的格式撰寫提示。
</li>
</ol>

## 1. 選擇框架

Gemma 執行於廣泛的生態系工具。哪一個適合您，取決於您的硬體——
雲端 GPU 或筆電——以及您想寫 Python 還是開啟應用程式。以下
每個選項執行的都是相同的權重，因此您可以之後再切換。

<!-- Generated from data/frameworks.yml -- edit that file, not these grids. -->
{% macro engines(ids) %}
<ul class="engine-grid engine-grid--detailed">
{% for id in ids %}{% for fw in frameworkdata.frameworks if fw.id == id %}
{% set page = fw.get('page') %}
  <li><a class="engine" href="{% if page %}../{{ page }}{% else %}{{ fw.url }}{% endif %}"
         {% if not page %}target="_blank" rel="noopener"{% endif %}>
    <span class="engine-mark" aria-hidden="true">
      {%- if fw.get('icon') %}{{ icon(fw.icon, 'engine-logo') }}{% else %}{{ fw.mark }}{% endif -%}
    </span>
    <span class="engine-body">
      <span class="engine-name">{{ fw.short_name | default(fw.name) }}{% if not page %} ↗{% endif %}</span>
      <span class="engine-blurb">{{ fw.blurb }}</span>
    </span>
  </a></li>
{% endfor %}{% endfor %}
</ul>
{% endmacro %}

### 免安裝直接試用

託管端點。無需下載、無需 GPU——適合在投入
硬體前先評估模型。

{{ engines(["ai-studio", "openrouter", "cerebras"]) }}

### 在自己的機器上執行

桌面應用程式與本機執行環境。這些使用 GGUF 等量化格式，因此筆電
通常就夠了。

{{ engines(["lmstudio", "ollama", "llamacpp", "mlx", "litert"]) }}

### 在 Python 中建構與微調

當 Gemma 是應用程式或訓練流程的一部分時，您會用到的函式庫。

{{ engines(["transformers", "unsloth", "keras"]) }}

另請參閱 [Tunix](https://tunix.readthedocs.io/en/latest/)，以在 JAX 中進行微調。

### 大規模服務

用於生產流量的推論伺服器與託管平台。

{{ engines(["vllm", "sglang", "google-cloud", "gemini-api"]) }}

對於企業部署，Google Cloud 也提供
[Gemini Enterprise Agent Platform](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/open-models/use-gemma)。

請確認您選擇的框架支援您預期的模型格式——Keras 內建、Safetensors 或 GGUF。

## 2. 選擇 Gemma 變體

Gemma 模型有多種變體與尺寸，包括
基礎或[核心](../models/gemma-4/index.md) Gemma 模型，以及更
專門的模型變體，例如
[PaliGemma](../models/paligemma/index.md) 與
[DataGemma](../models/datagemma/index.md)，還有
AI 開發者社群在
[Kaggle](https://www.kaggle.com/models?query=gemma) 與
[Hugging Face](https://huggingface.co/models?search=gemma) 等網站上建立的眾多變體。如果您不確定
該從哪個變體開始，請選擇參數數量最少的最新 Gemma
[核心](../models/gemma-4/index.md)指令微調（IT）模型。這類 Gemma 模型的運算需求較低，
能回應各種提示，無需額外開發。

選擇 Gemma 變體時，請考量以下因素：

- **Gemma 核心，以及 PaliGemma 與 EmbeddingGemma 等其他變體家族** ：*建議選 Gemma（核心）。* 核心版本以外的 Gemma 變體與核心模型架構相同，訓練時針對特定任務表現更好。除非您的應用或目標與特定 Gemma 變體的專長相符，否則最好從 Gemma 核心（或基礎）模型開始。
- **指令微調（IT）、預訓練（PT）、微調（FT）、混合
  （mix）** ：*建議選 IT。*
    - *指令微調*（IT）Gemma 變體是訓練來回應人類語言中各種指示或請求的模型。這些模型變體是最佳起點，因為它們無需進一步的模型訓練即可回應提示。
    - *預訓練*（PT）Gemma 變體是訓練來對語言或其他資料進行推論的模型，但未經訓練來遵循人類指示。這些模型需要額外的訓練或微調才能有效執行任務，適合想研究或開發模型能力與架構的研究人員或開發人員。
    - *微調*（FT）Gemma 變體可視為 IT 變體，但通常訓練來執行特定任務，或在特定生成式 AI 基準測試上有良好表現。PaliGemma 變體家族包含多個 FT 變體。
    - *混合*（mix）Gemma 變體是 PaliGemma 模型的版本，以多種指示進行指令微調，適合一般用途。
- **參數** ：*建議選數量最少的*。一般來說，模型的參數越多，能力越強。然而，執行較大的模型需要更大、更複雜的運算資源，且通常會減慢 AI 應用程式的開發速度。除非您已確定較小的 Gemma 模型無法滿足需求，否則請選擇參數數量較小的模型。
- **量化等級：***建議使用半精度（16 位元），微調除外*。量化是個複雜的主題，歸結為資料的大小與精度，以及生成式 AI 模型用於計算與生成回應的記憶體用量。模型以高精度資料（通常為 32 位元浮點資料）訓練後，像 Gemma 這樣的模型可以修改為使用較低精度的資料，例如 16、8 或 4 位元。這些量化的 Gemma 模型在運算與記憶體資源顯著減少的情況下仍可表現良好，視任務複雜度而定。然而，微調量化模型的工具有限，您的 AI 開發框架中可能沒有提供。通常，您必須以全精度微調像 Gemma 這樣的模型，然後將結果模型量化。

關於 Google 發布的主要 Gemma 模型清單，請參閱
[開始使用 Gemma 模型](../get-started/index.md) 的 Gemma 模型清單。

## 3. 執行生成與推論請求

選定 AI 執行框架與 Gemma 變體後，您就可以
開始執行模型，並提示它生成內容或完成任務。
關於如何以特定框架執行 Gemma 的更多資訊，請參閱
[選擇框架](index.md#1-選擇框架)一節連結的指南。

### 提示格式

所有指令微調的 Gemma 變體都有特定的提示格式
要求。其中一些格式要求由您使用的框架自動處理，但當您直接向分詞器發送提示
資料時，必須加上特定的標籤，且標籤要求可能因您使用的 Gemma 變體而異。請參閱
以下指南，了解 Gemma 變體的提示格式與系統指示：

- [Gemma 提示與系統指示](../get-started/prompt-formatting.md)
- [PaliGemma 提示與系統指示](../models/paligemma/prompting.md)
- [FunctionGemma 格式與最佳實務](../models/functiongemma/formatting.md)

--8<-- "glossary.md"
