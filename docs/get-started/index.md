---
title: 快速入門
description: 用一條指令完成第一次 Gemma 4 生成，再挑選適合您硬體的引擎、模型尺寸與檢查點。
models: [gemma-4]
frameworks: [transformers, llamacpp, ollama, mlx, vllm, lmstudio]
---

# 快速入門

<img class="page-mark" src="../assets/gemma-mark.png"
     alt="" width="132" height="132">

Gemma 是輕量且技術領先的開放模型家族，基於開發 [Gemini](https://deepmind.google/models/gemini/) 模型的相同研究與技術打造。Gemma 由 Google DeepMind 與 Google 其他團隊共同開發，名稱源自拉丁文 *gemma*，意為寶石。Gemma 模型權重搭配開發者工具，促進創新、協作，以及人工智慧（AI）的負責任使用。

Gemma 家族的最新成員是 [Gemma 4](../models/gemma-4/index.md)，支援多模態（文字、圖像、音訊）輸入。

!!! tip "第一次接觸 Gemma？"
    想深入了解 Gemma 模型家族，請造訪[模型家族](../models/index.md)頁面一覽。

## 選擇模型

Gemma 4 提供五種尺寸。它們共用相同的分詞器與提示格式，因此您可以從小尺寸開始，日後無需改寫任何內容即可升級。

<!-- Generated from data/models.yml -- edit that file, not this table. -->

| 尺寸 | 輸入 ➔ 輸出 | 上下文 | 適用平台 |
|---|---|---|---|
{% for size in modeldata.gemma4.sizes -%}
| **{{ size.name }}** | {{ size.modalities.input | map('capitalize') | join(', ') }} ➔ {{ size.modalities.output | map('capitalize') | join(', ') }} | {{ size.context }} | {{ size.platform }} |
{% endfor %}

## 執行第一次推論

=== "Hugging Face Transformers"
    安裝 [`transformers`](https://huggingface.co/docs/transformers/installation)，然後執行：

    ```python { .model-swap }
    from transformers import pipeline

    pipe = pipeline(
        "text-generation",
        model="google/gemma-4-12B-it",
        device_map="auto"
    )
    output = pipe("Explain quantum computing simply:", max_new_tokens=150)
    print(output[0]["generated_text"])
    ```

=== "llama.cpp"
    安裝 [`llama.cpp`](https://llama.app/)，然後執行：

    ```bash { .model-swap }
    llama-cli -hf google/gemma-4-12B-it-qat-q4_0-gguf --prompt "Write a poem about the Kraken."
    ```

=== "Ollama"
    安裝 [`ollama`](https://ollama.com/download)，然後執行：

    ```bash { .model-swap }
    ollama run gemma4:12b "Explain quantum computing simply:"
    ```

=== "Apple MLX"

    安裝 [`MLX LM`](https://github.com/ml-explore/mlx-lm)，然後執行：

    ```bash { .model-swap }
    mlx_lm.generate --model mlx-community/gemma-4-12B-it-qat-4bit \
        --prompt "Explain quantum computing simply:"
    ```

=== "vLLM"

    安裝 [`vLLM`](https://docs.vllm.ai/en/latest/getting_started/installation/)，然後執行：

    ```bash { .model-swap }
    vllm serve google/gemma-4-26B-A4B-it
    ```

=== "LMStudio"
    安裝 [`LMStudio`](https://lmstudio.ai/)，然後執行：

    ```bash { .model-swap }
    lms chat google/gemma-4-12b
    ```

!!! example "使用程式碼代理？"

    安裝 gemma-dev skill，讓您的代理始終跟上最新模型：

    ```bash
    npx skills add google-gemma/gemma-skills --skill gemma-dev --global
    ```

## 供應商

若想在其他堆疊上執行 Gemma 4，可從以下選項中選擇：

<!-- Generated from data/frameworks.yml -- edit that file, not this list. -->
{% set featured = ["ollama", "llamacpp", "lmstudio", "mlx", "litert", "transformers",
                   "keras", "unsloth", "vllm", "sglang", "gemini-api", "google-cloud"] %}
<ul class="engine-grid">
{% for id in featured %}{% for fw in frameworkdata.frameworks if fw.id == id %}
  <li><a class="engine" href="../{{ fw.page }}" title="{{ fw.blurb }}">
    <span class="engine-mark" aria-hidden="true">
      {%- if fw.get('icon') %}{{ icon(fw.icon, 'engine-logo') }}{% else %}{{ fw.mark }}{% endif -%}
    </span>
    <span class="engine-name">{{ fw.short_name | default(fw.name) }}</span>
  </a></li>
{% endfor %}{% endfor %}
  <li><a class="engine engine--all" href="../deployment/index.md">全部執行方式 →</a></li>
</ul>


## 取得權重

每種尺寸都提供全精度權重與[量化感知訓練（Quantization-Aware Training）](qat.md)（QAT）檢查點；在 4 位元下，QAT 檢查點的表現遠優於一般的訓練後量化。

| 哪裡 | 有什麼 |
|---|---|
| [Hugging Face][hf-gemma4] | Safetensors、GGUF 與 QAT 集合 |
| [Kaggle][kaggle-gemma4] | 相同的檢查點，鏡像版本 |
| [Ollama][ollama-gemma4] | 預先建置的標籤，首次執行時自動拉取 |

檢查點命名遵循 `google/gemma-4-{SIZE}-it{SUFFIX}` —— 請依引擎選擇適合的後綴：

| 執行環境 | 後綴 |
|---|---|
| llama.cpp, LM Studio, Ollama | `-qat-q4_0-gguf` |
| vLLM, SGLang | `-qat-w4a16-ct` |
| 行動裝置 | `-qat-mobile-transformers`, `-qat-mobile-ct` |


## 接下來

<div class="card-grid">
<a class="card" href="best-practices.md">
  <span class="card-icon">{{ icon('lucide/sliders-horizontal') }}</span>
  <div class="card-title">最佳做法</div>
  <p class="card-desc">取樣參數、思考模式詞元、模態順序與圖像詞元預算。</p>
</a>
<a class="card" href="capabilities/text.md">
  <span class="card-icon">{{ icon('lucide/type') }}</span>
  <div class="card-title">功能</div>
  <p class="card-desc">文字、推理、工具呼叫、視覺、影片與音訊——皆附可執行的程式碼。</p>
</a>
<a class="card" href="../models/index.md">
  <span class="card-icon">{{ icon('lucide/boxes') }}</span>
  <div class="card-title">模型家族</div>
  <p class="card-desc">每一代與每個變體，以及各自源自哪個基礎模型。</p>
</a>
</div>

--8<-- "glossary.md"
