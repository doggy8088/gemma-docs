---
title: Gemma 模型家族
description: 認識 Gemma 模型家族：介紹 Gemma 是什麼、各世代之間的演進關係，以及每個變體衍生自哪個基礎模型。
---

# Gemma 模型家族

Gemma 是 Google DeepMind 的**開放模型**家族：權重已公開，您可以下載並在自己的硬體上運行——筆記型電腦、手機，或您控制的叢集。除非您選擇傳送，否則不會有任何資料傳到伺服器。

此家族有兩部分。**主流世代**——Gemma 1 到 Gemma 4——是大多數人想要的通用模型。每個世代周圍有一組**變體**：為單一任務重新訓練或重構的相同架構，例如嵌入、安全分類或視覺語言任務。變體總是衍生自某個特定世代，下方的樹狀圖即呈現此關係。

## 家族結構

{% set generations = modeldata.families | selectattr('generation', 'defined') | sort(attribute='generation') %}
{% set independent = modeldata.families | selectattr('based_on', 'defined')
                                        | selectattr('based_on', 'none') | list %}

<!-- Generated from data/models.yml -- edit that file, not this tree. -->
{% set counts = [] %}
{% for gen in generations %}{% set _ = counts.append(
     modeldata.families | selectattr('based_on', 'defined')
                        | rejectattr('based_on', 'none')
                        | selectattr('based_on', 'equalto', gen.id) | list | length) %}{% endfor %}
<div class="model-tree">
<ol class="tree-spine" style="--tree-rows: {{ counts | max + 1 }}">
{% for gen in generations %}
{% set kids = modeldata.families | selectattr('based_on', 'defined')
                                 | rejectattr('based_on', 'none')
                                 | selectattr('based_on', 'equalto', gen.id) | list %}
  <li class="tree-gen">
    <a class="tree-node tree-node--gen" href="../{{ gen.page }}">
      <span class="tree-gen-name">{{ gen.name }}</span>
      <span class="tree-gen-note">{{ gen.tagline }}</span>
    </a>
    {% if kids %}
    <ul class="tree-branch">
    {% for kid in kids %}
      <li><a class="tree-node" href="../{{ kid.page }}">
        <span class="tree-name">{{ kid.name }}</span>
        <span class="tree-tagline">{{ kid.tagline }}</span>
      </a></li>
    {% endfor %}
    </ul>
    {% else %}
    <p class="tree-none">此世代沒有衍生變體。</p>
    {% endif %}
  </li>
{% endfor %}
</ol>

{% if independent %}
<ul class="tree-aside">
{% for fam in independent %}
  <li><a class="tree-node" href="../{{ fam.page }}">
    <span class="tree-name">{{ fam.name }}</span>
    <span class="tree-tagline">{{ fam.tagline }}</span>
  </a></li>
{% endfor %}
</ul>
{% endif %}
</div>

<p class="tree-legend">
每個變體都歸在其模型頁面所標示的基礎世代之下。
{% if independent %}虛線下方的模型是 Gemma 模型，但其頁面描述的是不同架構而非 Gemma 世代。{% endif %}
</p>

## Gemma 4

目前的世代；除非您需要某個變體的特定功能，否則請從這裡開始。

<!-- Generated from data/models.yml -- edit that file, not this table. -->

| 變體 | 參數量 | 輸入 | 上下文 | 目標平台 |
|---|---|---|---|---|
{% for size in modeldata.gemma4.sizes -%}
| **{{ size.name }}** | {{ size.parameters }} | {{ size.modalities.input | map('capitalize') | join(', ') }} | {{ size.context }} | {{ size.platform_short }} |
{% endfor %}
!!! note "音訊與影片支援"
    音訊輸入為 **E2B、E4B 與 12B** 的原生功能。**26B A4B** 與 **31B** 模型可將影片視為逐幀影像（即圖像）接受，但不處理音訊。

[立即執行](../get-started/index.md)或閱讀 [Gemma 4 模型卡](gemma-4/model-card.md)以了解完整技術細節。

## 先前世代

仍在支援中；當您只需要較小或純文字模型時，它們仍是合適的選擇。

<!-- Generated from data/models.yml -- edit that file, not these tables. -->
{% set n3 = modeldata.families | selectattr('id', 'equalto', 'gemma-3n') | list | first %}
### {{ n3.name }}

Gemma 3 世代的裝置端成員。

| 大小 | 輸入 ➔ 輸出 | 目標平台 |
|---|---|---|
{% for size in n3.sizes -%}
| **{{ size.name }}** | {{ size.input | capitalize }} ➔ Text | {{ size.platform }} |
{% endfor %}

{% set generations = modeldata.families | selectattr('generation', 'defined')
                                        | sort(attribute='generation', reverse=true) %}
{% for gen in generations if gen.generation != 4 %}
{% set sized = (gen.sizes | default([])) | selectattr('platform', 'defined') | list %}
{% if sized %}
### {{ gen.name }}

| 大小 | 輸入 ➔ 輸出 | 目標平台 |
|---|---|---|
{% for size in sized -%}
| **{{ size.name }}** | {{ size.input | capitalize }} ➔ Text | {{ size.platform }} |
{% endfor %}
{% endif %}
{% endfor %}


## 如何選擇

- **剛入門？** 選擇目前世代的指令微調版本，挑選符合您硬體的最小尺寸。它無需額外訓練即可回答各種提示。
- **需要完成特定任務？** 先確認是否已有變體可勝任——嵌入（[EmbeddingGemma](embeddinggemma/index.md)）、安全標註（[ShieldGemma](shieldgemma/index.md)）、工具呼叫（[FunctionGemma](functiongemma/index.md)）——再考慮自行微調核心模型。
- **在手機或邊緣裝置上？** [Gemma 3n](gemma-3n/index.md) 與 Gemma 4 E2B、E4B 尺寸正是為此打造。

如需所有版本的完整發布時程，請見[版本發布紀錄](releases.md)。

--8<-- "glossary.md"
