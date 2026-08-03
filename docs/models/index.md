# Gemma Model Family

The Gemma family of open models spans multiple generations and specialized variants.

## Gemma 4 (Latest)

Gemma 4 spans four distinct architectures:

<!-- Generated from data/models.yml -- edit that file, not this table. -->

| Variant | Parameters | Input | Context | Intended Platform |
|---|---|---|---|---|
{% for size in models.gemma4.sizes -%}
| **{{ size.name }}** | {{ size.parameters }} | {{ size.modalities.input | map('capitalize') | join(', ') }} | {{ size.context }} | {{ size.platform_short }} |
{% endfor %}
!!! note "Audio and video support"
    Audio input is native to **E2B, E4B and 12B**. The **26B A4B** and **31B** models accept
    video as individual frames — that is, as images — but do not process audio.

## Previous Generations

- [Gemma 3](gemma-3/index.md) — Core text and image models (270M to 27B)
- [Gemma 2](gemma-2/index.md) — Text models (2B, 9B, 27B)
- [PaliGemma](paligemma/index.md) — Vision-language models

## Specialized Models

- [DiffusionGemma](diffusiongemma/index.md) — Diffusion-based text generation
- [EmbeddingGemma](embeddinggemma/index.md) — Text embedding models
- [ShieldGemma](shieldgemma/index.md) — Safety classification models

## Research Models

- [RecurrentGemma](recurrentgemma/index.md) — Recurrent architecture models based on Griffin
- [DataGemma](datagemma/index.md) — Models for grounding generation in real-world data
- [Gemma Scope](gemma-scope/index.md) — Interpretability tools for understanding Gemma model internals

For full technical details, see the individual model pages linked above or the [Gemma 4 model card](gemma-4/model-card.md).


