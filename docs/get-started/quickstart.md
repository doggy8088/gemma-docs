---
hide:
  - toc
---

# Quick Start

The Gemma family of open models includes a range of model sizes, capabilities, and task-specialized variations to help you build custom generative solutions.

*The latest generation featuring multimodal (text, image, audio) inputs.*

<!-- Generated from data/models.yml -- edit that file, not this table. -->

| Size | Variant | Input ➔ Output | Context | Intended Platform |
|---|---|---|---|---|
{% for size in models.gemma4.sizes -%}
| **{{ size.name }}** | [Gemma 4 (core)](../models/gemma-4/index.md) | {{ size.modalities.input | map('capitalize') | join(', ') }} ➔ {{ size.modalities.output | map('capitalize') | join(', ') }} | {{ size.context }} | {{ size.platform }} |
{% endfor %}

!!! tip "New to Gemma?"
    A good place to start is the Gemma 4 12B model, which can be used for many tasks and runs on laptop hardware.

## Download models

You can download Gemma models from:

- [Kaggle](https://www.kaggle.com/models?query=gemma-4&publisher=google)
- [Hugging Face](https://huggingface.co/collections/google/gemma-4)

## Run your first inference

=== "Hugging Face Transformers"

    ```bash
    pip install --upgrade transformers accelerate
    ```

    Then use:

    ```python
    from transformers import pipeline

    pipe = pipeline(
        "text-generation",
        model="google/gemma-4-12b",
        device_map="auto"
    )
    output = pipe("Explain quantum computing simply:", max_new_tokens=150)
    print(output[0]["generated_text"])
    ```

=== "Ollama"

    ```bash
    ollama run gemma4:12b "Explain quantum computing simply:"
    ```

=== "Apple MLX"

    ```bash
    pip install mlx-lm
    mlx_lm.generate --model mlx-community/gemma-4-12b-4bit \
        --prompt "Explain quantum computing simply:"
    ```

!!! example "Using a coding agent?"

    Install the gemma-dev skill so your agent stays current with the latest models:

    ```bash
    npx skills add google-gemma/gemma-skills --skill gemma-dev --global
    ```

