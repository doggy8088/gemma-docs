---
hide:
  - toc
---

# Quick Start

The Gemma family of open models includes a range of model sizes, capabilities, and task-specialized variations to help you build custom generative solutions.

*The latest generation featuring multimodal (text, image, audio) inputs.*

| Size | Variant | Input ➔ Output | Intended Platform |
|---|---|---|---|
| **E2B** | [Gemma 4 (core)](https://ai.google.dev/gemma/docs/core) | Text, images, audio ➔ Text | Mobile devices |
| **E4B** | [Gemma 4 (core)](https://ai.google.dev/gemma/docs/core) | Text, images, audio ➔ Text | Mobile devices and laptops |
| **12B** | [Gemma 4 (core)](https://ai.google.dev/gemma/docs/core) | Text, images, audio ➔ Text | Laptops, desktop computers and small servers |
| **A4B** | [Gemma 4 (core)](https://ai.google.dev/gemma/docs/core) | Text, images ➔ Text | Desktop computers and small servers |
| **31B** | [Gemma 4 (core)](https://ai.google.dev/gemma/docs/core) | Text, images ➔ Text | Large servers or server clusters |

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

## Next steps

Explore Gemma 4's core capabilities:

- [Basic Text Generation](capabilities/text.md)
- [Thinking & Reasoning](capabilities/thinking.md)
- [Function Calling](capabilities/function-calling.md)
- [Image Understanding](capabilities/image.md)
- [Video Understanding](capabilities/video.md)
- [Audio Understanding](capabilities/audio.md)
