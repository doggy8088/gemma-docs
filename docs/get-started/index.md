# Get Started with Gemma

The Gemma family of open models includes a range of model sizes, capabilities, and task-specialized variations to help you build custom generative solutions.

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

## Next steps

Explore Gemma 4's core capabilities:

- [Basic Text Generation](capabilities/text.md)
- [Thinking & Reasoning](capabilities/thinking.md)
- [Function Calling](capabilities/function-calling.md)
- [Vision Understanding](capabilities/vision.md)
- [Audio Understanding](capabilities/audio.md)
