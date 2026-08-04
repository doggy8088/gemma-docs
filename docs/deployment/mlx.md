---
title: Apple MLX
description: Run Gemma 4 on Apple silicon with MLX — text and vision generation, model repos, and the OpenAI-compatible server.
---

# Run Gemma with MLX

[MLX](https://github.com/ml-explore/mlx) is Apple's array framework for machine learning on
Apple silicon. It runs Gemma natively on the unified memory architecture, without a GGUF
conversion step.

## Install

```bash
pip install mlx mlx-lm mlx-vlm
```

## Run

=== "Text"

    ```bash
    mlx_lm.generate --model mlx-community/gemma-4-e2b-it-4bit --prompt "Who are you?"
    ```

=== "Vision"

    ```bash
    mlx_vlm.generate --model mlx-community/gemma-4-e2b-it-4bit \
      --prompt "Describe this image." --image <path_to_image>
    ```

## Model repositories

Community-converted weights live under
[mlx-community](https://huggingface.co/mlx-community):

| Size | Repository |
|---|---|
| E2B | `mlx-community/gemma-4-e2b-it-4bit` |
| E4B | `mlx-community/gemma-4-e4b-it-qat-OptiQ-4bit` |
| 12B | `mlx-community/gemma-4-12B-it-4bit`, `…-it-8bit` |
| 26B A4B | `mlx-community/gemma-4-26b-a4b-it-4bit` |

!!! note
    Coverage and naming vary — these are community conversions, not first-party releases.
    Search [mlx-community for Gemma 4](https://huggingface.co/mlx-community?search=gemma-4) for
    the current list. [Ollama](ollama.md) also ships MLX builds for every size under its `-mlx`
    tags, if you would rather not pick a repository.

## Server

```bash
mlx_vlm.server --model mlx-community/gemma-4-e2b-it-4bit --port 8080
```

OpenAI-compatible API at `http://localhost:8080/v1`.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="mlx-community/gemma-4-e2b-it-4bit",
    messages=[{"role": "user", "content": "Say this is a test"}],
)
print(response.choices[0].message.content)
```
