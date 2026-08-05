/* GENERATED FILE -- DO NOT EDIT.
   Written by scripts/generate_model_names.py from data/models.yml.
   Verified by: uv run python scripts/lint_docs.py --check generated

   Consumed by docs/javascripts/model-switcher.js. A model with no id for an engine
   has no published build there -- the switcher must leave such snippets alone
   rather than invent a repository name. */
window.GEMMA_MODELS = {
  "gemma-4/e2b": {
    "name": "E2B",
    "fullName": "Gemma 4 E2B",
    "context": "128K",
    "platform": "Mobile devices",
    "ids": {
      "hf": "google/gemma-4-E2B-it",
      "hf_base": "google/gemma-4-E2B",
      "hf_qat": "google/gemma-4-E2B-it-qat-w4a16-ct",
      "gguf": "ggml-org/gemma-4-E2B-it-GGUF",
      "gguf_qat": "google/gemma-4-E2B-it-qat-q4_0-gguf",
      "ollama": "gemma4:e2b",
      "mlx": "mlx-community/gemma-4-E2B-it-qat-4bit"
    },
    "inputs": [
      "text",
      "image",
      "audio",
      "video"
    ]
  },
  "gemma-4/e4b": {
    "name": "E4B",
    "fullName": "Gemma 4 E4B",
    "context": "128K",
    "platform": "Mobile and laptops",
    "ids": {
      "hf": "google/gemma-4-E4B-it",
      "hf_base": "google/gemma-4-E4B",
      "hf_qat": "google/gemma-4-E4B-it-qat-w4a16-ct",
      "gguf": "ggml-org/gemma-4-E4B-it-GGUF",
      "gguf_qat": "google/gemma-4-E4B-it-qat-q4_0-gguf",
      "ollama": "gemma4:e4b",
      "mlx": "mlx-community/gemma-4-E4B-it-qat-4bit"
    },
    "inputs": [
      "text",
      "image",
      "audio",
      "video"
    ]
  },
  "gemma-4/12b": {
    "name": "12B",
    "fullName": "Gemma 4 12B",
    "context": "256K",
    "platform": "Laptops and desktops",
    "ids": {
      "hf": "google/gemma-4-12B-it",
      "hf_base": "google/gemma-4-12B",
      "hf_qat": "google/gemma-4-12B-it-qat-w4a16-ct",
      "gguf": "ggml-org/gemma-4-12B-it-GGUF",
      "gguf_qat": "google/gemma-4-12B-it-qat-q4_0-gguf",
      "ollama": "gemma4:12b",
      "mlx": "mlx-community/gemma-4-12B-it-qat-4bit"
    },
    "inputs": [
      "text",
      "image",
      "audio",
      "video"
    ]
  },
  "gemma-4/26b-a4b": {
    "name": "26B A4B",
    "fullName": "Gemma 4 26B A4B",
    "context": "256K",
    "platform": "Desktops and servers",
    "ids": {
      "hf": "google/gemma-4-26B-A4B-it",
      "hf_base": "google/gemma-4-26B-A4B",
      "gguf": "ggml-org/gemma-4-26B-A4B-it-GGUF",
      "gguf_qat": "google/gemma-4-26B-A4B-it-qat-q4_0-gguf",
      "ollama": "gemma4:26b",
      "mlx": "mlx-community/gemma-4-26B-A4B-it-qat-4bit"
    },
    "inputs": [
      "text",
      "image",
      "video"
    ]
  },
  "gemma-4/31b": {
    "name": "31B",
    "fullName": "Gemma 4 31B",
    "context": "256K",
    "platform": "Servers",
    "ids": {
      "hf": "google/gemma-4-31B-it",
      "hf_base": "google/gemma-4-31B",
      "hf_qat": "google/gemma-4-31B-it-qat-w4a16-ct",
      "gguf": "ggml-org/gemma-4-31B-it-GGUF",
      "gguf_qat": "google/gemma-4-31B-it-qat-q4_0-gguf",
      "ollama": "gemma4:31b",
      "mlx": "mlx-community/gemma-4-31B-it-qat-4bit"
    },
    "inputs": [
      "text",
      "image",
      "video"
    ]
  }
};
