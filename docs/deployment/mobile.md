---
title: Deploy Gemma on mobile devices
description: This document outlines various methods and tools for deploying and running Gemma models on mobile devices, including using the Google AI Edge Gallery app and the MediaPipe LLM Inference API.
models: [gemma-4]
frameworks: [litert, android, ios]
source_url: "https://ai.google.dev/gemma/docs/integrations/mobile.md.txt"
---
# Deploy Gemma on mobile devices

This document outlines various methods and tools for deploying and running Gemma
models on mobile devices, including using the Google AI Edge Gallery app and the
MediaPipe LLM Inference API.

For information on converting a fine-tuned Gemma model to a LiteRT version, see
the [Conversion Guide](litert.md).

## Google AI Edge Gallery app

To see the LLM Inference APIs in action and test your Task Bundle model, you can
use the [Google AI Edge Gallery app](https://github.com/google-ai-edge/gallery).
This app provides a user interface for interacting with on-device LLMs,
allowing you to:

- **Import Models:** Load your custom `.task` models into the app.
- **Configure Parameters:** Adjust settings like temperature and top-k.
- **Generate Text:** Input prompts and view the model's responses.
- **Test Performance:** Evaluate the model's speed and accuracy.

For a detailed guide on how to use the Google AI Edge Gallery app, including
instructions for importing your own models, refer to the app's
[documentation](https://github.com/google-ai-edge/gallery/blob/main/README.md).

## MediaPipe LLM

You can run Gemma models on mobile devices with the
[MediaPipe LLM Inference API](https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference).
The LLM Inference API acts as a wrapper for large language models, enabling you
run Gemma models on-device for common text-to-text generation tasks like
information retrieval, email drafting, and document summarization.

The LLM Inference API is available on the following mobile platforms:

- [Android](https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference/android)
- [iOS](https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference/ios)

To learn more, refer to the
[MediaPipe LLM Inference documentation](https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference).
