---
title: Master Local Fine-Tuning with gemma-trainer
description: Take control of your AI models with our newest skill, designed to make local fine-tuning efficient.
models: [gemma-4]
frameworks: [unsloth, transformers]
source_url: "https://bebechien.github.io/cozy-corner-future/posts/master-local-fine-tuning-with-gemma-trainer/"
---

# Master Local Fine-Tuning with `gemma-trainer`

!!! note
    Originally published on
    [Cozy Corner Future](https://bebechien.github.io/cozy-corner-future/posts/master-local-fine-tuning-with-gemma-trainer/).

Remember back in May when I introduced the [`gemma-skills`](https://github.com/google-gemma/gemma-skills) repository? It's been rewarding to see how many of you have used [my previous post](https://bebechien.github.io/cozy-corner-future/posts/a-warm-welcome-to-gemma-skills/) to streamline your workflows. (And hey, even if we aren't swimming in GitHub stars yet, I think we're off to a great start! 😉)

But as I built more custom applications, I kept hitting the same roadblock: **how to take a great base model and adapt it to my specific needs.**

Fine-tuning a model usually requires wading through complex setups and confusing guides. To make this process straightforward and quick, we created our newest skill: **`gemma-trainer`**.

## What is `gemma-trainer`?

`gemma-trainer` is your blueprint for training and adapting Gemma models on your local hardware. It handles the "how-to" so you can focus on your specific project goals, whether you are teaching a model a new domain or aligning its behavior to your preferences.

## Why You'll Use It

- **Faster, Lighter Training**: We recommend using **[Unsloth](http://unsloth.ai/)** for single-GPU training, making it fast and using less memory so it runs easily on personal hardware.
- **Three Key Methods**: It guides you through Supervised Fine-Tuning (SFT) to teach new info, Direct Preference Optimization (DPO) to align with preferences, and Reward Modeling (RM) to rate responses.
- **Teach Models to See and Hear**: It includes clear instructions for training models with images and audio (multimodal learning) alongside text.
- **Run Anywhere**: Quickly convert your models to lightweight formats (like GGUF) and run them on mobile or smart devices (IoT) using **[LiteRT-LM](https://developers.google.com/edge/litert-lm/models/gemma-4)**.
- **Up-to-Date Best Practices**: The skill is continuously updated with the latest optimized settings and training techniques, ensuring you're always using the best methods.

## Practical Use Case

To see this in action, recall how we turned Gemma 4 into an expert translator for Classical Korean literature in [my previous post](https://bebechien.github.io/cozy-corner-future/posts/turning-gemma-4-into-an-old-korean-translator/). With `gemma-trainer`, you don't need to manually piece together a pipeline. You can simply ask your agent:

> **"Fine-tune Gemma 4 E2B on the dataset bebechien/HongGildongJeon."**

With the `gemma-trainer` skill, your agent will partner with you to:

1. **Verify your data**: Use the validation script to ensure your training data matches template requirements.
2. **Set up parameters**: Select the best LoRA settings to teach the model linguistic nuances without running out of video memory (VRAM).
3. **Run the training**: Launch the training session using optimized, resource-efficient defaults.
4. **Evaluate and iterate**: Review the model's performance and adjust settings to get the exact results you need.

Here is an example showing the agent starting a fine-tuning run on a Gemma 4 12B model for audio tasks:

![audio-tuning start](../assets/tuning/gemma-trainer-audio-tuning-start.png)

Once configured, the agent kicks off the training process using your designated dataset:

![audio-tuning training](../assets/tuning/gemma-trainer-audio-tuning-training.png)

Even if you make a mistake, the agent has your back. For instance, when I accidentally requested training a Gemma 4 31B model (which is a text-and-vision model and has no audio capability), it suggested using Gemma 4 E2B or 12B for audio tuning instead:

![audio-tuning fix](../assets/tuning/gemma-trainer-audio-tuning-fix.png)

Once training is complete, the agent presents the results and outlines the next steps:

![audio-tuning finish](../assets/tuning/gemma-trainer-audio-tuning-finish.png)

You can also ask your agent to write a custom evaluation script based on your specific requirements. In this case, I asked the agent to create a script that checks transcription similarity:

![audio-tuning eval](../assets/tuning/gemma-trainer-audio-tuning-eval.png)

Finally, you will receive a comprehensive report summarizing the training performance, making it clear where you can make improvements in the next run:

![audio-tuning report](../assets/tuning/gemma-trainer-audio-tuning-report.png)

## Let's try!

`gemma-trainer` is a living, structured document. Drop it into your agent's skills directory, and your AI assistant will immediately know how to guide you through the process.

Check out the [repository](https://github.com/google-gemma/gemma-skills), add the skill to your toolbox, and let's build something amazing!

Thanks for reading and happy training!
