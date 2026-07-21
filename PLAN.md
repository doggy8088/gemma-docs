# Gemma docs port

I want to create zensical documentation (see https://github.com/zensical/zensical and https://zensical.org/docs/get-started/) out of the Gemma 4 documentation that can originally be found here: https://ai.google.dev/gemma/docs

The goal is to create a documentation experience that goes beyond merely communicating factual data. We want a developer experience that makes users want to use Gemma as much as possible. An experience that drives curiosity and makes newcomers feel welcome to the Gemma ecosystem. Likewise, it should give partners additional methods for sharing their content/blogs/demos/etc. 
What we are looking for is a platform that creates a developer experience that makes Gemma 4 an easy-to-use solution for all types of developers. With interactive tutorials, animated explainers, videos, demos, etc., the documentation should contain a wealth of information without being overwhelming. 

## ToC

The idea is to port (part off) the documentation over and create an entirely new interface / user experience for the documentation. I'm thinking of the following nav bar at the top:

* Home
* Getting Started
* Models
* Tutorials
* Deployment
* Tuning
* Community
* Tools

With the following subsections:

* Getting Started
  * Overview
  * Quickstart
  * https://ai.google.dev/gemma/docs/capabilities/audio
  * https://ai.google.dev/gemma/docs/capabilities/text/basic
  * https://ai.google.dev/gemma/docs/capabilities/text/function-calling-gemma4
  * https://ai.google.dev/gemma/docs/capabilities/vision
  * https://ai.google.dev/gemma/docs/capabilities/vision/image
  * https://ai.google.dev/gemma/docs/capabilities/vision/video
  * https://ai.google.dev/gemma/docs/capabilities/thinking
* Tutorials
  * A Visual Guide to Gemma 4: https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4
  * A Visual Guide to Gemma 4 12B: https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4-12b
  * A Visual Guide to MTP for Gemma 4: https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4
  * A Visual Guide to DiffusionGemma 4: https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-diffusiongemma
* Models
  * Gemma 4: https://ai.google.dev/gemma/docs/core/model_card_4
  * Gemma 3: https://ai.google.dev/gemma/docs/core/model_card_3
  * Gemma 2: https://ai.google.dev/gemma/docs/core/model_card_2
  * Gemma: https://ai.google.dev/gemma/docs/core/model_card
  * EmbeddingGemma: https://ai.google.dev/gemma/docs/embeddinggemma/model_card
  * DiffusionGemma: https://ai.google.dev/gemma/docs/diffusiongemma/model_card
  * FunctionGemma: https://ai.google.dev/gemma/docs/functiongemma/model_card
  * PaliGemma: https://ai.google.dev/gemma/docs/paligemma/model_card
  * ShieldGemma: https://ai.google.dev/gemma/docs/shieldgemma/model_card
* Deployment
  * LMStudio
  * llama.cpp
  * vLLM
  * LiteRT
  * SGLang
  * HuggingFace
  * Ollama
* Tuning
  * https://ai.google.dev/gemma/docs/tune
  * https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora
  * https://ai.google.dev/gemma/docs/core/huggingface_vision_finetune_qlora
  * https://ai.google.dev/gemma/docs/core/huggingface_text_full_finetune
* Community
  * https://deepmind.google/models/gemma/gemmaverse/
  * https://ai.google.dev/gemma/docs/discord
* Ecosystem
* Demos
  * Concurrent: https://github.com/google-gemma/cookbook/tree/main/apps/concurrent
* Tools
  * (V)RAM calculator
  * Model Selector
  * Gemma Skills: https://github.com/google-gemma/gemma-skills


For all the google.dev links you can simply append them with .md.txt to get the raw file. 

## Requirements



### Agent-first

The documentation should be easily accessible for any Agent that you point at it but in particular the Gemma 4 models (with E2B as a potential in-browser variant). 

* **Every page is a skill**  
  * Yaml frontmatter contains a summary (larger than what is found in llms.txt) but also metadata for the agent, like total number of tokens, code languages on the page, etc.  
  * Llms.txt is then a skill to traverse the website, so the agent knows how to load the frontmatter first of a given page before reading the entire thing

**NOTE**: The current documentation site does have:

* [https://ai.google.dev/gemma/docs/llms.txt](https://ai.google.dev/gemma/docs/llms.txt)  
* [https://ai.google.dev/gemma/docs/core.md.txt](https://ai.google.dev/gemma/docs/core.md.txt) 

* **Use Gemma to traverse the Docs**  
  * Using transformers.js, we can add a chatbot to the site that allows for easily finding information

### Techstack

* Techstack  
  * [Zensical](https://github.com/zensical/zensical), [ReadtheDocs](https://about.readthedocs.com/), etc.  
  * ReadtheDocs POC  
    * [https://gemma-cookbook.readthedocs.io/en/latest/](https://gemma-cookbook.readthedocs.io/en/latest/)  
    * [Testing branch](https://github.com/bebechien/gemma-cookbook/tree/read-the-doc)  
    * [Converting script](https://user.git.corp.google.com/bebechien/gemma-doc/+/refs/heads/main/convert_md.py) (devsite markdown \-\> read-the-docs format)  
    * [Juyeong Ji](mailto:bebechien@google.com) to explore ideal toc with this tool  
    * Theme? E.g., [Qwen](https://qwen.readthedocs.io/en/latest/)

### Tutorials / Demos

Demos come in various formats and would benefit from having an overview where users can easily track the cool demos and tutorials to play with. 

* Notebook tutorials  
* GitHub repositories  
* [Juyeong Ji](mailto:bebechien@google.com) turtle-gemma / AIventure to readthedocs  
* Interactive applications (HF space)  
* Interactive applications on the docs itself (js-based VRAM calculator for instance)  
  * Great example is: [https://www.philschmid.de/how-managed-agents-work](https://www.philschmid.de/how-managed-agents-work) 

### Resources

We’ve created many resources with various launches (including images and videos) that are not consolidated in a single place. We propose to add a page with:

* Visual Guides  
* Resources for “train the trainer” (see [resources](https://critique.corp.google.com/cl/946846636) and [page](https://gemma-ttt.com/))  
* Videos

### Ecosystem

The ecosystem is what drives adoption of Gemma. The current documentation does not have a single page giving an overview of partners, inference tools, etc. We want to include everyone that we love working with:

* Open source: HF, llama.cpp, MLX, etc.   
* 1P: …

### Navigation

As addressed below in more detail, we propose a different navigation mechanism that makes it easier to traverse the documentation for users with more clearly defined headers:

* Getting Started  
* Models  
* Tutorials  
* Deployment  
* Fine-tuning  
* Ecosystem  
* Tools

